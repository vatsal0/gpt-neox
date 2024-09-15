# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2023 MegaBlocks authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu import get_model_parallel_group, get_model_parallel_rank, get_data_parallel_group
import megablocks.ops

from .sparsemixer import sparsemixerv2_routing, MoEAuxLossAutoScaler

class Router(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
    ):
        super().__init__()
        self.jitter_eps = neox_args.moe_jitter_eps
        self.top_k = neox_args.moe_top_k

        self.layer = torch.nn.Linear(
            neox_args.hidden_size,
            neox_args.moe_num_experts,
            bias=False,
            dtype=neox_args.params_dtype,
            device=torch.cuda.current_device(),
        )
        self.router_type = neox_args.moe_router_type
        self.neox_args = neox_args
        
        self.moe_aux_loss_coeff = neox_args.moe_aux_loss_coeff / neox_args.gradient_accumulation_steps
        init_method(self.layer.weight)

        self.num_experts = neox_args.moe_num_experts
        self.router_type = neox_args.moe_router_type
        self.tokens_per_batch = neox_args.seq_length * neox_args.train_micro_batch_size_per_gpu
        self.expert_parallel_group = get_model_parallel_group()
        self.expert_parallel_rank = get_model_parallel_rank()
        self.data_parallel_group = get_data_parallel_group()
        self.reset_logging_buffers()

    def jitter(self, x):
        low = 1.0 - self.jitter_eps
        high = 1.0 + self.jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)
    
    def _top_k(self, scores):
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)
    
    def switch_load_balancing_loss_func(self,
        probs: torch.Tensor, tokens_per_expert: torch.Tensor, topk: int, moe_aux_loss_coeff: float
    ):
        num_tokens = probs.shape[0]
        num_experts = probs.shape[1]

        aggregated_probs_per_expert = probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
        )
        return aux_loss

    def apply_load_balancing_loss(
            self,
            probs: torch.Tensor,
            num_local_tokens_per_expert: torch.Tensor,
            activation: torch.Tensor,
        ):
            aux_loss = self.switch_load_balancing_loss_func(
                probs, num_local_tokens_per_expert, self.top_k, self.moe_aux_loss_coeff
            )
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
            
            return activation, aux_loss

    def forward(self, x, router_type_override=None):
        router_type = router_type_override
        
        if router_type is None:
            router_type = self.router_type
        
        if self.training and self.jitter_eps is not None:
            x = x * self.jitter(x)

        logits = self.layer(x.view(-1, x.shape[-1]))
        
        if router_type == "topk" or router_type == "dense_approx" or router_type == "dense_approx_lsh" or router_type == "dense_approx_efficient" or router_type == "expert_prob_approx":
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)
        
        elif router_type == "sparsemixer":
            expert_weights, scores, expert_indices = sparsemixerv2_routing(logits, self.top_k, self.jitter_eps, self.training)

        elif router_type == "dense":
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = torch.topk(scores, self.num_experts, dim=-1)

        else:
            raise ValueError(f"Invalid MoE Router type {router_type}")
        
        with torch.no_grad():
            expert_indices_ft = expert_indices.flatten()
            tokens_per_expert = megablocks.ops.histogram(expert_indices_ft, self.num_experts)
        expert_weights, lbl_loss = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=expert_weights)
        lbl_loss_temp = lbl_loss.detach()
        global_lbl_loss = torch.distributed.all_reduce(
                lbl_loss_temp,
                group=self.data_parallel_group,
                op=torch.distributed.ReduceOp.SUM,
                async_op=True,
            )

        if router_type_override is None:
          # only do this for the actual forward pass
          routing_counts = torch.bincount(expert_indices.squeeze(1).flatten(),minlength=self.num_experts)
          global_routing_counts = torch.distributed.all_reduce(
                  routing_counts,
                  group=self.data_parallel_group,
                  op=torch.distributed.ReduceOp.SUM,
                  async_op=True,
              )

        global_lbl_loss.wait()
        if self.expert_parallel_rank == 0 and self.training:
            self.save_lbl_loss(lbl_loss_temp / (self.neox_args.global_num_gpus * self.moe_aux_loss_coeff))
        
        if router_type_override is None:
            global_routing_counts.wait()
            if self.expert_parallel_rank == 0 and self.training:
                self.save_routing_counts_train(routing_counts)

        return expert_weights, expert_indices, scores

    def reset_logging_buffers(self):
        self.train_routing_count_buffer = None
        self.lbl_loss_buffer = None
        self.z_loss_buffer = None

    def save_routing_counts_train(self, routing_counts):
        if self.train_routing_count_buffer == None:
            self.train_routing_count_buffer = routing_counts.unsqueeze(0)
        else:
            self.train_routing_count_buffer = torch.cat([self.train_routing_count_buffer,
                                                            routing_counts.unsqueeze(0)],
                                                            dim=0)

    def save_lbl_loss(self, new_loss):
        if self.lbl_loss_buffer == None:
            self.lbl_loss_buffer = new_loss.unsqueeze(0)
        else:
            self.lbl_loss_buffer = torch.cat([self.lbl_loss_buffer,
                                                new_loss.unsqueeze(0)],
                                                dim=0)