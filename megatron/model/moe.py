# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2023 MegaBlocks authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional

import megablocks.ops
import numpy as np
import torch

from megatron import mpu
from megatron.mpu import get_expert_token_counts_for_rank
from megatron.mpu import get_expert_tokens_for_rank
from megatron.mpu import copy_to_expert_model_parallel_region
from megatron.mpu import gather_from_expert_model_parallel_region
from megatron.neox_arguments.arguments import NeoXArgs

from .moe_mlp import ParallelGroupedLLaMAMLP, ParallelGroupedMLP
from .router import Router

from .lsh_triton import launch_lsh_approximation_kernel
print('start compiling')
kernel = torch.compile(launch_lsh_approximation_kernel)
print('done compiling')

from xformers.components.attention import ScaledDotProduct

class ParallelDroplessMLP(torch.nn.Module):
    """
    This class defines MoE expert computation, using tensor (model) parallel size as the expert parallel size

    The implication of this parallelism decision is that the expert weights can only be sharded within a single node
    """

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
    ):
        """

        Bias is currently not supported
        """
        super(ParallelDroplessMLP, self).__init__()

        # Calculate the number of experts to allocate on this rank
        world_size = mpu.get_model_parallel_world_size()
        assert neox_args.moe_num_experts % world_size == 0
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = self.num_experts // world_size
        self.top_k = neox_args.moe_top_k

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # decide which parallel grouped MLP implementation to use
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelGroupedMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        elif neox_args.mlp_type == "llama":
            self.mlp = ParallelGroupedLLaMAMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        else:
            raise KeyError(neox_args.mlp_type)

        self.total_approx_count = 0
        self.buffer = None
        self.scores = None
        self.approx = None

    def indices_and_bins(self, top_expert: torch.Tensor):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = megablocks.ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = megablocks.ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = megablocks.ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
        self,
        input_: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
        router_type: str,
        queries=None,
        keys=None
    ):
        """
        grouped_permute_and_compute

        torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)

        NOTE: Megablocks sets up all MLP tensors as column parallel and uses transposes on some of the grouped_gemm calls for the ops that would be row parallel. This seems to be fine and since we aren't using the underlying NeoX ColumnParallelLinear and RowParallelLinear classes, there doesn't seem to be a reason to change it...because that'd introduce a lot of additional complexity.

        column parallel linear forward

        ```python
        def forward(self, input_):
            if self.use_mup and self.mup_rescale_parameters:
                input_ /= self.width_mult()
            # Set up backprop all-reduce.
            input_parallel = copy_to_model_parallel_region(input_)
            # Matrix multiply.

            bias = self.bias if not self.skip_bias_add else None
            output_parallel = F.linear(input_parallel, self.weight, bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = gather_from_model_parallel_region(output_parallel)
            else:
                output = output_parallel
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias
        ```
        """
        # Route the tokens for MoE computation.
        ## stack (sl, bs, hs) into (sl * bs, hs)
        seq_len, batch_size = input_.shape[0], input_.shape[1]
        input_ = input_.view(-1, input_.shape[-1])

        ## repeat each token top_k times and shuffle tokens to group them by their respective experts
        input_ = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)

        # get tokens routed to this rank's experts only
        input_parallel = copy_to_expert_model_parallel_region(input_, tokens_per_expert)

        # get tokens_per_expert for this rank's experts only
        # with torch.no_grad():
        local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)
        # if torch.cuda.current_device() == 0:
        #     print(f"{torch.cuda.current_device()}: local_tokens_per_expert {local_tokens_per_expert}, global tokens {tokens_per_expert}")

        # Perform the expert computation for this rank's experts
        output_parallel = self.mlp(input_parallel, local_tokens_per_expert)

        # all gather masked results from across Tensor parallel ranks here and cat them together
        # this will replicate the calculation of each expert across all ranks
        # NOTE: this combined all_gather and torch.cat operation is performed by gather_from_model_parallel_region(output_parallel)
        # Unlike ColumnParallelLinear, it is nonsensical in the MoE world
        # to optionally return the output_parallel result...we still have to scatter the tokens back to their original positions
        output = gather_from_expert_model_parallel_region(
            output_parallel,
            tokens_per_expert,
        )
        # e.g. indices 0, 1, 2, 3 will all correspond to input 0 if top_k = 4
        input_indices = indices // top_k
        if self.buffer is None:
            self.buffer = torch.zeros((seq_len * batch_size * self.num_experts, output.size(-1))).to(output.device, dtype=output.dtype)
        self.buffer.zero_()

        if self.scores is None:
            self.scores = torch.zeros(self.num_experts, seq_len * batch_size // 64, seq_len * batch_size // 64).to(input_.device, dtype=input_.dtype)
        self.scores.zero_()
        if self.approx is None:
            self.approx = torch.zeros(self.num_experts, seq_len * batch_size // 64, output.size(-1)).to(output.device, dtype=output.dtype)
        self.approx.zero_()

        if router_type == "dense_approx_lsh":
            self.buffer = self.buffer.view(
                seq_len * batch_size, 
                self.num_experts, 
                output.size(-1)
            )

            nbits = 10
            # the example here samples uniform [-0.5, 0.5] https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/
            lsh_vectors = torch.rand((input_.shape[-1], nbits), dtype=input_.dtype, device=input_.device) - 0.5
            hash_vectors = torch.matmul(input_.detach(), lsh_vectors) > 0

            # since nbits is small, let's use tensor operations to turn each unique binary vector into an integer representation
            exponents = 2 ** torch.arange(nbits - 1, -1, -1).to(hash_vectors.device)
            hashes = torch.sum(exponents * hash_vectors, -1)

            # get buckets for all vectors
            bucket_ids, bucket_indices = megablocks.ops.sort(hashes, nbits)
            bucket_counts = megablocks.ops.histogram(hashes, 2**nbits)
            bucket_ends = megablocks.ops.inclusive_cumsum(bucket_counts, 0)

            scale = np.sqrt(input_.shape[-1])
            # print((bucket_counts**2).sum(), flush=True)
            for i in range(2**nbits):
                bucket_end = bucket_ends[i]
                bucket_size = bucket_counts[i]
                bucket_start = bucket_end - bucket_size
                this_bucket_indices = bucket_indices[bucket_start : bucket_end]

                if bucket_size > self.scores.shape[1]:
                    this_bucket_indices = this_bucket_indices[torch.randperm(bucket_size)[:self.scores.shape[1]]]
                    bucket_size = self.scores.shape[1]

                this_bucket_input = input_[this_bucket_indices].detach()
                this_bucket_expert_indices = bin_ids[this_bucket_indices]
                this_bucket_output = output[this_bucket_indices].detach()

                torch.matmul(this_bucket_input / scale, this_bucket_input.T / scale, out=self.scores[0, :bucket_size, :bucket_size])
                # zero out similarities between same vector, or any below threshold
                threshold = 1 - 2 / self.num_experts
                mask = torch.eye(bucket_size, dtype=torch.bool, device=self.scores.device).logical_or(self.scores[0, :bucket_size, :bucket_size] < threshold)
                self.scores[0, :bucket_size, :bucket_size].masked_fill_(mask, 0)

                self.scores[1:].copy_(self.scores[0])
                # broadcasting to nxn matrices, of all 0, all 1, ... all num_experts-1. each entry represents the expert
                expert_range = torch.arange(self.num_experts, device=self.scores.device).view(-1, 1, 1)
                # for element i of this tensor, we want approximations for expert i, for tokens not routed to expert i, using similarities between tokens to expert i
                # so if an entry's row matches the corresponding input's expert index (we don't want its value)
                # or if an entry's column doesn't match the corresponding input's expert index (we don't want to use its score)
                # set it to 0
                # two masked fills with num_experts x n x 1 and num_experts x 1 x n avoids materializing an n^2 mask
                self.scores[:, :bucket_size, :bucket_size].masked_fill_(expert_range.eq(this_bucket_expert_indices.unsqueeze(1)), 0)
                self.scores[:, :bucket_size, :bucket_size].masked_fill_(expert_range.ne(this_bucket_expert_indices.unsqueeze(0)), 0)

                # matmul (which does a weighted sum), then divide by counts (to make it a weighted average)
                score_counts = (self.scores > 0).sum(dim=-1)

                torch.matmul(self.scores[:, :bucket_size, :bucket_size], this_bucket_output, out=self.approx[:, :bucket_size])
                self.approx[score_counts > 0] /= score_counts[score_counts > 0].unsqueeze(1)
                self.total_approx_count += (score_counts > 0).sum()

                self.buffer.index_add_(0, this_bucket_indices, self.approx[:, :bucket_size].transpose(0, 1))
        else:
          # ith element of output will be added to index corresponding to input index, and associated expert
          self.buffer.index_add_(dim=0, index=self.num_experts * input_indices + bin_ids, source=output.detach())

        # Un-route the data for the MoE output
        return megablocks.ops.scatter(
            output,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
        ), self.buffer.view(
            seq_len * batch_size, 
            self.num_experts, 
            output.size(-1)
        )

    def forward(self, x, expert_weights, expert_indices, queries=None, keys=None, router_type=None):
        """
        grouped_forward_once

            x: [sl, bs, hs]
            expert_weights: [sl * bs, top-k]
            expert_indices: [sl * bs, top-k]
        """
        # save shape so we can re-shape the outputs later
        in_shape = x.size()

        # both are now (sl * bs * top_k)
        expert_weights = expert_weights.flatten()
        expert_indices = expert_indices.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(
                expert_indices
            )

        x, expert_output = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k if router_type != "dense" else self.num_experts,
            router_type,
            queries=queries,
            keys=keys
        )

        # restore input shape
        x = x.view(in_shape)
        return x, expert_output


def cast_if_autocast_enabled(tensor: torch.Tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class ParallelDroplessMoE(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
    ):
        super(ParallelDroplessMoE, self).__init__()

        self.router = Router(neox_args, init_method)

        self.experts = ParallelDroplessMLP(
            neox_args,
            init_method,
            output_layer_init_method,
        )

        self.attention = ScaledDotProduct().cuda()
        self.routed_mask = None

    def forward(self, x, attention_scores, queries=None, keys=None, router_type_override=None):
        router_type = router_type_override
        
        if router_type is None:
            router_type = self.router.router_type
        # we expect inputs as (sl, bs, hs)
        # neox provides inputs as torch.Size([2048, 4, 768])
        # (sl, bs, hs)

        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        expert_weights, expert_indices, scores = self.router(x, router_type_override=router_type_override)

        # return value should be
        output, expert_output = self.experts(x, expert_weights, expert_indices, queries=queries, keys=keys, router_type=router_type)

        if router_type == "dense_approx":
          attention_scores = attention_scores.mean(dim=1).unsqueeze(1).expand(-1, self.experts.num_experts, -1, -1).clone()

          notrouted_mask = torch.ones(expert_indices.size(0), self.experts.num_experts, dtype=torch.bool).to(expert_indices.device)
          notrouted_mask.scatter_(1, expert_indices, 0)
          notrouted_mask = notrouted_mask.view(attention_scores.size(2), attention_scores.size(0), self.experts.num_experts).permute(1, 2, 0)

          attention_scores.masked_fill_(notrouted_mask.unsqueeze(2), torch.finfo(attention_scores.dtype).min)
          attention_probs = attention_scores.softmax(dim=-1)
          attention_probs.masked_fill_(~notrouted_mask.unsqueeze(3), 0)

          expert_output = expert_output.view(attention_probs.size(2), attention_probs.size(0), self.experts.num_experts, -1).permute(1, 2, 0, 3)
          notrouted_output = torch.bmm(attention_probs.view(-1, attention_probs.size(2), attention_probs.size(3)), expert_output.view(-1, expert_output.size(2), expert_output.size(3)))
          notrouted_output = notrouted_output.view(attention_probs.size(0), self.experts.num_experts, expert_output.size(2), expert_output.size(3))

          notrouted_output = notrouted_output.permute(2, 0, 1, 3).contiguous().view(-1, self.experts.num_experts, expert_output.size(-1))
          # note notrouted_output is all 0 for indices corresponding to expert_indices, so those scores won't apply
          approx_output = (scores.unsqueeze(-1) * notrouted_output).sum(dim=1).view(x.shape)

          return output + approx_output, None

        if router_type == "dense_approx_lsh":
            approx_output = (scores.unsqueeze(-1) * expert_output).sum(dim=1).view(x.shape)
            return output + approx_output, None
        
        if router_type == "dense_approx_efficient":
            if self.routed_mask is None:
                self.routed_mask = torch.zeros(expert_indices.size(0), self.experts.num_experts, dtype=torch.bool).to(expert_indices.device)
            with torch.no_grad():
                self.routed_mask.zero_().scatter_(1, expert_indices, 1)
            
            att_mask = self.routed_mask.view(queries.shape[1], queries.shape[0], -1).transpose(0, 1).transpose(1, 2)

            attn_result = self.attention(
                # bs x sl x nheads x head dim -> bs x nexperts x sl x head dim
                queries.mean(dim=2, keepdim=True).expand(-1, -1, self.experts.num_experts, -1).transpose(1, 2),
                # bs x sl x nheads x head dim -> bs x nexperts x sl x head dim
                keys.mean(dim=2, keepdim=True).expand(-1, -1, self.experts.num_experts, -1).transpose(1, 2), 
                # sl*bs x nexperts x hidden dim -> bs x nexperts x sl x hidden dim
                expert_output.view(queries.shape[1], queries.shape[0], self.experts.num_experts, -1).transpose(0, 1).transpose(1, 2),
                # select columns of routed tokens
                att_mask=att_mask.unsqueeze(2)
            )
            # -> bs x nexperts x sl x hidden dim
            # remove value rows of routed tokens
            # rarely a whole sequence can be masked out (all tokens routed to that expert) resulting in nans
            # in either case mask out approximations
            attn_result = torch.where(torch.logical_or(att_mask.unsqueeze(3), torch.isnan(attn_result)), 0, attn_result)

            approx_output = (scores.view(*x.shape[:2], -1).unsqueeze(-1) * attn_result.transpose(1, 2).transpose(0, 1)).sum(dim=2)

            return output + approx_output, None

        if router_type == "expert_prob_approx":
            # for expert 0
            # get all inputs that didn't go expert 0
            # get highest probs to expert 0, grouped by expert
            pass
        return output, None
