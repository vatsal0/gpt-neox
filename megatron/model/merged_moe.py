import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.neox_arguments.arguments import NeoXArgs
from megatron.mpu.layers import _initialize_affine_weight_gpu
from megablocks import grouped_gemm_util as gg

class ParallelGroupedMergedMLP(nn.Module):
  def __init__(
    self,
    neox_args: NeoXArgs,
    init_method,
    output_layer_init_method,
    stride=1
  ):
    super().__init__()

    intermediate_size = neox_args.hidden_size * 8 // 3
    
    self.w1 = nn.Parameter(
      torch.empty(
        neox_args.moe_num_experts,
        neox_args.hidden_size,
        intermediate_size,
        device=torch.cuda.current_device(),
        dtype=neox_args.params_dtype,
      )
    )
    _initialize_affine_weight_gpu(
        self.w1, init_method, partition_dim=0, stride=stride
    )

    self.w3 = nn.Parameter(
      torch.empty(
        neox_args.moe_num_experts,
        neox_args.hidden_size,
        intermediate_size,
        device=torch.cuda.current_device(),
        dtype=neox_args.params_dtype,
      )
    )
    _initialize_affine_weight_gpu(
        self.w3, init_method, partition_dim=0, stride=stride
    )

    self.w2 = nn.Parameter(
      torch.empty(
        neox_args.moe_num_experts,
        intermediate_size,
        neox_args.hidden_size,
        device=torch.cuda.current_device(),
        dtype=neox_args.params_dtype,
      )
    )
    _initialize_affine_weight_gpu(
        self.w2, output_layer_init_method, partition_dim=0, stride=stride
    )

  def forward(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    expert_weights = expert_weights.view(expert_weights.size(0), expert_weights.size(1), 1, 1) # BS*S x N x 1 x 1

    # weighted average of each FFN parameters
    merged_w1 = (expert_weights * self.w1).sum(dim=1) # BS*S x N x 1 x 1, N x HS x IS -> BS*S x HS x IS
    merged_w2 = (expert_weights * self.w2).sum(dim=1)
    merged_w3 = (expert_weights * self.w3).sum(dim=1)

    # hidden_states: L x BS*S x HS
    output1 = F.silu(torch.bmm(hidden_states.transpose(0, 1), merged_w1)) # L x BS*S x HS, BS*S x HS x IS -> BS*S x L x IS
    output3 = torch.bmm(hidden_states.transpose(0, 1), merged_w3)
    output2 = torch.bmm(output1 * output3, merged_w2) # -> BS*S x L x IS, BS*S x IS x HS -> BS*S x L x HS
    
    return output2.transpose(0, 1)

class Router(nn.Module):
  def __init__(
    self,
    neox_args: NeoXArgs,
    init_method,
  ):
    super().__init__()

    self.layer = torch.nn.Linear(
      neox_args.hidden_size,
      neox_args.moe_num_experts,
      bias=False,
      dtype=neox_args.params_dtype,
      device=torch.cuda.current_device(),
    )
    init_method(self.layer.weight)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.layer(hidden_states)

class SoftMergingMoE(nn.Module):
  def __init__(
    self,
    neox_args: NeoXArgs,
    init_method,
    output_layer_init_method,
    segment_length=256
  ):
    super().__init__()
    self.num_experts = neox_args.moe_num_experts
    self.segment_length = segment_length
    
    self.router = Router(neox_args, init_method)
    self.experts = ParallelGroupedMergedMLP(neox_args, init_method, output_layer_init_method)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    sequence_length, batch_size, hidden_dim = hidden_states.shape # SL x BS x HS
    
    # split into segments and get average representation across each segment
    num_segments = sequence_length // self.segment_length

    segment_hidden_states = hidden_states.view(self.segment_length, batch_size * num_segments, hidden_dim) # L x BS*S x HS
    mean_segment_hidden_states = segment_hidden_states.mean(dim=0) # BS*S x HS
    
    # router results for each averaged segment
    expert_weights = F.softmax(self.router(mean_segment_hidden_states), dim=-1) # BS*S x N
    
    # for each element in the batch save the router weights for the first segment
    first_weights = expert_weights.view(batch_size, num_segments, self.num_experts)[0, :]
    
    # causal part of the segment routing, now each segment's expert weights are the result of the previous segment
    expert_weights = expert_weights.roll(1)
    
    # but for the first segment in each sequence put back that corresponding segment's weights (no causal relationship between batch elements)
    # and stop the gradient flow backwards
    expert_weights = expert_weights.view(batch_size, num_segments, self.num_experts)
    expert_weights[0, :] = first_weights.detach()
    expert_weights = expert_weights.view(batch_size * num_segments, self.num_experts)

    output_hidden_states = self.experts(segment_hidden_states, expert_weights)

    # rearrange segments back into their sequences
    return output_hidden_states.reshape(sequence_length, batch_size, hidden_dim), None