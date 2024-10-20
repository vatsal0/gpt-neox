import torch
import megablocks.ops

import triton
import triton.language as tl

BLOCK_SIZE = 32

@triton.jit
def approx_vals_kernel(
  output_ptr,
  approx_indices_ptr,
  mask_ptr,
  total_counts_ptr,
  approx_vals_ptr,
  N: tl.constexpr,
  D: tl.constexpr,
  K: tl.constexpr,
  M: tl.constexpr,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(0)
  
  start_idx = pid * BLOCK_SIZE
  
  for i in range(BLOCK_SIZE):
    if start_idx + i < N and tl.load(mask_ptr + start_idx + i) != 0:
      output = tl.load(output_ptr + (start_idx + i) * D + tl.arange(0, D))
      output_fp32 = tl.cast(output, tl.float32)

      for k in range(K):
        write_index = tl.load(approx_indices_ptr + (start_idx + i) * K + k)
        scale = tl.load(total_counts_ptr + write_index)
        if write_index < M:
          tl.atomic_add(approx_vals_ptr + write_index * D + tl.arange(0, D), output_fp32 / scale)

@triton.jit
def approx_output_kernel(
  approx_vals_ptr,
  approx_indices_ptr,
  expert_weights_ptr,
  approx_output_ptr,
  N: tl.constexpr,
  D: tl.constexpr,
  K: tl.constexpr,
  E: tl.constexpr,
  E2: tl.constexpr,
  BLOCK_SIZE: tl.constexpr
):
  pid = tl.program_id(0)
  
  start_idx = pid * BLOCK_SIZE

  for i in range(BLOCK_SIZE):
    if start_idx + i < N:
      expert_index = tl.arange(0, E2)
      expert_mask = expert_index < E

      group_expert_index = tl.arange(0, K)[:, None] * E + expert_index[None, :]
      approx_indices = tl.load(approx_indices_ptr + (start_idx + i) * K * E + group_expert_index, mask=expert_mask[None, :])

      approx_vals = tl.load(approx_vals_ptr + approx_indices[:, :, None] * D + tl.arange(0, D)[None, None, :])
      expert_weights = tl.load(expert_weights_ptr + (start_idx + i) * E + expert_index, mask=expert_mask)
      result = tl.sum(tl.sum(expert_weights[None, :, None] * approx_vals, axis=0), axis=0)
      tl.store(approx_output_ptr + (start_idx + i) * D + tl.arange(0, D), -result/K)

@torch.compile
def expert_approx(output: torch.Tensor, input_indices: torch.Tensor, bin_ids: torch.Tensor, scores: torch.Tensor, num_experts: int, top_k: int, group_topk: int):
  _, group_indices = scores.topk(k=group_topk, dim=-1)

  # use this output given for expert i only if the corresponding input belongs to group i
  output_group_indices = group_indices[input_indices]
  mask = (output_group_indices == bin_ids.unsqueeze(1)).any(dim=-1)
  # 2d index; row is the group we are approxing for and column is the expert we are approxing
  approx_indices = output_group_indices * num_experts + bin_ids.unsqueeze(1)
  total_counts = megablocks.ops.histogram(approx_indices[mask].flatten(), num_experts * num_experts).clamp(min=1).unsqueeze(1)

  N, D = output.shape
  K = group_topk
  M = num_experts * num_experts
  
  approx_vals = torch.zeros((M, D), dtype=torch.float32, device=output.device)
  
  grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
  approx_vals_kernel[grid](
    output, approx_indices, mask, total_counts, approx_vals,
    N, D, K, M,
    BLOCK_SIZE=BLOCK_SIZE
  )
  approx_vals = approx_vals.to(dtype=output.dtype)

  # for each input: which approx is it gonna use, and with which corresponding weight
  missing_expert_weights, missing_expert_indices = (-scores).topk(k=num_experts - top_k, dim=-1)
  # tokens x experts

  # same 2d index, tokens x group x experts
  missing_approx_indices = (group_indices.unsqueeze(2) * num_experts + missing_expert_indices.unsqueeze(1))

  # corresponding approx for tokens x group x expert, weights token x expert -> tokens x 1 x expert x 1, weighted sum of experts, averaged over groups
  N, K, E = missing_approx_indices.shape
  approx_output = torch.zeros((N, D), dtype=output.dtype, device=output.device)

  grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
  approx_output_kernel[grid](
    approx_vals, missing_approx_indices, missing_expert_weights, approx_output,
    N, D, K, E, triton.next_power_of_2(E),
    BLOCK_SIZE=BLOCK_SIZE
  )

  return approx_output