import torch
import megablocks.ops

@torch.compile
def expert_approx(output: torch.Tensor, input_indices: torch.Tensor, bin_ids: torch.Tensor, scores: torch.Tensor, num_experts: int, top_k: int, group_topk: int):
  _, group_indices = scores.topk(k=group_topk, dim=-1)

  # use this output given for expert i only if the corresponding input belongs to group i
  output_group_indices = group_indices[input_indices]
  mask = (output_group_indices == bin_ids.unsqueeze(1)).any(dim=-1)
  # 2d index; row is the group we are approxing for and column is the expert we are approxing
  approx_indices = output_group_indices * num_experts + bin_ids.unsqueeze(1)
  total_counts = megablocks.ops.histogram(approx_indices[mask].flatten(), num_experts * num_experts).clamp(min=1).unsqueeze(1)

  approx_select = torch.zeros(output.shape[0], num_experts * num_experts, dtype=torch.bool, device=output.device)
  approx_select.scatter_(1, approx_indices, 1)
  approx_select.masked_fill_(~mask.unsqueeze(1), 0)

  approx_vals = torch.einsum('nd,na->ad', output, approx_select.to(dtype=output.dtype)) / total_counts

  # for each input: which approx is it gonna use, and with which corresponding weight
  missing_expert_weights, missing_expert_indices = (-scores).topk(k=num_experts - top_k, dim=-1)
  # tokens x experts

  # same 2d index, tokens x group x experts
  missing_approx_indices = (group_indices.unsqueeze(2) * num_experts + missing_expert_indices.unsqueeze(1))

  # corresponding approx for tokens x group x expert, weights token x expert -> tokens x 1 x expert x 1, weighted sum of experts, averaged over groups
  approx_output = torch.einsum('nged,ne->nd', approx_vals[missing_approx_indices], -missing_expert_weights)/group_topk

  return approx_output