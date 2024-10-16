import torch

def expert_approx(output, input_indices, bin_ids, scores, num_experts, top_k, group_topk):
  _, group_indices = scores.topk(k=group_topk, dim=-1)
  sums = torch.zeros(num_experts * num_experts, output.shape[-1], dtype=output.dtype, device=output.device)

  # use this output given for expert i only if the corresponding input belongs to group i
  mask = (group_indices[input_indices] == bin_ids.unsqueeze(1)).any(dim=-1)
  # 2d index; row is the group we are approxing for and column is the expert we are approxing
  approx_indices = group_indices[input_indices] * num_experts + bin_ids.unsqueeze(1)
  total_counts = approx_indices[mask].flatten().bincount().clamp(min=1).unsqueeze(1)

  # two options: scale outputs by approximation weight first, or scale after summing (the latter should be less precise?)
  approx_vals = sums.index_add(0, approx_indices[mask].flatten(), output[mask].repeat_interleave(group_topk, dim=0) / total_counts[approx_indices[mask].flatten()])
  # approx_vals = sums.index_add(0, approx_indices[mask].flatten(), output[mask].repeat_interleave(group_topk, dim=0)) / total_counts

  # for each input: which approx is it gonna use, and with which corresponding weight
  missing_expert_weights, missing_expert_indices = (-scores).topk(k=num_experts - top_k, dim=-1)
  # tokens x experts

  # same 2d index, tokens x group x experts
  missing_approx_indices = (group_indices.unsqueeze(2) * num_experts + missing_expert_indices.unsqueeze(1))

  # corresponding approx for tokens x group x expert, weights token x expert -> tokens x 1 x expert x 1, weighted sum of experts, averaged over groups
  approx_output = (approx_vals[missing_approx_indices] * -missing_expert_weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2).mean(dim=1)

  return approx_output