import torch
import torch.testing
import megablocks
from expert_approx import group_approx, approx_vals_kernel

num_experts = 32
top_k = 4
group_topk = 2
BLOCK_SIZE = 32
N=64

output = torch.randn(N*top_k, 128).cuda()
output.requires_grad_(True)
scores = torch.randn(N, num_experts).cuda().softmax(dim=-1)
scores.requires_grad_(True)
input_indices = torch.arange(N).repeat_interleave(top_k).cuda()
bin_ids = scores.topk(k=top_k, dim=-1).indices.flatten()

_, group_indices = scores.topk(k=group_topk, dim=-1)

output_group_indices = group_indices[input_indices]
mask = (output_group_indices == bin_ids.unsqueeze(1)).any(dim=-1)
approx_indices = output_group_indices * num_experts + bin_ids.unsqueeze(1)
total_counts = megablocks.ops.histogram(approx_indices[mask].flatten(), num_experts * num_experts).clamp(min=1).unsqueeze(1)

approx_select = torch.zeros(output.shape[0], num_experts * num_experts, dtype=torch.bool, device=output.device)
approx_select.scatter_(1, approx_indices, 1)
approx_select.masked_fill_(~mask.unsqueeze(1), 0)
approx_vals = torch.einsum('nd,na->ad', output, approx_select.to(dtype=output.dtype)) / total_counts
orig_grad = torch.autograd.grad(outputs=approx_vals, inputs=output, grad_outputs=torch.ones_like(approx_vals))[0]

new_approx_vals = group_approx.apply(output, approx_indices, mask, total_counts)
new_grad = torch.autograd.grad(outputs=new_approx_vals, inputs=output, grad_outputs=torch.ones_like(new_approx_vals))[0]

# grad: mask.unsqueeze(1) * (upstream_grad[approx_indices] / total_counts[approx_indices]).sum(dim=1)

torch.testing.assert_close(orig_grad, new_grad)