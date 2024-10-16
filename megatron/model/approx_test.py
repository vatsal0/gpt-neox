import torch
import torch.testing
from expert_approx import expert_approx

num_experts = 4
top_k = 2
group_topk = 2

output = torch.randn(6, 128)
# x, y, z: x routed to 1, 3; y routed to 1, 2; z routed to 2, 3
input_indices = torch.Tensor([0, 0, 1, 1, 2, 2]).long()
bin_ids = torch.Tensor([1, 3, 1, 2, 2, 3]).int()

scores = torch.Tensor([[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2], [0.1, 0.2, 0.4, 0.3]])

approx = expert_approx(output, input_indices, bin_ids, scores, num_experts, top_k, group_topk)

torch.testing.assert_close(approx[0], 0.2 * (output[3] + output[4]) / 2)
