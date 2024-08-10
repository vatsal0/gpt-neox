import triton
import triton.language as tl

from pdb import set_trace
import os
# os.environ["TRITON_INTERPRET"] = '1'

os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

@triton.jit
def lsh_approximation_kernel(
    input, output, bin_ids, bucket_indices,
    bucket_starts, bucket_ends, reshaped_output,
    input_stride, output_stride, bin_ids_stride, bucket_indices_stride,
    num_experts: tl.constexpr, input_dim: tl.constexpr, output_dim: tl.constexpr, bucket_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    bucket = bucket_offset + tl.program_id(0)
    expert = tl.program_id(1)
    
    bucket_start = tl.load(bucket_starts + bucket)
    bucket_end = tl.load(bucket_ends + bucket)
    
    bucket_size = bucket_end - bucket_start

    assert bucket_size <= BLOCK_SIZE
    
    # Load indices for this block
    bucket_locs = tl.arange(0, BLOCK_SIZE)
    bucket_mask = bucket_locs < bucket_size
    this_bucket_indices = tl.load(bucket_indices + bucket_start + bucket_locs, mask=bucket_mask)
    
    # Load input for this block
    input_dims = tl.arange(0, triton.next_power_of_2(input_dim))
    input_locs = this_bucket_indices[:, None] * input_stride + input_dims[None, :]
    input_mask = bucket_mask[:, None] & (input_dims < input_dim)[None, :]
    input_block = tl.load(input + input_locs, mask=input_mask)
    
    # Compute scores
    scale = tl.sqrt(float(input_dim))
    scores = tl.dot(input_block / scale, tl.trans(input_block / scale))
    
    # Apply mask
    mask = tl.arange(0, BLOCK_SIZE)[:, None] == tl.arange(0, BLOCK_SIZE)[None, :]
    threshold = 1 - 2 / num_experts
    scores = tl.where((mask | (scores < threshold)), 0.0, scores)
    
    # Load expert indices for this block
    expert_indices = tl.load(bin_ids + this_bucket_indices * bin_ids_stride, mask=bucket_mask)
    # Compute approximations for each expert
    expert_mask = expert_indices == expert
    not_expert_mask = expert_indices != expert
    expert_scores = tl.where(expert_mask[:, None] | not_expert_mask[None, :], 0.0, scores)
    
    # Load output for this block
    output_dims = tl.arange(0, triton.next_power_of_2(output_dim))
    output_locs = this_bucket_indices[:, None] * output_stride + output_dims[None, :]
    output_mask = bucket_mask[:, None] & (output_dims < output_dim)[None, :]
    output_block = tl.load(output + output_locs, mask=output_mask)
    
    # Compute approximation
    approx = tl.dot(expert_scores, output_block.to(tl.float32))
    score_counts = tl.sum(expert_scores > 0, axis=1)
    approx = tl.where(score_counts[:, None] > 0, approx / score_counts[:, None], approx)
    
    # Store approximation
    store_locs = this_bucket_indices[:, None] * num_experts * output_dim + expert * output_dim + output_dims[None, :]
    store_mask = bucket_mask[:, None] & (output_dims < output_dim)[None, :]
    tl.atomic_add(reshaped_output + store_locs, approx)
    # tl.store(reshaped_output + store_locs, approx, mask=store_mask)

# Launch the kernel
def launch_lsh_approximation_kernel(input_, output, bin_ids, bucket_indices, bucket_starts, bucket_ends, reshaped_output, num_experts, num_buckets):
    # BLOCK_SIZE = 32  # Adjust based on your hardware and input size
    BLOCK_SIZE = triton.next_power_of_2(int((bucket_ends - bucket_starts).max().item()))
    print(f"{BLOCK_SIZE=}")

    BUCKETS_PER_BATCH = 1

    assert num_buckets % BUCKETS_PER_BATCH == 0

    for bucket_offset in range(0, num_buckets, BUCKETS_PER_BATCH):
    
      grid = (BUCKETS_PER_BATCH, num_experts)
      
      lsh_approximation_kernel[grid](
          input_, output, bin_ids, bucket_indices, bucket_starts, bucket_ends, reshaped_output,
          input_.stride(0), output.stride(0), bin_ids.stride(0), bucket_indices.stride(0),
          num_experts, input_.shape[-1], output.shape[-1], bucket_offset,
          BLOCK_SIZE=BLOCK_SIZE
      )
      print('did batch')
