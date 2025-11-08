/*
 * DCNv4 TensorRT Plugin - CUDA Kernels
 * Adapted from DCNv4 official implementation
 *
 * Provides:
 * - Format conversion: [N,C,H,W] (TensorRT) <-> [N,H,W,C] (DCNv4)
 * - DCNv4 deformable convolution forward pass
 * - Optimized for RTX 4090 Ada Lovelace (Compute Capability 8.9)
 */

#include "common.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// ============================================================================
// DCNv4 Forward Kernel (adapted from dcnv4_im2col_cuda.cuh)
// ============================================================================

// Register-based forward kernel (no shared memory for small K)
template <typename scalar_t, int d_stride, typename transfer_t, int L, int K,
          bool softmax>
__global__ void forward_kernel_dcn_reg(
    const scalar_t *p_value, const scalar_t *p_offset, scalar_t *p_output,
    const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    const int block_multiplier, const int padded_offset_dim) {

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;
  constexpr int li = 0;

  opmath_t p_mask_shm[K] = {0.};
  opmath_t p_out_shm[d_stride] = {0.};

  const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*3;
  const int mask_length = K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;

  for (int i=0; i < K; i++){
    p_mask_shm[i] = *(p_offset_ptr + K*2 + i);
  }

  if (softmax) {
    // Calculate softmax over L and K
      opmath_t softmax_max = -1e100;
      opmath_t softmax_sum = 0.0;
      // get max
      for (int j = 0; j < K; j++) {
        softmax_max = max(softmax_max, p_mask_shm[j]);
      }

      // get sumexp
      for (int j = 0; j < K; j++) {
        opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
        p_mask_shm[j] = exp_results;
        softmax_sum += exp_results;
      }

      // normalize
      for (int j = 0; j < K; j++) {
        p_mask_shm[j] /= softmax_sum;
      }
  }

  int offset_idx = 0;
  int mask_idx = 0;

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr =
      p_value + (bi * (height_in * width_in)) * (G * D);

  const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                   (qi % width_out) * stride_w;
  const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                   (qi / width_out) * stride_h;
  const opmath_t p0_w_ =
      p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
  const opmath_t p0_h_ =
      p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
  const int center_h = kernel_h / 2;
  const int center_w = kernel_w / 2;

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  for (int i = 0; i < kernel_w; ++i) {
    for (int j = 0; j < kernel_h; ++j) {
      if (i != center_w || j != center_h || !remove_center) {
        const opmath_t w_im =
            p0_w_ + (i * dilation_w + (opmath_t)p_offset_ptr[offset_idx]) *
                        offset_scale;
        const opmath_t h_im =
            p0_h_ + (j * dilation_h + (opmath_t)p_offset_ptr[offset_idx + 1]) *
                        offset_scale;
        const opmath_t attn = p_mask_shm[mask_idx];

        if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
          ms_deform_attn_im2col_bilinear<scalar_t, transfer_t, d_stride>(
              p_out_shm, p_value_ptr, height_in, width_in, h_im, w_im, attn,
              w_stride, base_ptr);
        }
        offset_idx += 2;
        mask_idx += 1;
      }
    }
  }
  scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    fp16_regs[ds] = p_out_shm[ds];
  }

  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

// ============================================================================
// Template dispatch for different d_stride values
// ============================================================================

template <typename scalar_t, typename stride_type, int d_stride>
void _dcnv4_im2col_cuda(cudaStream_t stream,
                              const scalar_t *value,    // B, H * W, (G * D)
                              const scalar_t *p_offset, // B, H * W, G * K * 3)
                              scalar_t *output,         // B, H_out*W_out, G * D
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w,
                              const int pad_h, const int pad_w,
                              const int dilation_h, const int dilation_w,
                              const int G, const int D, const int B,
                              const int height_in, const int width_in,
                              const int height_out, const int width_out,
                              const opmath_t offset_scale,
                              const int remove_center, const int block_thread,
                              const int softmax,
                              const int padded_offset_dim) {

  constexpr int L = 1;

  auto kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, true>;

  int N = height_in * width_in;
  int Q = height_out * width_out;
  int K = kernel_h * kernel_w;

  if (remove_center) {
    K -= 1;
  }
  if (softmax) {
    switch (K) {
    case 9:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, true>;
      break;
    case 8:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 8, true>;
      break;
    default:
      printf("K=%d not supported\n", K);
      return;
    }
  } else {
    switch (K) {
    case 9:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, false>;
      break;
    case 8:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 8, false>;
    break;
    default:
      printf("K=%d not supported\n", K);
      return;
    }
  }

  const int block_multiplier = block_thread / (D / d_stride) / G;
  if ((B*Q) % block_multiplier != 0) {
    printf("ERROR: (B*Q) %% block_multiplier != 0\n");
    return;
  }

  dim3 num_blocks(B*Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  int shm_size = 0;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, p_offset, output, G, D, Q, kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w, dilation_h, dilation_w, height_in, width_in, height_out,
      width_out, offset_scale, remove_center, block_multiplier, padded_offset_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in dcnv4_im2col_cuda: %s\n", cudaGetErrorString(err));
    printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
           "shm_size=%d\n\n",
           num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
           num_threads.y, num_threads.z, shm_size);
  }
}

// Main entry point for DCNv4 im2col operation
template <typename scalar_t>
void dcnv4_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,    // B, H * W, (G * D)
    const scalar_t *p_offset, // B, H * W, G * K * 3)
    scalar_t *output,         // B, H_out*W_out, G * D
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int G, const int D, const int B,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    const int d_stride, const int block_thread, const bool softmax,
    const int padded_offset_dim) {

  if (D % d_stride != 0) {
    printf("ERROR: D %% d_stride != 0\n");
    return;
  }

  // FP16 path (sizeof(half) == 2)
  if (sizeof(scalar_t) == 2) {
    switch (d_stride) {
    case 1:
      _dcnv4_im2col_cuda<scalar_t, scalar_t, 1>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_im2col_cuda<scalar_t, uint, 2>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_im2col_cuda<scalar_t, uint2, 4>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_im2col_cuda<scalar_t, uint4, 8>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 16:
      _dcnv4_im2col_cuda<scalar_t, ulonglong4, 16>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    default:
      printf("ERROR: Unsupported d_stride=%d for FP16\n", d_stride);
    }
  }
  // FP32 path (sizeof(float) == 4)
  else if (sizeof(scalar_t) == 4) {
    switch (d_stride) {
    case 1:
      _dcnv4_im2col_cuda<scalar_t, uint, 1>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_im2col_cuda<scalar_t, uint2, 2>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_im2col_cuda<scalar_t, uint4, 4>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_im2col_cuda<scalar_t, ulonglong4, 8>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    default:
      printf("ERROR: Unsupported d_stride=%d for FP32\n", d_stride);
    }
  }
  else {
    printf("ERROR: Unsupported data type size: %zu\n", sizeof(scalar_t));
  }
}

// ============================================================================
// Format Conversion Kernels: [N,C,H,W] <-> [N,H,W,C]
// ============================================================================

// Convert [N, C, H, W] (TensorRT) -> [N, H, W, C] (DCNv4)
template <typename T>
__global__ void nchw_to_nhwc_kernel(
    const T* __restrict__ input,  // [N, C, H, W]
    T* __restrict__ output,       // [N, H, W, C]
    int N, int C, int H, int W) {

    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int n = idx / (C * H * W);

        int out_idx = ((n * H + h) * W + w) * C + c;
        output[out_idx] = input[idx];
    }
}

// Convert [N, H, W, C] (DCNv4) -> [N, C, H, W] (TensorRT)
template <typename T>
__global__ void nhwc_to_nchw_kernel(
    const T* __restrict__ input,  // [N, H, W, C]
    T* __restrict__ output,       // [N, C, H, W]
    int N, int C, int H, int W) {

    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        int c = idx % C;
        int w = (idx / C) % W;
        int h = (idx / (C * W)) % H;
        int n = idx / (H * W * C);

        int out_idx = ((n * C + c) * H + h) * W + w;
        output[out_idx] = input[idx];
    }
}

// Format conversion wrappers
template <typename T>
void convert_nchw_to_nhwc(
    const T* input,
    T* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    nchw_to_nhwc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}

template <typename T>
void convert_nhwc_to_nchw(
    const T* input,
    T* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    nhwc_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}

// ============================================================================
// Zero Offset Generator (Temporary for Phase 2G-2)
// ============================================================================

/**
 * Generates zero offsets for DCNv4 kernel testing.
 * This is a temporary solution to enable pipeline validation without learned offset parameters.
 *
 * Output format: [B, H*W, G*K*3] where:
 *   - Each pixel (H*W positions) has G groups
 *   - Each group has K kernel points (kernel_size²)
 *   - Each point has 3 values: (offset_x, offset_y, attention_weight)
 *   - With zero offsets, this becomes identity mapping for debugging
 */
__global__ void zero_offset_kernel(__half* offsets, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        offsets[idx] = __float2half(0.0f);
    }
}

void generate_zero_offsets(
    __half* offsets,           // Output: [B, H*W, G*K*3]
    int batch,                 // B
    int height,                // H
    int width,                 // W
    int group,                 // G
    int kernel_points,         // K (kernel_size²)
    cudaStream_t stream) {

    int spatial_size = height * width;
    int offset_dim = group * kernel_points * 3;  // 3 = (offset_x, offset_y, attention_weight)
    int total_elements = batch * spatial_size * offset_dim;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    zero_offset_kernel<<<blocks, threads, 0, stream>>>(offsets, total_elements);
}

// ============================================================================
// Main TensorRT Plugin Entry Point
// ============================================================================

extern "C" void dcnv4_forward_cuda(
    const void* input,       // [N, C, H, W] FP16
    void* output,            // [N, C, H, W] FP16
    void* workspace,         // Temporary workspace
    int batch,               // N
    int channels,            // C
    int height,              // H
    int width,               // W
    int kernel_size,         // K (e.g., 3)
    int stride,              // S (usually 1)
    int pad,                 // P (usually 1)
    int group,               // G (number of groups)
    float offset_scale,      // Offset scaling
    bool remove_center,      // Remove center pixel
    cudaStream_t stream) {

    // ========================================================================
    // DCNv4 Forward Pass Implementation (Phase 2G-2)
    // ========================================================================

    // Calculate derived parameters
    int spatial_size = height * width;
    int D = channels / group;  // Channels per group
    int kernel_points = kernel_size * kernel_size;
    if (remove_center) {
        kernel_points -= 1;  // Exclude center point
    }

    // Calculate padded offset dimension (must be multiple of 8 for tensor cores)
    int base_offset_dim = group * kernel_points * 3;  // 3 = (offset_x, offset_y, attention_weight)
    int padded_offset_dim = ((base_offset_dim + 7) / 8) * 8;  // Round up to multiple of 8

    // Calculate output spatial dimensions
    int height_out = (height + 2 * pad - kernel_size) / stride + 1;
    int width_out = (width + 2 * pad - kernel_size) / stride + 1;

    // ========================================================================
    // Step 1: Setup workspace memory layout
    // ========================================================================
    __half* workspace_base = (__half*)workspace;
    size_t offset = 0;

    // Input buffer in NHWC format: [B, H, W, C]
    __half* input_nhwc = workspace_base + offset;
    offset += batch * spatial_size * channels;

    // Offset tensor: [B, H*W, padded_offset_dim]
    __half* offsets = workspace_base + offset;
    offset += batch * spatial_size * padded_offset_dim;

    // Output buffer in NHWC format: [B, H_out, W_out, C]
    __half* output_nhwc = workspace_base + offset;
    // Note: output size should be batch * height_out * width_out * channels
    // For stride=1, pad=1, kernel=3, this equals input size

    // ========================================================================
    // Step 2: Convert input from NCHW [N,C,H,W] to NHWC [N,H,W,C]
    // ========================================================================
    convert_nchw_to_nhwc<__half>(
        (const __half*)input,
        input_nhwc,
        batch,
        channels,
        height,
        width,
        stream
    );

    // ========================================================================
    // Step 3: Generate zero offsets (temporary for Phase 2G-2)
    // ========================================================================
    // Note: In Phase 2G-3, this will be replaced with learned offsets from multi-input plugin
    generate_zero_offsets(
        offsets,
        batch,
        height,
        width,
        group,
        kernel_points,
        stream
    );

    // ========================================================================
    // Step 4: Call DCNv4 im2col kernel
    // ========================================================================
    // Input: [B, H*W, C]  (input_nhwc reshaped)
    // Offsets: [B, H*W, G*K*3]  (offsets)
    // Output: [B, H_out*W_out, C]  (output_nhwc reshaped)

    dcnv4_im2col_cuda<__half>(
        stream,                    // CUDA stream
        input_nhwc,                // value: [B, H*W, G*D]
        offsets,                   // spatial_offsets: [B, H*W, G*K*3]
        output_nhwc,               // output: [B, H_out*W_out, G*D]
        kernel_size,               // kernel_h
        kernel_size,               // kernel_w
        stride,                    // stride_h
        stride,                    // stride_w
        pad,                       // pad_h
        pad,                       // pad_w
        1,                         // dilation_h (always 1 for RFC Net)
        1,                         // dilation_w (always 1 for RFC Net)
        group,                     // group (G)
        D,                         // group_channels (D)
        batch,                     // im2col_step (B)
        height,                    // height_in
        width,                     // width_in
        height_out,                // height_out
        width_out,                 // width_out
        offset_scale,              // offset_scale
        remove_center,             // remove_center
        8,                         // d_stride (optimized for FP16, must be power of 2)
        256,                       // block_thread (optimal for RTX 4090)
        true,                      // softmax (enable attention weights)
        padded_offset_dim          // padded_offset_dim
    );

    // ========================================================================
    // Step 5: Convert output from NHWC [N,H,W,C] back to NCHW [N,C,H,W]
    // ========================================================================
    convert_nhwc_to_nchw<__half>(
        output_nhwc,
        (__half*)output,
        batch,
        channels,
        height_out,
        width_out,
        stream
    );

    // ========================================================================
    // Note: CUDA errors are checked by TensorRT infrastructure
    // cudaGetLastError() will be called after kernel execution
    // ========================================================================
}

// ============================================================================
// Multi-Input Entry Point (Phase 2G-3)
// ============================================================================

extern "C" void dcnv4_forward_cuda_with_offsets(
    const void* input,       // [N, C, H, W] FP16
    const void* offsets_nchw, // [N, G*K*3, H, W] FP16
    void* output,            // [N, C, H, W] FP16
    void* workspace,         // Temporary workspace
    int batch,               // N
    int channels,            // C
    int height,              // H
    int width,               // W
    int kernel_size,         // K (e.g., 3)
    int stride,              // S (usually 1)
    int pad,                 // P (usually 1)
    int group,               // G (number of groups)
    float offset_scale,      // Offset scaling
    bool remove_center,      // Remove center pixel
    cudaStream_t stream) {

    // ========================================================================
    // DCNv4 Forward Pass with Offset Input (Phase 2G-3)
    // ========================================================================

    // Calculate derived parameters
    int spatial_size = height * width;
    int D = channels / group;  // Channels per group
    int kernel_points = kernel_size * kernel_size;
    if (remove_center) {
        kernel_points -= 1;  // Exclude center point
    }

    // Calculate padded offset dimension (must be multiple of 8 for tensor cores)
    int base_offset_dim = group * kernel_points * 3;  // 3 = (offset_x, offset_y, attention_weight)
    int padded_offset_dim = ((base_offset_dim + 7) / 8) * 8;  // Round up to multiple of 8

    // Calculate output spatial dimensions
    int height_out = (height + 2 * pad - kernel_size) / stride + 1;
    int width_out = (width + 2 * pad - kernel_size) / stride + 1;

    // ========================================================================
    // Step 1: Setup workspace memory layout
    // ========================================================================
    __half* workspace_base = (__half*)workspace;
    size_t offset = 0;

    // Input buffer in NHWC format: [B, H, W, C]
    __half* input_nhwc = workspace_base + offset;
    offset += batch * spatial_size * channels;

    // Offset tensor in reshaped format: [B, H*W, padded_offset_dim]
    __half* offsets_reshaped = workspace_base + offset;
    offset += batch * spatial_size * padded_offset_dim;

    // Output buffer in NHWC format: [B, H_out, W_out, C]
    __half* output_nhwc = workspace_base + offset;

    // ========================================================================
    // Step 2: Convert input from NCHW [N,C,H,W] to NHWC [N,H,W,C]
    // ========================================================================
    convert_nchw_to_nhwc<__half>(
        (const __half*)input,
        input_nhwc,
        batch,
        channels,
        height,
        width,
        stream
    );

    // ========================================================================
    // Step 3: Reshape offsets from NCHW [N, G*K*3, H, W] to [N, H*W, G*K*3]
    // ========================================================================
    // This is essentially a transpose: (N, C, H, W) -> (N, H, W, C) -> (N, H*W, C)
    convert_nchw_to_nhwc<__half>(
        (const __half*)offsets_nchw,
        offsets_reshaped,
        batch,
        base_offset_dim,  // Treat G*K*3 as channels
        height,
        width,
        stream
    );

    // ========================================================================
    // Step 4: Call DCNv4 im2col kernel
    // ========================================================================
    dcnv4_im2col_cuda<__half>(
        stream,                    // CUDA stream
        input_nhwc,                // value: [B, H*W, G*D]
        offsets_reshaped,          // spatial_offsets: [B, H*W, G*K*3]
        output_nhwc,               // output: [B, H_out*W_out, G*D]
        kernel_size,               // kernel_h
        kernel_size,               // kernel_w
        stride,                    // stride_h
        stride,                    // stride_w
        pad,                       // pad_h
        pad,                       // pad_w
        1,                         // dilation_h (always 1 for RFC Net)
        1,                         // dilation_w (always 1 for RFC Net)
        group,                     // group (G)
        D,                         // group_channels (D)
        batch,                     // im2col_step (B)
        height,                    // height_in
        width,                     // width_in
        height_out,                // height_out
        width_out,                 // width_out
        offset_scale,              // offset_scale
        remove_center,             // remove_center
        8,                         // d_stride (optimized for FP16)
        256,                       // block_thread (optimal for RTX 4090)
        true,                      // softmax (enable attention weights)
        padded_offset_dim          // padded_offset_dim
    );

    // ========================================================================
    // Step 5: Convert output from NHWC [N,H,W,C] back to NCHW [N,C,H,W]
    // ========================================================================
    convert_nhwc_to_nchw<__half>(
        output_nhwc,
        (__half*)output,
        batch,
        channels,
        height_out,
        width_out,
        stream
    );
}

// Explicit template instantiations for FP16
template void dcnv4_im2col_cuda<__half>(
    cudaStream_t, const __half*, const __half*, __half*,
    const int, const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const int,
    const int, const int, const int, const int, const opmath_t, const int,
    const int, const int, const bool, const int);

template void convert_nchw_to_nhwc<__half>(
    const __half*, __half*, int, int, int, int, cudaStream_t);

template void convert_nhwc_to_nchw<__half>(
    const __half*, __half*, int, int, int, int, cudaStream_t);
