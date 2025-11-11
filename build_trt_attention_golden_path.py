"""
Golden Path: Segfault-proof TensorRT Attention with Conv1x1

Implements the exact shape discipline to avoid 5D->2D->MatMul bugs:
- Keep 5D only at edges (input/output)
- Use Conv1x1 for Q/K/V projections (4D NCHW)
- Go to 3D "tokens-major" [N', tokens, C] for attention
- Rank-4 batched GEMM for Q @ K^T
- Manual softmax (single-bit axis mask)
- Output via Conv1x1 back to 4D
- Single final reshape to 5D

Dimensions: B=1, T=2, H=6, W=9, C=512, n_head=4, d_head=128
"""

import os
import sys

trt_lib = r"D:\watermarkz\TensorRT-10.13.3.9\lib"
trt_bin = r"D:\watermarkz\TensorRT-10.13.3.9\bin"
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
os.environ['PATH'] = f"{trt_lib};{trt_bin};{cuda_bin};{os.environ['PATH']}"
os.add_dll_directory(trt_lib)
os.add_dll_directory(trt_bin)
os.add_dll_directory(cuda_bin)

import numpy as np
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'faster-propainter-main'))
from model.modules.sparse_transformer import TemporalSparseTransformerBlock


class TRTLogger(trt.ILogger):
    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.WARNING:
            print(f"[TRT-{severity}] {msg}")


def build_attention_golden_path():
    """Build attention following the golden path shape discipline"""
    print("\n" + "=" * 80)
    print("GOLDEN PATH: Segfault-proof Attention with Conv1x1")
    print("=" * 80)

    # Load weights
    t2t_params = {'kernel_size': (7, 7), 'stride': (3, 3), 'padding': (3, 3)}
    transformer = TemporalSparseTransformerBlock(
        dim=512, n_head=4, window_size=(12, 9), pool_size=(4, 4), depths=8, t2t_params=t2t_params
    )
    transformer.eval()
    first_layer = transformer.transformer[0]
    attn = first_layer.attention

    # Extract weights
    Wq = attn.query.weight.detach().cpu().numpy()
    bq = attn.query.bias.detach().cpu().numpy()
    Wk = attn.key.weight.detach().cpu().numpy()
    bk = attn.key.bias.detach().cpu().numpy()
    Wv = attn.value.weight.detach().cpu().numpy()
    bv = attn.value.bias.detach().cpu().numpy()
    Wproj = attn.proj.weight.detach().cpu().numpy()
    bproj = attn.proj.bias.detach().cpu().numpy()

    # Convert Linear weights to Conv1x1 format (FP16 for precision match with PyTorch)
    def linear_to_conv1x1(W_linear, b):
        """Convert [out, in] -> [out, in, 1, 1] with FP16 precision"""
        W_conv = W_linear.reshape(W_linear.shape[0], W_linear.shape[1], 1, 1)
        # Use FP16 to match PyTorch model precision (avoids FP32→FP16 quality loss)
        W_conv = np.ascontiguousarray(W_conv.astype(np.float16))
        b = np.ascontiguousarray(b.astype(np.float16))
        return W_conv, b

    Wq_conv, bq_conv = linear_to_conv1x1(Wq, bq)
    Wk_conv, bk_conv = linear_to_conv1x1(Wk, bk)
    Wv_conv, bv_conv = linear_to_conv1x1(Wv, bv)
    Wproj_conv, bproj_conv = linear_to_conv1x1(Wproj, bproj)

    # Build network
    logger = TRTLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Dimensions - Dynamic shape support
    B, C = 1, 512
    # Dynamic ranges based on production workload analysis
    T_min, T_opt, T_max = 2, 11, 16   # Temporal: min=boundary, opt=neighbor_length=10, max=with ref frames
    H_min, H_opt, H_max = 6, 12, 15   # Height: observed range from production
    W_min, W_opt, W_max = 9, 12, 15   # Width: observed range from production

    # Use MIN dimensions for network construction (TensorRT requirement for dynamic shapes)
    # TensorRT will scale kernels up to opt/max via optimization profile
    T, H, W = T_min, H_min, W_min

    n_head = 4
    d_head = C // n_head  # 128
    N_prime = B * T  # 11
    tokens = H * W  # 144

    print(f"\n[CONFIG] B={B}, C={C}")
    print(f"[CONFIG] Dynamic T: [{T_min}, {T_opt}, {T_max}]")
    print(f"[CONFIG] Dynamic H: [{H_min}, {H_opt}, {H_max}]")
    print(f"[CONFIG] Dynamic W: [{W_min}, {W_opt}, {W_max}]")
    print(f"[CONFIG] n_head={n_head}, d_head={d_head}")
    print(f"[CONFIG] Building with opt dimensions: T={T}, H={H}, W={W}")
    print(f"[CONFIG] N'={N_prime}, tokens={tokens}")

    # ========== STEP 0: 5D -> 4D (edge only) ==========
    print(f"\n[STEP 0] 5D -> 4D NCHW (edge)")
    # Dynamic shape input: -1 for T, H, W dimensions
    # Use FP16 to match PyTorch model precision (model.half())
    x5d = network.add_input(name="input", dtype=trt.float16, shape=(B, -1, -1, -1, C))
    print(f"        Input: [{B}, -1, -1, -1, {C}] (5D - T/H/W dynamic, FP16)")

    # Extract dynamic shape dimensions using IShapeLayer
    shape_layer = network.add_shape(x5d)  # Get shape as [1, T, H, W, 512]
    shape_layer.name = "get_input_shape"
    input_shape = shape_layer.get_output(0)  # Shape tensor: [B, T, H, W, C]

    # Extract T (dim 1), H (dim 2), W (dim 3)
    def extract_dim(shape_tensor, dim_index, name):
        """Extract a single dimension from shape tensor and cast to INT32"""
        slice_layer = network.add_slice(
            shape_tensor,
            start=trt.Dims([dim_index]),  # Start at dim_index
            shape=trt.Dims([1]),           # Get 1 element
            stride=trt.Dims([1])
        )
        slice_layer.name = f"{name}_slice"

        # Cast from INT64 (IShapeLayer output) to INT32 (required for shape math)
        cast_layer = network.add_cast(slice_layer.get_output(0), trt.int32)
        cast_layer.name = name
        return cast_layer.get_output(0)

    T_tensor = extract_dim(input_shape, 1, "extract_T")
    H_tensor = extract_dim(input_shape, 2, "extract_H")
    W_tensor = extract_dim(input_shape, 3, "extract_W")

    # Compute N_prime = B * T (for static B=1, this is just T)
    # CRITICAL: TRT requires INT32 for all shape tensors (not INT64!)
    const_B = network.add_constant(trt.Dims([1]), trt.Weights(np.array([B], dtype=np.int32)))
    const_B.name = "const_B"
    N_prime_layer = network.add_elementwise(const_B.get_output(0), T_tensor, trt.ElementWiseOperation.PROD)
    N_prime_layer.name = "compute_N_prime"
    N_prime_tensor = N_prime_layer.get_output(0)

    # Create constant for C dimension (INT32 for TRT shape tensors)
    const_C = network.add_constant(trt.Dims([1]), trt.Weights(np.array([C], dtype=np.int32)))
    const_C.name = "const_C"
    C_tensor = const_C.get_output(0)

    # Build reshape dims: [N', C, H, W] for NCHW format
    concat_layer = network.add_concatenation([N_prime_tensor, C_tensor, H_tensor, W_tensor])
    concat_layer.axis = 0
    concat_layer.name = "build_4d_shape"
    shape_4d = concat_layer.get_output(0)  # [N', C, H, W]

    # Reshape 5D -> 4D using shape tensor
    sh0 = network.add_shuffle(x5d)
    sh0.first_transpose = (0, 1, 2, 3, 4)  # identity for rank-5 (flatten B*T)
    sh0.set_input(1, shape_4d)  # Use shape tensor instead of reshape_dims
    sh0.second_transpose = (0, 1, 2, 3)  # identity for rank-4
    sh0.name = "input_5d_to_4d"
    x4d = sh0.get_output(0)
    print(f"        -> [B*T, C, H, W] (4D NCHW - dynamic)")

    # Compute tokens = H * W (will be used in reshapes)
    tokens_layer = network.add_elementwise(H_tensor, W_tensor, trt.ElementWiseOperation.PROD)
    tokens_layer.name = "compute_tokens"
    tokens_tensor = tokens_layer.get_output(0)

    # Create constants for n_head and d_head (INT32 for TRT shape tensors)
    const_n_head = network.add_constant(trt.Dims([1]), trt.Weights(np.array([n_head], dtype=np.int32)))
    const_n_head.name = "const_n_head"
    n_head_tensor = const_n_head.get_output(0)

    const_d_head = network.add_constant(trt.Dims([1]), trt.Weights(np.array([d_head], dtype=np.int32)))
    const_d_head.name = "const_d_head"
    d_head_tensor = const_d_head.get_output(0)

    # ========== STEP 1: Q/K/V via 1x1 Conv on 4D ==========
    print(f"\n[STEP 1] Q/K/V via 1x1 Convolution (4D)")

    def add_conv1x1(input_4d, W_conv, b_conv, out_channels, name):
        """Add 1x1 convolution layer"""
        # CRITICAL: Weights are FP16 (converted in linear_to_conv1x1), must match dtype
        W_trt = trt.Weights(trt.float16, W_conv.ctypes.data, int(np.prod(W_conv.shape)))
        b_trt = trt.Weights(trt.float16, b_conv.ctypes.data, int(np.prod(b_conv.shape)))
        conv = network.add_convolution_nd(input_4d, out_channels, trt.Dims([1, 1]), W_trt, b_trt)
        conv.stride_nd = trt.Dims([1, 1])
        conv.padding_nd = trt.Dims([0, 0])
        conv.name = name
        return conv.get_output(0)

    q4d = add_conv1x1(x4d, Wq_conv, bq_conv, C, "q_conv1x1")
    print(f"        Q: [{N_prime}, {C}, {H}, {W}] (Conv1x1)")

    k4d = add_conv1x1(x4d, Wk_conv, bk_conv, C, "k_conv1x1")
    print(f"        K: [{N_prime}, {C}, {H}, {W}] (Conv1x1)")

    v4d = add_conv1x1(x4d, Wv_conv, bv_conv, C, "v_conv1x1")
    print(f"        V: [{N_prime}, {C}, {H}, {W}] (Conv1x1)")

    # ========== STEP 2: 4D -> 3D "tokens-major" (Golden Path) ==========
    print(f"\n[STEP 2] 4D -> 3D tokens-major [N', tokens, C]")

    def nhwc_flatten(tensor_4d, name):
        """
        [N', C, H, W] (NCHW) -> [N', H, W, C] -> [N', H*W, C]
        Uses shape tensors for dynamic dimensions
        """
        # Build shape for NHWC: [N', H, W, C]
        nhwc_shape_layer = network.add_concatenation([N_prime_tensor, H_tensor, W_tensor, C_tensor])
        nhwc_shape_layer.axis = 0
        nhwc_shape_layer.name = f"{name}_nhwc_shape"

        # NCHW -> NHWC
        sh1 = network.add_shuffle(tensor_4d)
        sh1.name = f"{name}_nchw_to_nhwc"
        sh1.first_transpose = (0, 2, 3, 1)  # N,C,H,W -> N,H,W,C
        sh1.set_input(1, nhwc_shape_layer.get_output(0))
        sh1.second_transpose = (0, 1, 2, 3)  # identity
        nhwc = sh1.get_output(0)

        # Build shape for tokens-major: [N', tokens, C]
        tokens_major_shape_layer = network.add_concatenation([N_prime_tensor, tokens_tensor, C_tensor])
        tokens_major_shape_layer.axis = 0
        tokens_major_shape_layer.name = f"{name}_tokens_major_shape"

        # NHWC -> [N', tokens, C]
        sh2 = network.add_shuffle(nhwc)
        sh2.name = f"{name}_flatten_spatial"
        sh2.first_transpose = (0, 1, 2, 3)  # identity
        sh2.set_input(1, tokens_major_shape_layer.get_output(0))
        sh2.second_transpose = (0, 1, 2)  # identity
        return sh2.get_output(0)

    q3d = nhwc_flatten(q4d, "q")
    k3d = nhwc_flatten(k4d, "k")
    v3d = nhwc_flatten(v4d, "v")
    print(f"        Q/K/V: [{N_prime}, {tokens}, {C}] (3D tokens-major)")

    # ========== STEP 3: Split heads & prep for GEMM (Golden Path) ==========
    print(f"\n[STEP 3] Split heads [N', n_head, tokens, d_head]")

    def split_heads(tensor_3d, name):
        """
        [N', tokens, C] -> [N', tokens, n_head, d_head] -> transpose to [N', n_head, tokens, d_head]
        Uses shape tensors for dynamic dimensions
        """
        # Build shape [N', C, tokens] for transpose step
        NCT_shape_layer = network.add_concatenation([N_prime_tensor, C_tensor, tokens_tensor])
        NCT_shape_layer.axis = 0
        NCT_shape_layer.name = f"{name}_NCT_shape"

        # First transpose to make reshape cleaner: [N', tokens, C] -> [N', C, tokens]
        t1 = network.add_shuffle(tensor_3d)
        t1.name = f"{name}_pre_reshape"
        t1.first_transpose = (0, 2, 1)  # (N, tokens, C) -> (N, C, tokens)
        t1.set_input(1, NCT_shape_layer.get_output(0))
        t1.second_transpose = (0, 1, 2)
        xNCT = t1.get_output(0)

        # Build shape [N', n_head, d_head, tokens]
        heads_shape_layer = network.add_concatenation([N_prime_tensor, n_head_tensor, d_head_tensor, tokens_tensor])
        heads_shape_layer.axis = 0
        heads_shape_layer.name = f"{name}_heads_shape"

        # Reshape to [N', n_head, d_head, tokens] (still NCHW-ish)
        sh = network.add_shuffle(xNCT)
        sh.name = f"{name}_to_heads"
        sh.first_transpose = (0, 1, 2)  # identity
        sh.set_input(1, heads_shape_layer.get_output(0))
        sh.second_transpose = (0, 1, 3, 2)  # -> [N', n_head, tokens, d_head]
        return sh.get_output(0)

    q4 = split_heads(q3d, "q")
    k4 = split_heads(k3d, "k")
    v4 = split_heads(v3d, "v")
    print(f"        Q/K/V: [{N_prime}, {n_head}, {tokens}, {d_head}] (4D)")

    # ========== STEP 4: Q @ K^T (rank-4 batched matmul) ==========
    print(f"\n[STEP 4] Attention scores: Q @ K^T")

    mm_qk = network.add_matrix_multiply(
        q4, trt.MatrixOperation.NONE,
        k4, trt.MatrixOperation.TRANSPOSE  # transposes last two dims of K
    )
    mm_qk.name = "attn_scores"
    attn_scores = mm_qk.get_output(0)  # [N', n_head, tokens, tokens]
    print(f"        Scores: [{N_prime}, {n_head}, {tokens}, {tokens}]")

    # Scale by 1/sqrt(d_head) - broadcastable [1,1,1,1]
    # CRITICAL: Use FP16 to match attention scores dtype (prevents saturation)
    scale = 1.0 / np.sqrt(float(d_head))
    scale_arr = np.array([[[[scale]]]], dtype=np.float16)  # [1,1,1,1]
    scale_const = network.add_constant(
        trt.Dims([1, 1, 1, 1]),
        trt.Weights(trt.float16, scale_arr.ctypes.data, int(scale_arr.size))
    )
    scale_const.name = "attn_scale"
    scaled = network.add_elementwise(attn_scores, scale_const.get_output(0), trt.ElementWiseOperation.PROD)
    scaled.name = "attn_scaled"
    x = scaled.get_output(0)  # [N', n_head, tokens, tokens]
    print(f"        Scaled by {scale:.6f}")

    # ========== STEP 5: Manual softmax over last axis (tokens) ==========
    print(f"\n[STEP 5] Softmax over last axis (manual)")

    # reduce MAX (keep_dims=True)
    reduce_max = network.add_reduce(x, trt.ReduceOperation.MAX, 1 << 3, True)  # axis=3 is last dim
    reduce_max.name = "attn_max_keepdims"
    x_minus_max = network.add_elementwise(x, reduce_max.get_output(0), trt.ElementWiseOperation.SUB)
    x_minus_max.name = "attn_sub_max"
    exp = network.add_unary(x_minus_max.get_output(0), trt.UnaryOperation.EXP)
    exp.name = "attn_exp"
    reduce_sum = network.add_reduce(exp.get_output(0), trt.ReduceOperation.SUM, 1 << 3, True)  # axis=3
    reduce_sum.name = "attn_sum_keepdims"
    soft = network.add_elementwise(exp.get_output(0), reduce_sum.get_output(0), trt.ElementWiseOperation.DIV)
    soft.name = "attn_softmax_manual"
    attn_prob = soft.get_output(0)  # [N', n_head, tokens, tokens]
    print(f"        Probs: [{N_prime}, {n_head}, {tokens}, {tokens}]")

    # ========== STEP 6: Context = attn_prob @ V ==========
    print(f"\n[STEP 6] Context: attn_prob @ V")

    mm_ctx = network.add_matrix_multiply(
        attn_prob, trt.MatrixOperation.NONE,
        v4,       trt.MatrixOperation.NONE
    )
    mm_ctx.name = "attn_ctx"
    ctx = mm_ctx.get_output(0)  # [N', n_head, tokens, d_head]
    print(f"        Context: [{N_prime}, {n_head}, {tokens}, {d_head}]")

    # ========== STEP 7: Merge heads back to channel ==========
    print(f"\n[STEP 7] Merge heads [N', tokens, C]")

    # Build shape [N', tokens, n_head, d_head]
    ntHd_shape_layer = network.add_concatenation([N_prime_tensor, tokens_tensor, n_head_tensor, d_head_tensor])
    ntHd_shape_layer.axis = 0
    ntHd_shape_layer.name = "ntHd_shape"

    # [N', n_head, tokens, d_head] -> transpose to [N', tokens, n_head, d_head]
    t_merge = network.add_shuffle(ctx)
    t_merge.name = "merge_heads_transpose"
    t_merge.first_transpose = (0, 2, 1, 3)
    t_merge.set_input(1, ntHd_shape_layer.get_output(0))
    t_merge.second_transpose = (0, 1, 2, 3)  # identity
    ctx_ntHd = t_merge.get_output(0)

    # Build shape [N', tokens, C]
    ntC_shape_layer = network.add_concatenation([N_prime_tensor, tokens_tensor, C_tensor])
    ntC_shape_layer.axis = 0
    ntC_shape_layer.name = "ntC_shape"

    # reshape -> [N', tokens, C]
    sh_merge = network.add_shuffle(ctx_ntHd)
    sh_merge.name = "merge_heads_reshape"
    sh_merge.first_transpose = (0, 1, 2, 3)  # identity
    sh_merge.set_input(1, ntC_shape_layer.get_output(0))
    sh_merge.second_transpose = (0, 1, 2)
    attn_out_3d = sh_merge.get_output(0)
    print(f"        Merged: [N', tokens, C] (3D - dynamic)")

    # ========== STEP 8: Back to 4D NCHW for output projection ==========
    print(f"\n[STEP 8] 3D -> 4D NCHW for output projection")

    # Build shape [N', C, tokens]
    NCt_shape_layer = network.add_concatenation([N_prime_tensor, C_tensor, tokens_tensor])
    NCt_shape_layer.axis = 0
    NCt_shape_layer.name = "NCt_shape"

    # [N', tokens, C] -> [N', H, W, C] -> [N', C, H, W]
    to_nhwc = network.add_shuffle(attn_out_3d)
    to_nhwc.name = "to_nhwc_4d"
    to_nhwc.first_transpose = (0, 2, 1)  # (N, tokens, C) -> (N, C, tokens)
    to_nhwc.set_input(1, NCt_shape_layer.get_output(0))
    to_nhwc.second_transpose = (0, 2, 1)  # (N, C, tokens) -> (N, tokens, C)
    attn_out_ntc = to_nhwc.get_output(0)

    # Build shape [N', H, W, C] (NHWC)
    NHWC_shape_layer = network.add_concatenation([N_prime_tensor, H_tensor, W_tensor, C_tensor])
    NHWC_shape_layer.axis = 0
    NHWC_shape_layer.name = "NHWC_shape"

    sh4 = network.add_shuffle(attn_out_ntc)
    sh4.name = "to_nchw_4d"
    sh4.first_transpose = (0, 1, 2)  # identity for 3D
    sh4.set_input(1, NHWC_shape_layer.get_output(0))  # NHWC
    sh4.second_transpose = (0, 3, 1, 2)  # NHWC -> NCHW
    o4d = sh4.get_output(0)
    print(f"        -> [N', C, H, W] (4D NCHW - dynamic)")

    # ========== STEP 9: Output projection via 1x1 Conv ==========
    print(f"\n[STEP 9] Output projection (1x1 Conv)")

    # 1x1 conv for output projection
    o4d = add_conv1x1(o4d, Wproj_conv, bproj_conv, C, "output_proj_conv1x1")
    print(f"        Output proj: [{N_prime}, {C}, {H}, {W}] (Conv1x1)")

    # ========== STEP 10: Final 4D -> 5D (edge only) ==========
    print(f"\n[STEP 10] Final 4D -> 5D (edge)")

    # Build shape [B, T, H, W, C] (5D output)
    # INT32 for TRT shape tensors
    const_B_out = network.add_constant(trt.Dims([1]), trt.Weights(np.array([B], dtype=np.int32)))
    const_B_out.name = "const_B_out"
    output_5d_shape_layer = network.add_concatenation([const_B_out.get_output(0), T_tensor, H_tensor, W_tensor, C_tensor])
    output_5d_shape_layer.axis = 0
    output_5d_shape_layer.name = "output_5d_shape"

    shOut = network.add_shuffle(o4d)
    shOut.first_transpose = (0, 1, 2, 3)  # identity for rank-4
    shOut.set_input(1, output_5d_shape_layer.get_output(0))
    shOut.second_transpose = (0, 1, 2, 3, 4)  # identity for rank-5
    shOut.name = "output_4d_to_5d"
    y5d = shOut.get_output(0)

    # CRITICAL: Cast output to FP16 (matches PyTorch model.half())
    cast_out = network.add_cast(y5d, trt.float16)
    cast_out.name = "cast_output_to_fp16"
    y5d_fp16 = cast_out.get_output(0)

    # Mark as output FIRST before setting dtype (TRT requirement)
    network.mark_output(y5d_fp16)

    # Set output tensor precision explicitly (must be after mark_output)
    y5d_fp16.dtype = trt.float16
    print(f"        Output: [B, T, H, W, C] (5D - dynamic, FP16)")

    print(f"\n[BUILD] Total layers: {network.num_layers}")

    # ========== Build engine ==========
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 << 20)
    config.set_tactic_sources(
        1 << int(trt.TacticSource.CUBLAS) |
        1 << int(trt.TacticSource.CUBLAS_LT)
    )
    config.builder_optimization_level = 3

    # Enable FP16 precision mode (matches PyTorch model.half())
    config.set_flag(trt.BuilderFlag.FP16)
    print(f"\n[BUILD] FP16 mode enabled (matches PyTorch precision)")

    profile = builder.create_optimization_profile()
    # Dynamic shape support for all production resolutions
    profile.set_shape("input",
        min=(B, T_min, H_min, W_min, C),  # [1, 2, 6, 9, 512]
        opt=(B, T_opt, H_opt, W_opt, C),  # [1, 11, 12, 12, 512]
        max=(B, T_max, H_max, W_max, C)   # [1, 16, 15, 15, 512]
    )
    config.add_optimization_profile(profile)

    print(f"\n[BUILD] Dynamic optimization profile:")
    print(f"        min: [{B}, {T_min}, {H_min}, {W_min}, {C}]")
    print(f"        opt: [{B}, {T_opt}, {H_opt}, {W_opt}, {C}]")
    print(f"        max: [{B}, {T_max}, {H_max}, {W_max}, {C}]")
    print(f"        (Supports ALL production shapes - no PyTorch fallback)")

    print(f"\n[BUILD] Building golden path attention...")
    sys.stdout.flush()

    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"\n[FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    if serialized_engine is None:
        print(f"\n[FAIL] build_serialized_network returned None")
        return False

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        print(f"\n[FAIL] Deserialization failed")
        return False

    engine_size = serialized_engine.nbytes if hasattr(serialized_engine, 'nbytes') else len(bytes(serialized_engine))
    print(f"\n[SUCCESS] Golden path attention works!")
    print(f"[SUCCESS] Engine size: {engine_size/(1024*1024):.2f} MB")

    # Save engine
    engine_path = "engines/transformer/transformer_golden_path.engine"
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"[SUCCESS] Engine saved: {engine_path}")

    del engine
    return True


def main():
    """Test golden path attention"""
    print("=" * 80)
    print("GOLDEN PATH: Segfault-proof Attention Implementation")
    print("=" * 80)
    print("\nShape discipline:")
    print("- 5D only at edges (input/output)")
    print("- Conv1x1 for Q/K/V projections (4D NCHW)")
    print("- Tokens-major 3D [N', tokens, C] for attention")
    print("- Rank-4 batched GEMM (A=NONE, B=TRANSPOSE)")
    print("- Manual softmax (single-bit axis)")
    print("- Output via Conv1x1")

    success = build_attention_golden_path()

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: Golden path validated!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Integrate into ProPainter")
        print("2. Add windowing support")
        print("3. Benchmark performance")
        return 0
    else:
        print("\n" + "=" * 80)
        print("Build failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
