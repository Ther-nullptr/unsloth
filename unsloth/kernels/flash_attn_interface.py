import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .compress_function import *

def efficient_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    iteration=0, calibration_step=5, register_target=None,
    rank=0, outlier_ratio=0.005, quantize_bit=8, quantize_method='per-channel'
):
    return EfficientFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        iteration, calibration_step, register_target,
        rank, outlier_ratio, quantize_bit, quantize_method
    )


class EfficientFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        iteration=0, calibration_step=5, register_target=None,
        rank=0, outlier_ratio=0.005, quantize_bit=8, quantize_method='per-channel'
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        #! deal with X, e, g
        if iteration < calibration_step:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                outlier_X, L_X, R_X, scale_X = get_statistics_compress(q, iteration, outlier_ratio, 1., quantize_bit, quantize_method, rank)
                outlier_e, L_e, R_e, scale_e = get_statistics_compress(k, iteration, outlier_ratio, 1., quantize_bit, quantize_method, rank)
                outlier_g, L_g, R_g, scale_g = get_statistics_compress(v, iteration, outlier_ratio, 1., quantize_bit, quantize_method, rank)
            if iteration == 0: # register
                # for X
                register_target.register_buffer('outlier_X', outlier_X)
                register_target.register_buffer('L_X', L_X)
                register_target.register_buffer('R_X', R_X)
                register_target.register_buffer('scale_X', scale_X)
                # for e
                register_target.register_buffer('outlier_e', outlier_e)
                register_target.register_buffer('L_e', L_e)
                register_target.register_buffer('R_e', R_e)
                register_target.register_buffer('scale_e', scale_e)
                # for g
                register_target.register_buffer('outlier_g', outlier_g)
                register_target.register_buffer('L_g', L_g)
                register_target.register_buffer('R_g', R_g)
                register_target.register_buffer('scale_g', scale_g)
            else: # compute the average
                # for X
                register_target.outlier_X = (register_target.outlier_X * iteration + outlier_X) / (iteration + 1)
                register_target.L_X = (register_target.L_X * iteration + L_X) / (iteration + 1)
                register_target.R_X = (register_target.R_X * iteration + R_X) / (iteration + 1)
                register_target.scale_X = (register_target.scale_X * iteration + scale_X) / (iteration + 1)
                # for e
                register_target.outlier_e = (register_target.outlier_e * iteration + outlier_e) / (iteration + 1)
                register_target.L_e = (register_target.L_e * iteration + L_e) / (iteration + 1)
                register_target.R_e = (register_target.R_e * iteration + R_e) / (iteration + 1)
                register_target.scale_e = (register_target.scale_e * iteration + scale_e) / (iteration + 1)
                # for g
                register_target.outlier_g = (register_target.outlier_g * iteration + outlier_g) / (iteration + 1)
                register_target.L_g = (register_target.L_g * iteration + L_g) / (iteration + 1)
                register_target.R_g = (register_target.R_g * iteration + R_g) / (iteration + 1)
                register_target.scale_g = (register_target.scale_g * iteration + scale_g) / (iteration + 1)
        
        outlier_X_compressed, quantized_X_compressed = low_rank_subtraction_fuse_compression_quantization(
            l = register_target.L_X, 
            r = register_target.R_X,
            x = q,
            s = register_target.scale_X, 
            quantize_bit = quantize_bit, 
            outlier = register_target.outlier_X, 
        )
        outlier_e_compressed, quantized_e_compressed = low_rank_subtraction_fuse_compression_quantization(
            l = register_target.L_e,
            r = register_target.R_e,
            x = k,
            s = register_target.scale_e,
            quantize_bit = quantize_bit,
            outlier = register_target.outlier_e,
        )
        outlier_g_compressed, quantized_g_compressed = low_rank_subtraction_fuse_compression_quantization(
            l = register_target.L_g,
            r = register_target.R_g,
            x = v,
            s = register_target.scale_g,
            quantize_bit = quantize_bit,
            outlier = register_target.outlier_g,
        )
        ctx.quantize_bit = quantize_bit
        ctx.num_heads = q.shape[2]
        ctx.k_num_heads = k.shape[2]
        
        ctx.save_for_backward(
            outlier_X_compressed, quantized_X_compressed, register_target.L_X, register_target.R_X, scale_X,
            outlier_e_compressed, quantized_e_compressed, register_target.L_e, register_target.R_e, scale_e,
            outlier_g_compressed, quantized_g_compressed, register_target.L_g, register_target.R_g, scale_g,
            out_padded, softmax_lse, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        outlier_X_compressed, quantized_X_compressed, X_L, X_R, scale_X, \
            outlier_e_compressed, quantized_e_compressed, e_L, e_R, scale_e, \
            outlier_g_compressed, quantized_g_compressed, g_L, g_R, scale_g, out, softmax_lse, rng_state = ctx.saved_tensors
        q = low_rank_addition_fuse_decompression_dequantization(
            l=X_L,
            r=X_R,
            q=quantized_X_compressed,
            o=outlier_X_compressed,
            s=scale_X,
            quantize_bit=ctx.quantize_bit,
            is_head=True,
            num_heads=ctx.num_heads
        )
        k = low_rank_addition_fuse_decompression_dequantization(
            l=e_L,
            r=e_R,
            q=quantized_e_compressed,
            o=outlier_e_compressed,
            s=scale_e,
            quantize_bit=ctx.quantize_bit,
            is_head=True,
            num_heads=ctx.num_k_heads
        )
        v = low_rank_addition_fuse_decompression_dequantization(
            l=g_L,
            r=g_R,
            q=quantized_g_compressed,
            o=outlier_g_compressed,
            s=scale_g,
            quantize_bit=ctx.quantize_bit,
            is_head=True,
            num_heads=ctx.num_k_heads
        )
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None
