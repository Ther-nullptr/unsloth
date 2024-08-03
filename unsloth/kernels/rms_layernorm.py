# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch
from .utils import calculate_settings

from .compress_function import *

@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)#.to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)
pass


@triton.heuristics({"GEMMA": lambda args: args["GEMMA"],})
@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X,   X_row_stride,
    W,   W_row_stride,
    r,   r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    GEMMA      : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask = mask)
pass


@triton.jit
def _gemma_rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr,
):
    # Copies https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L31
    # and https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L33
    # exactly. Essentially all in float32!
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)
pass


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma = False,
                iteration=0, calibration_step=5, register_target=None,
                rank=4, outlier_ratio=0.005, quantize_bit=8, quantize_method='per-channel'):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = "cuda:0")
        r = torch.empty(n_rows, dtype = torch.float32, device = "cuda:0")

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        fx[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.GEMMA = gemma
        
        # reshape the X
        X = X.view(*shape)
        
        # if iteration < calibration_step, we nedd to collect the statistics
        if iteration < calibration_step:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                outlier_X, L_X, R_X, scale_X = get_statistics_compress(X, iteration, outlier_ratio, 1., quantize_bit, quantize_method, rank)
            if iteration == 0: # register
                register_target.register_buffer('outlier', torch.tensor(outlier_X).to(torch.bfloat16))
                register_target.register_buffer('L', L_X)
                register_target.register_buffer('R', R_X)
                register_target.register_buffer('scale', scale_X)
            else: # compute the average
                register_target.outlier = (register_target.outlier * iteration + outlier_X) / (iteration + 1)
                register_target.L = (register_target.L * iteration + L_X) / (iteration + 1)
                register_target.R = (register_target.R * iteration + R_X) / (iteration + 1)
                register_target.scale = (register_target.scale * iteration + scale_X) / (iteration + 1)
        
        outlier_X_compressed, quantized_X_compressed = low_rank_subtraction_fuse_compression_quantization(
            l = register_target.L, 
            r = register_target.R, 
            x = X, 
            s = register_target.scale, 
            quantize_bit = quantize_bit,
            outlier = register_target.outlier, 
        )
        ctx.quantize_bit = quantize_bit        
        ctx.save_for_backward(outlier_X_compressed, quantized_X_compressed, register_target.L, register_target.R, register_target.scale, W, r)
        
        return Y.view(*shape)
    pass

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        outlier_X_compressed, quantized_X_compressed, X_L, X_R, scale_X, W, r = ctx.saved_tensors
        
        X = low_rank_addition_fuse_decompression_dequantization(
            l=X_L,
            r=X_R,
            q=quantized_X_compressed,
            o=outlier_X_compressed,
            s=scale_X,
            quantize_bit=ctx.quantize_bit,
        )
        
        dim = shape[-1]
        X = X.view(-1, dim)
        
        n_rows, n_cols = dY.shape
        dW = X

        _rms_layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X .stride(0),
            W,  W .stride(0),
            r,  r .stride(0),
            dW, dW.stride(0),
            n_cols, ctx.eps,
            GEMMA      = ctx.GEMMA,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None, None, None, None, None, None, None, None
    pass
pass


def fast_rms_layernorm(layernorm, X, gemma = False):
    W   = layernorm.weight
    eps = layernorm.variance_epsilon
    out = Fast_RMS_Layernorm.apply(
        X, W, eps, gemma,
        layernorm.iteration, layernorm.calibration_step, layernorm
    )
    layernorm.iteration += 1
    return out
pass
