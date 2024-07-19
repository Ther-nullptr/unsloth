import torch
from .compress_function_kernel import *


def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim) #.transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, seq_len, num_heads, head_dim = x.shape
    return x.reshape(bsz, seq_len, -1)


def low_rank_subtraction_fuse_compression_quantization(l, r, x, s, quantize_bit=8, outlier=5.):
    # Check constraints.
    assert l.shape[1] == r.shape[0], "Incompatible dimensions"
    assert l.is_contiguous(), "Matrix A must be contiguous"
    assert r.is_contiguous(), "Matrix B must be contiguous"
    
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)
    
    M, K = l.shape
    K, N = r.shape
    B, _, _ = x.shape
    outlier = outlier.item()
    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    o = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    low_rank_subtraction_fuse_compression_quantization_kernel[grid](
        l, r, x, o, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2), 
        o.stride(0), o.stride(1), o.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        outlier, quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    o = o.view(-1).to_sparse() # view to save the index memory
    return o, q


def low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit=8, is_head=False, num_heads=1):
    assert l.shape[1] == r.shape[0], "Incompatible dimensions"
    assert l.is_contiguous(), "Matrix A must be contiguous"
    assert r.is_contiguous(), "Matrix B must be contiguous"
    M, K = l.shape
    K, N = r.shape
    B, _, _ = q.shape
    
    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16
    
    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=l.device, dtype=torch.bfloat16)
    x_temp = torch.empty((B, M, N), device=l.device, dtype=torch.uint8)
    o = o.to_dense().view(B, M, N)
    
    low_rank_addition_fuse_decompression_dequantization_kernel[grid](
        l, r, x, x_temp, o, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        x_temp.stride(0), x_temp.stride(1), x_temp.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    del x_temp
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


@torch.no_grad
def true_compress_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.to_sparse()
    return x_outlier_sparse


@torch.no_grad
def true_decompress_softmax(x_sparse: torch.Tensor):
    return x_sparse.to_dense()


def prune_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    return x_outlier


def profile_memory(name):
    print(f'---------------------{name}---------------------')
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
    print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    print('--------------------------------------------------')


@torch.no_grad
def get_statistics_compress(x: torch.Tensor, iteration: int, outlier_ratio: float, sub_outlier_ratio: float, sub_outlier_bit: int = 8, sub_outlier_quantize_method: str = 'per-tensor', svd_rank: int = 16):    
    if len(x.shape) == 4:
        batch, seq_len, num_head, sep_dim = x.shape
        x = x.reshape(batch, seq_len, num_head * sep_dim)
    if svd_rank > 0:
        # profile_memory('before_svd')
        with torch.no_grad():
            U, S, V = torch.svd_lowrank(x[0].to(torch.float32), q=svd_rank, niter=16)
            L = U
            L = L.contiguous()
            R = torch.diag(S) @ V.T
            R = R.contiguous()
            x = x - L @ R
            del U, S, V
    else:
        L = torch.zeros((x.shape[-2], 1)).to(x.device).to(x.dtype)
        R = torch.zeros((1, x.shape[-1])).to(x.device).to(x.dtype)

    outlier = torch.kthvalue(x[0].flatten().to(torch.float32), int(x[0].numel() * (1 - outlier_ratio))).values
    x_outlier = x * (x.abs() > outlier)
    x = x - x_outlier
    
    if sub_outlier_ratio > 0 and sub_outlier_bit != 16:
        x_sub_outlier = x[0]
        if sub_outlier_quantize_method == 'per-tensor':
            # TODO: set the scale factor to per channel or per tensor?
            scale = (x_sub_outlier.max() - x_sub_outlier.min()) / (2 ** sub_outlier_bit - 1)
        elif sub_outlier_quantize_method == 'per-channel':
            # channel dimension: -2
            scale = (x_sub_outlier.max(dim=-2, keepdim=True).values - x_sub_outlier.min(dim=-2, keepdim=True).values) / (2 ** sub_outlier_bit - 1)
        elif sub_outlier_quantize_method == 'per-token':
            # token dimension: -1
            scale = (x_sub_outlier.max(dim=-1, keepdim=True).values - x_sub_outlier.min(dim=-1, keepdim=True).values) / (2 ** sub_outlier_bit - 1)
        else:
            raise "Unsupport Quantize Method"
    else:
        scale = torch.tensor(1.).cuda()
    
    return outlier.item(), L.to(torch.bfloat16), R.to(torch.bfloat16), scale.to(torch.bfloat16)


@torch.no_grad
def get_statistics_softmax(x: torch.Tensor, iteration: int, outlier_ratio: float):
    outlier = torch.kthvalue(x[0].float().flatten(), int(x[0].numel() * (1 - outlier_ratio))).values
    return outlier


@torch.no_grad
def pad_cut_L(src_L, tgt_L):
    seq_len_1, r = src_L.shape
    seq_len_2, _ = tgt_L.shape
    if seq_len_1 < seq_len_2:
        src_L = torch.cat((src_L, torch.zeros(seq_len_2 - seq_len_1, r).to(src_L.dtype).to(src_L.device)), dim=0)
    elif seq_len_1 > seq_len_2:
        src_L = src_L[0:seq_len_2, :]
    return src_L.contiguous()
