import torch
from kernels import get_kernel

activation = get_kernel("kernels-community/activation")

def gelu_fast_kernel(x: torch.Tensor) ->  torch.Tensor:
    y = torch.empty_like(x)
    activation.gelu_fast(y, x)
    return y


# -------------------------
# Kernel-backed wrappers
# -------------------------
_activation_k = None
_flash_k = None


def _lazy_load_kernels():
    global _activation_k, _flash_k
    if get_kernel is None:
        return
    if _activation_k is None:
        try:
            # Example pack: fast GELU/SILU/etc. (may vary by hub repo)
            _activation_k = get_kernel("kernels-community/activation")
        except Exception:
            _activation_k = None
    if _flash_k is None:
        try:
            # Example pack: FlashAttention wrapper
            _flash_k = get_kernel("kernels-community/flash-attn")
        except Exception:
            _flash_k = None


def gelu_fast_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Fast GELU via HF kernels when possible; otherwise fall back.
    (Your model uses ReLU, but this enables speedups if you switch to GELU.)
    """
    _lazy_load_kernels()
    if _activation_k is None or (not x.is_cuda) or x.dtype == torch.float32:
        return F.gelu(x)
    y = torch.empty_like(x)
    if hasattr(_activation_k, "gelu_fast"):
        _activation_k.gelu_fast(y, x)
        return y
    return F.gelu(x)


def silu_fast_kernel(x: torch.Tensor) -> torch.Tensor:
    _lazy_load_kernels()
    if _activation_k is None or (not x.is_cuda) or x.dtype == torch.float32:
        return F.silu(x)
    y = torch.empty_like(x)
    if hasattr(_activation_k, "silu_fast"):
        _activation_k.silu_fast(y, x)
        return y
    return F.silu(x)


def sdpa_flash_kernel(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """
    Conservative FlashAttention drop-in for F.scaled_dot_product_attention.
    Only used if shapes/dtypes/conditions are friendly; otherwise fallback.
    (Your current MLP doesn’t call SDPA, but this supports attention variants.)
    """
    _lazy_load_kernels()
    use_flash = (
        _flash_k is not None
        and attn_mask is None
        and q.is_cuda and k.is_cuda and v.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype and v.dtype == q.dtype
        and dropout_p == 0.0
    )
    if not use_flash:
        return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)

    softmax_scale = 1.0 if scale is None else float(scale)
    out = _flash_k.mha_fwd(q=q, k=k, v=v, p_dropout=0.0, is_causal=is_causal, softmax_scale=softmax_scale)[0]
    return out