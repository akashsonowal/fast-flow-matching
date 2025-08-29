# train_aot_backend.py
# ------------------------------------------------------------
# Custom AOTAutograd backend for TRAINING.
# - Compiles forward & backward graphs (AOTAutograd).
# - Uses TorchInductor when available; else falls back to eager.
# - Returns BOXED functions (required by AOTAutograd).
# - Minimal, diff-safe FX rewrite pass.
# - AMP-ready training loop to demo end-to-end.
# ------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule
from torch._dynamo.backends.common import aot_autograd

# make_boxed_func is required for AOTAutograd backends
try:
    from functorch.compile import make_boxed_func
except Exception:
    # older/newer PyTorch builds sometimes place it here
    from torch._functorch.compile import make_boxed_func  # type: ignore

# Optional TorchInductor compile_fx (best speed); keep a safe fallback
try:
    from torch._inductor.compile_fx import compile_fx as inductor_compile_fx  # type: ignore
except Exception:
    inductor_compile_fx = None


# -------------------------
# Model (simple flow-ish MLP)
# -------------------------
class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super().__init__()
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t: torch.Tensor, max_positions: int = 10000):
        # keep dtype/device consistent (important for AMP / mixed precision)
        device, dtype = t.device, t.dtype
        half_dim = self.channels_t // 2

        log_max = torch.log(torch.tensor(max_positions, device=device, dtype=dtype))
        scale = log_max / (half_dim - 1)

        inv_freq = torch.exp(-torch.arange(half_dim, device=device, dtype=dtype) * scale)
        t_scaled = t * torch.tensor(max_positions, device=device, dtype=dtype)

        emb = t_scaled[:, None] * inv_freq[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels_t % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, t):
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x


# -------------------------
# Differentiable helpers (safe swaps)
# -------------------------
def gelu_fast_tanh(x: torch.Tensor) -> torch.Tensor:
    # Fully differentiable GELU approx; if you use GELU in your model,
    # the rewriter (below) can swap to this.
    return F.gelu(x, approximate="tanh")


# -------------------------
# FX rewrite pass for TRAINING
# Keep things differentiable & friendly to AOTAutograd.
# -------------------------
def _rewrite_fx_train(gm: GraphModule) -> GraphModule:
    g = gm.graph
    for n in list(g.nodes):
        if n.op == "call_function":
            if n.target is F.gelu:
                n.target = gelu_fast_tanh
            # (ReLU is fine; SDPA/FlashAttention swaps should be avoided here
            #  unless you have differentiable wrappers.)
    g.lint()
    gm.recompile()
    return gm


# -------------------------
# Compilers handed to AOTAutograd
# Must RETURN BOXED CALLABLES (make_boxed_func)
# -------------------------
def my_fw_compiler(gm: GraphModule, example_inputs):
    # Optional: inspect the graph during dev
    # print("FW Graph:"); gm.graph.print_tabular()

    gm = _rewrite_fx_train(gm)

    # Best: Inductor (if available)
    if inductor_compile_fx is not None:
        compiled = inductor_compile_fx(gm, example_inputs)
        return make_boxed_func(compiled)

    # Fallback: just run the FX module as-is
    return make_boxed_func(gm.forward)


def my_bw_compiler(gm: GraphModule, example_inputs):
    # Backward graph produced by AOTAutograd (core aten ops)
    gm = _rewrite_fx_train(gm)

    if inductor_compile_fx is not None:
        compiled = inductor_compile_fx(gm, example_inputs)
        return make_boxed_func(compiled)

    return make_boxed_func(gm.forward)


# Wrap our compilers with AOTAutograd; this backend goes into torch.compile(...)
my_training_backend = aot_autograd(
    fw_compiler=my_fw_compiler,
    bw_compiler=my_bw_compiler,  # omit to default to fw if you prefer
    # You can also pass partition_fn/decompositions here if needed.
)


# -------------------------
# Minimal AMP training demo
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = MLP().to(device)  # keep weights in fp32 for stability
    model.train()

    # Compile WITH our AOTAutograd training backend
    model = torch.compile(model, backend=my_training_backend)  # fullgraph=True optional

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16  # prefer bf16 on modern GPUs; no GradScaler needed

    steps, batch = 40, 64
    for step in range(steps):
        x = torch.randn(batch, 2, device=device)
        t = torch.rand(batch, device=device)
        target = torch.zeros(batch, 2, device=device)

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x, t)
                loss = F.mse_loss(out, target)
        else:
            out = model(x, t)
            loss = F.mse_loss(out, target)

        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step:03d} | loss {loss.item():.6f}")

    # quick eval
    model.eval()
    with torch.inference_mode():
        x = torch.randn(8, 2, device=device)
        t = torch.rand(8, device=device)
        y = model(x, t)
    print("Eval:", y.shape, y.dtype)


if __name__ == "__main__":
    main()