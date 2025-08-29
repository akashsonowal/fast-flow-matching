# torch.manual_seed(42)
# model.eval().requires_grad_(False)
# xt = torch.randn(1000, 2)
# steps = 1000
# plot_every = 100
# for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
#     pred = model(xt, t.expand(xt.size(0)))
#     xt = xt + (1 / steps) * pred
#     if i % plot_every == 0:
#         plt.figure(figsize=(6, 6))
#         plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color="red", marker="o")
#         plt.scatter(xt[:, 0], xt[:, 1], color="green", marker="o")
#         plt.show()
# model.train().requires_grad_(True)
# print("Done Sampling")

import torch

from hf_compiler import hf_kernels_compiler as hf


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose one mixed-precision dtype for best perf on GPU.
    # bfloat16 is often more numerically stable than float16.
    run_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = MLP().to(device, dtype=run_dtype)

    # Optionally load a checkpoint (uncomment if you have it)
    # ckpt = torch.load("models/model_500k.pt", map_location=device)
    # model.load_state_dict(ckpt)

    # Compile with our HF-kernel-aware backend
    model = torch.compile(model, backend=hf_kernelize)

    # Create inputs in the SAME dtype/device as the model
    x = torch.randn(8, 2, device=device, dtype=run_dtype)
    t = torch.rand(8, device=device, dtype=run_dtype)

    with torch.inference_mode():
        y = model(x, t)

    print("Output:", y.shape, y.dtype)


if __name__ == "__main__":
    main()