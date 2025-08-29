from torch.fx import GraphModule
from torch._dynamo import register_backend

from hf_kernels import gelu_fast_kernel, silu_fast_kernel, sdpa_fast_kernel


# def _rewrite_graph_with_hf_kernels(gm: GraphModule) -> GraphModule:
#     """
#     Replace:
#       - F.gelu -> gelu_fast_kernel
#       - F.silu -> silu_fast_kernel
#       - F.scaled_dot_product_attention -> sdpa_flash_kernel
#     (ReLU is already quite cheap; keep as-is.)
#     """
#     graph = gm.graph
#     for node in list(graph.nodes):
#         if node.op == "call_function":
#             if node.target is F.gelu:
#                 node.target = gelu_fast_kernel
#             elif node.target is F.silu:
#                 node.target = silu_fast_kernel
#             elif node.target is F.scaled_dot_product_attention:
#                 node.target = sdpa_flash_kernel
#     graph.lint()
#     gm.recompile()
#     return gm


# @register_backend
# def hf_kernelize(gm: GraphModule, example_inputs):
#     # For visibility during development:
#     print("my_compiler() called with FX graph:")
#     gm.graph.print_tabular()

#     # Optional: TF32 matmul on Ampere+ (safe for inference)
#     try:
#         torch.set_float32_matmul_precision("high")
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#     except Exception:
#         pass

#     gm = _rewrite_graph_with_hf_kernels(gm)
#     # Return a callable. We just forward the re-written GraphModule.
#     return gm.forward

@register_backend
def hf_kernels_compiler(gm: GraphModule, example_inputs):
    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_function":
            print(node.target)
    return gm.forward