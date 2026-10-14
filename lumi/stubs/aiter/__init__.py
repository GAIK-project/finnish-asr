# Shadows the container's aiter (AMD kernels) in the nemo env: NeMo 3.0 reaches it through torch.ops,
# and its JIT build needs a C++ compiler the LUMI PyTorch container does not have.
raise ImportError("aiter disabled for this eval: its JIT build needs a C++ compiler the container lacks")
