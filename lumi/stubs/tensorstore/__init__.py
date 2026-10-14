# Shadows the container's tensorstore in the nemo env: it is built against NumPy 1 (the container has NumPy 2)
# and NeMo only needs it for distributed checkpoints, not for inference.
raise ImportError("tensorstore disabled for this eval: built against NumPy 1")
