import torch
print(torch.__version__)
print(torch.backends.mps.is_available())
# python -m torch.utils.collect_env