import torch

# 假设 tensors 是一个包含了需要相加的张量的列表
tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]

# 使用 sum 函数逐元素相加
result = sum(tensors)

print(result)