import torch
import time

device = torch.device("cuda")

x = torch.randn(5000, 5000).to(device)

start = time.time()
for _ in range(50):
    y = torch.matmul(x, x)

torch.cuda.synchronize()
end = time.time()

print("Time taken:", end - start)