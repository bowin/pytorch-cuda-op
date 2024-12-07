import torch
import matrix_add

# Create two random matrices on CUDA
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')

# Call the custom CUDA operator
C = torch.ops.matrix_add.matrix_add(A, B)

# Verify the result
expected = A + B
print(torch.allclose(C, expected))
