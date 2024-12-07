// matrix_add_kernel.h
#ifndef MATRIX_ADD_KERNEL_H
#define MATRIX_ADD_KERNEL_H

#include <torch/torch.h>
#include <cuda_runtime.h>

__global__ void matrixAddKernel(const float* A, const float* B, float* C, int numElements);
torch::Tensor matrix_add_cuda(torch::Tensor A, torch::Tensor B);
#endif // MATRIX_ADD_KERNEL_H
