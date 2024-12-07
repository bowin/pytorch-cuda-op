// matrix_add.cpp
#include <torch/torch.h>
#include "matrix_add_kernel.h"

torch::Tensor matrix_add(torch::Tensor A, torch::Tensor B) {
    if (A.device().type() == torch::kCUDA && B.device().type() == torch::kCUDA) {
        return matrix_add_cuda(A, B);
    } else {
        throw std::runtime_error("Both tensors must be on CUDA devices.");
    }
}
