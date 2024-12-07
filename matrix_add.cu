// matrix_add.cu
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// #include <pybind11/pybind11.h>

__global__ void matrixAddKernel(const float* A, const float* B, float* C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

torch::Tensor matrix_add_cuda(torch::Tensor A, torch::Tensor B) {
    int numElements = A.numel();  // 获取元素数量
    auto C = torch::empty_like(A);  // 创建输出张量

    const float* Adata = A.data_ptr<float>();
    const float* Bdata = B.data_ptr<float>();
    float* Cdata = C.data_ptr<float>();

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(Adata, Bdata, Cdata, numElements);

    return C;
}

// PYBIND11_MODULE(matrix_add, m) {
//     m.def("matrix_add_cuda", &matrix_add_cuda, "A function to add two matrices using CUDA");
// }

// Registers CUDA implementations for mymuladd, mymul, myadd_out

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(matrix_add, m) {
  m.def("matrix_add(Tensor a, Tensor b) -> Tensor");
}


TORCH_LIBRARY_IMPL(matrix_add, CUDA, m) {
  m.impl("matrix_add", &matrix_add_cuda);
}