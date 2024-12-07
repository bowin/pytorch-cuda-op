# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# 获取 PyTorch 的 include 路径
torch_include = torch.utils.cpp_extension.include_paths()

setup(
    name='matrix_add',
    ext_modules=[
        CUDAExtension('matrix_add', [
            'matrix_add_kernel.cc',
            'matrix_add.cu',
        ],
        include_dirs=torch_include),  # 添加 PyTorch 的 include 路径
    ],
    cmdclass={
        'build_ext': BuildExtension
    })