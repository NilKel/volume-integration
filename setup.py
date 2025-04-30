#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
base_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="volume_integration",
    version="0.1.0",
    description="Volume integration library based on CUDA and PyTorch",
    packages=find_packages(include=['volume_integration', 'volume_integration.*']),
    ext_modules=[
        CUDAExtension(
            name="volume_integration._C",
            sources=[
                "cuda_integrator/integrator_impl.cu",
                "cuda_integrator/forward.cu",
                "integrate_primitives.cu",
                "ext.cpp",
                "cuda_integrator/backward.cu"
            ],
            include_dirs=[
                os.path.join(base_dir, "cuda_integrator"),
                os.path.join(base_dir, "third_party/glm")
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
