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

cmake_minimum_required(VERSION 3.20)

project(VolumeIntegration LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaIntegratorLib
	cuda_integrator/backward.h
	cuda_integrator/forward.h
	cuda_integrator/forward.cu
	cuda_integrator/auxiliary.h
	cuda_integrator/integrator_impl.cu
	cuda_integrator/integrator_impl.h
	cuda_integrator/integrator.h
	cuda_integrator/config.h
)

set_target_properties(CudaIntegratorLib PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaIntegratorLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_integrator)
target_include_directories(CudaIntegratorLib PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
