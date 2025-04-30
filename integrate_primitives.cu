/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_integrator/config.h"
#include "cuda_integrator/integrator.h"
#include <fstream>
#include <string>
#include <functional>
#include "integrate_primitives.h"
#include <c10/cuda/CUDAStream.h>

// Define constants for block dimensions if not defined elsewhere
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

// Helper function to create resizing lambdas for buffers allocated on GPU
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        // Resize the tensor to the requested byte size N
        // Note: Tensor size is in number of elements, so divide N by element size if needed.
        // Assuming the buffer will store bytes, resize to N directly.
        // Ensure the tensor has at least N bytes capacity.
        // If N is 0, resize to 0 elements.
        if (N == 0) {
             t.resize_({0});
        } else {
             // Resize based on bytes. Assuming dtype is uint8 (byte).
             t.resize_({(long long)N});
        }
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// Updated forward function for PyTorch binding
// Returns: num_rendered, out_color, out_features, visibility_info,
//          geomBuffer, binningBuffer, imgBuffer (state buffers for backward)
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IntegratePrimitivesCUDA(
    // Primitive Data
    const torch::Tensor& means3D,               // (P, 3) Primitive centers
    const torch::Tensor& primitive_confidences, // (P, grid_size^3) Confidence grids
    const float primitive_scale,                // Uniform scale for primitives
    // Camera Parameters
    const torch::Tensor& viewmatrix,            // (4, 4) View matrix
    const torch::Tensor& projmatrix,            // (4, 4) Projection matrix
    const torch::Tensor& cam_pos,               // (3) Camera position vector
    const float tan_fovx,                       // tan(fov_x / 2)
    const float tan_fovy,                       // tan(fov_y / 2)
    const int image_height,                     // Image height H
    const int image_width,                      // Image width W
    const float near_plane,                     // Near clipping plane distance
    const float max_distance,                   // Far clipping plane distance (max ray dist)
    // Hashgrid Data
    const torch::Tensor& feature_table,         // (T * L * F) Hashgrid features (flat) - Assuming flat layout
    const torch::Tensor& resolution,            // (L) Hashgrid resolutions per level
    const torch::Tensor& do_hash,               // (L) Flags for hashing per level
    const torch::Tensor& primes,                // (3) Hashing primes
    const int feature_offset,                   // Hash table size T per level
    // MLP Data (Linear Layer)
    const torch::Tensor& linear_weights,        // (input_linear_dim, output_linear_dim) MLP weights
    const torch::Tensor& linear_bias,           // (output_linear_dim) MLP biases
    // Integration Parameters
    const int stencil_genus,                    // Stencil type (1 or 2)
    const int grid_size,                        // Dimension of the confidence grid (e.g., 3)
    const int max_primitives_per_ray,           // Max primitives per ray
    const float occupancy_threshold,            // Confidence -> Occupancy threshold
    // Background Color
    const torch::Tensor& bg_color,              // (num_output_channels) Background color
    // Runtime Dimensions
    const uint32_t input_feature_dim,           // F: Feature dimension in hashgrid
    const uint32_t output_feature_dim,          // Output feature dimension (excluding RGB, density)
    const uint32_t hashgrid_levels,             // L: Number of hashgrid levels
    const uint32_t num_output_channels,         // Number of color channels (e.g., 3 for RGB)
    // Misc
    const bool debug                            // Debug flag
)
{


  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3, "means3D must have shape (P, 3)");
  // Add more checks as needed...

  // --- Options for creating new tensors ---
  auto options_float = means3D.options().dtype(torch::kFloat32);
  auto options_int = means3D.options().dtype(torch::kInt32);
  auto options_byte = means3D.options().dtype(torch::kUInt8); // Use kUInt8 for byte buffers
  torch::Device device(torch::kCUDA); // Ensure tensors are on CUDA

  // --- Allocate Output Tensors ---
  // Assuming CHW format for outputs based on typical PyTorch conventions
  // Adjust if your convention is HWC
  torch::Tensor out_color = torch::zeros({(long long)num_output_channels, H, W}, options_float.device(device));
  torch::Tensor out_features = torch::zeros({(long long)output_feature_dim, H, W}, options_float.device(device));
  torch::Tensor visibility_info = torch::zeros({H, W}, options_float.device(device));

  // --- Allocate Intermediate State Buffers (Resized by CUDA code, returned for backward) ---
  torch::Tensor geomBuffer = torch::empty({0}, options_byte.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options_byte.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options_byte.device(device));
  // Internal radii buffer (managed by Integrator::forward if needed)
  torch::Tensor radii_internal = torch::empty({0}, options_int.device(device)); // Will be resized if needed

  // Create resizing functions for forward pass state buffers
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  // <<< Instantiate the Integrator >>>
  CudaIntegrator::Integrator integrator; // Create an instance

  int num_rendered = 0;
  if(P != 0 && H > 0 && W > 0) // Check for valid primitives and image dimensions
  {
	  // <<< Call forward using the instance >>>
	  num_rendered = integrator.forward(
	    /* Buffer funcs */    geomFunc, binningFunc, imgFunc,
        /* Primitive Count */ P,
        /* Image Dim */       W, H,
        /* Primitive Data */  means3D.contiguous().data<float>(),
                              primitive_scale,
        /* Camera Params */   viewmatrix.contiguous().data<float>(),
                              projmatrix.contiguous().data<float>(),
                              cam_pos.contiguous().data<float>(),
                              near_plane, max_distance,
                              tan_fovx, tan_fovy,
        /* Confidence/Feat */ primitive_confidences.contiguous().data<float>(),
                              feature_table.contiguous().data<float>(),
                              resolution.contiguous().data<int>(),
                              do_hash.contiguous().data<int>(),
                              primes.contiguous().data<int>(),
                              feature_offset,
        /* MLP Data */        linear_weights.contiguous().data<float>(),
                              linear_bias.contiguous().data<float>(),
        /* Integ Params */    stencil_genus, grid_size, max_primitives_per_ray, occupancy_threshold,
        /* Background */      bg_color.contiguous().data<float>(),
        /* Outputs */         out_color.contiguous().data<float>(),
                              out_features.contiguous().data<float>(),
                              visibility_info.contiguous().data<float>(),
        /* Runtime Dims */    input_feature_dim, output_feature_dim, hashgrid_levels, num_output_channels,
        /* Optional Radii */  radii_internal.data<int>(), // Pass pointer, might be null if size 0
        /* Misc */            debug
      );
  } else {
      if (debug) std::cout << "Skipping CUDA forward: P=" << P << ", H=" << H << ", W=" << W << std::endl;
      // Ensure buffers are size 0 if skipped
      geomBuffer.resize_({0});
      binningBuffer.resize_({0});
      imgBuffer.resize_({0});
  }

  // <<< Add explicit synchronization before returning >>>
  if(debug)
  {
    cudaDeviceSynchronize();
  }

  // Return final outputs and the state buffers needed for backward
  return std::make_tuple(num_rendered, out_color, out_features, visibility_info, geomBuffer, binningBuffer, imgBuffer);
}


// Backward pass function definition for PyTorch binding
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IntegratePrimitivesBackwardCUDA(
    // --- Input Gradients (from Python) ---
    const torch::Tensor& dL_dout_color,         // (H, W, 3) or (3, H, W) - Ensure contiguous in Python
    const torch::Tensor& dL_dout_features,      // (H, W, F_out) or (F_out, H, W) - Ensure contiguous
    // --- Saved Forward Pass Data (from Python ctx) ---
    // State Buffers
    const torch::Tensor& geomBuffer,            // Saved geometry state buffer
    const torch::Tensor& binningBuffer,         // Saved binning state buffer
    const torch::Tensor& imgBuffer,             // Saved image state buffer
    // Original Inputs
	const torch::Tensor& means3D,
	const torch::Tensor& primitive_confidences, // Flat (P, G^3)
	const float primitive_scale,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& cam_pos,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const float near_plane,
	const float max_distance,
	const torch::Tensor& feature_table,
	const torch::Tensor& resolution,
	const torch::Tensor& do_hash,
	const torch::Tensor& primes,
	const int feature_offset,
	const torch::Tensor& linear_weights,
	const torch::Tensor& linear_bias,
	const int stencil_genus,
	const int grid_size,
	const int max_primitives_per_ray,
	const float occupancy_threshold,
	const torch::Tensor& bg_color,
    // --- Runtime Dimensions ---
	const uint32_t input_feature_dim,
	const uint32_t output_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels,
    const uint32_t feature_table_size, // Total elements in feature_table (T*L*F)
	// Misc
	const bool debug
)
{
    // // --- Input Tensor Checks ---
    // CHECK_CUDA_INPUT(dL_dout_color);
    // CHECK_CUDA_INPUT(dL_dout_features);
    // CHECK_CUDA_INPUT(geomBuffer);    // Check saved state buffers too
    // CHECK_CUDA_INPUT(binningBuffer);
    // CHECK_CUDA_INPUT(imgBuffer);
    // CHECK_CUDA_INPUT(means3D);
    // CHECK_CUDA_INPUT(primitive_confidences);
    // CHECK_CUDA_INPUT(viewmatrix);
    // CHECK_CUDA_INPUT(projmatrix);
    // CHECK_CUDA_INPUT(cam_pos);
    // CHECK_CUDA_INPUT(feature_table);
    // CHECK_CUDA_INPUT(resolution);
    // CHECK_CUDA_INPUT(do_hash);
    // CHECK_CUDA_INPUT(primes);
    // CHECK_CUDA_INPUT(linear_weights);
    // CHECK_CUDA_INPUT(linear_bias);
    // CHECK_CUDA_INPUT(bg_color);


    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    // --- Options for creating new tensors ---
    auto options_float = means3D.options().dtype(torch::kFloat32);
    auto options_byte = means3D.options().dtype(torch::kUInt8);
    torch::Device device(torch::kCUDA);

    // --- Allocate Output Gradient Tensors ---
    // Gradients should be zero-initialized as kernels might use atomic adds
    torch::Tensor dL_dmeans3D = torch::zeros_like(means3D, options_float.device(device));
    torch::Tensor dL_dprimitive_confidences = torch::zeros_like(primitive_confidences, options_float.device(device)); // Matches flat input
    torch::Tensor dL_dfeature_table = torch::zeros_like(feature_table, options_float.device(device));
    torch::Tensor dL_dlinear_weights = torch::zeros_like(linear_weights, options_float.device(device));
    torch::Tensor dL_dlinear_bias = torch::zeros_like(linear_bias, options_float.device(device));
    // Note: Gradient w.r.t primitive_scale is not computed here. Add if needed.

    // --- Allocate Temporary Buffer for Backward Pass Internal Use ---
    torch::Tensor tempBuffer = torch::empty({0}, options_byte.device(device));
    std::function<char*(size_t)> tempFunc = resizeFunctional(tempBuffer);

    // --- Create resizing functions for the SAVED forward state buffers ---
    // These lambdas capture the tensors passed *in* from Python (geomBuffer, binningBuffer, imgBuffer)
    // The integrator.backward call will use these to *read* the saved state.
    // Pass const tensors to the lambda capture
    const torch::Tensor& geomBufferConst = geomBuffer;
    const torch::Tensor& binningBufferConst = binningBuffer;
    const torch::Tensor& imgBufferConst = imgBuffer;
    std::function<char*(size_t)> geomFunc = [&geomBufferConst](size_t N) {
        // Backward only reads, so just return data_ptr. Size check might be useful.
        // TORCH_CHECK(geomBufferConst.nbytes() >= N, "geomBuffer too small");
        return reinterpret_cast<char*>(geomBufferConst.contiguous().data_ptr());
    };
     std::function<char*(size_t)> binningFunc = [&binningBufferConst](size_t N) {
        // TORCH_CHECK(binningBufferConst.nbytes() >= N, "binningBuffer too small");
        return reinterpret_cast<char*>(binningBufferConst.contiguous().data_ptr());
    };
     std::function<char*(size_t)> imgFunc = [&imgBufferConst](size_t N) {
        // TORCH_CHECK(imgBufferConst.nbytes() >= N, "imgBuffer too small");
        return reinterpret_cast<char*>(imgBufferConst.contiguous().data_ptr());
    };


    // <<< Instantiate the Integrator >>>
    CudaIntegrator::Integrator integrator;

    // Get the current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();


    if (P > 0 && H > 0 && W > 0 && geomBuffer.numel() > 0) // Also check if buffers are non-empty
    {
        // <<< Call backward using the instance >>>
        integrator.backward(
            /* Buffer funcs */    geomFunc, binningFunc, imgFunc, tempFunc,
            /* Primitive Count */ P,
            /* Image Dim */       W, H,
            /* Original Inputs*/  means3D.contiguous().data<float>(),
                                  primitive_scale,
                                  viewmatrix.contiguous().data<float>(),
                                  projmatrix.contiguous().data<float>(),
                                  cam_pos.contiguous().data<float>(),
                                  near_plane, max_distance,
                                  tan_fovx, tan_fovy,
                                  primitive_confidences.contiguous().data<float>(), // X
                                  feature_table.contiguous().data<float>(),         // Φ
                                  resolution.contiguous().data<int>(),
                                  do_hash.contiguous().data<int>(),
                                  primes.contiguous().data<int>(),
                                  feature_offset,
                                  linear_weights.contiguous().data<float>(),
                                  linear_bias.contiguous().data<float>(),
                                  stencil_genus, grid_size, max_primitives_per_ray, occupancy_threshold,
                                  bg_color.contiguous().data<float>(),
            /* Fwd Pass Data */   nullptr, // forward_primitive_features_7D <<< Placeholder
                                  nullptr, // forward_primitive_alphas      <<< Placeholder
                                  nullptr, // forward_final_Ts              <<< Placeholder
                                  nullptr, // forward_primitive_features_D  <<< Placeholder
            /* Runtime Dims */    input_feature_dim, output_feature_dim, hashgrid_levels, num_output_channels,
                                  feature_table_size, // Pass total size
            /* Input Gradients*/  dL_dout_color.contiguous().data<float>(),
                                  dL_dout_features.contiguous().data<float>(),
            /* Output Gradients*/ dL_dmeans3D.contiguous().data<float>(),
                                  dL_dprimitive_confidences.contiguous().data<float>(), // dL_dX
                                  dL_dfeature_table.contiguous().data<float>(),         // dL_dΦ
                                  dL_dlinear_weights.contiguous().data<float>(),
                                  dL_dlinear_bias.contiguous().data<float>(),
            /* Stream */          stream, // <<< Pass the stream
            /* Misc */            debug
        );
    } else {
         if (debug) std::cout << "Skipping CUDA backward: P=" << P << ", H=" << H << ", W=" << W << ", geomBuffer=" << geomBuffer.numel() << std::endl;
         // Output gradients remain zero
    }

    // <<< Add explicit synchronization before returning >>>
    if(debug)
    {
      // Sync stream before checking error
      cudaError_t err = cudaStreamSynchronize(stream);
      if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: Backward sync failed: %s\n", cudaGetErrorString(err));
      }
    }

    // Return the computed gradients
    return std::make_tuple(dL_dmeans3D, dL_dprimitive_confidences, dL_dfeature_table,
                           dL_dlinear_weights, dL_dlinear_bias);
}


// Mark visible function - kept as is
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{

  const int P = means3D.size(0);

  // Output tensor to mark visibility (boolean)
  auto options_bool = means3D.options().dtype(at::kBool);
  torch::Tensor present = torch::full({P}, false, options_bool.device(torch::kCUDA));

  if(P != 0)
  {
    // <<< Instantiate the Integrator >>>
    CudaIntegrator::Integrator integrator; // Create an instance

    // <<< Call markVisible using the instance >>>
    integrator.markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>()); // Pass pointer to boolean data
  }

  return present;
}

// namespace BACKWARD {
//     void backwardFeatureAndConfidence(
//         int P,
//         int D,
//         // ... other arguments ...
//         cudaStream_t stream
//     ) {
//         // <<< IMPLEMENTATION CODE HERE >>>
//     }
// } // namespace BACKWARD