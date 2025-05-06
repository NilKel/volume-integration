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
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
>
IntegratePrimitivesBackwardCUDA(
    // --- Input Gradients (from Python) ---
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_features,
    // --- Saved Forward Pass Data (from Python ctx) ---
    const torch::Tensor& geomBuffer_t,      // Renamed to avoid conflict
    const torch::Tensor& binningBuffer_t,   // Renamed to avoid conflict
    const torch::Tensor& imgBuffer_t,       // Renamed to avoid conflict
    // Original Inputs needed for backward
	const torch::Tensor& means3D,
	const torch::Tensor& primitive_confidences, // This is the flat version passed from Python
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
	const torch::Tensor& resolution_t,
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
    const uint32_t feature_table_size,
    // --- Other Saved Context ---
    const int P,
    const int num_rendered,
	// Misc
	const bool debug
)
{
    // 1. Instantiate Integrator
    CudaIntegrator::Integrator integrator;

    // 2. Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 3. Prepare Output Gradient Tensors (Zero-initialized)
    auto options_float = means3D.options().dtype(torch::kFloat32);
    torch::Tensor dL_dmeans3D = torch::zeros_like(means3D, options_float);
    torch::Tensor dL_dprimitive_confidences = torch::zeros_like(primitive_confidences, options_float); // Use flat input tensor shape
    torch::Tensor dL_dfeature_table = torch::zeros_like(feature_table, options_float);
    torch::Tensor dL_dlinear_weights = torch::zeros_like(linear_weights, options_float);
    torch::Tensor dL_dlinear_bias = torch::zeros_like(linear_bias, options_float);

    // 4. Create Buffer Accessor Lambdas
    // Need mutable copies for the lambda capture if resizeFunctional modifies the tensor reference
    auto geomBuffer_t_mut = geomBuffer_t;
    auto binningBuffer_t_mut = binningBuffer_t;
    auto imgBuffer_t_mut = imgBuffer_t;
    auto geomBuffer = resizeFunctional(geomBuffer_t_mut);
    auto binningBuffer = resizeFunctional(binningBuffer_t_mut);
    auto imageBuffer = resizeFunctional(imgBuffer_t_mut);

    // Get data pointers
    const int* resolution_ptr = resolution_t.contiguous().data<int>();
    const int* do_hash_ptr = do_hash.contiguous().data<int>();

    // 5. Call the Integrator's backward method if needed
    if (num_rendered > 0)
    {
        integrator.backward(
            // <<< Pass the buffer lambdas >>>
            geomBuffer,
            binningBuffer,
            imageBuffer,
            // Primitive count & Dimensions
            P,
            num_rendered,
            image_width, image_height,
            // Input Data Pointers (matching forward pass)
            means3D.contiguous().data<float>(),
            primitive_scale,
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            cam_pos.contiguous().data<float>(),
            near_plane,
            max_distance,
            tan_fovx, tan_fovy,
            primitive_confidences.contiguous().data<float>(),
            feature_table.contiguous().data<float>(),
            resolution_ptr,
            do_hash_ptr,
            primes.contiguous().data<int>(),
            feature_offset,
            linear_weights.contiguous().data<float>(),
            linear_bias.contiguous().data<float>(),
            stencil_genus,
            grid_size,
            max_primitives_per_ray,
            occupancy_threshold,
            bg_color.contiguous().data<float>(),
            // Runtime Dimensions
            input_feature_dim,
            output_feature_dim,
            hashgrid_levels,
            num_output_channels,
            feature_table_size,
            // Input Gradients
            dL_dout_color.contiguous().data<float>(),
            dL_dout_features.contiguous().data<float>(),
            // Output Gradients (to be computed)
            dL_dmeans3D.data_ptr<float>(),
            dL_dprimitive_confidences.data_ptr<float>(),
            dL_dfeature_table.data_ptr<float>(),
            dL_dlinear_weights.data_ptr<float>(),
            dL_dlinear_bias.data_ptr<float>(),
            // CUDA Stream & Debug Flag
            stream,
            debug
        );
    } else {
        if(debug) std::cout << "Skipping CUDA backward: num_rendered=0" << std::endl;
    }

    // 6. Return computed gradients
    return std::make_tuple(
        dL_dmeans3D,
        dL_dprimitive_confidences, // Return the flat gradient tensor
        dL_dfeature_table,
        dL_dlinear_weights,
        dL_dlinear_bias
    );
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