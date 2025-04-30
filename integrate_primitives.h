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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <cstdint> // Include for uint32_t
	
// Updated forward function declaration for PyTorch binding
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
	const torch::Tensor& feature_table,         // (T * L * F) Hashgrid features (flat)
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
);

// Backward pass function declaration for PyTorch binding
// Returns gradients w.r.t.: means3D, primitive_confidences, feature_table, linear_weights, linear_bias
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IntegratePrimitivesBackwardCUDA(
    // --- Input Gradients (from Python) ---
    const torch::Tensor& dL_dout_color,         // (H, W, 3) or (3, H, W)
    const torch::Tensor& dL_dout_features,      // (H, W, F_out) or (F_out, H, W)
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
);

// Function to mark primitives potentially visible (frustum culling)
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);