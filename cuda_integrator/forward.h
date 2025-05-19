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

#ifndef CUDA_INTEGRATOR_FORWARD_H_INCLUDED
#define CUDA_INTEGRATOR_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <vector_types.h> // For dim3, uint2, etc.
#include <cstdint>        // For uint32_t

// Define constants needed for template parameters if not defined elsewhere
// These should match the definitions in common headers or the .cu files
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

// // <<< Define MAX constants if not defined elsewhere >>>
// #ifndef MAX_GRID_VOLUME
// #define MAX_GRID_VOLUME (10*10*10) // Example: Max 10x10x10 grid
// #endif
// #ifndef MAX_STENCIL_SIZE
// #define MAX_STENCIL_SIZE 27 // Example: Max 3x3x3 stencil
// #endif
// #ifndef MAX_INPUT_LINEAR_DIM
// #define MAX_INPUT_LINEAR_DIM 64 // Example
// #endif
// #ifndef MAX_OUTPUT_LINEAR_DIM
// #define MAX_OUTPUT_LINEAR_DIM 16 // Example (e.g., 3 color + 12 feature + 1 density)
// #endif

namespace FORWARD
{
	// Structure definition (if any specific needed for forward pass, like Frustum)
	// ...

	// Perform initial steps for each primitive prior to sorting.
	// Simplified signature for sorting stage.
	void preprocess(
		int P,
		const float* means3D,
		const float primitive_scale, // Added: Scale for axis-aligned cubes
		const float* viewmatrix,
		const float* projmatrix,
		const int W, int H,
		const float tan_fovx, float tan_fovy,
		const float focal_x, float focal_y,
		int* radii, // Output: Screen-space radius
		float2* means2D, // Output: Projected 2D center
		float* depths, // Output: View-space depth
		const dim3 grid, // Tile grid dimensions
		uint32_t* tiles_touched, // Output: Number of tiles touched per primitive
		bool prefiltered); // Input: Flag for pre-filtering

	// Main integration/rendering method.
	void render(
		// Grid/Block dimensions for kernel launch
		const dim3 grid, dim3 block,
		// Data from sorting
		const uint2* ranges,
		const uint32_t* point_list,
		// Image dimensions
		int W, int H,
		// Camera parameters
		const float* viewmatrix,
		const float* projmatrix,
		const float* camera_center_vec, // Changed from float3 to float*
		const float near_plane,
		const float max_distance,
		// Primitive data
		const float* primitive_centers,
		const float* primitive_confidences,
		const float primitive_scale,
		// Hashgrid data
		const float* feature_table,
		const int* resolution,
		const int* do_hash,
		const int* primes,
		const int feature_offset,
		// MLP data
		const float* linear_weights,
		const float* linear_bias,
		// Integration params
		const int stencil_genus,
		const int grid_size,
		const int max_primitives_per_ray, // <<< Name matches integrator param >>>
		const float occupancy_threshold,
		// Background color
		const float* bg_color,
		// Output buffers
		float* out_color,
		float* out_features, // Added
		float* visibility_info, // Added
		// <<< MODIFIED Output Buffers for Backward Pass >>>
		float* out_final_transmittance,       // Stores final T per pixel (W*H)
		int* out_num_contrib_per_pixel,     // Stores count per pixel (W*H)
		int* out_pixel_contributing_indices,// Stores primitive index per pixel contribution (W*H*max_primitives_per_ray)
		float* out_delta_t,                 // <<< ADD THIS LINE BACK >>>
		// Runtime dimensions
		const uint32_t input_feature_dim, // Added
		const uint32_t output_feature_dim, // Added
		const uint32_t hashgrid_levels, // Added
		const uint32_t num_output_channels // Added (e.g., 3 for RGB)
		);
	
}


#endif