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

#ifndef CUDA_INTEGRATOR_H_INCLUDED
#define CUDA_INTEGRATOR_H_INCLUDED

#include <vector>
#include <functional>
#include <cstdint> // Include for uint32_t
// #include <glm/glm.hpp> // Assuming glm is used - uncomment if needed

// Forward declaration
struct GeometryState;
struct BinningState;
struct ImageState;

namespace CudaIntegrator
{
	class Integrator
	{
	public:
		// Simplified forward declaration for sorting stage
		int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int width, int height,
			const float* means3D,
			const float primitive_scale,
			const float* viewmatrix,
			const float* projmatrix,
			const float* camera_center_vec,
			const float near_plane,
			const float max_distance,
			const float tan_fovx, float tan_fovy,
			const float* primitive_confidences,
			const float* feature_table,
			const int* resolution,
			const int* do_hash,
			const int* primes,
			const int feature_offset,
			const float* linear_weights,
			const float* linear_bias,
			const int stencil_genus,
			const int grid_size,
			const int max_primitives_per_ray,
			const float occupancy_threshold,
			const float* bg_color,
			float* out_color,
			float* out_features,
			float* visibility_info,
			const uint32_t input_feature_dim,
			const uint32_t output_feature_dim,
			const uint32_t hashgrid_levels,
			const uint32_t num_output_channels,
			int* radii, // Optional override
			bool debug);

		// Backward declaration
		void backward(
			// Buffer allocation callbacks
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> tempBuffer,     // Callback for temporary gradient buffers
			// Primitive count & Dimensions from Forward
			const int P,
			const int width, int height,
			// Input Data Pointers (matching forward pass)
			const float* means3D,
			const float primitive_scale, // Assuming fixed, not differentiated
			const float* viewmatrix,
			const float* projmatrix,
			const float* camera_center_vec,
			const float near_plane,
			const float max_distance,
			const float tan_fovx, float tan_fovy,
			const float* primitive_confidences, // Corresponds to 'X'
			const float* feature_table,         // Corresponds to 'Φ'
			const int* resolution,
			const int* do_hash,
			const int* primes,
			const int feature_offset,
			const float* linear_weights,
			const float* linear_bias,
			const int stencil_genus,
			const int grid_size,
			const int max_primitives_per_ray,
			const float occupancy_threshold,
			const float* bg_color,
			// --- Saved Forward Pass Data (PLACEHOLDERS - Ensure these are saved and passed) ---
			const float* forward_primitive_features_7D, // Output of linear layer (P * 7)
			const float* forward_primitive_alphas,      // Output of sigmoid (P)
			const float* forward_final_Ts,              // Final transmittance per pixel (W * H)
			const float* forward_primitive_features_D,  // Input to linear layer (P * D)
			// --- Runtime Dimensions ---
			const uint32_t input_feature_dim,      // D
			const uint32_t output_feature_dim,     // Assumed 4 (for features part of 7D)
			const uint32_t hashgrid_levels,        // L
			const uint32_t num_output_channels,    // Assumed 3 (for color part of 7D)
			const uint32_t feature_table_size,     // Needed for backwardFeatureAndConfidence
			// --- Input Gradients ---
			const float* dL_dout_color,            // Gradient w.r.t final color (W * H * 3)
			const float* dL_dout_features,         // Gradient w.r.t final features (W * H * 4)
			// --- Output Gradients (to be computed) ---
			float* dL_dmeans3D,
			float* dL_dprimitive_confidences,    // Output: dL_dX
			float* dL_dfeature_table,          // Output: dL_dΦ
			float* dL_dlinear_weights,
			float* dL_dlinear_bias,
			// Add other outputs if needed (e.g., dL_dprimitive_scale)
			cudaStream_t stream,
			bool debug
		);

		// Frustum culling function declaration
		void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);
	};
};

#endif