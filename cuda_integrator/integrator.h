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
			std::function<char* (size_t)> geomBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const int num_rendered,
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
			const uint32_t input_feature_dim,
			const uint32_t output_feature_dim,
			const uint32_t hashgrid_levels,
			const uint32_t num_output_channels,
			const uint32_t feature_table_size,
			const float* dL_dout_color,
			const float* dL_dout_features,
			float* dL_dmeans3D,
			float* dL_dprimitive_confidences,
			float* dL_dfeature_table,
			float* dL_dlinear_weights,
			float* dL_dlinear_bias,
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

	private:
		// Internal state, buffer management, etc.
		bool debug;
		size_t P; // Number of primitives
		size_t N; // Number of pixels

		// Buffer sizes
		size_t geometryStateSize;
		size_t binningStateSize;
		size_t imageStateSize;
		size_t sortingMemorySize; // Assuming this exists if sortingMemory accessor does

		// Device buffers
		char* geomBuffer_device;
		char* binningBuffer_device;
		char* imageBuffer_device;
		char* sortingMemory_device; // Assuming this exists

		// Buffer accessors (using std::function for flexibility)
		std::function<char*(size_t)> geomBuffer;      // <<< MUST BE DECLARED
		std::function<char*(size_t)> binningBuffer;   // <<< MUST BE DECLARED
		std::function<char*(size_t)> imageBuffer;     // <<< MUST BE DECLARED
		std::function<char*(size_t)> sortingMemory;   // <<< MUST BE DECLARED (if used)

		// Helper to allocate buffer and return accessor function
		void setupBuffer(char*& buffer_ptr, size_t& buffer_size, size_t required_size, std::function<char*(size_t)>& accessor);
	};
};

#endif