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

#ifndef CUDA_INTEGRATOR_IMPL_H_INCLUDED
#define CUDA_INTEGRATOR_IMPL_H_INCLUDED

#include <iostream>
#include <vector>
#include "integrator.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <glm/glm.hpp> // Include GLM
#include <cub/cub.cuh> // Add for cub::DeviceScan, cub::DeviceRadixSort

namespace CudaIntegrator
{
	// Helper function for buffer allocation (remains the same)
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// Helper structures for buffer management
	struct GeometryState
	{
		size_t scan_size; // Size needed for CUB scan
		float* depths;
		int* internal_radii; // Used for bounding box calculation
		float2* means2D;     // Used for bounding box calculation
		uint32_t* tiles_touched; // Output of preprocess kernel
		uint32_t* point_offsets; // Output of CUB scan
		char* scanning_space;    // Temporary storage for CUB scan

		// static size_t required(size_t P)
		// {
		// 	size_t size = 0;
		// 	size += CUB_ROUND_UP_NEAREST(P * sizeof(float), 128); // depths
		// 	size += CUB_ROUND_UP_NEAREST(P * sizeof(int), 128); // internal_radii
		// 	size += CUB_ROUND_UP_NEAREST(P * sizeof(float2), 128); // means2D
		// 	size += CUB_ROUND_UP_NEAREST(P * sizeof(uint32_t), 128); // tiles_touched
		// 	size_t scan_bytes;
		// 	cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, P);
		// 	size += CUB_ROUND_UP_NEAREST(scan_bytes, 128); // scanning_space
		// 	size += CUB_ROUND_UP_NEAREST(P * sizeof(uint32_t), 128); // point_offsets
		// 	return size;
		// }

		static inline GeometryState fromChunk(char*& chunk, size_t P) {
			GeometryState state;
			obtain(chunk, state.depths, P, 128);
			obtain(chunk, state.internal_radii, P, 128);
			obtain(chunk, state.means2D, P, 128);
			obtain(chunk, state.tiles_touched, P, 128);
			cub::DeviceScan::InclusiveSum(nullptr, state.scan_size, state.tiles_touched, state.tiles_touched, P);
			obtain(chunk, state.scanning_space, state.scan_size, 128);
			obtain(chunk, state.point_offsets, P, 128); 
			return state;
		}
	};

	struct ImageState
	{
		uint2* ranges; // Output of identifyTileRanges kernel (size = num_tiles)
		float* final_transmittance; // <<< NEW: Final T per pixel (size = N) >>>
		int* num_contrib_per_pixel; // size = N_pixels <<< NEW >>>
		int* pixel_contributing_indices; // size = N_pixels * max_contrib_per_pixel <<< NEW >>>
		float* delta_t; // <<< NEW: Delta T per pixel contribution (size = N_pixels * max_contrib_per_pixel) >>>

		// static size_t required(size_t N_tiles, size_t N_pixels, size_t max_contrib_per_pixel)
		// {
		// 	size_t size = 0;
		// 	size += CUB_ROUND_UP_NEAREST(N_tiles * sizeof(uint2), 128); // ranges
		// 	size += CUB_ROUND_UP_NEAREST(N_pixels * sizeof(float), 128); // final_transmittance <<< NEW >>>
		// 	size += CUB_ROUND_UP_NEAREST(N_pixels * sizeof(int), 128); // num_contrib_per_pixel <<< NEW >>>
		// 	size += CUB_ROUND_UP_NEAREST(N_pixels * max_contrib_per_pixel * sizeof(int), 128); // pixel_contributing_indices <<< NEW >>>
		// 	size += CUB_ROUND_UP_NEAREST(N_pixels * max_contrib_per_pixel * sizeof(float), 128); // delta_t <<< NEW >>>
		// 	return size;
		// }

		static size_t required(size_t N_pixels, size_t max_contrib_per_pixel)
		{
			char* size = nullptr;
			ImageState::fromChunk(size, N_pixels, max_contrib_per_pixel);
			return ((size_t)size) + 128;
		}

		static inline ImageState fromChunk(char*& chunk, size_t N_pixels, size_t max_contrib_per_pixel) {
			ImageState state;
			obtain(chunk, state.ranges, N_pixels, 128);
			obtain(chunk, state.final_transmittance, N_pixels, 128);
			obtain(chunk, state.num_contrib_per_pixel, N_pixels, 128);
			obtain(chunk, state.pixel_contributing_indices, N_pixels * max_contrib_per_pixel, 128);
			obtain(chunk, state.delta_t, N_pixels * max_contrib_per_pixel, 128);
			return state;
		}
	};

	struct BinningState
	{
		size_t sorting_size; // Size needed for CUB sort
		uint32_t* point_list_unsorted;      // Primitive indices before sort
		uint64_t* point_list_keys_unsorted; // Keys (Tile | Depth) before sort
		uint32_t* point_list;               // Primitive indices after sort
		uint64_t* point_list_keys;          // Keys (Tile | Depth) after sort
		char* list_sorting_space;           // Temporary storage for CUB sort

		// static size_t required(size_t P_prime)
		// {
		// 	size_t size = 0;
		// 	size += CUB_ROUND_UP_NEAREST(P_prime * sizeof(uint64_t), 128); // point_list_keys_unsorted
		// 	size += CUB_ROUND_UP_NEAREST(P_prime * sizeof(uint32_t), 128); // point_list_unsorted
		// 	size += CUB_ROUND_UP_NEAREST(P_prime * sizeof(uint64_t), 128); // point_list_keys
		// 	size += CUB_ROUND_UP_NEAREST(P_prime * sizeof(uint32_t), 128); // point_list
		// 	size_t sort_bytes;
		// 	cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, (uint64_t*)nullptr, (uint64_t*)nullptr, (uint32_t*)nullptr, (uint32_t*)nullptr, P_prime);
		// 	size += CUB_ROUND_UP_NEAREST(sort_bytes, 128); // list_sorting_space
		// 	return size;
		// }

		

		static inline BinningState fromChunk(char*& chunk, size_t P_prime) {
			BinningState state;
			obtain(chunk, state.point_list_keys_unsorted, P_prime, 128);
			obtain(chunk, state.point_list_unsorted, P_prime, 128);
			obtain(chunk, state.point_list_keys, P_prime, 128);
			obtain(chunk, state.point_list, P_prime, 128);
			cub::DeviceRadixSort::SortPairs(
				nullptr, state.sorting_size,
				state.point_list_keys_unsorted, state.point_list_keys,
				state.point_list_unsorted, state.point_list, P_prime);
			obtain(chunk, state.list_sorting_space, state.sorting_size, 128);
			return state;
		}
	};

	// Template specialization for required size calculation
	template<typename T>
	size_t required(size_t P_or_N); // General declaration (can be removed if only specializations are used)

	// template<> // Specialization for GeometryState
	// inline size_t required<GeometryState>(size_t P) {
	// 	return GeometryState::required(P); // Now GeometryState is fully defined
	// }

	// inline size_t requiredImage(size_t N_tiles, size_t N_pixels, size_t max_contrib_per_pixel) {
	// 	return ImageState::required(N_tiles, N_pixels, max_contrib_per_pixel); // Now ImageState is fully defined
	// }

	// template<> // Specialization for BinningState
	// inline size_t required<BinningState>(size_t P_prime) {
	// 	return BinningState::required(P_prime); // Now BinningState is fully defined
	// }

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	// Add a return struct definition (can be inside the namespace or globally)
	struct ForwardOutput {
		int num_rendered;
		void* geom_buffer_ptr;   // Use void* for generic pointer
		void* binning_buffer_ptr;
		void* image_buffer_ptr;
	};
}

#endif // CUDA_INTEGRATOR_IMPL_H_INCLUDED