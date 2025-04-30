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
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// Forward declaration
	struct GeometryState;
	struct ImageState;
	struct BinningState;

	// Calculates the size needed for a buffer based on the struct's fromChunk method.
	template<typename T>
	size_t required(size_t P_or_N) // P for Geometry/Binning, N for Image
	{
		char* size = nullptr;
		T::fromChunk(size, P_or_N);
		return ((size_t)size) + 128; // Add padding
	}

	// Helper structures for buffer management
	struct GeometryState
	{
		// --- UPDATED fromChunk signature ---
		static GeometryState fromChunk(char*& chunk, size_t P);

		size_t scan_size; // Size needed for CUB scan
		float* depths;
		int* internal_radii; // Used for bounding box calculation
		float2* means2D;     // Used for bounding box calculation
		uint32_t* tiles_touched; // Output of preprocess kernel
		uint32_t* point_offsets; // Output of CUB scan
		char* scanning_space;    // Temporary storage for CUB scan
		// Removed members not needed for sorting (e.g., cov3D, conic_opacity, rgb, clamped)
	};

	struct ImageState
	{
		// --- UPDATED fromChunk signature ---
		static ImageState fromChunk(char*& chunk, size_t N); // N = width * height

		uint2* ranges; // Output of identifyTileRanges kernel
		// Removed accum_alpha and n_contrib as they are not needed for sorting
	};

	struct BinningState
	{
		// --- UPDATED fromChunk signature ---
		static BinningState fromChunk(char*& chunk, size_t P_prime); // P_prime = num_rendered (total overlaps)

		size_t sorting_size; // Size needed for CUB sort
		uint32_t* point_list_unsorted;      // Primitive indices before sort
		uint64_t* point_list_keys_unsorted; // Keys (Tile | Depth) before sort
		uint32_t* point_list;               // Primitive indices after sort
		uint64_t* point_list_keys;          // Keys (Tile | Depth) after sort
		char* list_sorting_space;           // Temporary storage for CUB sort
	};
}

#endif // CUDA_INTEGRATOR_IMPL_H_INCLUDED