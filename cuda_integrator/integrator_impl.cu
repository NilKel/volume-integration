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

#include "integrator_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h" 
// #include "backward.h" // Commented out backward pass include

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all primitives that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Kernel to initialize the per-pixel contributing indices buffer
__global__ void initializePixelIndicesKernel(
	int* pixel_indices_buffer,
	int W, int H,
	int max_contrib_per_pixel)
{
	int pix_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pix_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (pix_x >= W || pix_y >= H) {
		return;
	}

	int pix_id = pix_y * W + pix_x;
	int base_idx = pix_id * max_contrib_per_pixel;

	// Check if base_idx is within bounds (optional safety)
	// if (base_idx + max_contrib_per_pixel <= W * H * max_contrib_per_pixel) {
		for (int k = 0; k < max_contrib_per_pixel; ++k) {
			pixel_indices_buffer[base_idx + k] = -1; // Initialize with -1
		}
	// }
}

// Generates one key/value pair for all primitive / tile overlaps.
// Run once per primitive (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* primitive_keys_unsorted,
	uint32_t* primitive_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible primitives (radii <= 0)
	if (radii[idx] > 0)
	{
		// Find this primitive's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the primitive. Sorting the values
		// with this key yields primitive IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				primitive_keys_unsorted[off] = key;
				primitive_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) primitive ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t tile_id = key >> 32;

	if (idx == 0)
	{
		// First element starts the first tile range.
		ranges[tile_id].x = 0;
	}
	else
	{
		// If the tile ID differs from the previous element's,
		// then the previous element was the end of its tile range,
		// and the current element is the start of the next range.
		uint64_t prev_key = point_list_keys[idx - 1];
		uint32_t prev_tile_id = prev_key >> 32;
		if (tile_id != prev_tile_id)
		{
			ranges[prev_tile_id].y = idx;
			ranges[tile_id].x = idx;
		}
	}

	if (idx == L - 1)
	{
		// Last element ends the last tile range.
		ranges[tile_id].y = L;
	}
}

// Mark primitives as visible/invisible, based on view frustum testing
void CudaIntegrator::Integrator::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	if (P > 0)
	{
		checkFrustum << <(P + 255) / 256, 256 >> > (P, means3D, viewmatrix, projmatrix, present);
	}
}

// --- UPDATED Forward Function Signature ---
// Definition WITHOUT static (Correct C++ style)
int CudaIntegrator::Integrator::forward(
	// Buffer allocation callbacks
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	// Primitive count
	const int P,
	// Image dimensions
	const int width, const int height,
	// Basic Primitive Data
	const float* means3D,
	const float primitive_scale,
	// Camera parameters
	const float* viewmatrix,
	const float* projmatrix,
	const float* camera_center_vec, // Use float* as passed from Python/C++ binding
	const float near_plane,
	const float max_distance,
	const float tan_fovx, const float tan_fovy,
	// Primitive Confidence/Feature Data
	const float* primitive_confidences,
	const float* feature_table,
	const int* resolution,
	const int* do_hash,
	const int* primes,
	const int feature_offset,
	const float* linear_weights,
	const float* linear_bias,
	// Integration Parameters
	const int stencil_genus,
	const int grid_size,
	const int max_primitives_per_ray,
	const float occupancy_threshold, // Corrected parameter name
	// Background Color
	const float* bg_color,
	// Output Buffers
	float* out_color,
	float* out_features,
	float* visibility_info,
	// Runtime Dimensions
	const uint32_t input_feature_dim,
	const uint32_t output_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels,
	// Misc
	int* radii_override, // Changed name for clarity, use this if not null
	bool debug
)
{
	printf("[Integrator::forward] Entered function. P=%d, W=%d, H=%d\n", P, width, height); fflush(stdout);

	// --- 0. Handle Empty Input ---
	if (P == 0 || width == 0 || height == 0) {
		// Handle cases with no primitives or zero image dimensions
		if (debug) printf("[Integrator::forward] Skipping integration due to P=0 or W=0 or H=0.\n"); fflush(stdout);
		// Ensure output buffers are cleared if necessary (might already be zero-initialized)
		// cudaMemset(out_color, 0, ...); // Example if clearing is needed
		return 0; // Return 0 rendered primitives/overlaps
	}

	// --- Calculate Tile and Block Sizes ---
	dim3 block(BLOCK_X, BLOCK_Y, 1); // BLOCK_X, BLOCK_Y defined in config.h or elsewhere
	

	// --- Allocate Buffers ---
	// Geometry buffer
	size_t geom_chunk_size = GeometryState::required(P); // Use static method
	char* geom_chunkptr = geometryBuffer(geom_chunk_size);
	GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

	// Image buffer <<< Uses updated fromChunk >>>
	const int num_pixels = width * height;
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const int num_tiles = tile_grid.x * tile_grid.y;
	size_t img_chunk_size = ImageState::required(num_tiles, num_pixels, max_primitives_per_ray); // Use static method
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, num_tiles, num_pixels, max_primitives_per_ray); // Call updated fromChunk

	// --- 1. Preprocess Primitives ---
	// Calculate 2D means, depths, radii, and tiles touched per primitive
	// Use radii_override if provided, otherwise use internal buffer geomState.internal_radii
	int* radii = (radii_override != nullptr) ? radii_override : geomState.internal_radii;

	// Calculate focal lengths from FoV tangents
	float focal_y = (float)height / (2.0f * tan_fovy);
	float focal_x = (float)width / (2.0f * tan_fovx);

	FORWARD::preprocess(
		P,
		means3D,
		primitive_scale,
		viewmatrix, projmatrix,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,                 // Use the potentially overridden radii
		geomState.means2D,
		geomState.depths,
		tile_grid,
		geomState.tiles_touched, // Output: tiles touched per primitive
		false // prefiltered flag (assuming false for now)
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::forward] Completed FORWARD::preprocess.\n"); fflush(stdout);


	// --- 2. Calculate Point Offsets ---
	// Perform inclusive scan on tiles_touched to get offsets for key-value pairs
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space, // Workspace
		geomState.scan_size,      // Workspace size
		geomState.tiles_touched,  // Input
		geomState.point_offsets,  // Output offsets
		P), debug);               // Number of items
	printf("[Integrator::forward] Completed CUB Scan (point offsets).\n"); fflush(stdout);


	// --- 3. Determine Total Overlaps & Allocate Sorting Buffer ---
	// Get the total number of overlaps (primitive-tile pairs) from the last element of the scan output
	int num_rendered = 0; // This will hold the total number of key-value pairs (overlaps)
	if (P > 0) {
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	}
	if (debug) printf("[Integrator::forward] Calculated num_rendered (overlaps) = %d\n", num_rendered); fflush(stdout);

	// If no overlaps, we can potentially skip sorting and rendering
	if (num_rendered == 0) {
		if (debug) printf("[Integrator::forward] Skipping sorting and rendering as num_rendered is 0.\n"); fflush(stdout);
		// Ensure output buffers are cleared (might already be zero)
		// cudaMemset(out_color, 0, ...); // Example
		return 0; // Return 0 overlaps
	}

	// Allocate binning buffer (unsorted keys/values, sorted keys/values, sort_workspace)
	size_t sorting_chunk_size = required<BinningState>(num_rendered);
	char* sorting_chunkptr = binningBuffer(sorting_chunk_size);
	BinningState binningState = BinningState::fromChunk(sorting_chunkptr, num_rendered);
	printf("[Integrator::forward] Allocated sorting buffer.\n"); fflush(stdout);


	// --- 4. Create Keys & Sort ---
	// Create key-value pairs (Key: TileID | Depth, Value: PrimitiveID)
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted, // Output keys
		binningState.point_list_unsorted,      // Output values (primitive IDs)
		radii, // Use the potentially overridden radii
		tile_grid);
	CHECK_CUDA(, debug);
	printf("[Integrator::forward] Completed duplicateWithKeys kernel.\n"); fflush(stdout);


	// Sort the key-value pairs based on the keys (TileID | Depth)
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,        // Workspace
		binningState.sorting_size,              // Workspace size
		binningState.point_list_keys_unsorted,  // Input keys
		binningState.point_list_keys,           // Output sorted keys
		binningState.point_list_unsorted,       // Input values
		binningState.point_list,                // Output sorted values
		num_rendered, 0, 32 + bit), debug);      // Sort based on 32 depth bits + tile bits
	printf("[Integrator::forward] Completed CUB Sort (SortPairs).\n"); fflush(stdout);


	// --- 5. Identify Tile Ranges ---
	// Reset tile ranges buffer (part of ImageState)
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Find the start/end index for each tile in the sorted list
	identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
		num_rendered,
		binningState.point_list_keys, // Input sorted keys
		imgState.ranges);             // Output tile ranges
	CHECK_CUDA(, debug);
	printf("[Integrator::forward] Completed identifyTileRanges kernel.\n"); fflush(stdout);


	// --- 6. Initialize Pixel Indices Buffer ---
	printf("[Integrator::forward] Initializing pixel indices buffer...\n"); fflush(stdout);
	dim3 init_grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	initializePixelIndicesKernel<<<init_grid, block>>>(
		imgState.pixel_contributing_indices,
		width, height,
		max_primitives_per_ray
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::forward] Completed initializing pixel indices buffer.\n"); fflush(stdout);


	// --- 7. Render Volume ---
	// Call the volume rendering kernel
	// Note: Assumes out_color, out_features, visibility_info are already allocated by the caller

	// <<< ADD CHECK FOR camera_center_vec >>>
	if (camera_center_vec == nullptr) {
		printf("[Integrator::forward] ERROR: camera_center_vec is NULL before calling FORWARD::render!\n"); fflush(stdout);
		// Handle error appropriately - maybe return early or throw exception
		return -1; // Indicate error
	} else {
		// Optional: Print the pointer address itself for debugging
		printf("[Integrator::forward] camera_center_vec pointer: %p\n", (void*)camera_center_vec); fflush(stdout);
		// Optional: Try a safe read *here* to see if it crashes earlier
		// float test_val;
		// cudaError_t err = cudaMemcpy(&test_val, camera_center_vec, sizeof(float), cudaMemcpyDeviceToHost);
		// if (err != cudaSuccess) {
		//     printf("[Integrator::forward] ERROR: Failed to read from camera_center_vec: %s\n", cudaGetErrorString(err)); fflush(stdout);
		// } else {
		//     printf("[Integrator::forward] Successfully read test value from camera_center_vec: %f\n", test_val); fflush(stdout);
		// }
	}
	// <<< END CHECK >>>

	printf("[Integrator::forward] About to call FORWARD::render.\n"); fflush(stdout); // <<< PRINT BEFORE RENDER CALL >>>
	FORWARD::render(
		tile_grid, block,              // Kernel launch config
		imgState.ranges,               // Tile ranges in sorted list
		binningState.point_list,       // Sorted primitive IDs
		width, height,                 // Image dimensions
		viewmatrix, projmatrix,        // Camera matrices
		camera_center_vec,             // Camera position (using the passed-in float*)
		near_plane, max_distance,      // Clipping planes (using passed-in values)
		means3D,                       // Primitive centers (using passed-in means3D)
		primitive_confidences,         // Primitive confidence grids (using passed-in value)
		primitive_scale,               // Primitive scale (using passed-in value)
		feature_table,                 // Hashgrid features (using passed-in value)
		resolution, do_hash, primes,   // Hashgrid params (using passed-in values)
		feature_offset,                // Hashgrid table size T (using passed-in value)
		linear_weights, linear_bias,   // MLP params (using passed-in values)
		stencil_genus, grid_size,      // Integration params (using passed-in values)
		max_primitives_per_ray,        // Integration param (using passed-in value)
		occupancy_threshold,           // Integration param (using passed-in value) - Corrected from scene_bounds
		bg_color,                      // Background color (using passed-in value)
		out_color, out_features,       // Output buffers (using passed-in values)
		visibility_info,               // Output buffer (using passed-in value)
		imgState.final_transmittance,   // <<< Pass the final transmittance pointer >>>
		imgState.num_contrib_per_pixel,  // <<< Pass the num_contrib_per_pixel pointer >>>
		imgState.pixel_contributing_indices, // <<< Pass the pixel_contributing_indices pointer >>>
		imgState.delta_t,                // <<< Pass the delta_t pointer from ImageState >>>
		input_feature_dim,             // Runtime dimension (using passed-in value)
		output_feature_dim,            // Runtime dimension (using passed-in value)
		hashgrid_levels,               // Runtime dimension (using passed-in value)
		num_output_channels            // Runtime dimension (using passed-in value)
	);
	// CHECK_CUDA(, debug); // Check is inside FORWARD::render now
	printf("[Integrator::forward] Returned from FORWARD::render.\n"); fflush(stdout); // <<< PRINT AFTER RENDER CALL >>>


	// --- 8. Return Number of Overlaps ---
	// This value indicates how many primitive-tile pairs were sorted and potentially rendered.
	printf("[Integrator::forward] Returning num_rendered = %d.\n", num_rendered); fflush(stdout);
	return num_rendered;
}

// Removed the __global__ void clearBuffer kernel definition as it's no longer called.

// --- UPDATED Backward Function ---
void CudaIntegrator::Integrator::backward(
	// Buffer allocation callbacks
	std::function<char* (size_t)> geomBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	// Primitive count
	const int P,
	const int num_rendered,
	// Image dimensions
	const int width, const int height,
	// Basic Primitive Data
	const float* means3D,
	const float primitive_scale,
	// Camera parameters
	const float* viewmatrix,
	const float* projmatrix,
	const float* camera_center_vec,
	const float near_plane,
	const float max_distance,
	const float tan_fovx,
	const float tan_fovy,
	// Primitive Confidence/Feature Data
	const float* primitive_confidences,
	const float* feature_table,
	const int* resolution,
	const int* do_hash,
	const int* primes,
	const int feature_offset,
	const float* linear_weights,
	const float* linear_bias,
	// Integration Parameters
	const int stencil_genus,
	const int grid_size,
	const int max_primitives_per_ray,
	const float occupancy_threshold,
	// Background Color
	const float* bg_color,
	// Runtime Dimensions
	const uint32_t input_feature_dim,
	const uint32_t output_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels,
	const uint32_t feature_table_size,
	// Input Gradients
	const float* dL_dout_color,
	const float* dL_dout_features,
	// Output Gradients (to be computed)
	float* dL_dmeans3D,
	float* dL_dprimitive_confidences,
	float* dL_dfeature_table,
	float* dL_dlinear_weights,
	float* dL_dlinear_bias,
	cudaStream_t stream,
	bool debug
)
{
	if (debug) printf("[Integrator::backward] Starting backward pass...\n"); fflush(stdout);

	// --- 1. Calculate Derived Dimensions ---
	const uint32_t grid_volume = (uint32_t)grid_size * grid_size * grid_size;
	const uint32_t input_linear_dim = input_feature_dim * hashgrid_levels;
	const uint32_t output_linear_dim = num_output_channels + output_feature_dim + 1; // +1 for density

	// --- 2. Zero Initialize Final Output Gradient Buffers ---
	// It's crucial these are zero before the kernel launch, as it uses atomicAdds.
	// Note: dL_dmeans3D is zeroed but not computed by compute_gradients yet.
	cudaMemsetAsync(dL_dmeans3D, 0, (size_t)P * 3 * sizeof(float), stream);
	cudaMemsetAsync(dL_dprimitive_confidences, 0, (size_t)P * grid_volume * sizeof(float), stream);
	// <<< Corrected size calculation for feature table gradient >>>
	// Total size is L*T*F, which is feature_table_size * input_feature_dim
	// However, the kernel expects feature_table_size to be the total number of floats (L*T*F)
	// So, the size passed to cudaMemsetAsync should just be feature_table_size * sizeof(float)
	// Ensure feature_table_size passed from Python/caller is indeed L*T*F.
	cudaMemsetAsync(dL_dfeature_table, 0, (size_t)feature_table_size * sizeof(float), stream);
	cudaMemsetAsync(dL_dlinear_weights, 0, (size_t)input_linear_dim * output_linear_dim * sizeof(float), stream);
	cudaMemsetAsync(dL_dlinear_bias, 0, (size_t)output_linear_dim * sizeof(float), stream);

	// --- 3. Retrieve State from Forward Pass Buffers ---
	// Get pointers to the state saved during forward pass.
	char* geomChunk = geomBuffer(0);
	GeometryState geomState = GeometryState::fromChunk(geomChunk, P);

	char* binningChunk = binningBuffer(0);
	BinningState binningState = BinningState::fromChunk(binningChunk, num_rendered);

	char* imageChunk = imageBuffer(0);
	const int num_pixels = width * height;
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const int num_tiles = tile_grid.x * tile_grid.y;
	ImageState imageState = ImageState::fromChunk(imageChunk, num_tiles, num_pixels, max_primitives_per_ray);

	if (debug) printf("[Integrator::backward] Retrieved state from buffers.\n"); fflush(stdout);

	// --- 4. Execute Unified Backward Kernel ---
	// This single kernel computes gradients for confidences, features, weights, bias.
	if (debug) printf("[Integrator::backward] Calling BACKWARD::compute_gradients...\n"); fflush(stdout);

	dim3 backward_grid = tile_grid;
	dim3 backward_block(BLOCK_X, BLOCK_Y, 1);

	BACKWARD::compute_gradients(
		backward_grid, backward_block,
		// Input Gradients
		dL_dout_color, dL_dout_features,
		// Forward State
		imageState.final_transmittance,
		imageState.num_contrib_per_pixel,
		imageState.pixel_contributing_indices,
		imageState.delta_t,
		binningState.point_list,
		P, num_rendered,
		// Image Dimensions
		width, height,
		// Camera Parameters (Note: tan_fovx/y are NOT passed to compute_gradients currently)
		viewmatrix, projmatrix, camera_center_vec, near_plane, max_distance,
		// Primitive Data
		means3D, primitive_confidences, primitive_scale,
		// Hashgrid Data
		feature_table,
		resolution,
		do_hash,
		primes,
		feature_offset,
		feature_table_size,
		// MLP Data
		linear_weights, linear_bias,
		// Integration Parameters
		stencil_genus, grid_size, max_primitives_per_ray,
		occupancy_threshold,
		// Background
		bg_color,
		// Runtime Dimensions
		input_feature_dim, output_feature_dim, hashgrid_levels, num_output_channels,
		// Output Gradients
		dL_dlinear_weights,
		dL_dlinear_bias,
		dL_dfeature_table,
		dL_dprimitive_confidences
	);

	// === Add Synchronization and Error Check ===
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("[Integrator::backward] CUDA Error after compute_gradients launch: %s\n", cudaGetErrorString(err));
		fflush(stdout);
		// Optionally, throw an exception or handle the error appropriately
		// throw std::runtime_error("CUDA Error after compute_gradients launch");
	}

	if (debug) printf("[Integrator::backward] Synchronizing device...\n"); fflush(stdout);
	err = cudaDeviceSynchronize(); // Wait for kernel to complete and check for runtime errors
	if (err != cudaSuccess) {
		printf("[Integrator::backward] CUDA Error during compute_gradients execution (sync): %s\n", cudaGetErrorString(err));
		fflush(stdout);
		// Optionally, throw an exception or handle the error appropriately
		// throw std::runtime_error("CUDA Error during compute_gradients execution");
	} else {
		if (debug) printf("[Integrator::backward] Device synchronization successful.\n"); fflush(stdout);
	}
	// ==========================================

	if (debug) printf("[Integrator::backward] Completed BACKWARD::compute_gradients.\n"); fflush(stdout);

	// --- 5. Backward Preprocess (Placeholder) ---
	// This step is still needed to compute dL_dmeans3D.
	// It would likely take gradients computed by compute_gradients (e.g., dL/dPosWorld if added)
	// and propagate them back to the original means3D.
	if (debug) printf("[Integrator::backward] PLACEHOLDER: Backward preprocess step needed for dL_dmeans3D.\n"); fflush(stdout);


	// --- 6. Finalize ---
	// Gradients dL_dlinear_weights, dL_dlinear_bias, dL_dfeature_table,
	// dL_dprimitive_confidences are now computed and stored in the output pointers.
	// dL_dmeans3D requires the missing backward preprocess step.

	if (debug) printf("[Integrator::backward] Backward pass completed (excluding preprocess).\n"); fflush(stdout);
}

// Removed the __global__ void clearBuffer kernel definition as it's no longer called.