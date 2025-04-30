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

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
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

CudaIntegrator::GeometryState CudaIntegrator::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.point_offsets, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaIntegrator::ImageState CudaIntegrator::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaIntegrator::BinningState CudaIntegrator::BinningState::fromChunk(char*& chunk, size_t P_prime)
{
	BinningState binning;
	obtain(chunk, binning.point_list_keys_unsorted, P_prime, 128);
	obtain(chunk, binning.point_list_unsorted, P_prime, 128);
	obtain(chunk, binning.point_list_keys, P_prime, 128);
	obtain(chunk, binning.point_list, P_prime, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P_prime);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
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
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	// --- Allocate Buffers ---
	// Geometry buffer (depths, radii, means2D, tiles_touched, point_offsets, scan_workspace)
	size_t geom_chunk_size = required<GeometryState>(P);
	char* geom_chunkptr = geometryBuffer(geom_chunk_size);
	GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

	// Image buffer (tile ranges)
	size_t img_chunk_size = required<ImageState>(tile_grid.x * tile_grid.y);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, tile_grid.x * tile_grid.y);

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


	// --- 6. Render Volume ---
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
		input_feature_dim,             // Runtime dimension (using passed-in value)
		output_feature_dim,            // Runtime dimension (using passed-in value)
		hashgrid_levels,               // Runtime dimension (using passed-in value)
		num_output_channels            // Runtime dimension (using passed-in value)
	);
	// CHECK_CUDA(, debug); // Check is inside FORWARD::render now
	printf("[Integrator::forward] Returned from FORWARD::render.\n"); fflush(stdout); // <<< PRINT AFTER RENDER CALL >>>


	// --- 7. Return Number of Overlaps ---
	// This value indicates how many primitive-tile pairs were sorted and potentially rendered.
	printf("[Integrator::forward] Returning num_rendered = %d.\n", num_rendered); fflush(stdout);
	return num_rendered;
}

// Removed the __global__ void clearBuffer kernel definition as it's no longer called.

// --- Backward Pass Function ---
void CudaIntegrator::Integrator::backward(
	// Buffer allocation callbacks
	std::function<char* (size_t)> geometryBuffer, // Contains forward-pass geom data
	std::function<char* (size_t)> binningBuffer,  // Contains forward-pass binning data
	std::function<char* (size_t)> imageBuffer,    // Contains forward-pass image data
	std::function<char* (size_t)> tempBuffer,     // Callback for temporary gradient buffers
	// Primitive count & Dimensions from Forward
	const int P,
	const int width, int height,
	// Input Data Pointers (matching forward pass)
	const float* means3D,
	const float primitive_scale, // Assuming fixed, not differentiated (add dL_dscale if needed)
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
	const int grid_size, // Assumed cubic: GRID_SIZE_X = GRID_SIZE_Y = GRID_SIZE_Z = grid_size
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
	// const float* dL_dvisibility_info,   // Gradient w.r.t visibility (if needed)
	// --- Output Gradients (to be computed) ---
	float* dL_dmeans3D,
	float* dL_dprimitive_confidences,    // Output: dL_dX
	float* dL_dfeature_table,          // Output: dL_dΦ
	float* dL_dlinear_weights,
	float* dL_dlinear_bias,
	// Add other outputs if needed (e.g., dL_dprimitive_scale)
	cudaStream_t stream, // <<< THIS PARAMETER
	bool debug
)
{
	printf("[Integrator::backward] Entered function. P=%d, W=%d, H=%d\n", P, width, height); fflush(stdout);

	// --- 0. Check for empty input ---
	if (P == 0 || width == 0 || height == 0) {
		if (debug) printf("[Integrator::backward] Skipping backward pass due to P=0 or W=0 or H=0.\n"); fflush(stdout);
		// Ensure output gradient buffers are zeroed if necessary by the caller.
		return;
	}

	// --- Calculate Runtime Dimensions ---
	const uint32_t D = input_feature_dim * hashgrid_levels;
	const uint32_t L = hashgrid_levels;
	const uint32_t D_per_level = input_feature_dim; // Assuming input_feature_dim is features per level
	const int GRID_SIZE_X = grid_size; // Assuming grid_size is uniform
	const int GRID_SIZE_Y = grid_size;
	const int GRID_SIZE_Z = grid_size;
	const int BLEND_DIM = num_output_channels + output_feature_dim; // e.g., 3 + 4 = 7
	const int GRID_VOLUME = GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
	// Check if BLEND_DIM is 7 (as assumed by 7D outputs/inputs)
	if (BLEND_DIM != 7) {
		printf("Warning: BLEND_DIM (%d) is not 7. Backward pass assumes 7D features (Color+Alpha+Features).\n", BLEND_DIM);
		// Decide how to handle this - error out or proceed with caution?
	}
	if (D % L != 0) {
		printf("Warning: Total feature dimension D (%d) is not divisible by number of levels L (%d).\n", D, L);
	}

	// --- 1. Reconstruct States from Forward Buffers ---
	// These contain necessary sorting info and potentially some forward data
	char* geom_buffer_ptr = geometryBuffer(0); // Get pointer without reallocating
	GeometryState geomState = GeometryState::fromChunk(geom_buffer_ptr, P);


	// Need num_rendered to get BinningState size
	int num_rendered = 0;
	if (P > 0) {
		// Assuming point_offsets is still valid in geomState from forward
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	}
	if (debug) printf("[Integrator::backward] Retrieved num_rendered = %d\n", num_rendered); fflush(stdout);
	if (num_rendered == 0) {
		if (debug) printf("[Integrator::backward] Skipping backward pass as num_rendered is 0.\n"); fflush(stdout);
		// Ensure output gradient buffers are zeroed if necessary by the caller.
		return;
	}

	char* binning_buffer_ptr = binningBuffer(0); // Get pointer
	BinningState binningState = BinningState::fromChunk(binning_buffer_ptr, num_rendered);

	// Calculate tile grid dimensions needed for ImageState size
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	size_t imageStateSize = tile_grid.x * tile_grid.y;

	char* image_buffer_ptr = imageBuffer(0); // Get pointer
	ImageState imgState = ImageState::fromChunk(image_buffer_ptr, imageStateSize);


	// --- 2. Allocate Temporary Buffers for Intermediate Gradients ---
	// Calculate sizes based on dimensions
	size_t size_dL_dpixels_7D = (size_t)width * height * BLEND_DIM * sizeof(float);
	size_t size_dL_dFeatures_7D = (size_t)P * BLEND_DIM * sizeof(float); // BLEND_DIM = 7
	size_t size_dL_dAlphas = (size_t)P * sizeof(float);
	size_t size_dL_dDensityFeature = (size_t)P * sizeof(float);
	size_t size_dL_dOutput_8D = (size_t)P * 8 * sizeof(float); // 7 features + 1 density
	size_t size_dL_dPrimitiveFeatures_D = (size_t)P * D * sizeof(float);
	// --- Gradients needed for BACKWARD::preprocess (PLACEHOLDERS) ---
	size_t size_dL_dmeans2D = (size_t)P * 2 * sizeof(float); // Assuming P*float2
	size_t size_dL_ddepths = (size_t)P * sizeof(float);      // Assuming P*float
	size_t size_dL_dPosWorld = (size_t)P * GRID_VOLUME * 3 * sizeof(float); // P * GridVol * float3

	size_t total_temp_size = size_dL_dpixels_7D + size_dL_dFeatures_7D + size_dL_dAlphas +
							 size_dL_dDensityFeature + size_dL_dOutput_8D + size_dL_dPrimitiveFeatures_D +
							 size_dL_dmeans2D + size_dL_ddepths + size_dL_dPosWorld; // Add sizes for preprocess grads

	char* temp_buffer_ptr = tempBuffer(total_temp_size);
	if (debug) printf("[Integrator::backward] Allocated temporary buffer of size %zu bytes.\n", total_temp_size); fflush(stdout);


	// Assign pointers into the temporary buffer
	float* dL_dpixels_7D = (float*)temp_buffer_ptr;
	float* dL_dFeatures_7D = (float*)(temp_buffer_ptr + size_dL_dpixels_7D);
	float* dL_dAlphas = (float*)(dL_dFeatures_7D + P * BLEND_DIM);
	float* dL_dDensityFeature = (float*)(dL_dAlphas + P);
	float* dL_dOutput_8D = (float*)(dL_dDensityFeature + P);
	float* dL_dPrimitiveFeatures_D = (float*)(dL_dOutput_8D + P * 8);
	// --- Pointers for preprocess grads ---
	float* dL_dmeans2D = (float*)(dL_dPrimitiveFeatures_D + P * D);
	float* dL_ddepths = (float*)(dL_dmeans2D + P * 2);
	float* dL_dPosWorld = (float*)(dL_ddepths + P);


	// --- 3. Combine Input Gradients (dL_dout_color, dL_dout_features -> dL_dpixels_7D) ---
	// This requires a kernel to interleave or concatenate the input gradient buffers.
	// Placeholder: Assume dL_dpixels_7D is prepared correctly.
	// Example (needs a kernel): combineGradientsKernel<<<...>>>(dL_dout_color, dL_dout_features, dL_dpixels_7D, width*height);
	// For now, let's copy dL_dout_color and dL_dout_features if they happen to be contiguous and match 7D.
	// THIS IS LIKELY INCORRECT - NEEDS A DEDICATED KERNEL.
	if (num_output_channels == 3 && output_feature_dim == 4) {
		// Simplistic copy - replace with actual combination kernel
		size_t color_bytes = (size_t)width * height * 3 * sizeof(float);
		size_t feature_bytes = (size_t)width * height * 4 * sizeof(float);
		CHECK_CUDA(cudaMemcpy(dL_dpixels_7D, dL_dout_color, color_bytes, cudaMemcpyDeviceToDevice), debug);
		CHECK_CUDA(cudaMemcpy(dL_dpixels_7D + (size_t)width * height * 3, dL_dout_features, feature_bytes, cudaMemcpyDeviceToDevice), debug);
		if (debug) printf("[Integrator::backward] Placeholder: Combined input gradients via memcpy.\n"); fflush(stdout);
	} else {
		 throw std::runtime_error("Backward input gradient combination assumes 3 color + 4 feature channels.");
	}


	// --- 4. Call Backward Kernels in Reverse Order ---


	// Step 1: Backward Render (Pixels -> Primitive Features/Alphas)
	// TODO: Modify BACKWARD::render to also output dL_dmeans2D, dL_ddepths
	printf("[Integrator::backward] Calling BACKWARD::render...\n"); fflush(stdout);
	
	BACKWARD::render<BLEND_DIM>(
		imgState.ranges,            // Use ranges from ImageState
		binningState.point_list,    // Use point_list from BinningState
		width, height,
		bg_color,
		dL_dpixels_7D,              // Input gradient (combined)
		dL_dFeatures_7D,            // Output grad features
		dL_dAlphas,                  // Output grad alphas
		stream                      // CUDA stream
		// TODO: Add dL_dmeans2D, dL_ddepths as outputs here
		// dL_dmeans2D,
		// dL_ddepths
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::backward] Completed BACKWARD::render.\n"); fflush(stdout);

	// Step 2: Backward Alpha Activation (Alphas -> Density Feature)
	printf("[Integrator::backward] Calling BACKWARD::backwardAlphaActivation...\n"); fflush(stdout);
	BACKWARD::backwardAlphaActivation(
		P,
		dL_dAlphas,                 // Input grad alphas
		forward_primitive_alphas,   // Needs saving from forward
		dL_dDensityFeature          // Output grad density
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::backward] Completed BACKWARD::backwardAlphaActivation.\n"); fflush(stdout);

	// Step 3: Construct 8D Gradient (Combine Feature/Density Grads)
	printf("[Integrator::backward] Calling BACKWARD::construct8DGradient...\n"); fflush(stdout);
	BACKWARD::construct8DGradient(
		P,
		dL_dFeatures_7D,            // Input grad features (7D)
		dL_dDensityFeature,         // Input grad density (1D)
		dL_dOutput_8D               // Output grad combined (8D)
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::backward] Completed BACKWARD::construct8DGradient.\n"); fflush(stdout);

	// Step 4: Backward Linear Layer (8D Output -> D Input, Weights, Bias)
	printf("[Integrator::backward] Calling BACKWARD::linearLayer...\n"); fflush(stdout);
	BACKWARD::linearLayer(
		P,
		D,
		dL_dOutput_8D,              // Input grad 8D
		forward_primitive_features_D, // Needs saving from forward (Input to linear layer)
		linear_weights,             // Forward weights
		dL_dPrimitiveFeatures_D,    // Output grad D (w.r.t. aggregated features)
		dL_dlinear_weights,         // Output grad weights
		dL_dlinear_bias             // Output grad bias
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::backward] Completed BACKWARD::linearLayer.\n"); fflush(stdout);

	// Step 5: Backward Feature/Confidence (Aggregated Features -> Feature Table, Confidence Grid)
	// Zero initialize output buffers for atomic adds (Responsibility of the caller or handled internally if needed)


	printf("[Integrator::backward] Calling BACKWARD::backwardFeatureAndConfidence...\n"); fflush(stdout);
	BACKWARD::backwardFeatureAndConfidence(
		P,                          // 1: Number of primitives
		D, L, D_per_level,          // 2, 3, 4: Dimensions
		GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z, // 5, 6, 7: Grid dimensions
		dL_dPrimitiveFeatures_D,    // 8: Input grad D (P * D)
		primitive_confidences,      // 9: Forward confidence grids 'X' (P * GridVolume)
		feature_table,              // 10: Forward feature table 'Φ' (L * T * D_per_level)
		// --- Hashgrid Parameters ---
		resolution,                 // 11: Per-level resolutions (const int*) - Assuming 'resolution' holds this
		feature_table_size,         // 12: Size T per level (uint32_t) - Assuming 'feature_table_size' holds this
		do_hash,                    // 13: Flags for hashing per level (const int*)
		primes,                     // 14: Primes for hashing (const int*)
		// --- Outputs ---
		dL_dfeature_table,          // 15: Output grad feature table (float*)
		dL_dprimitive_confidences,  // 16: Output grad confidence 'dL_dX' (float*)
		dL_dPosWorld,               // 17: Output grad world positions (float*)
		stream                      // 18: CUDA stream
		// <<< Removed extra stream argument >>>
	);
	CHECK_CUDA(, debug);
	printf("[Integrator::backward] Completed BACKWARD::backwardFeatureAndConfidence.\n"); fflush(stdout);


	// --- Step 6: Backward Preprocess (dL/dIntermediate -> dL/dmeans3D) ---
	// PLACEHOLDER: This step is crucial but missing from backward.h and backward.cu
	printf("[Integrator::backward] PLACEHOLDER: Calling BACKWARD::preprocess...\n"); fflush(stdout);
	// Ensure dL_dmeans2D, dL_ddepths, dL_dPosWorld are correctly computed by prior steps
	// BACKWARD::preprocess(
	// 	P, means3D, primitive_scale, viewmatrix, projmatrix, width, height, focal_x, focal_y, tan_fovx, tan_fovy, // Forward params
	// 	dL_dmeans2D,  // Input from modified BACKWARD::render
	// 	dL_ddepths,   // Input from modified BACKWARD::render
	// 	dL_dPosWorld, // Input from modified BACKWARD::backwardFeatureAndConfidence
	// 	dL_dmeans3D   // Output grad means3D
	// );
	// CHECK_CUDA(, debug);
	// printf("[Integrator::backward] Completed BACKWARD::preprocess.\n"); fflush(stdout);

	// --- 7. Finalize ---
	// Gradients dL_dlinear_weights, dL_dlinear_bias, dL_dfeature_table,
	// dL_dprimitive_confidences, and dL_dmeans3D should now be in the output pointers.
	// Temporary buffers allocated via tempBuffer will be automatically managed (or need explicit freeing depending on callback).

	printf("[Integrator::backward] Backward pass completed.\n"); fflush(stdout);
}

// Removed the __global__ void clearBuffer kernel definition as it's no longer called.