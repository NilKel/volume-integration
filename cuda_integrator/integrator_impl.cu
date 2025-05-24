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

// <<< ADDED: Attempt to resolve CUB conflicts >>>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
// <<< END ADDED SECTION >>>

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
#include <cub/device/device_reduce.cuh>
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
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
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
	
	
	// <<< END DEBUG PRINTS >>>

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
	const uint32_t intermediate_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels,
	// Misc
	int* radii_override, // Changed name for clarity, use this if not null
	bool debug
)
{
	

	// --- Calculate Tile and Block Sizes ---
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1); // BLOCK_X, BLOCK_Y defined in config.h or elsewhere
	

	// --- Allocate Buffers ---
	// Geometry buffer
	size_t geom_chunk_size = required<GeometryState>(P); // Use static method
	char* geom_chunkptr = geometryBuffer(geom_chunk_size);
	GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

	// Image buffer <<< Uses updated fromChunk >>>
	const int num_pixels = width * height;
	const int num_tiles = tile_grid.x * tile_grid.y;

	size_t img_chunk_size = ImageState::required(num_pixels, max_primitives_per_ray); // Use static method
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, num_pixels, max_primitives_per_ray); // Call updated fromChunk

	// --- 1. Preprocess Primitives ---
	// Calculate 2D means, depths, radii, and tiles touched per primitive
	// Use radii_override if provided, otherwise use internal buffer geomState.internal_radii
	int* radii = (radii_override != nullptr) ? radii_override : geomState.internal_radii;

	// Calculate focal lengths from FoV tangents
	float focal_y = (float)height / (2.0f * tan_fovy);
	float focal_x = (float)width / (2.0f * tan_fovx);

	
	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D,
		primitive_scale,
		viewmatrix, projmatrix,
		width, height,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		
		radii,                 // Use the potentially overridden radii
		geomState.means2D,
		geomState.depths,
		tile_grid,
		geomState.tiles_touched, // Output: tiles touched per primitive
		false // prefiltered flag (assuming false for now)
	), debug);
	// printf("[Integrator::forward] Completed FORWARD::preprocess.\n"); fflush(stdout);


	// --- 2. Calculate Point Offsets ---
	// Perform inclusive scan on tiles_touched to get offsets for key-value pairs
	// CHECK_CUDA(cub::DeviceScan::InclusiveSum(
	// 	geomState.scanning_space, // Workspace
	// 	geomState.scan_size,      // Workspace size
	// 	geomState.tiles_touched,  // Input
	// 	geomState.point_offsets,  // Output offsets
	// 	P), debug);               // Number of items
	// printf("[Integrator::forward] Completed CUB Scan (point offsets).\n"); fflush(stdout);

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	// --- 3. Determine Total Overlaps & Allocate Sorting Buffer ---
	// Get the total number of overlaps (primitive-tile pairs) from the last element of the scan output
	int num_rendered; // This will hold the total number of key-value pairs (overlaps)
	if (P > 0) {
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	}
	// if (debug) printf("[Integrator::forward] Calculated num_rendered (overlaps) = %d\n", num_rendered); fflush(stdout);

	

	// Allocate binning buffer (unsorted keys/values, sorted keys/values, sort_workspace)
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);
	// printf("[Integrator::forward] Allocated sorting buffer.\n"); fflush(stdout);


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
		tile_grid)
		CHECK_CUDA(, debug);
	// printf("[Integrator::forward] Completed duplicateWithKeys kernel.\n"); fflush(stdout);


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
	// printf("[Integrator::forward] Completed CUB Sort (SortPairs).\n"); fflush(stdout);


	// --- 5. Identify Tile Ranges ---
	// Reset tile ranges buffer (part of ImageState)
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Find the start/end index for each tile in the sorted list
	if (num_rendered > 0) {
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys, // Input sorted keys
			imgState.ranges);             // Output tile ranges
		CHECK_CUDA(, debug);
		// printf("[Integrator::forward] Completed identifyTileRanges kernel.\n"); fflush(stdout);
	}

    // >>> ADD THIS DEBUG SECTION <<<
    // Allocate host memory for ranges
    uint2* h_ranges = new uint2[tile_grid.x * tile_grid.y];
    // Copy ranges from device to host
    CHECK_CUDA(cudaMemcpy(h_ranges, imgState.ranges, tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToHost), debug);

    // Find min and max values
    uint32_t min_range_x = UINT32_MAX;
    uint32_t max_range_x = 0;
    uint32_t min_range_y = UINT32_MAX;
    uint32_t max_range_y = 0;

    for (int i = 0; i < tile_grid.x * tile_grid.y; i++) {
        if (h_ranges[i].x < min_range_x) min_range_x = h_ranges[i].x;
        if (h_ranges[i].x > max_range_x) max_range_x = h_ranges[i].x;
        if (h_ranges[i].y < min_range_y) min_range_y = h_ranges[i].y;
        if (h_ranges[i].y > max_range_y) max_range_y = h_ranges[i].y;
    }

    // printf("[DEBUG RANGES] After identifyTileRanges:\n");
    // printf("  Min range.x: %u\n", min_range_x);
    // printf("  Max range.x: %u\n", max_range_x);
    // printf("  Min range.y: %u\n", min_range_y);
    // printf("  Max range.y: %u\n", max_range_y);
    // printf("  Total tiles: %d\n", tile_grid.x * tile_grid.y);
    // printf("  num_rendered: %d\n", num_rendered);
    fflush(stdout);

    // Clean up
    delete[] h_ranges;
    // <<< END DEBUG SECTION >>>

    // >>> ADD THIS SYNCHRONIZATION <<<
    cudaDeviceSynchronize(); 
    // cudaError_t sync_err = cudaGetLastError(); // Optional: Check for errors after sync
    // if (sync_err != cudaSuccess) {
    //    fprintf(stderr, "CUDA error after cudaDeviceSynchronize: %s\n", cudaGetErrorString(sync_err));
    // }
    // fflush(stdout); // Ensure prints appear before potential crashes if any

    // printf("[Integrator::forward] Completed identifyTileRanges kernel AND SYNCED GPU.\n"); // Modified/New log
    // fflush(stdout);


	// --- 6. Initialize Pixel Indices Buffer ---
	// printf("[Integrator::forward] Initializing pixel indices buffer...\n"); fflush(stdout);
	// dim3 init_grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	// initializePixelIndicesKernel<<<init_grid, block>>>(
	// 	imgState.pixel_contributing_indices,
	// 	width, height,
	// 	max_primitives_per_ray
	// );
	// CHECK_CUDA(, debug);
	// printf("[Integrator::forward] Completed initializing pixel indices buffer.\n"); fflush(stdout);


	// --- 7. Render Volume ---
	// Call the volume rendering kernel
	// Note: Assumes out_color, out_features, visibility_info are already allocated by the caller

	

	// printf("[Integrator::forward] About to call FORWARD::render.\n"); fflush(stdout); // <<< PRINT BEFORE RENDER CALL >>>
	FORWARD::render(
		tile_grid, block,              // Kernel launch config
		imgState.ranges,               // Tile ranges in sorted list
		binningState.point_list,       // Sorted primitive IDs
		width, height, P,              // Image dimensions
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
		intermediate_feature_dim,
		hashgrid_levels,               // Runtime dimension (using passed-in value)
		num_output_channels            // Runtime dimension (using passed-in value)
	);
	// CHECK_CUDA(, debug); // Check is inside FORWARD::render now
	// printf("[Integrator::forward] Returned from FORWARD::render.\n"); fflush(stdout); // <<< PRINT AFTER RENDER CALL >>>


	// --- 8. Return Number of Overlaps ---
	// This value indicates how many primitive-tile pairs were sorted and potentially rendered.
	// printf("[Integrator::forward] Returning num_rendered = %d.\n", num_rendered); fflush(stdout);
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
	// if (debug) {
    //     printf("[Integrator::backward] Received Params: P=%d, num_rendered=%d, W=%d, H=%d, max_primitives_per_ray=%d\n",
    //            P, num_rendered, width, height, max_primitives_per_ray);
    //     fflush(stdout);
	// }
	// if (debug) printf("[Integrator::backward] Entered Integrator::backward.\n"); fflush(stdout);
	if (num_rendered == 0 && debug) {
		// printf("[Integrator::backward] num_rendered is 0, skipping backward pass logic.\n"); fflush(stdout);
		// Still need to zero out gradients if they were allocated by the caller
        // The PyTorch binding zeros them, so this might not be strictly necessary here,
        // but good practice if this function could be called from elsewhere.
        cudaMemsetAsync(dL_dmeans3D, 0, (size_t)P * 3 * sizeof(float), stream);
        const int grid_volume_bw = grid_size * grid_size * grid_size; // Corrected variable name
        cudaMemsetAsync(dL_dprimitive_confidences, 0, (size_t)P * grid_volume_bw * sizeof(float), stream);
        cudaMemsetAsync(dL_dfeature_table, 0, (size_t)feature_table_size * sizeof(float), stream); // feature_table_size is uint64_t
        const uint32_t input_linear_dim_bw = input_feature_dim * hashgrid_levels;
        const uint32_t output_linear_dim_bw = num_output_channels + output_feature_dim + 1; // Matches kernel's output_linear_dim
        cudaMemsetAsync(dL_dlinear_weights, 0, (size_t)input_linear_dim_bw * output_linear_dim_bw * sizeof(float), stream);
        cudaMemsetAsync(dL_dlinear_bias, 0, (size_t)output_linear_dim_bw * sizeof(float), stream);
		return;
	}

	// Calculate grid_volume for confidence gradients
	const int grid_volume = grid_size * grid_size * grid_size;
	const int input_linear_dim = input_feature_dim * hashgrid_levels;
	const int output_linear_dim = num_output_channels;


	// --- 1. Check for Empty Input ---
	if (P == 0 || width == 0 || height == 0) {
		if (debug) printf("[Integrator::backward] Skipping backward due to P=0 or W=0 or H=0.\n"); fflush(stdout);
		// Ensure gradient buffers are zeroed if they were to be accumulated into
		// (already done by PyTorch or explicit zeroing before calling this function)
		return;
	}
	if (num_rendered == 0 && stencil_genus > 0) { // stencil_genus > 0 implies rendering happened
		if (debug) printf("[Integrator::backward] Skipping backward as num_rendered is 0 but stencil_genus > 0 (no actual rendering occurred).\n"); fflush(stdout);
		return;
	}


	// --- 2. Zero Initialize Final Output Gradient Buffers ---
	// It's crucial these are zero before the kernel launch, as it uses atomicAdds.
	cudaMemsetAsync(dL_dmeans3D, 0, (size_t)P * 3 * sizeof(float), stream);
	cudaMemsetAsync(dL_dprimitive_confidences, 0, (size_t)P * grid_volume * sizeof(float), stream);
	cudaMemsetAsync(dL_dfeature_table, 0, (size_t)feature_table_size * sizeof(float), stream);
	cudaMemsetAsync(dL_dlinear_weights, 0, (size_t)input_linear_dim * output_linear_dim * sizeof(float), stream);
	cudaMemsetAsync(dL_dlinear_bias, 0, (size_t)output_linear_dim * sizeof(float), stream);
	// if (debug) printf("[Integrator::backward] Reached this point...\n"); fflush(stdout);

	// --- 3. Retrieve State from Forward Pass Buffers ---
	char* geom_chunkptr = geomBuffer(0);
	// <<< ADD NULL CHECK FOR geom_chunkptr >>>
	if (geom_chunkptr == nullptr) {
		printf("[Integrator::backward] FATAL: geom_chunkptr is NULL after geomBuffer(0) call!\n"); fflush(stdout);
	} else if (debug) {
		// printf("[Integrator::backward] geom_chunkptr: %p\n", (void*)geom_chunkptr); fflush(stdout);
	}
	// <<< END NULL CHECK >>>
	GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

	char* binningChunk = binningBuffer(0);
	// <<< ADD NULL CHECK FOR binningChunk >>>
	if (binningChunk == nullptr) {
		printf("[Integrator::backward] FATAL: binningChunk is NULL after binningBuffer(0) call!\n"); fflush(stdout);
	} else if (debug) {
		// printf("[Integrator::backward] binningChunk: %p\n", (void*)binningChunk); fflush(stdout);
	}
	// <<< END NULL CHECK >>>
	BinningState binningState = BinningState::fromChunk(binningChunk, num_rendered);

	char* imageChunk = imageBuffer(0);
	// <<< ADD NULL CHECK FOR imageChunk >>>
	if (imageChunk == nullptr) {
		printf("[Integrator::backward] FATAL: imageChunk is NULL after imageBuffer(0) call!\n"); fflush(stdout);
		// If this happens, ImageState::fromChunk will use a null base pointer, leading to crashes.
		// Consider returning or throwing to prevent further execution.
	} else if (debug) {
		// printf("[Integrator::backward] imageChunk: %p\n", (void*)imageChunk); fflush(stdout);
	}
	// <<< END NULL CHECK >>>
	const int num_pixels = width * height;
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const int num_tiles = tile_grid.x * tile_grid.y;
	// <<< PRINT PARAMS FOR ImageState::fromChunk >>>
	// if (debug) {
	// 	printf("[Integrator::backward] ImageState::fromChunk params: num_tiles=%d, num_pixels=%d, max_primitives_per_ray=%d\n",
	// 		   num_tiles, num_pixels, max_primitives_per_ray);
	// 	fflush(stdout);
	// }
	// <<< END PRINT PARAMS >>>
	ImageState imageState = ImageState::fromChunk(imageChunk, num_pixels, max_primitives_per_ray);

	// if (debug) printf("[Integrator::backward] Retrieved state from buffers.\n"); fflush(stdout);

    // <<< ADDED: CUB Reduction for num_contrib_per_pixel in backward >>>
    if (debug && P > 0 && height > 0 && width > 0 && imageState.num_contrib_per_pixel != nullptr) {
        // Allocate device memory for sum and max results
        int* d_sum_contrib_bw = nullptr;
        int* d_max_contrib_bw = nullptr;
        cudaMallocAsync(&d_sum_contrib_bw, sizeof(int), stream);
        cudaMallocAsync(&d_max_contrib_bw, sizeof(int), stream);

        // Allocate temporary storage for CUB reduction
        void* d_temp_storage_sum_bw = nullptr;
        size_t temp_storage_bytes_sum_bw = 0;
        cub::DeviceReduce::Sum(d_temp_storage_sum_bw, temp_storage_bytes_sum_bw, imageState.num_contrib_per_pixel, d_sum_contrib_bw, num_pixels, stream);
        cudaMallocAsync(&d_temp_storage_sum_bw, temp_storage_bytes_sum_bw, stream);
        cub::DeviceReduce::Sum(d_temp_storage_sum_bw, temp_storage_bytes_sum_bw, imageState.num_contrib_per_pixel, d_sum_contrib_bw, num_pixels, stream);

        void* d_temp_storage_max_bw = nullptr;
        size_t temp_storage_bytes_max_bw = 0;
        cub::DeviceReduce::Max(d_temp_storage_max_bw, temp_storage_bytes_max_bw, imageState.num_contrib_per_pixel, d_max_contrib_bw, num_pixels, stream);
        cudaMallocAsync(&d_temp_storage_max_bw, temp_storage_bytes_max_bw, stream);
        cub::DeviceReduce::Max(d_temp_storage_max_bw, temp_storage_bytes_max_bw, imageState.num_contrib_per_pixel, d_max_contrib_bw, num_pixels, stream);

        // Copy results to host
        int h_sum_contrib_bw = 0;
        int h_max_contrib_bw = 0;
        cudaMemcpyAsync(&h_sum_contrib_bw, d_sum_contrib_bw, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_max_contrib_bw, d_max_contrib_bw, sizeof(int), cudaMemcpyDeviceToHost, stream);
        
        // Synchronize stream before printing, but after launching the main backward kernel if possible.
        // For this debug print, we'll sync here.
        cudaStreamSynchronize(stream);


        // printf("[Backward Check] Loaded num_contrib_per_pixel: SUM = %d, MAX = %d (over %d pixels)\n",
            //    h_sum_contrib_bw, h_max_contrib_bw, num_pixels);
        fflush(stdout);

        // Free temporary device memory
        cudaFreeAsync(d_temp_storage_sum_bw, stream);
        cudaFreeAsync(d_temp_storage_max_bw, stream);
        cudaFreeAsync(d_sum_contrib_bw, stream);
        cudaFreeAsync(d_max_contrib_bw, stream);
    } else if (debug && imageState.num_contrib_per_pixel == nullptr) {
        printf("[Backward Check] imageState.num_contrib_per_pixel is NULL.\n");
        fflush(stdout);
    }
    // <<< END ADDED CUB REDUCTION >>>

	// --- 4. Execute Unified Backward Kernel ---
	// This single kernel computes gradients for confidences, features, weights, bias.
	// if (debug) printf("[Integrator::backward] Calling BACKWARD::compute_gradients...\n"); fflush(stdout);

	dim3 backward_grid = tile_grid;
	dim3 backward_block(BLOCK_X, BLOCK_Y, 1);

	BACKWARD::compute_gradients(
		backward_grid, backward_block,
		imageState.ranges,
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

	// if (debug) printf("[Integrator::backward] Synchronizing device...\n"); fflush(stdout);
	err = cudaDeviceSynchronize(); // Wait for kernel to complete and check for runtime errors
	if (err != cudaSuccess) {
		printf("[Integrator::backward] CUDA Error during compute_gradients execution (sync): %s\n", cudaGetErrorString(err));
		fflush(stdout);
		// Optionally, throw an exception or handle the error appropriately
		// throw std::runtime_error("CUDA Error during compute_gradients execution");
	} else {
		// if (debug) printf("[Integrator::backward] Device synchronization successful.\n"); fflush(stdout);
	}
	// ==========================================

	// if (debug) printf("[Integrator::backward] Completed BACKWARD::compute_gradients.\n"); fflush(stdout);

	// --- 5. Backward Preprocess (Placeholder) ---
	// This step is still needed to compute dL_dmeans3D.
	// It would likely take gradients computed by compute_gradients (e.g., dL/dPosWorld if added)
	// and propagate them back to the original means3D.
	// if (debug) printf("[Integrator::backward] PLACEHOLDER: Backward preprocess step needed for dL_dmeans3D.\n"); fflush(stdout);


	// --- 6. Finalize ---
	// Gradients dL_dlinear_weights, dL_dlinear_bias, dL_dfeature_table,
	// dL_dprimitive_confidences are now computed and stored in the output pointers.
	// dL_dmeans3D requires the missing backward preprocess step.

	// if (debug) printf("[Integrator::backward] Backward pass completed (excluding preprocess).\n"); fflush(stdout);
}

// Removed the __global__ void clearBuffer kernel definition as it's no longer called.