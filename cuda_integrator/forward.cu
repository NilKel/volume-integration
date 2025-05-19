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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "hashgrid.h"
namespace cg = cooperative_groups;

// Commenting out defines related to rendering kernel for now
// #define MAX_INPUT_LINEAR_DIM 256 // Placeholder, adjust if needed
// #define MAX_INPUT_FEATURE_DIM 32 // Placeholder, adjust if needed


// <<< NEW HELPER FUNCTIONS START >>> (Commented out as they are for rendering)

// Perform initial steps for each primitive prior to sorting.
__global__ void preprocessCUDA(int P,
	const float* orig_points,
	const float primitive_scale, // All primitives have the same scale
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	// Removed outputs: cov3Ds, rgb, conic_opacity, clamped
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered) // Keep prefiltered argument for potential future use
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this primitive will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near/far culling (and potentially frustum culling).
	float3 p_view; // p_view is point in view space
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// --- Primitive Bounding Sphere Calculation ---
	// Calculate the radius of the bounding sphere for the cube primitive
	// world_radius = (scale / 2) * sqrt(3)
	float world_radius = primitive_scale * 0.86602540378f ; // sqrt(3)/2

	// Project the world radius to screen space. Use the max focal length
	// and the view space depth. Add small epsilon to avoid division by zero.
	// Use absolute value of depth as view space z is typically negative.
	float view_depth = abs(p_view.z) + 1e-6;
	float screen_radius_x = world_radius * focal_x / view_depth;
	float screen_radius_y = world_radius * focal_y / view_depth;
	// Use the larger radius to ensure the bounding box covers the projection.
	// Ceil ensures we cover pixels partially overlapped.
	float my_radius_f = max(screen_radius_x, screen_radius_y);
	int my_radius = ceilf(my_radius_f);
	// --- End Bounding Sphere Calculation ---


	// --- Removed Covariance Calculation ---
	// const float* cov3D; ...
	// float3 cov = computeCov2D(...);
	// float det = ...
	// float3 conic = ...
	// --- End Removed Covariance Calculation ---


	// Calculate screen coordinates and tile bounding box
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    // float2 point_image = { (-p_proj.x+1)*0.5*W, (-p_proj.y+1)*0.5*H };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid); // Use calculated radius

	// If the primitive covers zero tiles, it is culled.
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// --- Removed Color Calculation ---
	// if (colors_precomp == nullptr) ...
	// --- End Removed Color Calculation ---

	// Store required data for sorting and potential future rendering.
	depths[idx] = p_view.z;
	radii[idx] = my_radius; // Store integer radius
	points_xy_image[idx] = point_image;
	// Removed conic_opacity storage
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// ===================================================================================
// == Rendering Kernel and Wrapper (Commented Out for Preprocess-Only Build) ==
// ===================================================================================

// New helper function to determine occupancy of a grid point for a specific pixel
__device__ char pointOccupancy(
    const float3& sample_pos_world,
    const float* viewmatrix,
    const float* projmatrix,
    int W, int H,
    uint2 pix,
    float near_plane,
    float max_distance,
    float confidence_value,
    float occupancy_threshold,
    const bool check_frustum // Added to control frustum check, useful for stencil neighbors
) {
    if (confidence_value <= occupancy_threshold) {
        return 0;
    }
    if (check_frustum) { // Only check frustum for the central point of the stencil, not necessarily all neighbors
        if (isPointInPyramidalFrustum(sample_pos_world, viewmatrix, projmatrix, W, H, pix, near_plane, max_distance)) {
            return 1;
        }
        return 0;
    }
    return 1; // If not checking frustum explicitly, and confidence is high, consider it occupied (for stencil purposes)
}


// Main Volume Integration Kernel
// Templated on the number of output color channels (e.g., 3 for RGB).
// Runtime dimensions (feature dims, levels) are passed as arguments.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    // Data from sorting
    const uint2* __restrict__ ranges,          // Tile ranges in the sorted point list
    const uint32_t* __restrict__ point_list,    // Sorted list of primitive indices
    // Image dimensions
    const int W, const int H,
    // Camera parameters
    const float* __restrict__ viewmatrix,      // View matrix (float[16])
    const float* __restrict__ projmatrix,      // Projection matrix (float[16])
    const float* camera_center_vec,             // Camera position in world space (float*)
    const float near_plane,                 // Near plane distance
    const float max_distance,               // Far plane distance (max ray distance)
    // Primitive data
    const float* __restrict__ primitive_centers,     // Primitive centers (P, 3)
    const float* __restrict__ primitive_confidences, // Primitive confidence grids (P, grid_size^3)
    const float primitive_scale,            // Scale of the primitives (uniform)
    // Hashgrid data
    const float* __restrict__ feature_table,   // Hashgrid feature table (flat)
    const int* __restrict__ resolution,        // Hashgrid resolutions per level (L)
    const int* __restrict__ do_hash,           // Flags for hashing per level (L)
    const int* __restrict__ primes,            // Hashing primes (3)
    const int feature_offset,               // Hash table size (T) per level
    // MLP data (Linear Layer)
    const float* __restrict__ linear_weights,  // MLP weights (flat)
    const float* __restrict__ linear_bias,     // MLP biases (flat)
    // Integration parameters
    const int stencil_genus,                // Stencil type (1 or 2)
    const int grid_size,                    // Dimension of the confidence grid (e.g., 3 for 3x3x3)
    const int max_primitives_per_ray,       // Max primitives to process per ray
    const float occupancy_threshold,        // NEW: Threshold for confidence -> occupancy
    // Background color
    const float* __restrict__ bg_color,        // Background color (CHANNELS)
    // Output buffers
    float* __restrict__ out_color,             // Output color image (CHANNELS, H, W)
    float* __restrict__ out_features,          // Output feature image (output_feature_dim, H, W) - Optional
    float* __restrict__ visibility_info,       // Output visibility (1-T) or accumulated alpha (H, W) - Optional
    float* __restrict__ out_final_transmittance,   // Stores final T per pixel (W*H)
    int* __restrict__ out_num_contrib_per_pixel,     // Stores count per pixel (W*H)
    int* __restrict__ out_pixel_contributing_indices, // Stores primitive index per pixel contribution (W*H*max_primitives_per_ray)
    float* __restrict__ out_delta_t,                 // Stores delta_t per pixel contribution (W*H*max_primitives_per_ray)
    // Runtime dimensions
    const uint32_t input_feature_dim,
    const uint32_t output_feature_dim,
    const uint32_t hashgrid_levels
)
{
    // <<< Read camera center from pointer inside kernel >>>
    const float3 camera_center = {camera_center_vec[0], camera_center_vec[1], camera_center_vec[2]};

    // --- Define Maximum Compile-Time Dimensions ---
    const uint32_t MAX_INPUT_FEATURE_DIM = 4;
    const uint32_t MAX_OUTPUT_FEATURE_DIM = 4;
    const uint32_t MAX_HASHGRID_LEVELS = 8;
    const uint32_t MAX_GRID_SIZE = 5;
    const uint32_t MAX_GRID_VOLUME = MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE;

    // --- Runtime Dimension Calculations ---
    // Total number of elements in the confidence grid (e.g., 3*3*3 = 27)
    const uint32_t grid_volume = (uint32_t)grid_size * grid_size * grid_size;
    // Dimension of the MLP input layer (concatenated features)
    const uint32_t input_linear_dim = input_feature_dim * hashgrid_levels;
    // Dimension of the MLP output layer (RGB + Density + Output Features)
    const uint32_t output_linear_dim = CHANNELS + output_feature_dim + 1;
    const uint32_t MAX_OUTPUT_LINEAR_DIM = CHANNELS + MAX_OUTPUT_FEATURE_DIM + 1;
    const uint32_t MAX_INPUT_LINEAR_DIM = MAX_INPUT_FEATURE_DIM * MAX_HASHGRID_LEVELS;
    const uint32_t MAX_STENCIL_SIZE = 5;


    // --- Shared Memory Allocation ---
    __shared__ float stencil_coeffs_x[MAX_STENCIL_SIZE];
    __shared__ float stencil_coeffs_y[MAX_STENCIL_SIZE];
    __shared__ float stencil_coeffs_z[MAX_STENCIL_SIZE];
    __shared__ int shared_stencil_size;

    // MLP Weights and Biases (using MAX dimensions for allocation)
    // Weights: (input_linear_dim, output_linear_dim) -> flat size = MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM
    // Bias: (output_linear_dim) -> flat size = MAX_OUTPUT_LINEAR_DIM
    __shared__ float shared_linear_weights[MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM];
    __shared__ float shared_linear_bias[MAX_OUTPUT_LINEAR_DIM];

    __attribute__((shared)) float sh_confidence[MAX_GRID_VOLUME];
    // --- SHARED MEMORY for projected coordinates. .x == -1 means culled. ---
    __attribute__((shared)) float2 sh_screen_coords[MAX_GRID_VOLUME];
    
    __attribute__((shared)) float sh_interpolated_features_for_primitive[MAX_GRID_VOLUME * MAX_INPUT_LINEAR_DIM];
    __attribute__((shared)) float sh_mlp_output_per_grid_point[MAX_GRID_VOLUME * MAX_OUTPUT_LINEAR_DIM];

    __shared__ uint32_t collected_primitive_indices[BLOCK_X * BLOCK_Y]; 

    // --- Thread/Block Identification ---
    auto block = cg::this_thread_block();
    uint32_t thread_idx_in_block = block.thread_rank();
    uint32_t threads_per_block = block.size(); // BLOCK_X * BLOCK_Y

    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint32_t tile_idx = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;

    bool inside = pix.x < W && pix.y < H;
    bool done_processing_pixel = !inside; // Thread is done if its pixel is outside image

    // <<< DEBUG: Target a specific pixel for detailed prints >>>
    uint32_t target_debug_pixel_id = (H / 2) * W + (W / 2); // Center pixel

    // <<< PRINT INFERRED POINT LIST LENGTH >>>
    if (tile_idx == 0 && thread_idx_in_block == 0) { // Print only once
        uint32_t num_tiles = gridDim.x * gridDim.y;
        if (num_tiles > 0) {
            // The .y component of the last range is the total number of rendered points
            uint32_t inferred_num_rendered = ranges[num_tiles - 1].y;
            printf("[POINT LIST LENGTH] Inferred from ranges: %u\n", inferred_num_rendered);
        } else {
            printf("[POINT LIST LENGTH] Cannot infer, num_tiles is 0.\n");
        }
    }
    // <<< END PRINT INFERRED POINT LIST LENGTH >>>

    // <<< ADD THIS DEBUG PRINT for RANGES (as per your previous version) >>>
    // Calculate the tile_idx for the target_debug_pixel_id
    // This only needs to be done once, e.g. by thread 0 of the block containing the target pixel,
    // or just let all threads in that block print it (it will be redundant but fine for debug).
    uint32_t target_pix_y = target_debug_pixel_id / W;
    uint32_t target_pix_x = target_debug_pixel_id % W;
    uint32_t target_tile_gidx_y = target_pix_y / BLOCK_Y;
    uint32_t target_tile_gidx_x = target_pix_x / BLOCK_X;
    uint32_t tile_idx_for_target_pixel = target_tile_gidx_y * gridDim.x + target_tile_gidx_x;

    if (tile_idx == tile_idx_for_target_pixel && thread_idx_in_block == 0 && inside) {
        // num_rendered needs to be passed as a kernel argument if not already available
        // For now, let's assume you can get it or hardcode it for this debug print if necessary
        // uint32_t actual_num_rendered = ...; // Get this value
        // For consistency with the above print, let's also infer it here for the debug message
        uint32_t inferred_num_rendered_for_check = 0;
        uint32_t num_tiles_for_check = gridDim.x * gridDim.y;
        if (num_tiles_for_check > 0) {
            inferred_num_rendered_for_check = ranges[num_tiles_for_check - 1].y;
        }
        printf("[FWD RANGES CHECK TILE %u for P:%u] range.x: %u, range.y: %u (inferred num_rendered: %u)\n",
               tile_idx, target_debug_pixel_id,
               ranges[tile_idx].x, ranges[tile_idx].y, inferred_num_rendered_for_check );
    }
    // <<< END ADDED DEBUG PRINT for RANGES >>>

    // --- Load Shared Data (Stencil, MLP Weights/Biases) ---
    if (thread_idx_in_block == 0) {
        int stencil_size_local;
        if (stencil_genus == 1) {
            stencil_size_local = 3;
            // Check bounds before writing (using MAX_STENCIL_SIZE)
            if (stencil_size_local <= MAX_STENCIL_SIZE) {
                stencil_coeffs_x[0] = -0.5f; stencil_coeffs_x[1] = 0.0f; stencil_coeffs_x[2] = 0.5f;
                stencil_coeffs_y[0] = -0.5f; stencil_coeffs_y[1] = 0.0f; stencil_coeffs_y[2] = 0.5f;
                stencil_coeffs_z[0] = -0.5f; stencil_coeffs_z[1] = 0.0f; stencil_coeffs_z[2] = 0.5f;
            }
        } else if (stencil_genus == 2) {
            stencil_size_local = 5;
            if (stencil_size_local <= MAX_STENCIL_SIZE) {
                stencil_coeffs_x[0] = 1.0f/12.0f; stencil_coeffs_x[1] = -2.0f/3.0f; stencil_coeffs_x[2] = 0.0f; stencil_coeffs_x[3] = 2.0f/3.0f; stencil_coeffs_x[4] = -1.0f/12.0f;
                stencil_coeffs_y[0] = 1.0f/12.0f; stencil_coeffs_y[1] = -2.0f/3.0f; stencil_coeffs_y[2] = 0.0f; stencil_coeffs_y[3] = 2.0f/3.0f; stencil_coeffs_y[4] = -1.0f/12.0f;
                stencil_coeffs_z[0] = 1.0f/12.0f; stencil_coeffs_z[1] = -2.0f/3.0f; stencil_coeffs_z[2] = 0.0f; stencil_coeffs_z[3] = 2.0f/3.0f; stencil_coeffs_z[4] = -1.0f/12.0f;
            }
        } else {
            stencil_size_local = 3; // Default
            if (stencil_size_local <= MAX_STENCIL_SIZE) {
                stencil_coeffs_x[0] = -0.5f; stencil_coeffs_x[1] = 0.0f; stencil_coeffs_x[2] = 0.5f;
                stencil_coeffs_y[0] = -0.5f; stencil_coeffs_y[1] = 0.0f; stencil_coeffs_y[2] = 0.5f;
                stencil_coeffs_z[0] = -0.5f; stencil_coeffs_z[1] = 0.0f; stencil_coeffs_z[2] = 0.5f;
            }
        }
        shared_stencil_size = stencil_size_local;
    }

    // Cooperatively load linear layer weights and biases
    int total_weights = input_linear_dim * output_linear_dim;
    for (int i = thread_idx_in_block; i < total_weights; i += threads_per_block) {
        if (i < MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM) {
             shared_linear_weights[i] = linear_weights[i];
        }
    }
    int total_biases = output_linear_dim;
     for (int i = thread_idx_in_block; i < total_biases; i += threads_per_block) {
         if (i < MAX_OUTPUT_LINEAR_DIM) {
            shared_linear_bias[i] = linear_bias[i];
         }
    }
    block.sync();

    const int stencil_size = shared_stencil_size;
    const int stencil_offset = (stencil_size - 1) / 2;

    // --- Per-Pixel Initialization ---
    float T = 1.0f;
    float C[CHANNELS];
    float F[MAX_OUTPUT_FEATURE_DIM]; // Accumulated features

    #pragma unroll
    for(int i=0; i<CHANNELS; ++i) C[i] = 0.0f;
    for(uint32_t i=0; i<output_feature_dim; ++i) F[i] = 0.0f;

    uint32_t contributor_count = 0;
    float last_view_depth = near_plane;

    // --- Process Primitives for this Tile (Batched Approach) ---
    uint2 range = ranges[tile_idx];
    const int num_primitives_in_tile = range.y - range.x;
    const int rounds = (num_primitives_in_tile + threads_per_block - 1) / threads_per_block;
    int primitives_to_do_in_tile = num_primitives_in_tile;
    // Debug: Print the number of primitives to do in the tile
    if (thread_idx_in_block == 0) {
        printf("[FWD DEBUG] num_primitives_in_tile: %u, rounds: %u, threads_per_block: %u, range.x: %u, range.y: %u\n", num_primitives_in_tile, rounds, threads_per_block, range.x, range.y);
    }
    for (int r = 0; r < rounds; ++r, primitives_to_do_in_tile -= threads_per_block) {
        // Early exit if all threads in the block are done with their pixels
        int num_done_threads = __syncthreads_count(done_processing_pixel);
        if (num_done_threads == threads_per_block) { // threads_per_block is block.size()
            break;
        }
        if (done_processing_pixel) { // If this specific thread is done, it still participates in syncs
             // but won't do heavy computation. It can help in fetching if needed.
        }


        // Collectively fetch primitive indices for the current batch
        int current_batch_offset = r * threads_per_block;
        if (thread_idx_in_block < primitives_to_do_in_tile && (current_batch_offset + thread_idx_in_block < num_primitives_in_tile) ) {
            collected_primitive_indices[thread_idx_in_block] = point_list[range.x + current_batch_offset + thread_idx_in_block];
        }
        block.sync();

        // // <<< DEBUG: Print range.x, range.y, batch offset, thread_idx_in_block, and current batch offset >>>
        // if (thread_idx_in_block == 0) {
        //     printf("[FWD DEBUG] range.x: %u, range.y: %u, batch_offset: %u, thread_idx_in_block: %u, threads_per_block: %u, r: %u\n",
        //            range.x, range.y, current_batch_offset, thread_idx_in_block, threads_per_block, r);
        // }
        // DEBUG: Print range.x and range.y
        // if (thread_idx_in_block == 0) {
        //     printf("[FWD DEBUG] range.x: %u, range.y: %u\n", range.x, range.y);
        // }
        // Iterate over primitives in the current fetched batch
        for (int j_batch = 0; j_batch < min((int)threads_per_block, primitives_to_do_in_tile); ++j_batch) {
            if (done_processing_pixel || contributor_count >= max_primitives_per_ray || T < 1e-4f) {
                // This thread is done with its pixel, but needs to stay in sync for shared memory ops for other threads
                // or for subsequent primitives in the batch if not all threads are done.
                // A simple break here might desync, better to skip heavy work.
                // The outer cg::all check handles full block exit.
                // If this thread is done, it won't contribute further.
            }

            if (!done_processing_pixel && contributor_count < max_primitives_per_ray && T >= 1e-4f) {
                // --- Get Primitive Data for current primitive in batch ---
                uint32_t primitive_idx = collected_primitive_indices[j_batch]; 

                float3 center = {
                    primitive_centers[primitive_idx * 3 + 0],
                    primitive_centers[primitive_idx * 3 + 1],
                    primitive_centers[primitive_idx * 3 + 2]
                };

                float3 p_view_center = transformPoint4x3(center, viewmatrix); 
                float view_depth_center = p_view_center.z; 
                float delta_t = max(0.0f, view_depth_center - last_view_depth);

                // --- Cooperatively Load Confidence, Project Grid Points, Interpolate Features, AND COMPUTE MLP per Grid Point ---
                const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
                const float grid_offset_factor = (grid_size - 1) / 2.0f;
                
                for (uint32_t flat_idx_sh = thread_idx_in_block; flat_idx_sh < grid_volume; flat_idx_sh += threads_per_block) {
                    if (flat_idx_sh >= MAX_GRID_VOLUME) continue;

                    // Initialize features and MLP output for this grid point to 0.0f
                    for (uint32_t f_idx = 0; f_idx < MAX_INPUT_LINEAR_DIM; ++f_idx) {
                        sh_interpolated_features_for_primitive[flat_idx_sh * MAX_INPUT_LINEAR_DIM + f_idx] = 0.0f;
                    }
                    for (uint32_t o_idx = 0; o_idx < MAX_OUTPUT_LINEAR_DIM; ++o_idx) {
                        sh_mlp_output_per_grid_point[flat_idx_sh * MAX_OUTPUT_LINEAR_DIM + o_idx] = 0.0f;
                    }

                    sh_confidence[flat_idx_sh] = primitive_confidences[(uint64_t)primitive_idx * grid_volume + flat_idx_sh];

                
                    int z_coord = flat_idx_sh / (grid_size * grid_size);
                    int rem = flat_idx_sh % (grid_size * grid_size);
                    int y_coord = rem / grid_size;
                    int x_coord = rem % grid_size;

                    float sample_rel_x = (x_coord - grid_offset_factor) * grid_step;
                    float sample_rel_y = (y_coord - grid_offset_factor) * grid_step;
                    float sample_rel_z = (z_coord - grid_offset_factor) * grid_step;
                    float3 sample_pos_world_sh = {
                        center.x + sample_rel_x,
                        center.y + sample_rel_y,
                        center.z + sample_rel_z
                    };
                    
                    float4 p_hom_grid = transformPoint4x4(sample_pos_world_sh, projmatrix);
                    
                    float p_w_grid = 1.0f / p_hom_grid.w;
                    float3 p_proj_grid = { p_hom_grid.x * p_w_grid, p_hom_grid.y * p_w_grid, p_hom_grid.z * p_w_grid };
                    sh_screen_coords[flat_idx_sh] = { ndc2Pix(p_proj_grid.x, W), ndc2Pix(p_proj_grid.y, H) };
                    
                    for (int current_level_loop = 0; current_level_loop < hashgrid_levels; ++current_level_loop) {
                        hashInterpolateShared( 
                            sample_pos_world_sh, current_level_loop, feature_table, resolution,
                            feature_offset, do_hash[current_level_loop], primes,
                            &sh_interpolated_features_for_primitive[flat_idx_sh * MAX_INPUT_LINEAR_DIM],
                            input_feature_dim, hashgrid_levels,
                            MAX_INPUT_LINEAR_DIM, input_linear_dim);
                    }

                    // --- Perform MLP for this grid point (flat_idx_sh) ---
                    float* current_grid_point_features = &sh_interpolated_features_for_primitive[flat_idx_sh * MAX_INPUT_LINEAR_DIM];
                    float* current_grid_point_mlp_output = &sh_mlp_output_per_grid_point[flat_idx_sh * MAX_OUTPUT_LINEAR_DIM];

                    for (uint32_t out_f = 0; out_f < output_linear_dim; ++out_f) {
                        if (out_f < MAX_OUTPUT_LINEAR_DIM) {
                            float dot_prod = shared_linear_bias[out_f]; // Initialize with bias
                            for (uint32_t in_f = 0; in_f < input_linear_dim; ++in_f) {
                                if (in_f < MAX_INPUT_LINEAR_DIM) { // Check against compile-time max for safety
                                    // Ensure shared_linear_weights indexing is correct:
                                    // It's typically (out_dim, in_dim) flattened row-major.
                                    // So, index is out_f * MAX_INPUT_LINEAR_DIM + in_f
                                    dot_prod += shared_linear_weights[out_f * MAX_INPUT_LINEAR_DIM + in_f] * current_grid_point_features[in_f];
                                }
                            }
                            current_grid_point_mlp_output[out_f] = dot_prod;
                        }
                    }
                    
                    
                } // End cooperative loop
                block.sync(); // Ensure all shared memory writes (confidence, screen_coords, features, MLP_outputs) are complete
                //DEBUG: Print range.x and range.y
                // if (thread_idx_in_block == 0) {
                //     printf("[FWD DEBUG] range.x: %u, range.y: %u\n", range.x, range.y);
                // }
                // --- STAGE 2: Per-Thread: Accumulate Weighted MLP Outputs from Grid Points ---
                float primitive_outputs[MAX_OUTPUT_LINEAR_DIM]; 
                for(uint32_t k=0; k<output_linear_dim; ++k) {
                    if (k < MAX_OUTPUT_LINEAR_DIM) primitive_outputs[k] = 0.0f;
                }

                const int stencil_offset = (stencil_size - 1) / 2; // e.g., for size 3, offset is 1 (s goes 0,1,2 -> stencil_idx -1,0,1)

                for (uint32_t flat_idx_center = 0; flat_idx_center < grid_volume; ++flat_idx_center) { // Renamed flat_idx to flat_idx_center for clarity
                    if (flat_idx_center >= MAX_GRID_VOLUME) continue;

                    // --- Calculate Occupancy Gradient using sh_screen_coords ---
                    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

                    // Deconstruct flat_idx_center to get its 3D grid coordinates (x_c, y_c, z_c)
                    int z_c = flat_idx_center / (grid_size * grid_size);
                    int rem_c = flat_idx_center % (grid_size * grid_size);
                    int y_c = rem_c / grid_size;
                    int x_c = rem_c % grid_size;

                    if (stencil_size > 0) { 
                        for (int s = 0; s < stencil_size; s++) {
                            // Relative stencil offset (e.g., -1, 0, 1 for 3-point stencil)
                            int stencil_relative_offset = s - stencil_offset; 

                            char occ_x_neighbor = 0, occ_y_neighbor = 0, occ_z_neighbor = 0;

                            // X-gradient neighbor
                            int nx_coord = x_c + stencil_relative_offset;
                            if (nx_coord >= 0 && nx_coord < grid_size) { // Check bounds for neighbor grid coord
                                uint32_t flat_nx_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)nx_coord;
                                if (flat_nx_idx < MAX_GRID_VOLUME) { // Check bounds for flat index
                                    float2 neighbor_screen_coords = sh_screen_coords[flat_nx_idx];
                                    // Check if the neighbor_screen_coords (pre-projected) fall into the current pixel
                                    if (neighbor_screen_coords.x >= pix.x && neighbor_screen_coords.x < (pix.x + 1.0f) &&
                                        neighbor_screen_coords.y >= pix.y && neighbor_screen_coords.y < (pix.y + 1.0f)) {
                                        occ_x_neighbor = 1;
                                    }
                                }
                            }
                            grad_x += stencil_coeffs_x[s] * (float)occ_x_neighbor;
 
                            // Y-gradient neighbor
                            int ny_coord = y_c + stencil_relative_offset;
                            if (ny_coord >= 0 && ny_coord < grid_size) { 
                                uint32_t flat_ny_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)ny_coord * grid_size + (uint32_t)x_c;
                                if (flat_ny_idx < MAX_GRID_VOLUME) {
                                    float2 neighbor_screen_coords = sh_screen_coords[flat_ny_idx];
                                    if (neighbor_screen_coords.x >= pix.x && neighbor_screen_coords.x < (pix.x + 1.0f) &&
                                        neighbor_screen_coords.y >= pix.y && neighbor_screen_coords.y < (pix.y + 1.0f)) {
                                        occ_y_neighbor = 1;
                                    }
                                }
                            }
                            grad_y += stencil_coeffs_y[s] * (float)occ_y_neighbor;
 
                            // Z-gradient neighbor
                            int nz_coord = z_c + stencil_relative_offset;
                            if (nz_coord >= 0 && nz_coord < grid_size) { 
                                uint32_t flat_nz_idx = (uint32_t)nz_coord * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)x_c;
                                if (flat_nz_idx < MAX_GRID_VOLUME) {
                                    float2 neighbor_screen_coords = sh_screen_coords[flat_nz_idx];
                                    if (neighbor_screen_coords.x >= pix.x && neighbor_screen_coords.x < (pix.x + 1.0f) &&
                                        neighbor_screen_coords.y >= pix.y && neighbor_screen_coords.y < (pix.y + 1.0f)) {
                                        occ_z_neighbor = 1;
                                    }
                                }
                            }
                            grad_z += stencil_coeffs_z[s] * (float)occ_z_neighbor;
                        }
                    } else { // No stencil / point sampling - check only the center point
                        float2 center_screen_coords = sh_screen_coords[flat_idx_center];
                        if (center_screen_coords.x >= pix.x && center_screen_coords.x < (pix.x + 1.0f) &&
                            center_screen_coords.y >= pix.y && center_screen_coords.y < (pix.y + 1.0f)) {
                            // If point sampling, and it's in the pixel, it has full "gradient" or contribution
                            // The concept of gradient is a bit different here.
                            // Let's assume a default contribution if it's in the pixel.
                            // This part might need refinement based on desired behavior for stencil_size == 0
                            grad_x = 1.0f; // Or some other default to indicate presence
                            grad_y = 1.0f;
                            grad_z = 1.0f; 
                            // Or, occupancy_grad_magnitude could be set directly to 1.0f here.
                        }
                    }

                    float occupancy_grad_magnitude = fabsf(grad_x) + fabsf(grad_y) + fabsf(grad_z) + 1.0f; // L1 norm
                    // Alternative: float occupancy_grad_magnitude = sqrtf(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z); // L2 norm

                    if (occupancy_grad_magnitude > 1e-7f) {
                        float weight = sh_confidence[flat_idx_center] * occupancy_grad_magnitude; 

                        float* mlp_output_for_grid_point = &sh_mlp_output_per_grid_point[flat_idx_center * MAX_OUTPUT_LINEAR_DIM];
                        for (uint32_t k = 0; k < output_linear_dim; ++k) { // Loop for color and features
                            if (k < output_linear_dim) { // Check against shared memory max bound
                                primitive_outputs[k] += mlp_output_for_grid_point[k] * weight + shared_linear_bias[k];
                                primitive_outputs[k] = (1.0f / (1.0f + expf(-primitive_outputs[k]))); // Sigmoid
                            }
                        }
                        // Density (last element) - often handled differently (e.g. ReLU or exp)
                        // uint32_t density_output_idx = output_linear_dim - 1;
                        // if (density_output_idx < MAX_OUTPUT_LINEAR_DIM) {
                        //      primitive_outputs[density_output_idx] += mlp_output_for_grid_point[density_output_idx] * weight + shared_linear_bias[density_output_idx];
                        //      // Density activation (e.g., expf for NeRF-like density, or ReLU) will be applied later
                        // }
                    }
                } // End loop over flat_idx_center for accumulating weighted MLP outputs
                
                
                uint32_t density_runtime_idx = output_linear_dim - 1; // Assuming density is the last output
                float primitive_density = primitive_outputs[density_runtime_idx];
                // if (density_runtime_idx < output_linear_dim) { // Check against shared memory max bound
                //     primitive_density = expf(primitive_outputs[density_runtime_idx]); // Apply exp to final accumulated density logit
                //     // primitive_density = 1.0f;
                //     // primitive_density = max(0.0f, primitive_outputs[density_runtime_idx]); // Or ReLU
                // }

                // --- Extract Activated Outputs (Color and Features) ---
                float primitive_rgb[CHANNELS];
                float primitive_features[MAX_OUTPUT_FEATURE_DIM];
                #pragma unroll
                for(int c=0; c<CHANNELS; ++c) primitive_rgb[c] = primitive_outputs[c];
                for(uint32_t f=0; f<output_feature_dim; ++f) {
                    uint32_t source_idx = CHANNELS + f;
                    if (source_idx < MAX_OUTPUT_LINEAR_DIM) {
                        primitive_features[f] = primitive_outputs[source_idx];
                    } else {
                        primitive_features[f] = 0.0f;
                    }
                }

                // --- Calculate Alpha and Composite ---
                float alpha = 1.0f - expf(-primitive_density * delta_t);
                alpha = min(max(alpha, 0.0f), 1.0f); // Clamp alpha
                float comp_weight = T * alpha; // Renamed from 'weight' to avoid confusion

                #pragma unroll
                for (int c = 0; c < CHANNELS; c++) C[c] += comp_weight * primitive_rgb[c];
                // for (int c = 0; c < CHANNELS; c++) C[c] += comp_weight * 1.0f;
                for (uint32_t f = 0; f < output_feature_dim; f++) F[f] += comp_weight * primitive_features[f];
                T *= (1.0f - alpha);

                // --- Store Primitive Index and Delta T for Backward Pass ---
                // Note: primitive_idx here is the global index. The problem asks for 'i' which was the loop counter
                // in the original code (index within the tile's point_list).
                // We need to store the original index from point_list for this tile.
                // The current primitive being processed is point_list[range.x + current_batch_offset + j_batch]
                // So, the index to store is (range.x + current_batch_offset + j_batch)
                // DEBUG: Print range.x and range.y
                if (thread_idx_in_block == 0) {
                    printf("[FWD DEBUG] range.x: %u, range.y: %u\n", range.x, range.y);
                }
                if (out_pixel_contributing_indices != nullptr) { // 'inside' check is implicitly handled by done_processing_pixel
                     int write_idx = pix_id * max_primitives_per_ray + contributor_count;
                     out_pixel_contributing_indices[write_idx] = range.x + current_batch_offset + j_batch; // Store index from original point_list for the tile
                     if(pix_id == target_debug_pixel_id) {
                        printf("[FWD DEBUG] write_idx: %d, pix_id: %d, contributor_count: %d, range.x: %u, current_batch_offset: %u, j_batch: %u, point_list[range.x + current_batch_offset + j_batch]: %u\n",
                               write_idx, pix_id, contributor_count, range.x, current_batch_offset, j_batch, range.x + current_batch_offset + j_batch);
                     }
                     if (out_delta_t != nullptr) {
                         out_delta_t[write_idx] = delta_t;
                     }
                }
                last_view_depth = view_depth_center;
                contributor_count++;

                if (T < 1e-4f || contributor_count >= max_primitives_per_ray) {
                    done_processing_pixel = true; // This pixel is done
                }

                // <<< DEBUG: Print values for the target pixel >>>
                if (inside && pix_id == target_debug_pixel_id && alpha > 0.0f) {
                    printf("[renderCUDA DBG P:%d Prim:%u] contrib_idx:%d, delta_t:%.4f, density:%.4f, alpha:%.4f, T_before:%.4f\n",
                           pix_id, (unsigned int)(ranges[tile_idx].x + current_batch_offset + j_batch), contributor_count,
                           delta_t, primitive_density, alpha, T);
                }

                if (inside && pix_id == target_debug_pixel_id) {
                     printf("[renderCUDA DBG P:%d Prim:%u] T_after:%.4f\n", pix_id, (unsigned int)(ranges[tile_idx].x + current_batch_offset + j_batch), T);
                }

                if (inside) { // Only write if originally an 'inside' pixel
                    // <<< DEBUG: Print final contributor_count for the target pixel >>>
                    if (pix_id == target_debug_pixel_id) {
                        printf("[renderCUDA DBG P:%d] Final contributor_count: %d, Final T: %.4f\n",
                               pix_id, contributor_count, T);
                    }

                    #pragma unroll
                    for (int c = 0; c < CHANNELS; c++) {
                        float bg = (bg_color != nullptr) ? bg_color[c] : 0.0f;
                        C[c] += comp_weight * primitive_rgb[c];
                        out_color[c * H * W + pix_id] = C[c];
                    }

                    if (out_features != nullptr) {
                        for (uint32_t f = 0; f < output_feature_dim; f++) {
                            out_features[f * H * W + pix_id] = F[f];
                        }
                    }
                    if (visibility_info != nullptr) {
                        visibility_info[pix_id] = 1.0f - T;
                    }
                    if (out_final_transmittance != nullptr) {
                        out_final_transmittance[pix_id] = T;
                    }
                    if (out_num_contrib_per_pixel != nullptr) {
                        out_num_contrib_per_pixel[pix_id] = contributor_count;
                    }
                }
            } // end if !done_processing_pixel
             block.sync(); // Sync after each primitive in batch to ensure all threads are ready for next primitive's shared mem load
                           // or to correctly evaluate cg::all for early exit.
        }
        block.sync(); // --- End Loop Over Primitives in Batch (j_batch) ---
    } // --- End Loop Over Batches (r) ---
    block.sync();
    // --- Finalize Pixel Color and Features ---
    if (inside) { // Only write if originally an 'inside' pixel
        // <<< DEBUG: Print final contributor_count for the target pixel >>>
        if (pix_id == target_debug_pixel_id) {
            printf("[renderCUDA DBG P:%d] Final contributor_count: %d, Final T: %.4f\n",
                   pix_id, contributor_count, T);
        }

        #pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
            float bg = (bg_color != nullptr) ? bg_color[c] : 0.0f;
            C[c] += T * bg;
            out_color[c * H * W + pix_id] = C[c];
        }

        if (out_features != nullptr) {
            for (uint32_t f = 0; f < output_feature_dim; f++) {
                out_features[f * H * W + pix_id] = F[f];
            }
        }
        if (visibility_info != nullptr) {
            visibility_info[pix_id] = 1.0f - T;
        }
        if (out_final_transmittance != nullptr) {
            out_final_transmittance[pix_id] = T;
        }
        if (out_num_contrib_per_pixel != nullptr) {
            out_num_contrib_per_pixel[pix_id] = contributor_count;
        }
    }
}

// Wrapper for the render/integration kernel launch.
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* viewmatrix,
	const float* projmatrix,
	const float* camera_center_vec,
	const float near_plane,
	const float max_distance,
	const float* primitive_centers,
	const float* primitive_confidences,
	const float primitive_scale,
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
	float* out_final_transmittance,
	int* out_num_contrib_per_pixel,
	int* out_pixel_contributing_indices,
    float* out_delta_t,
	const uint32_t input_feature_dim,
	const uint32_t output_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels
)
{
    printf("[FORWARD::render] About to launch renderCUDA kernel...\n");
    fflush(stdout);

	// Determine which kernel template specialization to launch based on num_output_channels
	if (num_output_channels == 3) {
		renderCUDA<3> <<<grid, block>>> (
			ranges,
			point_list,
			W, H,
			viewmatrix,
			projmatrix,
			camera_center_vec,
			near_plane,
			max_distance,
			primitive_centers,
			primitive_confidences,
			primitive_scale,
			feature_table,
			resolution,
			do_hash,
			primes,
			feature_offset,
			linear_weights,
			linear_bias,
			stencil_genus,
			grid_size,
			max_primitives_per_ray,
            occupancy_threshold,
			bg_color,
			out_color,
			out_features,
			visibility_info,
			out_final_transmittance,
			out_num_contrib_per_pixel,
			out_pixel_contributing_indices,
            out_delta_t,
			input_feature_dim,
			output_feature_dim,
			hashgrid_levels);
	} else {
		// Handle unsupported channel count...
		printf("Error: Unsupported number of output channels (%d) in FORWARD::render. Only 3 is currently supported.\n", num_output_channels);
	}

    // Check for kernel launch errors immediately
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("[FORWARD::render] CUDA kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        fflush(stdout);
    } else {
        printf("[FORWARD::render] renderCUDA kernel launch successful (or error check passed).\n");
        fflush(stdout);
    }
}

// Wrapper for the preprocess kernel launch. Updated signature.
void FORWARD::preprocess(int P, // Removed D, M
	const float* orig_points,
	const float primitive_scale, // All primitives have the same scale
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
    const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	
	int* radii,
	float2* points_xy_image,
	float* depths,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > ( // Removed template arg <NUM_CHANNELS>
		P, // Removed D, M
		orig_points,
		primitive_scale,
		viewmatrix,
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		points_xy_image,
		depths,
		grid,
		tiles_touched,
		prefiltered
		);
}

// Removed placeholder comments for volume integration kernel
// <<< VOLUME INTEGRATION KERNEL START >>>
// <<< VOLUME INTEGRATION KERNEL END >>>

// ... rest of the file (preprocessCUDA kernel, FORWARD::preprocess wrapper, FORWARD::render wrapper) ...
// ... rest of the file (preprocessCUDA kernel, FORWARD::preprocess wrapper, FORWARD::render wrapper) ...