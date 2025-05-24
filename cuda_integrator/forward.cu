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
__global__ void preprocessCUDA(
    int P,
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
    const int W, const int H, const int P,
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
    const uint32_t intermediate_feature_dim,
    const uint32_t hashgrid_levels
)
{
    // <<< Read camera center from pointer inside kernel >>>
    const float3 camera_center = {camera_center_vec[0], camera_center_vec[1], camera_center_vec[2]};
    // printf("input feature dim: %d, output feature dim: %d, hashgrid levels: %d\n", input_feature_dim, output_feature_dim, hashgrid_levels);
    
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
    const uint32_t input_linear_dim = intermediate_feature_dim;
    // Dimension of the MLP output layer (RGB + Density + Output Features)
    const uint32_t output_linear_dim = CHANNELS + output_feature_dim + 1;
    const uint32_t MAX_OUTPUT_LINEAR_DIM = CHANNELS + MAX_OUTPUT_FEATURE_DIM + 1;
    const uint32_t MAX_INPUT_LINEAR_DIM = MAX_INPUT_FEATURE_DIM * MAX_HASHGRID_LEVELS;
    const uint32_t MAX_STENCIL_SIZE = 5;
    // printf("input_linear_dim: %d, output_linear_dim: %d, input_feature_dim: %d, output_feature_dim: %d, intermediate_feature_dim: %d\n", input_linear_dim, output_linear_dim    , input_feature_dim, output_feature_dim, intermediate_feature_dim);
    // printf("MAX_INPUT_LINEAR_DIM: %d, MAX_OUTPUT_LINEAR_DIM: %d\n", MAX_INPUT_LINEAR_DIM, MAX_OUTPUT_LINEAR_DIM);
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
            // printf("[POINT LIST LENGTH] Inferred from ranges: %u\n", inferred_num_rendered);
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
        // printf("[FWD RANGES CHECK TILE %u for P:%u] range.x: %u, range.y: %u (inferred num_rendered: %u)\n",
        //        tile_idx, target_debug_pixel_id,
        //        ranges[tile_idx].x, ranges[tile_idx].y, inferred_num_rendered_for_check );
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

    const int stencil_size = shared_stencil_size;
    // const int stencil_offset = (stencil_size - 1) / 2; // Will be defined inside STAGE 2 where it's first used

    // --- Per-Pixel Initialization ---
    float T = 1.0f;
    float C[CHANNELS];
    float F[MAX_OUTPUT_FEATURE_DIM]; // Accumulated features

    #pragma unroll
    for(int i=0; i<CHANNELS; ++i) C[i] = 0.0f;
    for(uint32_t i=0; i<output_feature_dim; ++i) F[i] = 0.0f; // Use runtime output_feature_dim

    uint32_t contributor_count = 0;
    // bool inside = pix.x < W && pix.y < H; // Already defined earlier

    // State for deferred accumulation
    bool has_pending_primitive = false;
    float pending_primitive_properties[MAX_OUTPUT_LINEAR_DIM]; // Stores sigmoid(color), sigmoid(features), density_logit
    float pending_view_depth = near_plane; // Depth of the pending primitive
    uint32_t pending_point_list_idx_for_output; // Index in the original point_list (e.g. range.x + offset + j_batch)
    float T_at_pending_primitive_start; // Stores T *before* the pending primitive would be composited


    // --- Process Primitives for this Tile (Batched Approach) ---
    const uint2 range = ranges[tile_idx];
    const int num_primitives_in_tile = range.y - range.x;
    const int rounds = (num_primitives_in_tile + threads_per_block - 1) / threads_per_block;
    int primitives_to_do_in_tile = num_primitives_in_tile;
    for (int r = 0; r < rounds; ++r, primitives_to_do_in_tile -= threads_per_block) {
        // Early exit if all threads in the block are done with their pixels
        int num_done_threads = __syncthreads_count(done_processing_pixel);
        if (num_done_threads == threads_per_block) {
            break;
        }
        // If this specific thread is done, it still participates in syncs
        // but won't do heavy computation.

        // Collectively fetch primitive indices for the current batch
        int current_batch_offset = r * threads_per_block;
        if (thread_idx_in_block < primitives_to_do_in_tile && (current_batch_offset + thread_idx_in_block < num_primitives_in_tile) ) {
            collected_primitive_indices[thread_idx_in_block] = point_list[range.x + current_batch_offset + thread_idx_in_block];
        }
        block.sync(); // Sync after loading collected_primitive_indices

        for (int j_batch = 0; j_batch < min((int)threads_per_block, primitives_to_do_in_tile); ++j_batch) {
            // --- Get Primitive Data (common for all threads in block for STAGE 1) ---
            uint32_t primitive_idx_from_point_list = collected_primitive_indices[j_batch];

            float3 center = {
                primitive_centers[primitive_idx_from_point_list * 3 + 0],
                primitive_centers[primitive_idx_from_point_list * 3 + 1],
                primitive_centers[primitive_idx_from_point_list * 3 + 2]
            };

            // --- STAGE 1: Cooperatively Load Confidence, Project Grid Points, Interpolate Features, AND COMPUTE MLP per Grid Point ---
            // This stage computes `sh_mlp_output_per_grid_point` which contains raw logits from MLP.
            const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
            const float grid_offset_factor = (grid_size - 1) / 2.0f;
            // DEBUG FOR A POINT PRINT THE sample_pos_world_sh
            // if (pix_id == target_debug_pixel_id){
            //     printf("sample_pos_world_sh: %f, %f, %f\n", sample_pos_world_sh.x, sample_pos_world_sh.y, sample_pos_world_sh.z);
            // }
            for (uint32_t flat_idx_sh = thread_idx_in_block; flat_idx_sh < grid_volume; flat_idx_sh += threads_per_block) {
                if (flat_idx_sh >= MAX_GRID_VOLUME || flat_idx_sh < 0) continue;

                // Initialize features and MLP output for this grid point to 0.0f
                for (uint32_t f_idx = 0; f_idx < input_linear_dim; ++f_idx) {
                    sh_interpolated_features_for_primitive[flat_idx_sh * input_linear_dim + f_idx] = 0.0f;
                }
                for (uint32_t o_idx = 0; o_idx < output_linear_dim; ++o_idx) {
                    sh_mlp_output_per_grid_point[flat_idx_sh * output_linear_dim + o_idx] = 0.0f;
                }

                sh_confidence[flat_idx_sh] = primitive_confidences[(uint64_t)primitive_idx_from_point_list * grid_volume + flat_idx_sh];
                // if(primitive_idx_from_point_list>P){
                
                // }
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

                float p_w_grid = 1.0f / (p_hom_grid.w+ 0.0000001f);
                float3 p_proj_grid = { p_hom_grid.x * p_w_grid, p_hom_grid.y * p_w_grid, p_hom_grid.z * p_w_grid };
                if (flat_idx_sh >= 0 && flat_idx_sh < MAX_GRID_VOLUME) // Bounds check for flat_idx_sh
                
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
                        float dot_prod = 0.0f; // Initialize with bias
                        
                        for (uint32_t in_f = 0; in_f < input_linear_dim; ++in_f) {
                            if (in_f < MAX_INPUT_LINEAR_DIM) {
                                dot_prod += shared_linear_weights[out_f * MAX_INPUT_LINEAR_DIM + in_f] * current_grid_point_features[in_f];
                            }
                        }
                        current_grid_point_mlp_output[out_f] = dot_prod; // Store raw weighted sum of features
                    }
                }
            } // End cooperative loop for STAGE 1
            block.sync(); // Ensure all shared memory writes from STAGE 1 (sh_mlp_output_per_grid_point etc.) are complete

            // --- Per-Pixel Processing (if this thread's pixel is not done) ---
            if (!done_processing_pixel) {
                float3 p_view_center_thread = transformPoint4x3(center, viewmatrix); // Thread-local view transform
                float current_primitive_view_depth = p_view_center_thread.z;
                uint32_t current_primitive_point_list_idx = range.x + current_batch_offset + j_batch;

                // --- STAGE 2: Per-Thread: Accumulate Weighted MLP Outputs from Grid Points, Apply Bias & Activation ---
                float current_properties_activated[MAX_OUTPUT_LINEAR_DIM]; // To store sigmoid(C), sigmoid(F), density_logit
                for(uint32_t k=0; k<output_linear_dim; ++k) { // Initialize accumulator
                    if (k < MAX_OUTPUT_LINEAR_DIM) current_properties_activated[k] = 0.0f;
                }

                const int current_stencil_offset = (stencil_size - 1) / 2;
                float occupancy_grad_sum = 0.0f;
                float2 current_proj = {0.0f, 0.0f};
                for (uint32_t flat_idx_center = 0; flat_idx_center < grid_volume; ++flat_idx_center) {
                    if (flat_idx_center >= MAX_GRID_VOLUME) continue;

                    // --- Calculate Occupancy Gradient using sh_screen_coords ---
                    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

                    // Deconstruct flat_idx_center to get its 3D grid coordinates (x_c, y_c, z_c)
                    int z_c = flat_idx_center / (grid_size * grid_size);
                    int rem_c = flat_idx_center % (grid_size * grid_size);
                    int y_c = rem_c / grid_size;
                    int x_c = rem_c % grid_size;
                    current_proj = sh_screen_coords[flat_idx_center];
                    if ((!(current_proj.x >= pix.x && current_proj.x < (pix.x + 1.0f) &&
                                        current_proj.y >= pix.y && current_proj.y < (pix.y + 1.0f)))  || (sh_confidence[flat_idx_center] <= 0.0f)){
                        continue;
                    }   
                    // if (stencil_size > 0) {
                    //     for (int s = 0; s < stencil_size; s++) {
                    //         int stencil_relative_offset = s - current_stencil_offset;
                    //         char occ_x_neighbor = 0, occ_y_neighbor = 0, occ_z_neighbor = 0;

                    //         // X-gradient neighbor
                    //         int nx_coord = x_c + stencil_relative_offset;
                    //         if (nx_coord >= 0 && nx_coord < grid_size) { // Check bounds for neighbor grid coord
                    //             uint32_t flat_nx_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)nx_coord;
                    //             if (flat_nx_idx < MAX_GRID_VOLUME) { // Check bounds for flat index
                    //                 float2 neighbor_screen_coords = sh_screen_coords[flat_nx_idx];
                    //                 // Check if the neighbor_screen_coords (pre-projected) fall into the current pixel
                    //                 if (neighbor_screen_coords.x >= pix.x && neighbor_screen_coords.x < (pix.x + 1.0f) &&
                    //                     neighbor_screen_coords.y >= pix.y && neighbor_screen_coords.y < (pix.y + 1.0f)) {
                    //                     grad_x += stencil_coeffs_x[s];
                    //                 }
                    //             }
                    //         }
 
                    //         // Y-gradient neighbor
                    //         int ny_coord = y_c + stencil_relative_offset;
                    //         if (ny_coord >= 0 && ny_coord < grid_size) { 
                    //             uint32_t flat_ny_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)ny_coord * grid_size + (uint32_t)x_c;
                    //             if (flat_ny_idx < MAX_GRID_VOLUME) {
                    //                 float2 neighbor_screen_coords = sh_screen_coords[flat_ny_idx];
                    //                 if (neighbor_screen_coords.x >= pix.x && neighbor_screen_coords.x < (pix.x + 1.0f) &&
                    //                     neighbor_screen_coords.y >= pix.y && neighbor_screen_coords.y < (pix.y + 1.0f)) {
                    //                     grad_y += stencil_coeffs_y[s];
                    //                 }
                    //             }
                    //         }
 
                    //         // Z-gradient neighbor
                    //         int nz_coord = z_c + stencil_relative_offset;
                    //         if (nz_coord >= 0 && nz_coord < grid_size) { 
                    //             uint32_t flat_nz_idx = (uint32_t)nz_coord * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)x_c;
                    //             if (flat_nz_idx < MAX_GRID_VOLUME) {
                    //                 float2 neighbor_screen_coords = sh_screen_coords[flat_nz_idx];
                    //                 if (neighbor_screen_coords.x >= pix.x && neighbor_screen_coords.x < (pix.x + 1.0f) &&
                    //                     neighbor_screen_coords.y >= pix.y && neighbor_screen_coords.y < (pix.y + 1.0f)) {
                    //                     grad_z += stencil_coeffs_z[s];
                    //                 }
                    //             }
                    //         }
                    //     }
                    // } else { // No stencil / point sampling - check only the center point
                    //     float2 center_screen_coords = sh_screen_coords[flat_idx_center];
                    //     if (center_screen_coords.x >= pix.x && center_screen_coords.x < (pix.x + 1.0f) &&
                    //         center_screen_coords.y >= pix.y && center_screen_coords.y < (pix.y + 1.0f)) {
                    //         // If point sampling, and it's in the pixel, it has full "gradient" or contribution
                    //         // The concept of gradient is a bit different here.
                    //         // Let's assume a default contribution if it's in the pixel.
                    //         // This part might need refinement based on desired behavior for stencil_size == 0
                    //         grad_x = 1.0f; // Or some other default to indicate presence
                    //         grad_y = 1.0f;
                    //         grad_z = 1.0f; 
                    //         // Or, occupancy_grad_magnitude could be set directly to 1.0f here.
                    //     }
                    // }

                    if (stencil_size > 0) {
                        for (int s = 0; s < stencil_size; s++) {
                            int stencil_relative_offset = s - current_stencil_offset;
                            char occ_x_neighbor = 0, occ_y_neighbor = 0, occ_z_neighbor = 0;

                            // X-gradient neighbor
                            int nx_coord = x_c + stencil_relative_offset;
                            if (nx_coord >= 0 && nx_coord < grid_size) { // Check bounds for neighbor grid coord
                                uint32_t flat_nx_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)nx_coord;
                                if (flat_nx_idx < MAX_GRID_VOLUME) { // Check bounds for flat index
                                    
                                    if (sh_confidence[flat_nx_idx] > 0.0f) {
                                        grad_x += stencil_coeffs_x[s];
                                    }
                                }
                            }
 
                            // Y-gradient neighbor
                            int ny_coord = y_c + stencil_relative_offset;
                            if (ny_coord >= 0 && ny_coord < grid_size) { 
                                uint32_t flat_ny_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)ny_coord * grid_size + (uint32_t)x_c;
                                if (flat_ny_idx < MAX_GRID_VOLUME) {
                                    if (sh_confidence[flat_ny_idx] > 0.0f) {
                                        grad_y += stencil_coeffs_y[s];
                                    }
                                }
                            }
 
                            // Z-gradient neighbor
                            int nz_coord = z_c + stencil_relative_offset;
                            if (nz_coord >= 0 && nz_coord < grid_size) { 
                                uint32_t flat_nz_idx = (uint32_t)nz_coord * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)x_c;
                                if (flat_nz_idx < MAX_GRID_VOLUME) {
                                    if (sh_confidence[flat_nz_idx] > 0.0f) {
                                        grad_z += stencil_coeffs_z[s];
                                    }
                                }
                            }
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
                    
                    float occupancy_grad_magnitude = fabsf(grad_x) + fabsf(grad_y) + fabsf(grad_z);
                    occupancy_grad_sum += occupancy_grad_magnitude;
                    if (occupancy_grad_magnitude > 1e-7f) { 
                        
                        float confidence_weight = sh_confidence[flat_idx_center] * occupancy_grad_magnitude;
                        float* mlp_output_for_grid_point = &sh_mlp_output_per_grid_point[flat_idx_center * MAX_OUTPUT_LINEAR_DIM];
                        for (uint32_t k = 0; k < output_linear_dim; ++k) {
                            if (k < MAX_OUTPUT_LINEAR_DIM) {
                                current_properties_activated[k] += mlp_output_for_grid_point[k] * confidence_weight;
                                
                                
                                
                            }
                        }
                                            
                    }
                } // End loop over flat_idx_center for STAGE 2 accumulation
                
                if (occupancy_grad_sum < 1e-7f){
                    continue;
                }

                for (uint32_t k = 0; k < output_linear_dim; ++k) {
                    if (k < MAX_OUTPUT_LINEAR_DIM) {
                        current_properties_activated[k] += shared_linear_bias[k];
                        if (k<output_linear_dim-1){
                            current_properties_activated[k] = (1.0f/(1.0f + expf(-current_properties_activated[k])));
                        }
                        else{
                            current_properties_activated[k] = expf(current_properties_activated[k]);
                        }   
                    }
                }
            
                
                // Now current_properties_activated holds: sigmoid(Color), sigmoid(Features), DensityLogit

                // --- Process Pending Primitive (if any) using current_primitive_view_depth ---
                if (has_pending_primitive) {
                    float delta_t_for_pending = max(0.0f, current_primitive_view_depth - pending_view_depth);

                    // Only contribute if segment is meaningful and we haven't hit max contributors
                    if (delta_t_for_pending > 1e-5f && contributor_count < max_primitives_per_ray) {
                        float pending_density = pending_primitive_properties[output_linear_dim - 1];
                        

                        float alpha = 1.0f - expf(-pending_density * delta_t_for_pending);
                        alpha = min(max(alpha, 0.0f), 1.0f); // Clamp alpha
                        float comp_weight = T_at_pending_primitive_start * alpha;

                        #pragma unroll
                        for (int c_idx = 0; c_idx < CHANNELS; c_idx++) C[c_idx] += comp_weight * pending_primitive_properties[c_idx];

                        for (uint32_t f_idx = 0; f_idx < output_feature_dim; f_idx++) {
                            uint32_t source_idx = CHANNELS + f_idx;
                            if (source_idx < output_linear_dim - 1 && source_idx < MAX_OUTPUT_LINEAR_DIM) { // Ensure it's a feature
                                 F[f_idx] += comp_weight * pending_primitive_properties[source_idx];
                            }
                        }
                        T = T_at_pending_primitive_start * (1.0f - alpha);

                        if (out_pixel_contributing_indices != nullptr) {
                             int write_idx = pix_id * max_primitives_per_ray + contributor_count;
                             out_pixel_contributing_indices[write_idx] = pending_point_list_idx_for_output;
                             if (out_delta_t != nullptr) {
                                 out_delta_t[write_idx] = delta_t_for_pending;
                             }
                        }
                        // if (inside && pix_id == target_debug_pixel_id && alpha > 0.01f) {
                        //     printf("[renderCUDA DBG P:%d Prim:%u (pending)] contrib#:%d, depth:%.2f->%.2f, dt:%.4f, dens:%.4f, alpha:%.4f, T_start:%.4f, T_end:%.4f\n",
                        //            pix_id, pending_point_list_idx_for_output, contributor_count,
                        //            pending_view_depth, current_primitive_view_depth, delta_t_for_pending, pending_density, alpha, T_at_pending_primitive_start, T);
                        // }
                        contributor_count++;
                    }
                } // End if (has_pending_primitive)

                // --- Current primitive becomes the new pending primitive ---
                // Only if pixel is still active (T high enough, contributor count not maxed)
                if (T >= 1e-4f && contributor_count < max_primitives_per_ray) {
                    for(uint32_t k=0; k<output_linear_dim; ++k) {
                        if (k < MAX_OUTPUT_LINEAR_DIM) {
                            pending_primitive_properties[k] = current_properties_activated[k];
                        }
                    }
                    pending_view_depth = current_primitive_view_depth;
                    pending_point_list_idx_for_output = current_primitive_point_list_idx;
                    has_pending_primitive = true;
                    T_at_pending_primitive_start = T; // Store current T for this new pending primitive
                } else {
                    // Pixel became inactive (T too low or max contributors reached by processing the previous pending primitive)
                    // So, this current primitive cannot become pending.
                    has_pending_primitive = false; // Ensure no new pending primitive is set
                    done_processing_pixel = true;  // Mark pixel as done
                }

                // Update done_processing_pixel based on current T and contributor_count
                // This check is somewhat redundant if the above 'else' sets done_processing_pixel,
                // but it's a good safeguard.
                if (T < 1e-4f || contributor_count >= max_primitives_per_ray) {
                    done_processing_pixel = true;
                }
            } // end if (!done_processing_pixel) for this thread's pixel work

            block.sync(); // Sync after each primitive in batch, for next primitive's STAGE 1 shared mem ops
        } // --- End Loop Over Primitives in Batch (j_batch) ---
    } // --- End Loop Over Batches (r) ---
    block.sync(); // Sync after all r-loops are done by all threads in block.

    // --- After the main loop, process the final pending primitive if any ---
    if (inside && has_pending_primitive && T >= 1e-4f && contributor_count < max_primitives_per_ray) {
        float delta_t_final = max(0.0f, max_distance - pending_view_depth);

        if (delta_t_final > 1e-5f) {
            float pending_density = pending_primitive_properties[output_linear_dim - 1];
            

            float alpha = 1.0f - expf(-pending_density * delta_t_final);
            alpha = min(max(alpha, 0.0f), 1.0f);
            float comp_weight = T_at_pending_primitive_start * alpha;
            // printf("[renderCUDA DBG P:%d Prim:%u (final pending)] contrib#:%d, depth:%.2f->%.2f, dt:%.4f, dens:%.4f, alpha:%.4f, T_start:%.4f, T_end:%.4f\n",
            //            pix_id, pending_point_list_idx_for_output, contributor_count,
            //            pending_view_depth, max_distance, delta_t_final, pending_density, alpha, T_at_pending_primitive_start, T);
            #pragma unroll
            for (int c_idx = 0; c_idx < CHANNELS; c_idx++) C[c_idx] += comp_weight * pending_primitive_properties[c_idx];
            for (uint32_t f_idx = 0; f_idx < output_feature_dim; f_idx++) {
                uint32_t source_idx = CHANNELS + f_idx;
                if (source_idx < output_linear_dim - 1 && source_idx < MAX_OUTPUT_LINEAR_DIM) {
                     F[f_idx] += comp_weight * pending_primitive_properties[source_idx];
                }
            }
            T = T_at_pending_primitive_start * (1.0f - alpha);

            if (out_pixel_contributing_indices != nullptr) {
                 int write_idx = pix_id * max_primitives_per_ray + contributor_count;
                 out_pixel_contributing_indices[write_idx] = pending_point_list_idx_for_output;
                 if (out_delta_t != nullptr) {
                     out_delta_t[write_idx] = delta_t_final;
                 }
            }
            // if (inside && pix_id == target_debug_pixel_id && alpha > 0.01f) {
            //     printf("[renderCUDA DBG P:%d Prim:%u (final pending)] contrib#:%d, depth:%.2f->%.2f, dt:%.4f, dens:%.4f, alpha:%.4f, T_start:%.4f, T_end:%.4f\n",
            //            pix_id, pending_point_list_idx_for_output, contributor_count,
            //            pending_view_depth, max_distance, delta_t_final, pending_density, alpha, T_at_pending_primitive_start, T);
            // }
            contributor_count++;
        }
    }

    // --- Finalize Pixel Color and Features (add background and write to output) ---
    if (inside) {
        #pragma unroll
        for (int c_idx = 0; c_idx < CHANNELS; c_idx++) {
            float bg = (bg_color != nullptr) ? bg_color[c_idx] : 0.0f;
            C[c_idx] += T * bg; // Add background contribution weighted by final T
            C[c_idx] = min(max(C[c_idx], 0.0f), 1.0f);
            out_color[c_idx * H * W + pix_id] = C[c_idx];
        }

        if (out_features != nullptr) {
            for (uint32_t f_idx = 0; f_idx < output_feature_dim; f_idx++) {
                if(f_idx < MAX_OUTPUT_FEATURE_DIM){
                    out_features[f_idx * H * W + pix_id] = F[f_idx];
                }
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
        // Final debug print for the target pixel
        // if (pix_id == target_debug_pixel_id) {
        //      printf("[renderCUDA DBG P:%d] Final Output. Contribs: %d, Final T: %.4f, Final C[0]: %.4f\n",
        //             pix_id, contributor_count, T, (CHANNELS > 0 ? C[0] : 0.0f) );
        // }
    }
}

// Wrapper for the render/integration kernel launch.
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int P,
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
    const uint32_t intermediate_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels
)
{
    // printf("[FORWARD::render] About to launch renderCUDA kernel...\n");
    // fflush(stdout);

	// Determine which kernel template specialization to launch based on num_output_channels
	if (num_output_channels == 3) {
		renderCUDA<3> <<<grid, block>>> (
			ranges,
			point_list,
			W, H, P,
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
			intermediate_feature_dim,
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
        // printf("[FORWARD::render] renderCUDA kernel launch successful (or error check passed).\n");
        // fflush(stdout);
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