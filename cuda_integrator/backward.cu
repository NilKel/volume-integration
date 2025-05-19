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

#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h> // Include if using half precision
#include <cstdint>     // For uint32_t
#include <cmath>       // For floorf, powf, expf, fabsf, max, min etc.
#include <vector_types.h> // Should be included via backward.h, but explicit doesn't hurt
#include "auxiliary.h"    // <<< Make sure this is included BEFORE the function below >>>
#include "hashgrid.h"     // <<< Includes hashInterpolate, hashComputeGradient, isPointInPyramidalFrustum >>>
#include "backward.h" // Include the corresponding header

namespace cg = cooperative_groups;

// Define constants consistent with forward pass if needed (e.g., for shared memory sizing)
const uint32_t MAX_INPUT_FEATURE_DIM_BW = 4;  // F_max
const uint32_t MAX_OUTPUT_FEATURE_DIM_BW = 4; // F_out_max
const uint32_t MAX_HASHGRID_LEVELS_BW = 8;    // L_max
const uint32_t MAX_GRID_SIZE_BW = 5;
const uint32_t MAX_GRID_VOLUME_BW = MAX_GRID_SIZE_BW * MAX_GRID_SIZE_BW * MAX_GRID_SIZE_BW;
const uint32_t MAX_STENCIL_SIZE_BW = 5;
// Assuming CHANNELS = 3 for color
const uint32_t MAX_OUTPUT_LINEAR_DIM_BW = 3 + MAX_OUTPUT_FEATURE_DIM_BW + 1; // RGB(3) + Feat(F_out) + Density(1)
const uint32_t MAX_INPUT_LINEAR_DIM_BW = MAX_INPUT_FEATURE_DIM_BW * MAX_HASHGRID_LEVELS_BW; // F_in * L
// <<< ADDED: Compile-time max blend dimension >>>
// Note: CHANNELS is a template parameter, so this works.
template <uint32_t CHANNELS>
constexpr uint32_t MAX_BLEND_DIM_BW = CHANNELS + MAX_OUTPUT_FEATURE_DIM_BW; // Renamed for clarity


// =========================================================================
// Backward Hash Interpolation Gradient Distribution (Declaration/Definition)
// =========================================================================
// Calculates the gradient contribution to the feature table entries based on
// the incoming gradient dL_dFeature_j and the interpolation weights.
// <<< REMOVED: backward_hash_interpolate_distribute function is deleted >>>

// =========================================================================
// Unified Backward Kernel
// =========================================================================
template <uint32_t CHANNELS> // Typically 3 for RGB
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
unifiedBackwardKernel(
    // --- Input Gradients (Pixel-based) ---
    const uint2* __restrict__ ranges,
    const float* __restrict__ dL_dOut_Color,    // (CHANNELS, H, W)
    const float* __restrict__ dL_dOut_Features, // (output_feature_dim, H, W) - Optional
    // --- Forward Pass State/Outputs (Pixel-based & Contribution-based) ---
    const float* __restrict__ final_T,                 // Final transmittance per pixel (H, W)
    const int* __restrict__ num_contrib_per_pixel,     // Contribution count per pixel (H, W)
    const int* __restrict__ pixel_contributing_indices,// GLOBAL index i from point_list per pixel contribution (H, W, max_primitives_per_ray)
    const float* __restrict__ store_delta_t,           // Stored delta_t per contribution (H, W, max_primitives_per_ray)
    // --- Data from Sorting ---
    const uint32_t* __restrict__ point_list,           // Sorted list of primitive indices (num_rendered) - Maps global index i to primitive_idx
    int P,                                // Total number of unique primitives
    int num_rendered,                     // Total number of contributions in point_list
    // --- Image Dimensions ---
    const int W, const int H,
    // --- Camera Parameters ---
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const float* camera_center_vec, // float3 passed as float*
    float near_plane,
    float max_distance,
    // --- Primitive Data (Needed for recomputation) ---
    const float* __restrict__ primitive_centers,     // (P, 3)
    const float* __restrict__ primitive_confidences, // (P, grid_volume)
    float primitive_scale,
    // --- Hashgrid Data ---
    const float* __restrict__ feature_table,         // (L * T * F) or similar layout
    const int* __restrict__ resolution,              // Array of resolutions per level (L)
    const int* __restrict__ do_hash,                 // Array of hash flags per level (L)
    const int* __restrict__ primes,                  // Hashing primes (3)
    int feature_offset,                 // Size T of hash table per level
    uint64_t feature_table_size,        // Total size of feature table (L*T*F)
    // --- MLP Data ---
    const float* __restrict__ linear_weights,        // (input_linear_dim, output_linear_dim)
    const float* __restrict__ linear_bias,           // (output_linear_dim)
    // --- Integration Parameters ---
    int stencil_genus,
    int grid_size,
    int max_primitives_per_ray,         // Max contributions considered per pixel
    float occupancy_threshold,
    // --- Background ---
    const float* __restrict__ bg_color,              // (CHANNELS)
    // --- Runtime Dimensions ---
    const uint32_t input_feature_dim,         // F
    const uint32_t output_feature_dim,        // F_out
    const uint32_t hashgrid_levels,           // L
    // --- FINAL Output Gradients (Accumulated atomically, MUST be zero-initialized before launch) ---
    float* dL_dLinearWeights,           // (input_linear_dim, output_linear_dim)
    float* dL_dLinearBias,              // (output_linear_dim)
    float* dL_dFeatureTable,            // (L * T * F) - Size matches feature_table_size
    float* dL_dprimitive_confidences    // (P * grid_volume)
)
{
    // --- Kernel Setup ---
    const float3 camera_center = {camera_center_vec[0], camera_center_vec[1], camera_center_vec[2]};

    // Runtime dimensions derived
    const uint32_t grid_volume = (uint32_t)grid_size * grid_size * grid_size;
    const uint32_t input_linear_dim = input_feature_dim * hashgrid_levels;
    const uint32_t output_linear_dim = CHANNELS + output_feature_dim + 1; // +1 for density
    const uint32_t blend_dim = CHANNELS + output_feature_dim; // Runtime blend dimension (Color + Features)

    // Compile-time max dimensions for stack/shared arrays
    const uint32_t KERNEL_MAX_OUTPUT_LINEAR_DIM_USED = MAX_OUTPUT_LINEAR_DIM_BW; // CHANNELS + F_out_max + 1
    const uint32_t KERNEL_MAX_BLEND_DIM_USED = MAX_BLEND_DIM_BW<CHANNELS>; // CHANNELS + F_out_max

    // Add a debug pixel ID
    const uint32_t DEBUG_PIX_ID = 320400; // Change this to inspect a different pixel

    // Shared memory
    __shared__ float stencil_coeffs_x[MAX_STENCIL_SIZE_BW];
    __shared__ float stencil_coeffs_y[MAX_STENCIL_SIZE_BW];
    __shared__ float stencil_coeffs_z[MAX_STENCIL_SIZE_BW];
    __shared__ int shared_stencil_size;

    // MLP Weights and Biases (using MAX dimensions for allocation)
    // Weights: (input_linear_dim, output_linear_dim) -> flat size = MAX_INPUT_LINEAR_DIM_BW * MAX_OUTPUT_LINEAR_DIM_BW
    // Bias: (output_linear_dim) -> flat size = MAX_OUTPUT_LINEAR_DIM_BW
    __shared__ float shared_linear_weights[MAX_INPUT_LINEAR_DIM_BW * MAX_OUTPUT_LINEAR_DIM_BW];
    __shared__ float shared_linear_bias[MAX_OUTPUT_LINEAR_DIM_BW];

    __attribute__((shared)) float sh_confidence[MAX_GRID_VOLUME_BW];
    // __attribute__((shared)) float sh_pixel_occupancy[MAX_GRID_VOLUME_BW]; // Occupancy for the *current* pixel - REMOVED, computed per-thread
    // __attribute__((shared)) float sh_occupancy_grad_mag[MAX_GRID_VOLUME_BW]; // Store grad magnitude per grid point - REMOVED, computed per-thread

    // Shared memory for collaboratively recomputed primitive data (mirroring forward pass)
    __attribute__((shared)) float2 sh_screen_coords[MAX_GRID_VOLUME_BW]; // Projected grid points
    __attribute__((shared)) float sh_interpolated_features_for_primitive[MAX_GRID_VOLUME_BW * MAX_INPUT_LINEAR_DIM_BW];
    __attribute__((shared)) float sh_mlp_output_per_grid_point[MAX_GRID_VOLUME_BW * MAX_OUTPUT_LINEAR_DIM_BW];


    // Thread/Block Identification
    auto block = cg::this_thread_block();
    uint32_t thread_idx_in_block = block.thread_rank();
    uint32_t threads_per_block = block.size(); // BLOCK_X * BLOCK_Y

    uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix = {pix_min.x + thread_idx_in_block % BLOCK_X, pix_min.y + thread_idx_in_block / BLOCK_X}; // Corrected pix calculation
    uint32_t pix_id = W * pix.y + pix.x; // Global pixel index (row-major)
    bool inside = pix.x < W && pix.y < H;

    // --- Load Shared Data (Stencil, MLP Weights/Bias) ---
    if (thread_idx_in_block == 0) {
        int stencil_size_local;
        if (stencil_genus == 1) {
            stencil_size_local = 3;
            if (stencil_size_local <= MAX_STENCIL_SIZE_BW) {
                stencil_coeffs_x[0] = -0.5f; stencil_coeffs_x[1] = 0.0f; stencil_coeffs_x[2] = 0.5f;
                stencil_coeffs_y[0] = -0.5f; stencil_coeffs_y[1] = 0.0f; stencil_coeffs_y[2] = 0.5f;
                stencil_coeffs_z[0] = -0.5f; stencil_coeffs_z[1] = 0.0f; stencil_coeffs_z[2] = 0.5f;
            } else { stencil_size_local = 0; }
        } else if (stencil_genus == 2) {
            stencil_size_local = 5;
            if (stencil_size_local <= MAX_STENCIL_SIZE_BW) {
                stencil_coeffs_x[0] = 1.0f/12.0f; stencil_coeffs_x[1] = -2.0f/3.0f; stencil_coeffs_x[2] = 0.0f; stencil_coeffs_x[3] = 2.0f/3.0f; stencil_coeffs_x[4] = -1.0f/12.0f;
                stencil_coeffs_y[0] = 1.0f/12.0f; stencil_coeffs_y[1] = -2.0f/3.0f; stencil_coeffs_y[2] = 0.0f; stencil_coeffs_y[3] = 2.0f/3.0f; stencil_coeffs_y[4] = -1.0f/12.0f;
                stencil_coeffs_z[0] = 1.0f/12.0f; stencil_coeffs_z[1] = -2.0f/3.0f; stencil_coeffs_z[2] = 0.0f; stencil_coeffs_z[3] = 2.0f/3.0f; stencil_coeffs_z[4] = -1.0f/12.0f;
            } else { stencil_size_local = 0; }
        } else { // Default genus 1
            stencil_size_local = 3;
             if (stencil_size_local <= MAX_STENCIL_SIZE_BW) {
                stencil_coeffs_x[0] = -0.5f; stencil_coeffs_x[1] = 0.0f; stencil_coeffs_x[2] = 0.5f;
                stencil_coeffs_y[0] = -0.5f; stencil_coeffs_y[1] = 0.0f; stencil_coeffs_y[2] = 0.5f;
                stencil_coeffs_z[0] = -0.5f; stencil_coeffs_z[1] = 0.0f; stencil_coeffs_z[2] = 0.5f;
            } else { stencil_size_local = 0; }
        }
        shared_stencil_size = stencil_size_local;
    }

    // Cooperatively load linear layer weights and biases
    int total_weights_to_load = input_linear_dim * output_linear_dim;
    for (int i = thread_idx_in_block; i < total_weights_to_load; i += threads_per_block) {
        // Assuming linear_weights is [output_linear_dim, input_linear_dim] row-major
        // And shared_linear_weights is also [MAX_OUTPUT_LINEAR_DIM_BW, MAX_INPUT_LINEAR_DIM_BW] row-major
        // Or, if both are flat [out * in_dim + in]
        // The forward pass loads shared_linear_weights[out * MAX_INPUT_LINEAR_DIM + in] = linear_weights[out * input_linear_dim + in]
        // Let's assume linear_weights is flat [idx] and shared_linear_weights is also flat [idx]
        // where idx = out_idx * actual_input_dim + in_idx for global,
        // and   idx = out_idx * MAX_INPUT_DIM + in_idx for shared.
        // For simplicity, assume flat layout matching:
        if (i < MAX_INPUT_LINEAR_DIM_BW * KERNEL_MAX_OUTPUT_LINEAR_DIM_USED) { // Check against allocated shared size
             shared_linear_weights[i] = linear_weights[i]; // Assumes linear_weights matches this flat layout up to total_weights_to_load
        }
    }
    int total_biases_to_load = output_linear_dim;
    for (int i = thread_idx_in_block; i < total_biases_to_load; i += threads_per_block) {
         if (i < KERNEL_MAX_OUTPUT_LINEAR_DIM_USED) { // Check against allocated shared size
            shared_linear_bias[i] = linear_bias[i];
         }
    }
    block.sync();
    const int stencil_size = shared_stencil_size;
    const int stencil_offset = (stencil_size > 0) ? (stencil_size - 1) / 2 : 0;

    // --- Per-Pixel Backward Initialization ---
    float dL_dOut_pix_local[KERNEL_MAX_BLEND_DIM_USED];
    float accum_blend_after[KERNEL_MAX_BLEND_DIM_USED];
    float T_current_pixel = 0.0f;
    int num_contrib_pixel = 0;
    int current_pixel_next_contrib_idx = -1; // Index for pixel_contributing_indices, counts down
    
    if (pix_id == DEBUG_PIX_ID) {
        printf("[BKWD_DBG P:%u InitCheck] Checking inside=%d\n", pix_id, inside);
    }
    if (inside) {
        num_contrib_pixel = num_contrib_per_pixel[pix_id];
        num_contrib_pixel = min(num_contrib_pixel, max_primitives_per_ray);
        current_pixel_next_contrib_idx = num_contrib_pixel - 1;

        if (pix_id == DEBUG_PIX_ID) {
            printf("[BKWD_DBG P:%u Init] Inside. num_contrib_pixel=%d, current_pixel_next_contrib_idx=%d\n",
                   pix_id, num_contrib_pixel, current_pixel_next_contrib_idx);
        }

        for(int c=0; c<CHANNELS; ++c) {
            dL_dOut_pix_local[c] = dL_dOut_Color[c * H * W + pix_id];
            accum_blend_after[c] = (bg_color != nullptr) ? bg_color[c] : 0.0f;
        }
        if (pix_id == DEBUG_PIX_ID && CHANNELS > 0) {
            printf("[BKWD_DBG P:%u Init] dL_dOut_Color[0] (dL_dOut_pix_local[0]) = %f\n", pix_id, dL_dOut_pix_local[0]);
        }
        if (dL_dOut_Features != nullptr) {
            for(uint32_t f=0; f<output_feature_dim; ++f) {
                dL_dOut_pix_local[CHANNELS + f] = dL_dOut_Features[f * H * W + pix_id];
                accum_blend_after[CHANNELS + f] = 0.0f; // Background features are zero
            }
        } else {
             for(uint32_t f=0; f<output_feature_dim; ++f) {
                dL_dOut_pix_local[CHANNELS + f] = 0.0f;
                accum_blend_after[CHANNELS + f] = 0.0f;
            }
        }
        // Initialize any remaining elements in the max-sized arrays to 0
        for(uint32_t d = blend_dim; d < KERNEL_MAX_BLEND_DIM_USED; ++d) {
             dL_dOut_pix_local[d] = 0.0f;
             accum_blend_after[d] = 0.0f;
        }
        T_current_pixel = final_T[pix_id];
        if (pix_id == DEBUG_PIX_ID) {
             printf("[BKWD_DBG P:%u Init] Initial T_current_pixel=%f\n", pix_id, T_current_pixel);
        }
    } else {
        if (pix_id == DEBUG_PIX_ID) {
            printf("[BKWD_DBG P:%u Init] Skipped (not inside).\n", pix_id);
        }
        for(uint32_t d=0; d<KERNEL_MAX_BLEND_DIM_USED; ++d) {
            dL_dOut_pix_local[d] = 0.0f;
            accum_blend_after[d] = 0.0f;
        }
        T_current_pixel = 1.0f; // Or 0.0f? If outside, final_T might be 1.0 if bg is black and T is for scene.
                               // Let's assume final_T[pix_id] is valid even if 'inside' is false, or init to 1.
                               // Matching forward pass, if not rendered, T remains 1.0.
        num_contrib_pixel = 0;
        current_pixel_next_contrib_idx = -1;
    }
    block.sync();
    // --- Iterate Through Primitives for this Tile (in reverse order of forward processing) ---
    uint2 range = ranges[tile_idx];
    const int num_entries_for_tile_in_point_list = range.y - range.x;

    for (int k_rev_tile = num_entries_for_tile_in_point_list - 1; k_rev_tile >= 0; k_rev_tile--) {
        int current_idx_in_global_point_list = range.x + k_rev_tile;
        uint32_t current_primitive_idx_for_block = point_list[current_idx_in_global_point_list];

        if (pix_id == DEBUG_PIX_ID) {
            printf("[BKWD_DBG P:%u Loop] k_rev_tile=%d, global_idx_in_point_list=%d, prim_idx_for_block=%u\n",
                   pix_id, k_rev_tile, current_idx_in_global_point_list, current_primitive_idx_for_block);
        }

        if (current_primitive_idx_for_block >= P) continue; // Should not happen with valid point_list

        float3 center_current_prim = {
            primitive_centers[current_primitive_idx_for_block * 3 + 0],
            primitive_centers[current_primitive_idx_for_block * 3 + 1],
            primitive_centers[current_primitive_idx_for_block * 3 + 2]
        };
        const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
        const float grid_offset_factor = (grid_size - 1) / 2.0f;

        // --- STAGE 1 (Collaborative): Recompute forward state for `current_primitive_idx_for_block` into shared memory ---
        for (uint32_t flat_idx_sh = thread_idx_in_block; flat_idx_sh < grid_volume; flat_idx_sh += threads_per_block) {
            if (flat_idx_sh >= MAX_GRID_VOLUME_BW) continue;

            // Initialize features and MLP output for this grid point to 0.0f
            for (uint32_t f_idx = 0; f_idx < MAX_INPUT_LINEAR_DIM_BW; ++f_idx) {
                sh_interpolated_features_for_primitive[flat_idx_sh * MAX_INPUT_LINEAR_DIM_BW + f_idx] = 0.0f;
            }
            for (uint32_t o_idx = 0; o_idx < MAX_OUTPUT_LINEAR_DIM_BW; ++o_idx) {
                sh_mlp_output_per_grid_point[flat_idx_sh * MAX_OUTPUT_LINEAR_DIM_BW + o_idx] = 0.0f;
            }

            sh_confidence[flat_idx_sh] = primitive_confidences[(uint64_t)current_primitive_idx_for_block * grid_volume + flat_idx_sh];

            int z_coord = flat_idx_sh / (grid_size * grid_size);
            int rem = flat_idx_sh % (grid_size * grid_size);
            int y_coord = rem / grid_size;
            int x_coord = rem % grid_size;

            float sample_rel_x = (x_coord - grid_offset_factor) * grid_step;
            float sample_rel_y = (y_coord - grid_offset_factor) * grid_step;
            float sample_rel_z = (z_coord - grid_offset_factor) * grid_step;
            float3 sample_pos_world_sh = {
                center_current_prim.x + sample_rel_x,
                center_current_prim.y + sample_rel_y,
                center_current_prim.z + sample_rel_z
            };
            
            float4 p_hom_grid = transformPoint4x4(sample_pos_world_sh, projmatrix); // Uses projmatrix (global)
            float p_w_grid = 1.0f / p_hom_grid.w;
            float3 p_proj_grid = { p_hom_grid.x * p_w_grid, p_hom_grid.y * p_w_grid, p_hom_grid.z * p_w_grid };
            sh_screen_coords[flat_idx_sh] = { ndc2Pix(p_proj_grid.x, W), ndc2Pix(p_proj_grid.y, H) };
            
            // Interpolate features
            for (int current_level_loop = 0; current_level_loop < hashgrid_levels; ++current_level_loop) {
                hashInterpolateShared( 
                    sample_pos_world_sh, current_level_loop, feature_table, resolution,
                    feature_offset, do_hash[current_level_loop], primes,
                    &sh_interpolated_features_for_primitive[flat_idx_sh * MAX_INPUT_LINEAR_DIM_BW], // Target in shared mem
                    input_feature_dim, hashgrid_levels,
                    MAX_INPUT_LINEAR_DIM_BW, // Max stride for shared memory
                    input_linear_dim         // Actual feature dimension to fill
                );
            }

            // Perform MLP for this grid point
            float* current_grid_point_features_sh = &sh_interpolated_features_for_primitive[flat_idx_sh * MAX_INPUT_LINEAR_DIM_BW];
            float* current_grid_point_mlp_output_sh = &sh_mlp_output_per_grid_point[flat_idx_sh * MAX_OUTPUT_LINEAR_DIM_BW];

            for (uint32_t out_f = 0; out_f < output_linear_dim; ++out_f) {
                if (out_f < MAX_OUTPUT_LINEAR_DIM_BW) { // Check against compile-time max for shared memory write
                    float dot_prod = shared_linear_bias[out_f]; // Initialize with bias (from shared)
                    for (uint32_t in_f = 0; in_f < input_linear_dim; ++in_f) {
                        if (in_f < MAX_INPUT_LINEAR_DIM_BW) { // Check against compile-time max for shared memory read
                            // shared_linear_weights is [MAX_OUTPUT_LINEAR_DIM_BW, MAX_INPUT_LINEAR_DIM_BW]
                            // Access: shared_linear_weights[out_f * MAX_INPUT_LINEAR_DIM_BW + in_f]
                            dot_prod += shared_linear_weights[out_f * MAX_INPUT_LINEAR_DIM_BW + in_f] * current_grid_point_features_sh[in_f];
                        }
                    }
                    current_grid_point_mlp_output_sh[out_f] = dot_prod;
                }
            }
        }
        block.sync(); // Ensure sh_confidence, sh_screen_coords, sh_interpolated_features, sh_mlp_output are ready

        // --- STAGE 2 (Per-Thread/Pixel processing) ---
        bool should_process_pixel = inside && current_pixel_next_contrib_idx >= 0;
        if (pix_id == DEBUG_PIX_ID) {
             printf("[BKWD_DBG P:%u S2 Check] Checking should_process_pixel: (inside=%d && current_pixel_next_contrib_idx=%d >= 0) -> %d\n",
                    pix_id, inside, current_pixel_next_contrib_idx, should_process_pixel);
        }
        if (should_process_pixel) {
            // Retrieve the stored point_list_idx for the current expected contributor
            uint32_t stored_point_list_idx_for_pixel = pixel_contributing_indices[pix_id * max_primitives_per_ray + current_pixel_next_contrib_idx];
            
            uint32_t expected_global_primitive_id_for_pixel = 0xFFFFFFFF; // Default to an invalid ID

            // Boundary check before using stored_point_list_idx_for_pixel to access point_list
            // num_rendered is the total size of point_list
            if (stored_point_list_idx_for_pixel < num_rendered) { 
                expected_global_primitive_id_for_pixel = point_list[stored_point_list_idx_for_pixel];
            }

            bool primitive_matches_pixel = (current_primitive_idx_for_block == expected_global_primitive_id_for_pixel);

            if (pix_id == DEBUG_PIX_ID) {
                // Print the stored_point_list_idx as well for clarity
                printf("[BKWD_DBG P:%u S2 MatchCheck] StoredPLI:%u, ExpectedGID:%u == CurrentGID:%u -> %d\n",
                       pix_id, stored_point_list_idx_for_pixel, expected_global_primitive_id_for_pixel, current_primitive_idx_for_block, primitive_matches_pixel);
            }

            if (primitive_matches_pixel) {
                // This pixel (thread) needs to process current_primitive_idx_for_block.
                float delta_t_k = store_delta_t[pix_id * max_primitives_per_ray + current_pixel_next_contrib_idx];

                // (A) Recompute Blend_k and primitive_density_k for THIS PIXEL using shared data
                float mlp_input_aggregated_pixel[MAX_INPUT_LINEAR_DIM_BW];
                for(uint32_t i=0; i<input_linear_dim; ++i) mlp_input_aggregated_pixel[i] = 0.0f;
                for(uint32_t i=input_linear_dim; i<MAX_INPUT_LINEAR_DIM_BW; ++i) mlp_input_aggregated_pixel[i] = 0.0f;


                // Loop over grid points of `current_primitive_idx_for_block`
                for (uint32_t flat_idx_center = 0; flat_idx_center < grid_volume; ++flat_idx_center) {
                    if (flat_idx_center >= MAX_GRID_VOLUME_BW) continue;

                    float grad_x_pixel = 0.0f, grad_y_pixel = 0.0f, grad_z_pixel = 0.0f;
                    int z_c = flat_idx_center / (grid_size * grid_size);
                    int rem_c = flat_idx_center % (grid_size * grid_size);
                    int y_c = rem_c / grid_size;
                    int x_c = rem_c % grid_size;

                    if (stencil_size > 0) {
                        for (int s = 0; s < stencil_size; s++) {
                            int stencil_relative_offset = s - stencil_offset;
                            char occ_x_n = 0, occ_y_n = 0, occ_z_n = 0;
                            int nx = x_c + stencil_relative_offset;
                            if (nx >= 0 && nx < grid_size) {
                                uint32_t flat_nx_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)nx;
                                if (flat_nx_idx < MAX_GRID_VOLUME_BW) {
                                    float2 n_scr = sh_screen_coords[flat_nx_idx];
                                    if (n_scr.x >= pix.x && n_scr.x < (pix.x + 1.0f) &&
                                        n_scr.y >= pix.y && n_scr.y < (pix.y + 1.0f)) occ_x_n=1;
                                }
                            } grad_x_pixel += stencil_coeffs_x[s] * (float)occ_x_n;
                            int ny = y_c + stencil_relative_offset;
                            if (ny >= 0 && ny < grid_size) {
                                uint32_t flat_ny_idx = (uint32_t)z_c * grid_size * grid_size + (uint32_t)ny * grid_size + (uint32_t)x_c;
                                if (flat_ny_idx < MAX_GRID_VOLUME_BW) {
                                    float2 n_scr = sh_screen_coords[flat_ny_idx];
                                    if (n_scr.x >= pix.x && n_scr.x < (pix.x + 1.0f) && n_scr.y >= pix.y && n_scr.y < (pix.y + 1.0f)) occ_y_n=1;
                                }
                            } grad_y_pixel += stencil_coeffs_y[s] * (float)occ_y_n;
                            int nz = z_c + stencil_relative_offset;
                            if (nz >= 0 && nz < grid_size) {
                                uint32_t flat_nz_idx = (uint32_t)nz * grid_size * grid_size + (uint32_t)y_c * grid_size + (uint32_t)x_c;
                                if (flat_nz_idx < MAX_GRID_VOLUME_BW) {
                                    float2 n_scr = sh_screen_coords[flat_nz_idx];
                                    if (n_scr.x >= pix.x && n_scr.x < (pix.x+1.0f) && n_scr.y >= pix.y && n_scr.y < (pix.y+1.0f)) occ_z_n=1;
                                }
                            } grad_z_pixel += stencil_coeffs_z[s] * (float)occ_z_n;
                        }
                    } else {
                        float2 c_scr = sh_screen_coords[flat_idx_center];
                        if (c_scr.x >= pix.x && c_scr.x < (pix.x + 1.0f) && c_scr.y >= pix.y && c_scr.y < (pix.y + 1.0f)) {
                            grad_x_pixel = 1.0f; grad_y_pixel = 1.0f; grad_z_pixel = 1.0f;
                        }
                    }
                    float occupancy_grad_magnitude_pixel = fabsf(grad_x_pixel) + fabsf(grad_y_pixel) + fabsf(grad_z_pixel) + 1.0f; // L1 norm + 1

                    bool check_occ_grad_mag = occupancy_grad_magnitude_pixel > 1e-7f;
                    if (pix_id == DEBUG_PIX_ID && flat_idx_center == 0) { // Print for first grid point only
                        printf("[BKWD_DBG P:%u S2A GridLoopCheck] flat_idx_center=%u, Checking occ_grad_mag_pixel=%f > 1e-7f -> %d\n",
                               pix_id, flat_idx_center, occupancy_grad_magnitude_pixel, check_occ_grad_mag);
                    }

                    if (check_occ_grad_mag) { // Note: was > 1e-7f, L1+1 ensures >1 if any contribution
                        float weight_pixel = sh_confidence[flat_idx_center] * occupancy_grad_magnitude_pixel;
                        float* interpolated_features_sh = &sh_interpolated_features_for_primitive[flat_idx_center * MAX_INPUT_LINEAR_DIM_BW];
                        for (uint32_t f_idx = 0; f_idx < input_linear_dim; ++f_idx) {
                             if (f_idx < MAX_INPUT_LINEAR_DIM_BW) { // Boundary check for mlp_input_aggregated_pixel
                                mlp_input_aggregated_pixel[f_idx] += interpolated_features_sh[f_idx] * weight_pixel;
                             }
                        }
                    }
                } // End loop over grid points for mlp_input_aggregated_pixel

                // Apply Linear Layer (MLP) for this pixel
                float primitive_outputs_pre_activation_pixel[MAX_OUTPUT_LINEAR_DIM_BW];
                for(uint32_t i=0; i<output_linear_dim; ++i) primitive_outputs_pre_activation_pixel[i] = 0.0f;
                for(uint32_t i=output_linear_dim; i<MAX_OUTPUT_LINEAR_DIM_BW; ++i) primitive_outputs_pre_activation_pixel[i] = 0.0f;


                for (uint32_t out_f = 0; out_f < output_linear_dim; ++out_f) {
                    if (out_f < MAX_OUTPUT_LINEAR_DIM_BW) { // Check against shared_linear_bias and primitive_outputs_pre_activation_pixel size
                        float dot_prod = shared_linear_bias[out_f];
                        for (uint32_t in_f = 0; in_f < input_linear_dim; ++in_f) {
                            if (in_f < MAX_INPUT_LINEAR_DIM_BW) { // Check against shared_linear_weights and mlp_input_aggregated_pixel size
                                dot_prod += shared_linear_weights[out_f * MAX_INPUT_LINEAR_DIM_BW + in_f] * mlp_input_aggregated_pixel[in_f];
                            }
                        }
                        primitive_outputs_pre_activation_pixel[out_f] = dot_prod;
                    }
                }

                // Apply Activations for this pixel
                float Blend_k_pixel[KERNEL_MAX_BLEND_DIM_USED]; // Max sized for safety
                for(uint32_t i=0; i<blend_dim; ++i) Blend_k_pixel[i] = 0.0f;
                for(uint32_t i=blend_dim; i<KERNEL_MAX_BLEND_DIM_USED; ++i) Blend_k_pixel[i] = 0.0f;

                float primitive_density_k_pixel = 0.0f;

                for (uint32_t k_blend = 0; k_blend < blend_dim; ++k_blend) {
                    if (k_blend < output_linear_dim && k_blend < MAX_OUTPUT_LINEAR_DIM_BW) { // Check pre_activation index
                        Blend_k_pixel[k_blend] = 1.0f / (1.0f + expf(-primitive_outputs_pre_activation_pixel[k_blend]));
                    }
                }
                uint32_t density_idx_mlp = output_linear_dim - 1;
                if (density_idx_mlp < MAX_OUTPUT_LINEAR_DIM_BW) { // Check pre_activation index
                    primitive_density_k_pixel = expf(primitive_outputs_pre_activation_pixel[density_idx_mlp]);
                    primitive_density_k_pixel = max(0.0f, primitive_density_k_pixel);
                }
                
                float alpha_k_pixel = 1.0f - expf(-primitive_density_k_pixel * delta_t_k);
                alpha_k_pixel = min(max(alpha_k_pixel, 0.0f), 1.0f);

                // (B) Perform backward step for this contribution
                float T_before_pixel = 0.0f;
                float one_minus_alpha_pixel = 1.0f - alpha_k_pixel;
                bool check_one_minus_alpha = fabsf(one_minus_alpha_pixel) > 1e-6f;
                 if (pix_id == DEBUG_PIX_ID) {
                    printf("[BKWD_DBG P:%u S2B TBeforeCheck] Checking fabsf(one_minus_alpha_pixel=%f) > 1e-6f -> %d\n",
                           pix_id, one_minus_alpha_pixel, check_one_minus_alpha);
                 }
                if (check_one_minus_alpha) {
                    T_before_pixel = T_current_pixel / one_minus_alpha_pixel;
                    T_before_pixel = min(max(T_before_pixel, 0.0f), 1.0f);
                } else { 
                    T_before_pixel = 0.0f; 
                    if (pix_id == DEBUG_PIX_ID) {
                        printf("[BKWD_DBG P:%u S2B TBeforeCheck] Result: T_before_pixel set to 0.\n", pix_id);
                    }
                }

                float dL_dBlend_k_pixel[KERNEL_MAX_BLEND_DIM_USED];
                for(int d=0; d<blend_dim; ++d) {
                    if (d < KERNEL_MAX_BLEND_DIM_USED) {
                        dL_dBlend_k_pixel[d] = dL_dOut_pix_local[d] * T_before_pixel * alpha_k_pixel;
                    } else { dL_dBlend_k_pixel[d] = 0.0f; }
                }

                float dL_dalpha_k_pixel = 0.0f;
                for(int d=0; d<blend_dim; ++d) {
                    if (d < KERNEL_MAX_BLEND_DIM_USED) {
                        dL_dalpha_k_pixel += dL_dOut_pix_local[d] * T_before_pixel * (Blend_k_pixel[d] - accum_blend_after[d]);
                    }
                }

                float dL_dLinearOutput_k_pixel[MAX_OUTPUT_LINEAR_DIM_BW];
                for(uint32_t i=0; i<output_linear_dim; ++i) dL_dLinearOutput_k_pixel[i] = 0.0f;
                for(uint32_t i=output_linear_dim; i<MAX_OUTPUT_LINEAR_DIM_BW; ++i) dL_dLinearOutput_k_pixel[i] = 0.0f;


                for (uint32_t k = 0; k < blend_dim; ++k) {
                    if (k < KERNEL_MAX_BLEND_DIM_USED && k < MAX_OUTPUT_LINEAR_DIM_BW) {
                        float y = Blend_k_pixel[k];
                        dL_dLinearOutput_k_pixel[k] = dL_dBlend_k_pixel[k] * y * (1.0f - y);
                    }
                }
                float dL_ddensity_pixel = dL_dalpha_k_pixel * expf(-primitive_density_k_pixel * delta_t_k) * delta_t_k;
                float dL_dx_density_pixel = dL_ddensity_pixel * primitive_density_k_pixel;
                if (density_idx_mlp < MAX_OUTPUT_LINEAR_DIM_BW) {
                    dL_dLinearOutput_k_pixel[density_idx_mlp] = dL_dx_density_pixel;
                }

                float dL_dLinearInput_k_pixel[MAX_INPUT_LINEAR_DIM_BW]; // Grad w.r.t mlp_input_aggregated_pixel
                for(uint32_t i=0; i<input_linear_dim; ++i) dL_dLinearInput_k_pixel[i] = 0.0f;
                for(uint32_t i=input_linear_dim; i<MAX_INPUT_LINEAR_DIM_BW; ++i) dL_dLinearInput_k_pixel[i] = 0.0f;

                for (uint32_t j = 0; j < input_linear_dim; ++j) {
                    if (j < MAX_INPUT_LINEAR_DIM_BW) {
                        float grad_sum = 0.0f;
                        for (uint32_t k_out = 0; k_out < output_linear_dim; ++k_out) {
                            if (k_out < MAX_OUTPUT_LINEAR_DIM_BW) {
                                uint32_t weight_idx = k_out * MAX_INPUT_LINEAR_DIM_BW + j; // Accessing shared_linear_weights
                                grad_sum += dL_dLinearOutput_k_pixel[k_out] * shared_linear_weights[weight_idx];
                            }
                        }
                        dL_dLinearInput_k_pixel[j] = grad_sum;
                    }
                }

                for (uint32_t k_out = 0; k_out < output_linear_dim; ++k_out) {
                    if (k_out >= MAX_OUTPUT_LINEAR_DIM_BW) continue;
                    float dL_dOutput_k = dL_dLinearOutput_k_pixel[k_out];
                    if (pix_id == DEBUG_PIX_ID && fabsf(dL_dOutput_k) > 1e-9f) { // Print if non-trivial
                        printf("[BKWD_DBG P:%u S2B BiasGrad] k_out=%u, PRE-ATOMIC dL_dLinearBias += %f\n", pix_id, k_out, dL_dOutput_k);
                    }
                    atomicAdd(&dL_dLinearBias[k_out], dL_dOutput_k); // dL_dLinearBias is global
                    for (uint32_t j_in = 0; j_in < input_linear_dim; ++j_in) {
                        if (j_in >= MAX_INPUT_LINEAR_DIM_BW) continue;
                        float input_j = mlp_input_aggregated_pixel[j_in];
                        float dL_dW_kj = dL_dOutput_k * input_j;
                        if (pix_id == DEBUG_PIX_ID && fabsf(dL_dW_kj) > 1e-9f) { // Print if non-trivial
                            printf("[BKWD_DBG P:%u S2B WeightGrad] k_out=%u, j_in=%u, PRE-ATOMIC dL_dLinearWeights += %f (dL_dOutput_k=%f, input_j=%f)\n",
                                   pix_id, k_out, j_in, dL_dW_kj, dL_dOutput_k, input_j);
                        }
                        uint64_t weight_idx_global = (uint64_t)k_out * input_linear_dim + j_in; // Global layout assumed [out_dim_runtime, in_dim_runtime]
                        if (weight_idx_global < (uint64_t)input_linear_dim * output_linear_dim) {
                             atomicAdd(&dL_dLinearWeights[weight_idx_global], dL_dW_kj); // dL_dLinearWeights is global
                        }
                    }
                }

                // (C) Backward Feature/Confidence Integration
                for (uint32_t flat_idx_gj = 0; flat_idx_gj < grid_volume; ++flat_idx_gj) {
                    if (flat_idx_gj >= MAX_GRID_VOLUME_BW) continue;

                    // Recalculate occupancy_grad_magnitude_pixel_gj for this pixel and grid_point_gj
                    // This is the same calculation as in section (A) for mlp_input_aggregated_pixel
                    float grad_x_pgj = 0.0f, grad_y_pgj = 0.0f, grad_z_pgj = 0.0f;
                    int z_c_gj = flat_idx_gj / (grid_size * grid_size);
                    int rem_c_gj = flat_idx_gj % (grid_size * grid_size);
                    int y_c_gj = rem_c_gj / grid_size;
                    int x_c_gj = rem_c_gj % grid_size;
                    if (stencil_size > 0) {
                        for (int s = 0; s < stencil_size; s++) {
                            int stencil_rel_off = s - stencil_offset;
                            char occ_x_n = 0, occ_y_n = 0, occ_z_n = 0;
                            int nx = x_c_gj + stencil_rel_off;
                            if (nx >= 0 && nx < grid_size) {
                                uint32_t flat_nx_idx = (uint32_t)z_c_gj * grid_size * grid_size + (uint32_t)y_c_gj * grid_size + (uint32_t)nx;
                                if (flat_nx_idx < MAX_GRID_VOLUME_BW) {
                                    float2 n_scr = sh_screen_coords[flat_nx_idx];
                                    if (n_scr.x >= pix.x && n_scr.x < (pix.x+1.0f) && n_scr.y >= pix.y && n_scr.y < (pix.y+1.0f)) occ_x_n=1;
                                }
                            } grad_x_pgj += stencil_coeffs_x[s] * (float)occ_x_n;
                            int ny = y_c_gj + stencil_rel_off;
                            if (ny >= 0 && ny < grid_size) {
                                uint32_t flat_ny_idx = (uint32_t)z_c_gj * grid_size * grid_size + (uint32_t)ny * grid_size + (uint32_t)x_c_gj;
                                if (flat_ny_idx < MAX_GRID_VOLUME_BW) {
                                    float2 n_scr = sh_screen_coords[flat_ny_idx];
                                    if (n_scr.x >= pix.x && n_scr.x < (pix.x+1.0f) && n_scr.y >= pix.y && n_scr.y < (pix.y+1.0f)) occ_y_n=1;
                                }
                            } grad_y_pgj += stencil_coeffs_y[s] * (float)occ_y_n;
                            int nz = z_c_gj + stencil_rel_off;
                            if (nz >= 0 && nz < grid_size) {
                                uint32_t flat_nz_idx = (uint32_t)nz * grid_size * grid_size + (uint32_t)y_c_gj * grid_size + (uint32_t)x_c_gj;
                                if (flat_nz_idx < MAX_GRID_VOLUME_BW) {
                                    float2 n_scr = sh_screen_coords[flat_nz_idx];
                                    if (n_scr.x >= pix.x && n_scr.x < (pix.x+1.0f) && n_scr.y >= pix.y && n_scr.y < (pix.y+1.0f)) occ_z_n=1;
                                }
                            } grad_z_pgj += stencil_coeffs_z[s] * (float)occ_z_n;
                        }
                    } else {
                        float2 c_scr = sh_screen_coords[flat_idx_gj];
                        if (c_scr.x >= pix.x && c_scr.x < (pix.x+1.0f) && c_scr.y >= pix.y && c_scr.y < (pix.y+1.0f)) {
                            grad_x_pgj=1.0f; grad_y_pgj=1.0f; grad_z_pgj=1.0f;
                        }
                    }
                    float occ_grad_mag_pixel_gj = fabsf(grad_x_pgj) + fabsf(grad_y_pgj) + fabsf(grad_z_pgj) + 1.0f;

                    bool check_occ_grad_mag_gj = occ_grad_mag_pixel_gj > 1e-7f;
                    if (pix_id == DEBUG_PIX_ID && flat_idx_gj == 0) { // Print for first grid point only
                        printf("[BKWD_DBG P:%u S2C ConfCheck] flat_idx_gj=%u, Checking occ_grad_mag_pixel_gj=%f > 1e-7f -> %d\n",
                               pix_id, flat_idx_gj, occ_grad_mag_pixel_gj, check_occ_grad_mag_gj);
                    }

                    if (check_occ_grad_mag_gj) {
                        float* phi_interp_gj_sh = &sh_interpolated_features_for_primitive[flat_idx_gj * MAX_INPUT_LINEAR_DIM_BW];
                        float dL_dX_gj = 0.0f;
                        for (uint32_t f_idx = 0; f_idx < input_linear_dim; ++f_idx) {
                            if (f_idx < MAX_INPUT_LINEAR_DIM_BW) { // Check dL_dLinearInput_k_pixel and phi_interp_gj_sh bounds
                                dL_dX_gj += dL_dLinearInput_k_pixel[f_idx] * phi_interp_gj_sh[f_idx];
                            }
                        }
                        dL_dX_gj *= occ_grad_mag_pixel_gj;
                        if (pix_id == DEBUG_PIX_ID && fabsf(dL_dX_gj) > 1e-9f) { // Print if non-trivial
                             printf("[BKWD_DBG P:%u S2C ConfGrad] flat_idx_gj=%u, PRE-ATOMIC dL_dprimitive_confidences += %f\n",
                                    pix_id, flat_idx_gj, dL_dX_gj);
                        }
                        atomicAdd(&dL_dprimitive_confidences[(uint64_t)current_primitive_idx_for_block * grid_volume + flat_idx_gj], dL_dX_gj);

                        float current_conf_gj_sh = sh_confidence[flat_idx_gj];
                        float weight_gj_pixel = occ_grad_mag_pixel_gj * current_conf_gj_sh;

                        bool check_weight_gj = fabsf(weight_gj_pixel) > 1e-9f;
                        if (pix_id == DEBUG_PIX_ID && flat_idx_gj == 0) { // Print for first grid point only
                            printf("[BKWD_DBG P:%u S2C FeatCheck] flat_idx_gj=%u, Checking fabsf(weight_gj_pixel=%f) > 1e-9f -> %d\n",
                                   pix_id, flat_idx_gj, weight_gj_pixel, check_weight_gj);
                        }

                        if (check_weight_gj) {
                            float dL_dPhi_level_gj[MAX_INPUT_FEATURE_DIM_BW]; // For one level
                            
                            int z_coord_gj = flat_idx_gj / (grid_size * grid_size);
                            int rem_gj = flat_idx_gj % (grid_size * grid_size);
                            int y_coord_gj = rem_gj / grid_size;
                            int x_coord_gj = rem_gj % grid_size;
                            float sample_rel_x_gj = (x_coord_gj - grid_offset_factor) * grid_step;
                            float sample_rel_y_gj = (y_coord_gj - grid_offset_factor) * grid_step;
                            float sample_rel_z_gj = (z_coord_gj - grid_offset_factor) * grid_step;
                            float3 pos_world_gj = {
                                center_current_prim.x + sample_rel_x_gj,
                                center_current_prim.y + sample_rel_y_gj,
                                center_current_prim.z + sample_rel_z_gj
                            };

                            for (int level = 0; level < hashgrid_levels; ++level) {
                                bool level_has_features = false;
                                for(uint32_t f_level = 0; f_level < input_feature_dim; ++f_level) {
                                    if (f_level < MAX_INPUT_FEATURE_DIM_BW) {
                                        uint32_t linear_idx = level * input_feature_dim + f_level;
                                        if (linear_idx < input_linear_dim && linear_idx < MAX_INPUT_LINEAR_DIM_BW) { // Check dL_dLinearInput_k_pixel bounds
                                            dL_dPhi_level_gj[f_level] = dL_dLinearInput_k_pixel[linear_idx] * weight_gj_pixel;
                                            level_has_features = true;
                                        } else {
                                            dL_dPhi_level_gj[f_level] = 0.0f;
                                        }
                                    } else { dL_dPhi_level_gj[f_level] = 0.0f; } // Should not happen if MAX_INPUT_FEATURE_DIM_BW >= input_feature_dim
                                }
                                if (pix_id == DEBUG_PIX_ID && flat_idx_gj == 0) { // Print for first grid point only
                                    printf("[BKWD_DBG P:%u S2C HashGradCheck] flat_idx_gj=%u, level=%d, Checking level_has_features -> %d\n",
                                           pix_id, flat_idx_gj, level, level_has_features);
                                }
                                if (level_has_features) {
                                    if (pix_id == DEBUG_PIX_ID && flat_idx_gj == 0 && input_feature_dim > 0 && fabsf(dL_dPhi_level_gj[0]) > 1e-9f) {
                                        printf("[BKWD_DBG P:%u S2C HashGradCall] flat_idx_gj=%u, level=%d, Calling hashComputeGradient with dL_dPhi_level_gj[0]=%f\n",
                                               pix_id, flat_idx_gj, level, dL_dPhi_level_gj[0]);
                                    }
                                    hashComputeGradient(
                                        pos_world_gj, level, resolution, (do_hash[level] != 0), primes,
                                        dL_dPhi_level_gj, input_feature_dim, hashgrid_levels,
                                        feature_offset, feature_table_size, dL_dFeatureTable
                                    );
                                }
                            }
                        }
                    }
                } // End loop over grid points for feature/confidence backward

                // (D) Update per-pixel backward state
                for(int d=0; d<blend_dim; ++d) {
                    if (d < KERNEL_MAX_BLEND_DIM_USED) {
                        dL_dOut_pix_local[d] *= one_minus_alpha_pixel;
                        accum_blend_after[d] = Blend_k_pixel[d] * alpha_k_pixel + accum_blend_after[d] * one_minus_alpha_pixel;
                    }
                }
                T_current_pixel = T_before_pixel;
                current_pixel_next_contrib_idx--;
                if (pix_id == DEBUG_PIX_ID) {
                    printf("[BKWD_DBG P:%u S2D Update] Next dL_dOut_pix_local[0]=%f, Next T_current_pixel=%f, next_contrib_idx=%d\n",
                           pix_id, (blend_dim > 0 ? dL_dOut_pix_local[0] : 0.0f), T_current_pixel, current_pixel_next_contrib_idx);
                }
            } // end if (primitive_matches_pixel)
        } // end if (should_process_pixel)
        block.sync(); // Ensure all threads complete work for current_primitive_idx_for_block
    } // --- End Loop Over Primitives for Tile (k_rev_tile) ---

    if (pix_id == DEBUG_PIX_ID) { // Final state for the debug pixel
        printf("[BKWD_DBG P:%u KERNEL END] Final T_current_pixel=%f, Final dL_dOut_pix_local[0]=%f\n",
               pix_id, T_current_pixel, (blend_dim > 0 ? dL_dOut_pix_local[0] : 0.0f));
    }
}


// =========================================================================
// Wrapper Function (Namespace BACKWARD)
// =========================================================================
namespace BACKWARD {

// Wrapper for the unified backward kernel
void compute_gradients(
    // <<< Kernel Launch Config >>>
    const dim3 grid, dim3 block,
    const uint2* ranges,
    // --- Input Gradients (Pixel-based) ---
    const float* dL_dOut_Color,
    const float* dL_dOut_Features,
    // --- Forward Pass State/Outputs ---
    const float* final_T,
    const int* num_contrib_per_pixel,
    const int* pixel_contributing_indices,
    const float* store_delta_t,
    // --- Data from Sorting ---
    const uint32_t* point_list,
    int P,
    int num_rendered,
    // --- Image Dimensions ---
    int W, int H,
    // --- Camera Parameters ---
    const float* viewmatrix,
    const float* projmatrix,
    const float* camera_center_vec,
    float near_plane,
    float max_distance,
    // --- Primitive Data ---
    const float* primitive_centers,
    const float* primitive_confidences,
    float primitive_scale,
    // --- Hashgrid Data ---
    const float* feature_table,
    const int* resolution,
    const int* do_hash,
    const int* primes,
    int feature_offset,
    uint32_t feature_table_size,
    // --- MLP Data ---
    const float* linear_weights,
    const float* linear_bias,
    // --- Integration Parameters ---
    int stencil_genus,
    int grid_size,
    int max_primitives_per_ray,
    float occupancy_threshold,
    // --- Background ---
    const float* bg_color,
    // --- Runtime Dimensions ---
    uint32_t input_feature_dim,
    uint32_t output_feature_dim,
    uint32_t hashgrid_levels,
    uint32_t num_output_channels,
    // --- FINAL Output Gradients ---
    float* dL_dLinearWeights,
    float* dL_dLinearBias,
    float* dL_dFeatureTable,
    float* dL_dprimitive_confidences
)
{
    // Ensure output gradient buffers are zero-initialized before calling this function!

    // Launch the appropriate template specialization
    if (num_output_channels == 3) {
         unifiedBackwardKernel<3><<<grid, block>>>(
             ranges,
             dL_dOut_Color, dL_dOut_Features,
             final_T, num_contrib_per_pixel, pixel_contributing_indices, store_delta_t,
             point_list, P, num_rendered,
             W, H,
             viewmatrix, projmatrix, camera_center_vec, near_plane, max_distance,
             primitive_centers, primitive_confidences, primitive_scale,
             feature_table, resolution, do_hash, primes, feature_offset, (uint64_t)feature_table_size, // Cast back to uint64_t for kernel
             linear_weights, linear_bias,
             stencil_genus, grid_size, max_primitives_per_ray, occupancy_threshold,
             bg_color,
             input_feature_dim, output_feature_dim, hashgrid_levels,
             dL_dLinearWeights, dL_dLinearBias, dL_dFeatureTable, dL_dprimitive_confidences
         );
         // cudaError_t err = cudaGetLastError();
         // if (err != cudaSuccess) printf("CUDA Error in BACKWARD::compute_gradients launch: %s\n", cudaGetErrorString(err));
    } else {
         printf("CUDA Error in BACKWARD::compute_gradients: Unsupported channel count %d. Only 3 is implemented.\n", num_output_channels);
         return;
    }
}

} // namespace BACKWARD



