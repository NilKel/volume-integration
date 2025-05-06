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
const uint32_t MAX_GRID_SIZE_BW = 3;
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
    const uint32_t KERNEL_MAX_OUTPUT_LINEAR_DIM = MAX_BLEND_DIM_BW<CHANNELS>;
    const uint32_t KERNEL_MAX_BLEND_DIM = MAX_BLEND_DIM_BW<CHANNELS>;

    // Shared memory
    __shared__ float stencil_coeffs_x[MAX_STENCIL_SIZE_BW];
    __shared__ float stencil_coeffs_y[MAX_STENCIL_SIZE_BW];
    __shared__ float stencil_coeffs_z[MAX_STENCIL_SIZE_BW];
    __shared__ int shared_stencil_size;
    __shared__ float shared_linear_weights[MAX_INPUT_LINEAR_DIM_BW * KERNEL_MAX_OUTPUT_LINEAR_DIM];
    __shared__ float shared_linear_bias[KERNEL_MAX_OUTPUT_LINEAR_DIM];
    __attribute__((shared)) float sh_confidence[MAX_GRID_VOLUME_BW];
    __attribute__((shared)) float sh_pixel_occupancy[MAX_GRID_VOLUME_BW]; // Occupancy for the *current* pixel
    __attribute__((shared)) float sh_occupancy_grad_mag[MAX_GRID_VOLUME_BW]; // Store grad magnitude per grid point

    // Thread/Block Identification
    auto block = cg::this_thread_block();
    uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x; // Global pixel index (row-major)
    bool inside = pix.x < W && pix.y < H;

    // --- Load Shared Data (Stencil, MLP Weights/Bias) ---
    // (Identical loading logic as in forward renderCUDA)
    
    if (block.thread_rank() == 0) {
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
    // block.sync();
    // return;
    // Load weights (output-major layout assumed for global linear_weights)
    // W_global[in * out_dim + out] -> W_shared[out * MAX_IN + in]
    // Let's assume global weights are stored flat matching the shared layout expectation:
    // W_global[out * in_dim + in] -> W_shared[out * MAX_IN + in]
    int total_weights = input_linear_dim * output_linear_dim;
    for (int i = block.thread_rank(); i < total_weights; i += block.size()) {
        if (i < MAX_INPUT_LINEAR_DIM_BW * KERNEL_MAX_OUTPUT_LINEAR_DIM) {
             shared_linear_weights[i] = linear_weights[i];
        }
    }
    int total_biases = output_linear_dim;
     for (int i = block.thread_rank(); i < total_biases; i += block.size()) {
         if (i < KERNEL_MAX_OUTPUT_LINEAR_DIM) {
            shared_linear_bias[i] = linear_bias[i];
         }
    }
    block.sync();
    const int stencil_size = shared_stencil_size;
    const int stencil_offset = (stencil_size > 0) ? (stencil_size - 1) / 2 : 0;

    // --- Per-Pixel Backward Initialization ---
    float dL_dOut_pix_local[KERNEL_MAX_BLEND_DIM]; // Gradient w.r.t. pixel color/features
    float accum_blend_after[KERNEL_MAX_BLEND_DIM]; // Color/features accumulated *after* current primitive
    float T_current = 0.0f; // Transmittance *after* current primitive
    int num_contrib = 0;    // Number of contributions for this pixel

    // <<< Your existing test accesses >>>
    num_contrib_per_pixel[pix_id];
    min(num_contrib, max_primitives_per_ray); // Note: This uses uninitialized 'num_contrib' here
    dL_dOut_pix_local[0];                     // Accesses local stack array (safe)
    dL_dOut_pix_local[KERNEL_MAX_BLEND_DIM - 1];              // Accesses local stack array (safe if CHANNELS < KERNEL_MAX_BLEND_DIM)
    dL_dOut_Color[0];                         // Accesses global memory (potentially unsafe if pointer invalid)
    dL_dOut_Color[(CHANNELS-1)*H*W+pix_id];   // Accesses global memory (unsafe if pointer invalid OR pix_id OOB)

    // <<< ADDED: Test access patterns for remaining reads from 'if (inside)' >>>

    // Test dL_dOut_Features access pattern (only if pointer is valid and features exist)
    // Reads the first and last potential feature elements for this pix_id
    if (dL_dOut_Features != nullptr && output_feature_dim > 0) {
        // Access the memory location, result is discarded
        (void)dL_dOut_Features[0 * H * W + pix_id];
        (void)dL_dOut_Features[(output_feature_dim - 1) * H * W + pix_id];
    }

    accum_blend_after[0];
    accum_blend_after[CHANNELS+output_feature_dim-1];
    // Test final_T access pattern
    // Access the memory location, result is discarded
    (void)final_T[pix_id];
    accum_blend_after[0] = (bg_color != nullptr) ? bg_color[0] : 0.0f;
    accum_blend_after[CHANNELS-1] = (bg_color != nullptr) ? bg_color[CHANNELS-1] : 0.0f;
    // <<< End Added Tests >>>

    // block.sync(); // Synchronize before returning
    // return;       // Exit kernel here for testing

    // The original if/else block is now effectively skipped by the return above
    if (inside) {
        num_contrib = num_contrib_per_pixel[pix_id];
        // Clamp num_contrib to max_primitives_per_ray to prevent out-of-bounds access
        // in pixel_contributing_indices if the forward pass stored more than allowed here.
        num_contrib = min(num_contrib, max_primitives_per_ray);

        // Load initial gradient from dL_dOut_Color and dL_dOut_Features
        for(int c=0; c<CHANNELS; ++c) {
            dL_dOut_pix_local[c] = dL_dOut_Color[c * H * W + pix_id];
        }
        if (dL_dOut_Features != nullptr) {
        for(uint32_t f=0; f<output_feature_dim; ++f) {
            dL_dOut_pix_local[CHANNELS + f] = dL_dOut_Features[f * H * W + pix_id];
        }
        } else {
             for(uint32_t f=0; f<output_feature_dim; ++f) {
                dL_dOut_pix_local[CHANNELS + f] = 0.0f; // Assume zero gradient if not provided
            }
        }
        // Initialize any remaining elements in the max-sized array to 0
        // block.sync();
        // return;
        // for(uint32_t d = blend_dim; d < KERNEL_MAX_BLEND_DIM; ++d) {
        //      dL_dOut_pix_local[d] = 0.0f;
        // }

        // Initialize accum_blend_after with background color/features (features assumed 0)
        for(int c=0; c<CHANNELS; ++c) {
            accum_blend_after[c] = (bg_color != nullptr) ? bg_color[c] : 0.0f;
        }
        for(uint32_t f=0; f<output_feature_dim; ++f) {
            accum_blend_after[CHANNELS + f] = 0.0f; // Background features are zero
        }
        // for(uint32_t d = blend_dim; d < KERNEL_MAX_BLEND_DIM; ++d) {
        //      accum_blend_after[d] = 0.0f;
        // }

        // Load final transmittance T after the last contribution
        T_current = final_T[pix_id];
        // block.sync();
        // return;
    } else {
        // block.sync();
        // return;
        // Initialize to zero if outside bounds
        for(uint32_t d=0; d<KERNEL_MAX_BLEND_DIM; ++d) {
            dL_dOut_pix_local[d] = 0.0f;
            accum_blend_after[d] = 0.0f;
        }
        T_current = 0.0f; // Or 1.0? Check forward pass logic. Assume 0.0 consistent with final_T.
        num_contrib = 0;
    }
    if (num_contrib <= 0) {
        return;
    }
    block.sync();
    return;
    // --- Iterate Backwards Through Primitives for this Pixel ---
    for (int c = num_contrib - 1; c > 0; c--) {

        // Get the global contribution index 'i' for this pixel's c-th contribution
        // This index 'i' corresponds to the entry in point_list and store_delta_t
        int global_contrib_idx = pixel_contributing_indices[pix_id * max_primitives_per_ray + c];

        // Basic check: Ensure index is valid
        if (global_contrib_idx < 0 || global_contrib_idx >= num_rendered) {
            // Should not happen if forward pass and inputs are correct
            // printf("Warning: Invalid global_contrib_idx %d for pixel %d, contrib %d\n", global_contrib_idx, pix_id, c);
            continue;
        }

        // Get the original primitive index from the sorted list
        uint32_t primitive_idx = point_list[global_contrib_idx];

        // Basic check: Ensure primitive index is valid
        if (primitive_idx >= P) {
             // printf("Warning: Invalid primitive_idx %u from point_list[%d] for pixel %d, contrib %d\n", primitive_idx, global_contrib_idx, pix_id, c);
             return;
        }
        block.sync();
        return;
        // Get primitive center
            float3 center = { primitive_centers[primitive_idx * 3 + 0], primitive_centers[primitive_idx * 3 + 1], primitive_centers[primitive_idx * 3 + 2] };

        // ====================================================
        // == Recompute Forward Pass for this Contribution ==
        // ====================================================
        
        // --- Initialize Primitive Feature Integral ---
            float primitive_feature_integral[MAX_INPUT_LINEAR_DIM_BW]; // Use BW max size
            for(uint32_t k=0; k<input_linear_dim; ++k) primitive_feature_integral[k] = 0.0f;

        // --- Load Confidence Grid into Shared Memory ---
            const uint64_t base_conf_idx_global = (uint64_t)primitive_idx * grid_volume;
            block.sync();
            return;      
            if(base_conf_idx_global < (uint64_t)P * grid_volume) {
            for (uint32_t j = 0; j < grid_volume; j ++) {
                // Check 1: Is the index 'j' within the allocated shared memory bounds?
                if (j < MAX_GRID_VOLUME_BW) {
                    // Check 2: Is the calculated global index within the global primitive_confidences bounds?
                    return;
                    // if (base_conf_idx_global + j < (uint64_t)P * grid_volume) {
                    //      // Write to shared memory from global memory
                    //      sh_confidence[j] = primitive_confidences[base_conf_idx_global + j];
                    // } else {
                    //      // Handle potential out-of-bounds read from global memory (e.g., padding primitives)
                    //      sh_confidence[j] = 0.0f;
                    // }
                }
                // <<< Potential Issue Area >>>
            }
            }
            block.sync();
            return;
        // --- Pass 1: Calculate Per-Pixel Occupancy (sh_pixel_occupancy) ---
        // This needs to be recalculated for *this specific pixel*
            const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
            const float grid_offset_factor = (grid_size - 1) / 2.0f;
        bool primitive_contributes_to_pixel = false; // Check if any grid point contributes

            for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
                if (flat_idx >= MAX_GRID_VOLUME_BW) continue;
                int z = flat_idx / (grid_size * grid_size); int rem = flat_idx % (grid_size * grid_size); int y = rem / grid_size; int x = rem % grid_size;
                char occupancy = 0;
            float confidence = sh_confidence[flat_idx]; // Read from shared

                if (confidence > occupancy_threshold) {
                    float sample_rel_x = (x - grid_offset_factor) * grid_step;
                    float sample_rel_y = (y - grid_offset_factor) * grid_step;
                    float sample_rel_z = (z - grid_offset_factor) * grid_step;
                    float3 sample_pos_world = { center.x + sample_rel_x, center.y + sample_rel_y, center.z + sample_rel_z };

                // Check if this grid point contributes to THIS pixel
                    if (isPointInPyramidalFrustum(sample_pos_world, viewmatrix, projmatrix, W, H, pix, near_plane, max_distance)) {
                        occupancy = 1;
                    primitive_contributes_to_pixel = true; // Set flag if any point contributes
                }
            }
            sh_pixel_occupancy[flat_idx] = occupancy; // Write occupancy for this pixel
        }
        block.sync(); // Ensure sh_pixel_occupancy is fully written for this pixel
        // return;
        // Optional: Reduce primitive_contributes_to_pixel across the block if needed,
        // but the gradient calculation below handles zero contribution implicitly.

        // --- Calculate and Store Occupancy Gradient Magnitude ---
        // Each thread calculates the gradient magnitude for its assigned grid points
        // using the completed sh_pixel_occupancy grid.
            for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
                if (flat_idx >= MAX_GRID_VOLUME_BW) continue;
                int z = flat_idx / (grid_size * grid_size); int rem = flat_idx % (grid_size * grid_size); int y = rem / grid_size; int x = rem % grid_size;

                float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
            if (stencil_size > 0) { // Check if stencil is valid
                for (int s = 0; s < stencil_size; s++) {
                    int sx = s - stencil_offset; int nx = x + sx;
                    int sy = s - stencil_offset; int ny = y + sy;
                    int sz = s - stencil_offset; int nz = z + sz;
                    uint32_t flat_nx_idx = z * grid_size * grid_size + y * grid_size + nx;
                    uint32_t flat_ny_idx = z * grid_size * grid_size + ny * grid_size + x;
                    uint32_t flat_nz_idx = nz * grid_size * grid_size + y * grid_size + x;
                    char occ_x_neighbor = (nx >= 0 && nx < grid_size && flat_nx_idx < MAX_GRID_VOLUME_BW) ? sh_pixel_occupancy[flat_nx_idx] : 0;
                    char occ_y_neighbor = (ny >= 0 && ny < grid_size && flat_ny_idx < MAX_GRID_VOLUME_BW) ? sh_pixel_occupancy[flat_ny_idx] : 0;
                    char occ_z_neighbor = (nz >= 0 && nz < grid_size && flat_nz_idx < MAX_GRID_VOLUME_BW) ? sh_pixel_occupancy[flat_nz_idx] : 0;
                    grad_x += stencil_coeffs_x[s] * (float)occ_x_neighbor;
                    grad_y += stencil_coeffs_y[s] * (float)occ_y_neighbor;
                    grad_z += stencil_coeffs_z[s] * (float)occ_z_neighbor;
                }
            }
            // Store the calculated magnitude in shared memory
            sh_occupancy_grad_mag[flat_idx] = fabsf(grad_x) + fabsf(grad_y) + fabsf(grad_z);
        }
        block.sync(); // Ensure all threads have written sh_occupancy_grad_mag
        // return;
        // --- Pass 2: Calculate Gradient & Accumulate Features ---
        // Uses sh_pixel_occupancy specific to the current pixel
        for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
            if (flat_idx >= MAX_GRID_VOLUME_BW) continue;
            int z = flat_idx / (grid_size * grid_size); int rem = flat_idx % (grid_size * grid_size); int y = rem / grid_size; int x = rem % grid_size;

            // <<< Read pre-calculated occupancy gradient magnitude >>>
            float occupancy_grad_magnitude = sh_occupancy_grad_mag[flat_idx];

                if (occupancy_grad_magnitude > 1e-6f) {
                    // Fetch Features from Hashgrid (Φ)
                float sample_features[MAX_INPUT_LINEAR_DIM_BW]; // Combined features F*L
                    for(uint32_t f=0; f<input_linear_dim; ++f) sample_features[f] = 0.0f;

                    float sample_rel_x = (x - grid_offset_factor) * grid_step;
                    float sample_rel_y = (y - grid_offset_factor) * grid_step;
                    float sample_rel_z = (z - grid_offset_factor) * grid_step;
                    float3 sample_pos_world = { center.x + sample_rel_x, center.y + sample_rel_y, center.z + sample_rel_z };

                    for (int level = 0; level < hashgrid_levels; ++level) {
                        float level_features[MAX_INPUT_FEATURE_DIM_BW];
                        hashInterpolate(sample_pos_world, level, feature_table, resolution, feature_offset,
                                    do_hash[level] != 0, primes, level_features, input_feature_dim, hashgrid_levels); 
                        // Concatenate
                        for (uint32_t f = 0; f < input_feature_dim; f++) {
                            uint32_t target_idx = level * input_feature_dim + f;
                        if (target_idx < input_linear_dim && target_idx < MAX_INPUT_LINEAR_DIM_BW) {
                                sample_features[target_idx] = level_features[f];
                            }
                        }
                    }
                    // Accumulate Weighted Features
                float current_confidence = sh_confidence[flat_idx]; // Read from shared
                    float weight = occupancy_grad_magnitude * current_confidence;
                    for (uint32_t f = 0; f < input_linear_dim; ++f) {
                    // Accumulation is thread-local
                        primitive_feature_integral[f] += sample_features[f] * weight;
                    }
            } // End if grad_magnitude > threshold
        } // End Pass 2 loop (flat_idx)
        // primitive_feature_integral is now computed for this primitive's contribution to this pixel

        // --- Apply Linear Layer (MLP) -> primitive_outputs_pre_act_k ---
        float primitive_outputs_pre_act_k[KERNEL_MAX_OUTPUT_LINEAR_DIM]; // Use BW max size
            for(uint32_t k=0; k<output_linear_dim; ++k) primitive_outputs_pre_act_k[k] = 0.0f;

            for (uint32_t out_idx = 0; out_idx < output_linear_dim; out_idx++) {
                float dot_prod = 0.0f;
                // Ensure indices stay within bounds of shared memory arrays
            uint32_t max_weight_idx = MAX_INPUT_LINEAR_DIM_BW * KERNEL_MAX_OUTPUT_LINEAR_DIM;
                uint32_t weight_row_base = out_idx * MAX_INPUT_LINEAR_DIM_BW; // Use compile-time max for indexing shared mem

            if (weight_row_base < max_weight_idx && out_idx < KERNEL_MAX_OUTPUT_LINEAR_DIM) { // Check row start and output index
                    for (uint32_t in_idx = 0; in_idx < input_linear_dim; in_idx++) {
                        uint32_t weight_idx = weight_row_base + in_idx;
                    if (weight_idx < max_weight_idx && in_idx < MAX_INPUT_LINEAR_DIM_BW) { // Check full weight index and input index
                            dot_prod += shared_linear_weights[weight_idx] * primitive_feature_integral[in_idx];
                        }
                    }
                    // Add bias, check bias index bounds
                if (out_idx < KERNEL_MAX_OUTPUT_LINEAR_DIM) {
                         primitive_outputs_pre_act_k[out_idx] = dot_prod + shared_linear_bias[out_idx];
                    } else {
                         primitive_outputs_pre_act_k[out_idx] = dot_prod; // Should not happen if output_linear_dim <= MAX
                    }
                }
            }

        // --- Apply Activations -> Blend_k (Color+Feat), primitive_density_k ---
        float Blend_k[KERNEL_MAX_BLEND_DIM]; // Activated Color + Features
            float primitive_density_k = 0.0f;
            // Sigmoid for Color (first CHANNELS) and Features (next output_feature_dim)
            for (uint32_t k = 0; k < blend_dim; ++k) {
             if (k < output_linear_dim && k < KERNEL_MAX_OUTPUT_LINEAR_DIM) { // Check bounds
                Blend_k[k] = 1.0f / (1.0f + expf(-primitive_outputs_pre_act_k[k]));
                 } else {
                    Blend_k[k] = 0.0f;
                 }
            }
            // Initialize rest of max-sized array (optional)
        for (uint32_t k = blend_dim; k < KERNEL_MAX_BLEND_DIM; ++k) {
                Blend_k[k] = 0.0f;
            }

            // Exp for density (last element of MLP output)
            uint32_t density_idx_mlp = output_linear_dim - 1; // Index of density in pre-activation output
        if (density_idx_mlp < KERNEL_MAX_OUTPUT_LINEAR_DIM) { // Check bounds against allocated size
                 primitive_density_k = expf(primitive_outputs_pre_act_k[density_idx_mlp]);
                 primitive_density_k = max(0.0f, primitive_density_k); // Ensure non-negative
            }

        // ====================================================
        // == Backward Pass Calculations for this Contribution ==
        // ====================================================

        // --- Fetch Delta T for this contribution ---
        // Read delta_t using the pixel-based index matching the forward pass storage
            float delta_t_k = 0.0f;
        int delta_t_read_idx = pix_id * max_primitives_per_ray + c;
        // Bounds check: Ensure the index is within the allocated buffer size (W*H*max_primitives_per_ray)
        // We already clamped num_contrib, so 'c' should be valid relative to max_primitives_per_ray for this pixel.
        // A full buffer bounds check could be added if allocation size is uncertain:
        // if (store_delta_t != nullptr && delta_t_read_idx < W * H * max_primitives_per_ray) {
        if (store_delta_t != nullptr) { // Assuming buffer is correctly sized
            delta_t_k = store_delta_t[delta_t_read_idx];
            } else {
            // Handle error: delta_t storage is required
            // printf("Error: store_delta_t is null!\n");
        }

        // --- Calculate Alpha ---
        // alpha_k = 1.0f - expf(-primitive_density_k * delta_t_k)
            float alpha_k = 1.0f - expf(-primitive_density_k * delta_t_k);
            alpha_k = min(max(alpha_k, 0.0f), 1.0f); // Clamp alpha

            // --- Backward Blending Calculation ---
            // T_current holds T *after* this primitive. T_before = T_after / (1 - alpha_k)
            float T_before = 0.0f;
            float one_minus_alpha = 1.0f - alpha_k;
        if (fabsf(one_minus_alpha) > 1e-6f) {
                 T_before = T_current / one_minus_alpha;
             T_before = min(max(T_before, 0.0f), 1.0f); // Clamp T_before
            } else {
             // If alpha_k is ~1, T_current should be ~0. Set T_before = 0.
             T_before = 0.0f;
        }

            // Compute dL/dBlend_k = dL/dOut_pix_local * T_before * alpha_k
        float dL_dBlend_k[KERNEL_MAX_BLEND_DIM]; // Gradient w.r.t activated Blend (Color+Feat)
            for(int d=0; d<blend_dim; ++d) {
            if (d < KERNEL_MAX_BLEND_DIM) {
                    dL_dBlend_k[d] = dL_dOut_pix_local[d] * T_before * alpha_k;
                } else {
                 dL_dBlend_k[d] = 0.0f;
                }
            }

            // Compute dL/dalpha_k = dot(dL/dOut_pix_local, T_before * (Blend_k - accum_blend_after))
            float dL_dalpha_k = 0.0f;
            for(int d=0; d<blend_dim; ++d) {
             if (d < KERNEL_MAX_BLEND_DIM) {
                     dL_dalpha_k += dL_dOut_pix_local[d] * T_before * (Blend_k[d] - accum_blend_after[d]);
                 }
            }

        // --- Backward Activation ---
        float dL_dLinearOutput_k[KERNEL_MAX_OUTPUT_LINEAR_DIM]; // Grad w.r.t. MLP pre-activation output

        // Backward Sigmoid for Blend (Color + Features)
        for (uint32_t k = 0; k < blend_dim; ++k) {
            // dL/dx = dL/dy * y * (1-y), where y = Blend_k[k] = sigmoid(x)
            if (k < KERNEL_MAX_BLEND_DIM && k < KERNEL_MAX_OUTPUT_LINEAR_DIM) {
                float y = Blend_k[k];
                dL_dLinearOutput_k[k] = dL_dBlend_k[k] * y * (1.0f - y);
            } else if (k < KERNEL_MAX_OUTPUT_LINEAR_DIM) {
                 dL_dLinearOutput_k[k] = 0.0f;
            }
        }

        // Backward Exp/Alpha for Density
        // dL/dx_density = dL/dalpha * dalpha/ddensity * ddensity/dx_density
    // dalpha/ddensity = exp(-density * delta_t) * delta_t
    // ddensity/dx_density = exp(x_density) = density
        float dL_ddensity = dL_dalpha_k * expf(-primitive_density_k * delta_t_k) * delta_t_k;
        float dL_dx_density = dL_ddensity * primitive_density_k; // Use activated density

        if (density_idx_mlp < KERNEL_MAX_OUTPUT_LINEAR_DIM) {
            dL_dLinearOutput_k[density_idx_mlp] = dL_dx_density;
        }
        // Zero-init remaining (if any)
        for(uint32_t k=output_linear_dim; k<KERNEL_MAX_OUTPUT_LINEAR_DIM; ++k) {
            dL_dLinearOutput_k[k] = 0.0f;
        }


        // --- Backward Linear Layer ---
        // Calculate dL/dLinearInput_k = dL/dLinearOutput_k @ Weights^T
        float dL_dLinearInput_k[MAX_INPUT_LINEAR_DIM_BW]; // Grad w.r.t feature integral for this contribution
    for (uint32_t j = 0; j < input_linear_dim; ++j) { // Input feature index
        float grad_sum = 0.0f;
            if (j < MAX_INPUT_LINEAR_DIM_BW) {
        for (uint32_t k = 0; k < output_linear_dim; ++k) { // Output feature index
                    // Weight index in shared memory: shared_linear_weights[k * MAX_IN + j]
                    uint32_t weight_idx = k * MAX_INPUT_LINEAR_DIM_BW + j;
                    if (k < KERNEL_MAX_OUTPUT_LINEAR_DIM && weight_idx < MAX_INPUT_LINEAR_DIM_BW * KERNEL_MAX_OUTPUT_LINEAR_DIM) {
                        grad_sum += dL_dLinearOutput_k[k] * shared_linear_weights[weight_idx];
                    }
                }
                dL_dLinearInput_k[j] = grad_sum;
            }
        }
        // Zero-init remaining (if any)
        for(uint32_t j=input_linear_dim; j<MAX_INPUT_LINEAR_DIM_BW; ++j) {
             dL_dLinearInput_k[j] = 0.0f;
        }


        // Accumulate dL/dWeights and dL/dBias using atomicAdd
    // dL/dWeight_kj = dL/dOutput_k * Input_j
    // dL/dBias_k = dL/dOutput_k
    for (uint32_t k = 0; k < output_linear_dim; ++k) { // Output feature index
            if (k >= KERNEL_MAX_OUTPUT_LINEAR_DIM) continue;
            float dL_dOutput_k = dL_dLinearOutput_k[k];

        // Accumulate bias gradient
        atomicAdd(&dL_dLinearBias[k], dL_dOutput_k);

        // Accumulate weight gradient
        for (uint32_t j = 0; j < input_linear_dim; ++j) { // Input feature index
                if (j >= MAX_INPUT_LINEAR_DIM_BW) continue;
                float input_j = primitive_feature_integral[j];
            float dL_dW_kj = dL_dOutput_k * input_j;
                // Weight index in global memory (matches shared layout assumption)
                uint64_t weight_idx = (uint64_t)k * input_linear_dim + j; // Or k * MAX_INPUT_LINEAR_DIM_BW + j if using max size indexing? Assume runtime size.
                // TODO: Verify global weight layout and indexing. Assuming [out * in_runtime + in]
                // Add bounds check if necessary
            uint64_t total_dL_weights_size = (uint64_t)input_linear_dim * output_linear_dim; // Calculate if not passed
            if (weight_idx < total_dL_weights_size) {
                atomicAdd(&dL_dLinearWeights[weight_idx], dL_dW_kj);
            } else {
                // Optional: Add printf for debugging OOB attempts
                // printf("OOB atomicAdd dL_dLinearWeights: idx %llu >= size %llu\n", weight_idx, total_dL_weights_size);
            }
            }
        }

        // --- Backward Feature/Confidence Integration ---
        // Propagate dL/dLinearInput_k back to dL/dX and dL/dFeatureTable
        // This involves iterating through the grid points again
        for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
            if (flat_idx >= MAX_GRID_VOLUME_BW) continue; // Bounds check shared memory access

            // <<< Read pre-calculated occupancy gradient magnitude >>>
            float occupancy_grad_magnitude = sh_occupancy_grad_mag[flat_idx];

            if (occupancy_grad_magnitude > 1e-6f) {
                // <<< Calculate world position ONCE per grid point here >>>
                int z = flat_idx / (grid_size * grid_size); int rem = flat_idx % (grid_size * grid_size); int y = rem / grid_size; int x = rem % grid_size;
                float sample_rel_x = (x - grid_offset_factor) * grid_step;
                float sample_rel_y = (y - grid_offset_factor) * grid_step;
                float sample_rel_z = (z - grid_offset_factor) * grid_step;
                float3 pos_world_j = { center.x + sample_rel_x, center.y + sample_rel_y, center.z + sample_rel_z };

                // Recompute interpolated features (Φ_interp_j) for this grid point
                float Phi_interp_j[MAX_INPUT_LINEAR_DIM_BW] = {0.0f}; // Concatenated features F*L
                for (int level = 0; level < hashgrid_levels; ++level) {
                    float level_features[MAX_INPUT_FEATURE_DIM_BW];
                    hashInterpolate(pos_world_j, level, feature_table, resolution, feature_offset,
                                    do_hash[level] != 0, primes, level_features, input_feature_dim, hashgrid_levels); 

                    // Concatenate features for this level
                    for(uint32_t f=0; f < input_feature_dim; ++f) {
                        if (level * input_feature_dim + f < MAX_INPUT_LINEAR_DIM_BW) { // Bounds check
                            Phi_interp_j[level * input_feature_dim + f] = level_features[f];
                        }
                    }
                }

                // Calculate dL/dX_j
                // FeatureIntegral = sum_j ( const_occupancy_grad_magnitude_j * X_j * Φ_interp_j )
                // dL/dX_j = dot(dL/dFeatureIntegral, dFeatureIntegral/dX_j)
                float dL_dX_j = 0.0f; // Gradient w.r.t. confidence X_j
                for (uint32_t k = 0; k < input_linear_dim; ++k) {
                    if (k < MAX_INPUT_LINEAR_DIM_BW) {
                        dL_dX_j += dL_dLinearInput_k[k] * Phi_interp_j[k];
                    }
                }
                dL_dX_j *= occupancy_grad_magnitude;

                // Accumulate dL/dX for this primitive's confidence grid
                uint64_t dX_write_idx = base_conf_idx_global + flat_idx;
                if (dX_write_idx < (uint64_t)P * grid_volume) {
                    atomicAdd(&dL_dprimitive_confidences[dX_write_idx], dL_dX_j);
                }

                // Calculate dL/dPhi_interp_j
                // dL/dPhi_interp_j = dot(dL/dFeatureIntegral, dFeatureIntegral/dPhi_interp_j)
                float current_confidence_j = sh_confidence[flat_idx]; // Read confidence for this point
                float weight_j = occupancy_grad_magnitude * current_confidence_j; // Correct weight
                float dL_dPhi_interp_j[MAX_INPUT_LINEAR_DIM_BW];
                for (uint32_t k = 0; k < input_linear_dim; ++k) {
                    if (k < MAX_INPUT_LINEAR_DIM_BW) {
                        dL_dPhi_interp_j[k] = dL_dLinearInput_k[k] * weight_j;
                    } else {
                        dL_dPhi_interp_j[k] = 0.0f;
                    }
                }

                // Distribute dL/dPhi_interp_j to dL/dFeatureTable
                if (fabsf(weight_j) > 1e-9f) { // Avoid distributing if weight is zero
                    for (int level = 0; level < hashgrid_levels; ++level) {
                        // Get the portion of dL/dPhi_interp_j for this level
                        float dL_dPhi_level[MAX_INPUT_FEATURE_DIM_BW];
                        bool level_has_features = false; // Check if this level contributes
                        for(uint32_t f=0; f<input_feature_dim; ++f) {
                            uint32_t linear_idx = level * input_feature_dim + f;
                            if (linear_idx < MAX_INPUT_LINEAR_DIM_BW && linear_idx < input_linear_dim) {
                                dL_dPhi_level[f] = dL_dPhi_interp_j[linear_idx];
                                level_has_features = true; // Mark that this level is used
                            } else {
                                dL_dPhi_level[f] = 0.0f;
                            }
                        }

                        // Only distribute if this level actually contributes features
                        if (level_has_features) {
                            // Call the corrected gradient computation helper function
                            hashComputeGradient(
                                pos_world_j,          // World position of the grid sample point
                                level,                // Current hash level
                                resolution,           // Resolution table
                                (do_hash[level] != 0),// Use hashing?
                                primes,               // Hashing primes
                                dL_dPhi_level,        // Gradient w.r.t. this level's interpolated features
                                input_feature_dim,    // F
                                hashgrid_levels,      // L
                                feature_offset,       // T (max entries per level)
                                feature_table_size,   // Total size of dL_dFeatureTable buffer
                                dL_dFeatureTable      // Output buffer for atomic adds
                            );
                        }
                    }
                }
            }
        } // End loop over grid points (flat_idx) for feature/confidence backward


        // ====================================================
        // == Update Backward State for Next (Earlier) Primitive ==
        // ====================================================
        // Update dL/dOut_pix_local: dL/dOut_prev = dL/dOut_curr * (1 - alpha_k)
        // This propagates the gradient backward through the alpha compositing step.
        for(int d=0; d<blend_dim; ++d) {
             if (d < KERNEL_MAX_BLEND_DIM) {
                 dL_dOut_pix_local[d] = dL_dOut_pix_local[d] * one_minus_alpha;
             }
        }

        // Update accum_blend_after: accum_blend_after_prev = Blend_k * alpha_k + accum_blend_after_curr * (1 - alpha_k)
        // This updates the color/feature accumulated *after* the next primitive to be processed.
        for(int d=0; d<blend_dim; ++d) {
             if (d < KERNEL_MAX_BLEND_DIM) {
                 accum_blend_after[d] = Blend_k[d] * alpha_k + accum_blend_after[d] * one_minus_alpha;
             }
        }

        // Update T_current for the next iteration (transmittance *before* the current primitive)
        T_current = T_before;

    } // --- End Backwards Loop Over Contributions (c) ---
}


// =========================================================================
// Wrapper Function (Namespace BACKWARD)
// =========================================================================
namespace BACKWARD {

// Wrapper for the unified backward kernel
void compute_gradients(
    // <<< Kernel Launch Config >>>
    const dim3 grid, dim3 block,
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
             dL_dOut_Color, dL_dOut_Features,
             final_T, num_contrib_per_pixel, pixel_contributing_indices, store_delta_t,
             point_list, P, num_rendered,
             W, H,
             viewmatrix, projmatrix, camera_center_vec, near_plane, max_distance,
             primitive_centers, primitive_confidences, primitive_scale,
             feature_table, resolution, do_hash, primes, feature_offset, feature_table_size,
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



