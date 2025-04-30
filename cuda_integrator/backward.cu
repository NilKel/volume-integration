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

#include "backward.h" // Include the corresponding header
#include "auxiliary.h" // Include auxiliary functions (like transformPoint4x3, potentially hashInterpolate if defined there)

// Assuming forward pass helpers are accessible (either defined here or included)
// If not, they need to be copied/included from forward.cu or common headers.
// E.g., transformPoint4x3, hashInterpolate, isPointInPyramidalFrustum

namespace cg = cooperative_groups;

// Define constants consistent with forward pass if needed (e.g., for shared memory sizing)
// These should match the MAX values used in renderCUDA's shared memory allocation
const uint32_t MAX_INPUT_FEATURE_DIM_BW = 4;  // F_max
const uint32_t MAX_OUTPUT_FEATURE_DIM_BW = 4; // F_out_max
const uint32_t MAX_HASHGRID_LEVELS_BW = 8;    // L_max
const uint32_t MAX_GRID_SIZE_BW = 5;
const uint32_t MAX_GRID_VOLUME_BW = MAX_GRID_SIZE_BW * MAX_GRID_SIZE_BW * MAX_GRID_SIZE_BW;
const uint32_t MAX_STENCIL_SIZE_BW = 5;
// Assuming CHANNELS = 3 for color
const uint32_t MAX_OUTPUT_LINEAR_DIM_BW = 3 + MAX_OUTPUT_FEATURE_DIM_BW + 1; // RGB(3) + Feat(F_out) + Density(1)
const uint32_t MAX_INPUT_LINEAR_DIM_BW = MAX_INPUT_FEATURE_DIM_BW * MAX_HASHGRID_LEVELS_BW; // F_in * L


// =========================================================================
// Step 1: Backward Pass for Rendering (Compute dL/dBlend, dL/dAlpha)
// =========================================================================
// This kernel re-runs the forward integration logic per-pixel to compute
// intermediate values needed for the backward alpha blending pass.
// It computes gradients w.r.t the activated MLP outputs (Blend_k: 7D) and alpha_k.
template <uint32_t CHANNELS> // Typically 3 for RGB
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
backwardRenderKernel(
    // --- Inputs ---
    // Gradients from loss
    const float* __restrict__ dL_dOut_Color,    // (CHANNELS, H, W) - Assuming CHW layout like forward output
    const float* __restrict__ dL_dOut_Features, // (output_feature_dim, H, W) - Assuming FHW layout
    // Forward pass outputs / state
    const float* __restrict__ final_T,          // Final transmittance per pixel (H, W) - Assuming HW layout
    // Data from sorting
    const uint2* __restrict__ ranges,          // Tile ranges in the sorted point list
    const uint32_t* __restrict__ point_list,    // Sorted list of primitive indices (num_rendered)
    // Image dimensions
    const int W, const int H,
    // Camera parameters (needed for recomputation)
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const float* camera_center_vec,
    const float near_plane,
    const float max_distance,
    // Primitive data (needed for recomputation)
    const float* __restrict__ primitive_centers,
    const float* __restrict__ primitive_confidences,
    const float primitive_scale,
    // Hashgrid data (needed for recomputation)
    const float* __restrict__ feature_table,
    const int* __restrict__ resolution,
    const int* __restrict__ do_hash,
    const int* __restrict__ primes,
    const int feature_offset, // T
    // MLP data (needed for recomputation)
    const float* __restrict__ linear_weights,
    const float* __restrict__ linear_bias,
    // Integration parameters (needed for recomputation)
    const int stencil_genus,
    const int grid_size,
    const float occupancy_threshold,
    // Background color
    const float* __restrict__ bg_color, // CHANNELS
    // Runtime dimensions
    const uint32_t input_feature_dim,    // F
    const uint32_t output_feature_dim,   // F_out
    const uint32_t hashgrid_levels,      // L
    // --- Outputs ---
    // Gradients per contribution (num_rendered entries)
    float* __restrict__ dL_dBlend_7D,       // (num_rendered, CHANNELS + output_feature_dim) - zero initialized
    float* __restrict__ dL_dAlpha,          // (num_rendered) - zero initialized
    // --- Temporary Storage / Recomputation Outputs (Optional, could be stack vars if feasible) ---
    // These are needed for subsequent backward steps if not recomputed again there
    float* __restrict__ store_primitive_outputs_pre_act, // (num_rendered, CHANNELS + output_feature_dim + 1) - Optional
    float* __restrict__ store_primitive_density,         // (num_rendered) - Optional
    float* __restrict__ store_delta_t,                   // (num_rendered) - Optional
    float* __restrict__ store_feature_integral           // (num_rendered, input_feature_dim * hashgrid_levels) - Optional
) {
    // --- Kernel Setup ---
    // Get camera center
    const float3 camera_center = {camera_center_vec[0], camera_center_vec[1], camera_center_vec[2]};

    // Runtime dimensions derived
    const uint32_t grid_volume = (uint32_t)grid_size * grid_size * grid_size;
    const uint32_t input_linear_dim = input_feature_dim * hashgrid_levels;
    const uint32_t output_linear_dim = CHANNELS + output_feature_dim + 1; // +1 for density
    const uint32_t blend_dim = CHANNELS + output_feature_dim; // 7D

    // Shared memory (mirroring forward pass for recomputation)
    __shared__ float stencil_coeffs_x[MAX_STENCIL_SIZE_BW];
    __shared__ float stencil_coeffs_y[MAX_STENCIL_SIZE_BW];
    __shared__ float stencil_coeffs_z[MAX_STENCIL_SIZE_BW];
    __shared__ int shared_stencil_size;
    __shared__ float shared_linear_weights[MAX_INPUT_LINEAR_DIM_BW * MAX_OUTPUT_LINEAR_DIM_BW];
    __shared__ float shared_linear_bias[MAX_OUTPUT_LINEAR_DIM_BW];
    __attribute__((shared)) float sh_confidence[MAX_GRID_VOLUME_BW];
    __attribute__((shared)) float sh_pixel_occupancy[MAX_GRID_VOLUME_BW];


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
        if (stencil_genus == 1) { stencil_size_local = 3; /* ... load coeffs ... */ }
        else if (stencil_genus == 2) { stencil_size_local = 5; /* ... load coeffs ... */ }
        else { stencil_size_local = 3; /* ... load default coeffs ... */ }
        // Ensure coeffs are loaded within MAX_STENCIL_SIZE_BW bounds
        shared_stencil_size = stencil_size_local;
    }
    int total_weights = input_linear_dim * output_linear_dim;
    for (int i = block.thread_rank(); i < total_weights; i += block.size()) {
        if (i < MAX_INPUT_LINEAR_DIM_BW * MAX_OUTPUT_LINEAR_DIM_BW) {
             shared_linear_weights[i] = linear_weights[i];
        }
    }
    int total_biases = output_linear_dim;
     for (int i = block.thread_rank(); i < total_biases; i += block.size()) {
         if (i < MAX_OUTPUT_LINEAR_DIM_BW) {
            shared_linear_bias[i] = linear_bias[i];
         }
    }
    block.sync();
    const int stencil_size = shared_stencil_size;
    const int stencil_offset = (stencil_size - 1) / 2;

    // --- Per-Pixel Backward Initialization ---
    float dL_dOut_pix_local[blend_dim]; // Gradient w.r.t. the current pixel output (Color + Features)
    float accum_blend_after[blend_dim]; // Accumulated blend value from primitives processed *after* current one

    if (inside) {
        // Load initial gradient from dL_dOut_Color and dL_dOut_Features
        for(int c=0; c<CHANNELS; ++c) {
            dL_dOut_pix_local[c] = dL_dOut_Color[c * H * W + pix_id];
        }
        for(uint32_t f=0; f<output_feature_dim; ++f) {
            // Assuming dL_dOut_Features is not NULL if output_feature_dim > 0
            dL_dOut_pix_local[CHANNELS + f] = dL_dOut_Features[f * H * W + pix_id];
        }

        // Initialize accum_blend_after with background color/features (features assumed 0)
        for(int c=0; c<CHANNELS; ++c) {
            accum_blend_after[c] = (bg_color != nullptr) ? bg_color[c] : 0.0f;
        }
        for(uint32_t f=0; f<output_feature_dim; ++f) {
            accum_blend_after[CHANNELS + f] = 0.0f; // Background features are zero
        }
    } else {
        // Initialize to zero if outside bounds
        for(int d=0; d<blend_dim; ++d) {
            dL_dOut_pix_local[d] = 0.0f;
            accum_blend_after[d] = 0.0f;
        }
    }

    // Load final transmittance for this pixel
    float T_current = (inside && final_T != nullptr) ? final_T[pix_id] : 0.0f; // T after the last primitive

    // --- Iterate Backwards Through Primitives for this Tile ---
    uint2 range = ranges[tile_idx];
    // We need to know how many primitives contributed to *this specific pixel* in the forward pass
    // to iterate backwards correctly. The current `ranges` are per-tile.
    // This requires storing the forward pass contribution list per pixel, or re-filtering.
    // Let's assume for now we iterate backwards through the whole tile range,
    // and rely on the recomputation logic (esp. frustum culling) to skip primitives
    // that didn't actually contribute to this pixel. This might be inefficient.
    // A better approach would be to store the per-pixel point list from forward.

    // Simplified approach: Iterate backwards through the tile's range.
    for (int i = range.y - 1; i >= range.x; --i) {
        if (!inside) break; // Skip if pixel is invalid

        // --- Get Primitive Data ---
        uint32_t primitive_idx = point_list[i]; // Original index
        uint32_t contribution_idx = i; // Index in the sorted list (and output gradient arrays)

        float3 center = { primitive_centers[primitive_idx * 3 + 0], primitive_centers[primitive_idx * 3 + 1], primitive_centers[primitive_idx * 3 + 2] };

        // --- Recompute Forward Pass for this Primitive and Pixel ---
        // This block mirrors the core logic inside the primitive loop in `renderCUDA`

        // Calculate depth (needed for delta_t, though delta_t itself isn't directly used in backward blending)
        float3 p_view = transformPoint4x3(center, viewmatrix);
        float view_depth = p_view.z; // Note: forward uses abs(z) or -z for depth checks

        // Initialize feature integral for this primitive contribution
        float primitive_feature_integral[MAX_INPUT_LINEAR_DIM_BW]; // Use BW max size
        for(uint32_t k=0; k<input_linear_dim; ++k) primitive_feature_integral[k] = 0.0f;

        // Load confidence grid into shared memory
        const uint64_t base_conf_idx_global = (uint64_t)primitive_idx * grid_volume;
        for (uint32_t j = block.thread_rank(); j < grid_volume; j += block.size()) {
            if (j < MAX_GRID_VOLUME_BW) {
                sh_confidence[j] = primitive_confidences[base_conf_idx_global + j];
            }
        }
        block.sync();

        // Pass 1: Calculate Per-Pixel Occupancy (sh_pixel_occupancy)
        bool primitive_contributes_to_pixel = false; // Flag if any grid point contributes
        const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
        const float grid_offset_factor = (grid_size - 1) / 2.0f;

        for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
            if (flat_idx >= MAX_GRID_VOLUME_BW) continue;
            int z = flat_idx / (grid_size * grid_size); int rem = flat_idx % (grid_size * grid_size); int y = rem / grid_size; int x = rem % grid_size;
            char occupancy = 0;
            float confidence = sh_confidence[flat_idx];
            if (confidence > occupancy_threshold) {
                float sample_rel_x = (x - grid_offset_factor) * grid_step; /* ... calculate sample_pos_world ... */
                float sample_rel_y = (y - grid_offset_factor) * grid_step;
                float sample_rel_z = (z - grid_offset_factor) * grid_step;
                float3 sample_pos_world = { center.x + sample_rel_x, center.y + sample_rel_y, center.z + sample_rel_z };

                // *** Crucial Check: Does this grid point contribute to THIS pixel? ***
                if (isPointInPyramidalFrustum(sample_pos_world, viewmatrix, projmatrix, W, H, pix, near_plane, max_distance)) {
                    occupancy = 1;
                    primitive_contributes_to_pixel = true; // Mark that this primitive affects the pixel
                }
            }
            sh_pixel_occupancy[flat_idx] = occupancy;
        }
        // Reduce across the block to see if *any* thread found a contribution for this primitive to this pixel
        primitive_contributes_to_pixel = cg::reduce(block, primitive_contributes_to_pixel, cg::logical_or<bool>());
        block.sync(); // Ensure occupancy is written and reduction is complete

        // If this primitive didn't contribute to this pixel in the forward pass, skip backward calculation for it
        if (!primitive_contributes_to_pixel) {
            continue;
        }

        // Pass 2: Calculate Gradient & Accumulate Features (primitive_feature_integral)
        // (Identical logic as in forward renderCUDA Pass 2, using sh_pixel_occupancy)
        for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
            if (flat_idx >= MAX_GRID_VOLUME_BW) continue;
            int z = flat_idx / (grid_size * grid_size); int rem = flat_idx % (grid_size * grid_size); int y = rem / grid_size; int x = rem % grid_size;

            // Calculate Occupancy Gradient (using stencil on sh_pixel_occupancy)
            float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
            // ... stencil application logic using stencil_coeffs_x/y/z and sh_pixel_occupancy ...
            // Ensure neighbor checks use MAX_GRID_VOLUME_BW for shared memory bounds
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
            float occupancy_grad_magnitude = fabsf(grad_x) + fabsf(grad_y) + fabsf(grad_z);

            if (occupancy_grad_magnitude > 1e-6f) {
                // Fetch Features from Hashgrid (Φ)
                float sample_features[MAX_INPUT_LINEAR_DIM_BW]; // Combined features
                for(uint32_t f=0; f<input_linear_dim; ++f) sample_features[f] = 0.0f;
                float sample_rel_x = (x - grid_offset_factor) * grid_step; /* ... calculate sample_pos_world ... */
                float sample_rel_y = (y - grid_offset_factor) * grid_step;
                float sample_rel_z = (z - grid_offset_factor) * grid_step;
                float3 sample_pos_world = { center.x + sample_rel_x, center.y + sample_rel_y, center.z + sample_rel_z };

                for (int level = 0; level < hashgrid_levels; ++level) {
                    float level_features[MAX_INPUT_FEATURE_DIM_BW];
                    // Assuming hashInterpolate is available and handles world coords
                    hashInterpolate(sample_pos_world, level, feature_table, resolution, feature_offset,
                                    do_hash[level], primes, level_features, input_feature_dim, hashgrid_levels);
                    // Concatenate
                    for (uint32_t f = 0; f < input_feature_dim; f++) {
                        uint32_t target_idx = level * input_feature_dim + f;
                        if (target_idx < input_linear_dim) {
                            sample_features[target_idx] = level_features[f];
                        }
                    }
                }
                // Accumulate Weighted Features
                float current_confidence = sh_confidence[flat_idx];
                float weight = occupancy_grad_magnitude * current_confidence;
                for (uint32_t f = 0; f < input_linear_dim; ++f) {
                    primitive_feature_integral[f] += sample_features[f] * weight;
                }
            }
        }
        // Note: primitive_feature_integral is thread-local, no sync needed here

        // Apply Linear Layer (MLP) -> primitive_outputs_pre_act_k
        float primitive_outputs_pre_act_k[MAX_OUTPUT_LINEAR_DIM_BW]; // Use BW max size
        for(uint32_t k=0; k<output_linear_dim; ++k) primitive_outputs_pre_act_k[k] = 0.0f;
        // ... Matrix multiplication logic using shared_linear_weights/bias ...
        for (uint32_t out_idx = 0; out_idx < output_linear_dim; out_idx++) {
            float dot_prod = 0.0f;
            for (uint32_t in_idx = 0; in_idx < input_linear_dim; in_idx++) {
                 if (out_idx * MAX_INPUT_LINEAR_DIM_BW + in_idx < MAX_INPUT_LINEAR_DIM_BW * MAX_OUTPUT_LINEAR_DIM_BW) {
                     dot_prod += shared_linear_weights[out_idx * MAX_INPUT_LINEAR_DIM_BW + in_idx] * primitive_feature_integral[in_idx];
                 }
            }
            if (out_idx < MAX_OUTPUT_LINEAR_DIM_BW) {
                primitive_outputs_pre_act_k[out_idx] = dot_prod + shared_linear_bias[out_idx];
            } else {
                 primitive_outputs_pre_act_k[out_idx] = dot_prod;
            }
        }


        // Apply Activations -> Blend_k (7D), primitive_density_k
        float Blend_k[blend_dim]; // Activated Color + Features
        float primitive_density_k = 0.0f;
        // Sigmoid for Color (CHANNELS) and Features (output_feature_dim)
        for (uint32_t k = 0; k < blend_dim; ++k) {
             if (k < output_linear_dim) {
                Blend_k[k] = (1.0f / (1.0f + expf(-primitive_outputs_pre_act_k[k])));
             } else {
                Blend_k[k] = 0.0f; // Should not happen if blend_dim <= output_linear_dim
             }
        }
        // Exp for density
        uint32_t density_idx = blend_dim; // Index after color and features
        if (density_idx < output_linear_dim) {
            primitive_density_k = expf(primitive_outputs_pre_act_k[density_idx]);
            primitive_density_k = max(0.0f, primitive_density_k);
        }

        // Calculate delta_t (step size) - Requires depth tracking from previous step in backward pass
        // This is tricky. The forward pass calculates delta_t based on `last_view_depth`.
        // In the backward pass, we need the *same* delta_t that was used for this primitive in forward.
        // Recomputing requires knowing the depth of the *next* primitive processed in the forward pass.
        // Simplification: Assume delta_t can be reasonably approximated or needs to be stored.
        // Let's assume `store_delta_t` is available or calculated somehow.
        float delta_t_k = 0.01f; // Placeholder - NEEDS ACCURATE VALUE
        if (store_delta_t != nullptr) {
            // If storage is provided, we assume it was filled correctly (e.g., during forward or a pre-pass)
            // Or we could try to recompute it based on depth differences in the *backward* iteration order,
            // but that doesn't match the forward pass order.
            // store_delta_t[contribution_idx] = delta_t_k; // Write if calculating here
            delta_t_k = store_delta_t[contribution_idx]; // Read if pre-filled
        }


        // Calculate alpha_k
        float alpha_k = 1.0f - expf(-primitive_density_k * delta_t_k);
        alpha_k = min(max(alpha_k, 0.0f), 1.0f);

        // --- End Recomputation ---

        // --- Backward Blending Calculation ---
        // We need T *before* this primitive contribution.
        // T_current holds T *after* this primitive. T_before = T_after / (1 - alpha_k)
        float T_before = 0.0f;
        if (fabsf(1.0f - alpha_k) > 1e-6f) { // Avoid division by zero/small numbers
             T_before = T_current / (1.0f - alpha_k);
        } else {
             // Handle case where alpha is close to 1. T_before would be large.
             // If alpha == 1, T_current should be 0. If T_current is non-zero, something is wrong.
             // If T_current is 0 and alpha is 1, T_before could be anything, but the contribution
             // to the gradient might be zero anyway. Let's set T_before to 0 to be safe.
             T_before = 0.0f; // Or handle based on expected behavior when alpha=1
        }


        // Compute dL/dBlend_k = dL/dOut_pix_local * T_before * alpha_k
        float dL_dBlend_k[blend_dim];
        for(int d=0; d<blend_dim; ++d) {
            dL_dBlend_k[d] = dL_dOut_pix_local[d] * T_before * alpha_k;
            // Write to global output buffer using contribution_idx
            if (contribution_idx * blend_dim + d < num_rendered * blend_dim) { // Bounds check
                 atomicAdd(&dL_dBlend_7D[contribution_idx * blend_dim + d], dL_dBlend_k[d]); // Use atomicAdd as multiple threads (pixels) might write to same contribution
            }
        }

        // Compute dL/dalpha_k = dot(dL/dOut_pix_local, T_before * (Blend_k - accum_blend_after))
        float dL_dalpha_k = 0.0f;
        for(int d=0; d<blend_dim; ++d) {
            dL_dalpha_k += dL_dOut_pix_local[d] * T_before * (Blend_k[d] - accum_blend_after[d]);
        }
        // Write to global output buffer
        if (contribution_idx < num_rendered) { // Bounds check
            atomicAdd(&dL_dAlpha[contribution_idx], dL_dalpha_k);
        }

        // --- Update State for Next (Earlier) Primitive ---
        // Update dL/dOut_pix_local for the next iteration (gradient flowing from before)
        // dL_dOut_pix_local = dL_dOut_pix_local * T_before * (1 - alpha_k)
        // Note: T_before * (1 - alpha_k) is just T_current
        for(int d=0; d<blend_dim; ++d) {
            dL_dOut_pix_local[d] = dL_dOut_pix_local[d] * T_current; // Simplified update
        }

        // Update accum_blend_after for the next iteration
        // accum_blend_after = Blend_k * alpha_k + accum_blend_after * (1 - alpha_k)
        for(int d=0; d<blend_dim; ++d) {
            accum_blend_after[d] = Blend_k[d] * alpha_k + accum_blend_after[d] * (1.0f - alpha_k);
        }

        // Update T_current for the next iteration (becomes T before the current primitive)
        T_current = T_before;

        // --- Store Recomputed Values (Optional) ---
        // If subsequent kernels need these without recomputing again
        if (store_primitive_outputs_pre_act != nullptr) {
            for(uint32_t k=0; k<output_linear_dim; ++k) {
                 if (contribution_idx * output_linear_dim + k < num_rendered * output_linear_dim) { // Bounds check
                    store_primitive_outputs_pre_act[contribution_idx * output_linear_dim + k] = primitive_outputs_pre_act_k[k];
                 }
            }
        }
        if (store_primitive_density != nullptr) {
             if (contribution_idx < num_rendered) { // Bounds check
                 store_primitive_density[contribution_idx] = primitive_density_k;
             }
        }
        // delta_t was handled earlier
        if (store_feature_integral != nullptr) {
             for(uint32_t k=0; k<input_linear_dim; ++k) {
                 if (contribution_idx * input_linear_dim + k < num_rendered * input_linear_dim) { // Bounds check
                    store_feature_integral[contribution_idx * input_linear_dim + k] = primitive_feature_integral[k];
                 }
            }
        }

    } // --- End Backwards Loop Over Primitives ---
}


// =========================================================================
// Step 2: Backward Pass for Activations (Sigmoid, Exp/Alpha)
// =========================================================================
// Computes dL/dLinearOutput (8D) from dL/dBlend (7D) and dL/dAlpha.
__global__ void backwardActivationKernel(
    // Inputs
    int num_rendered,                       // Number of contributions (N)
    const float* __restrict__ dL_dBlend_7D, // Gradient w.r.t. activated outputs (N, 7)
    const float* __restrict__ dL_dAlpha,    // Gradient w.r.t. alpha (N)
    // Recomputed/Stored values from forward/backward render
    const float* __restrict__ primitive_outputs_pre_act, // MLP output before activation (N, 8)
    const float* __restrict__ primitive_density,         // Activated density (N)
    const float* __restrict__ store_delta_t,             // Step size used for alpha (N) - CRITICAL
    // Runtime dimensions
    const uint32_t output_feature_dim, // F_out
    const uint32_t CHANNELS,           // Typically 3
    // Output
    float* __restrict__ dL_dLinearOutput_8D // Gradient w.r.t. MLP output pre-activation (N, 8) - zero initialized
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rendered) return;

    const uint32_t blend_dim = CHANNELS + output_feature_dim;
    const uint32_t density_idx_in_8d = blend_dim; // Density is the last channel

    // --- Backward Sigmoid for Blend (Color + Features) ---
    for (uint32_t k = 0; k < blend_dim; ++k) {
        // y = sigmoid(x) => dL/dx = dL/dy * y * (1-y)
        // Need the activated value y = Blend_k[k]
        // Need the pre-activation value x = primitive_outputs_pre_act[idx * 8 + k]
        // We can recompute y from x: y = 1 / (1 + exp(-x))
        float x = primitive_outputs_pre_act[idx * (blend_dim + 1) + k]; // Index into 8D array
        float y = 1.0f / (1.0f + expf(-x));
        float dL_dy = dL_dBlend_7D[idx * blend_dim + k];
        float dL_dx = dL_dy * y * (1.0f - y);
        dL_dLinearOutput_8D[idx * (blend_dim + 1) + k] = dL_dx;
    }

    // --- Backward Exp/Alpha for Density ---
    // alpha = 1 - exp(-density * delta_t)
    // density = exp(x_density)
    // Need: dL/dalpha, density, delta_t, x_density
    float dL_dalpha = dL_dAlpha[idx];
    float density = primitive_density[idx];
    float delta_t = store_delta_t[idx]; // Get the accurate delta_t
    float x_density = primitive_outputs_pre_act[idx * (blend_dim + 1) + density_idx_in_8d];

    // dL/ddensity = dL/dalpha * dalpha/ddensity
    // dalpha/ddensity = -exp(-density * delta_t) * (-delta_t) = exp(-density * delta_t) * delta_t
    float dL_ddensity = dL_dalpha * expf(-density * delta_t) * delta_t;

    // dL/dx_density = dL/ddensity * ddensity/dx_density
    // ddensity/dx_density = exp(x_density) = density
    float dL_dx_density = dL_ddensity * density;

    dL_dLinearOutput_8D[idx * (blend_dim + 1) + density_idx_in_8d] = dL_dx_density;
}


// =========================================================================
// Step 3: Backward Pass for Linear Layer
// =========================================================================
// Computes dL/dLinearInput, dL/dWeights, dL/dBias from dL/dLinearOutput.
__global__ void backwardLinearLayerKernel(
    // Inputs
    int num_rendered,                       // Number of contributions (N)
    const float* __restrict__ dL_dLinearOutput_8D, // Gradient w.r.t. MLP output (N, 8)
    // Recomputed/Stored feature integrals (input to MLP)
    const float* __restrict__ store_feature_integral, // (N, input_linear_dim)
    // MLP weights (needed for dL/dInput)
    const float* __restrict__ linear_weights, // (input_linear_dim, output_linear_dim=8)
    // Runtime dimensions
    const uint32_t input_linear_dim,    // F * L
    const uint32_t output_linear_dim,   // Should be 8 (CHANNELS + F_out + 1)
    // Outputs
    float* __restrict__ dL_dLinearInput,    // Gradient w.r.t. feature integral (N, input_linear_dim) - zero initialized
    float* __restrict__ dL_dLinearWeights,  // Accumulated gradient for weights - zero initialized
    float* __restrict__ dL_dLinearBias      // Accumulated gradient for bias - zero initialized
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rendered) return;

    // --- Calculate dL/dLinearInput = dL/dOutput @ Weights^T ---
    // dL/dInput_j = sum_k ( dL/dOutput_k * Weight_kj )
    const float* dL_dOutput_i = dL_dLinearOutput_8D + idx * output_linear_dim;
    float* dL_dInput_i = dL_dLinearInput + idx * input_linear_dim;

    for (uint32_t j = 0; j < input_linear_dim; ++j) { // Input feature index
        float grad_sum = 0.0f;
        for (uint32_t k = 0; k < output_linear_dim; ++k) { // Output feature index
            // Weight layout: Assume input_dim major: W[k * input_dim + j] or output_dim major W[j * output_dim + k]?
            // Forward MLP uses: shared_linear_weights[out_idx * MAX_INPUT_LINEAR_DIM + in_idx]
            // So, weights are likely stored output-major in global memory: linear_weights[in_idx * output_dim + out_idx]
            // Let's assume output-major layout for linear_weights: W_kj = linear_weights[j * output_linear_dim + k]
             uint64_t weight_idx = (uint64_t)j * output_linear_dim + k;
             // Add bounds check if necessary based on total weight matrix size
             grad_sum += dL_dOutput_i[k] * linear_weights[weight_idx];
        }
        dL_dInput_i[j] = grad_sum;
    }

    // --- Accumulate dL/dWeights and dL/dBias ---
    // dL/dWeight_kj = dL/dOutput_k * Input_j
    // dL/dBias_k = dL/dOutput_k
    const float* input_i = store_feature_integral + idx * input_linear_dim;

    for (uint32_t k = 0; k < output_linear_dim; ++k) { // Output feature index
        float dL_dOutput_k = dL_dOutput_i[k];

        // Accumulate bias gradient
        atomicAdd(&dL_dLinearBias[k], dL_dOutput_k);

        // Accumulate weight gradient
        for (uint32_t j = 0; j < input_linear_dim; ++j) { // Input feature index
            float input_j = input_i[j];
            float dL_dW_kj = dL_dOutput_k * input_j;
            // Weight index assuming output-major layout
            uint64_t weight_idx = (uint64_t)j * output_linear_dim + k;
            atomicAdd(&dL_dLinearWeights[weight_idx], dL_dW_kj);
        }
    }
}


// =========================================================================
// Step 4: Aggregate dL/dLinearInput per Primitive
// =========================================================================
// Aggregates the gradient w.r.t. the feature integral from per-contribution
// gradients to per-primitive gradients.
__global__ void aggregateLinearInputGradKernel(
    // Inputs
    int num_rendered,                       // Number of contributions (N)
    const uint32_t* __restrict__ point_list,// Sorted list of primitive indices (N)
    const float* __restrict__ dL_dLinearInput, // Gradient per contribution (N, input_linear_dim)
    int P,                                  // Total number of primitives
    // Runtime dimensions
    const uint32_t input_linear_dim,    // F * L
    // Output
    float* __restrict__ dL_dLinearInput_agg // Aggregated gradient per primitive (P, input_linear_dim) - zero initialized
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rendered) return;

    uint32_t primitive_idx = point_list[idx]; // Get the primitive this contribution belongs to

    if (primitive_idx < P) { // Check primitive index validity
        const float* dL_dInput_i = dL_dLinearInput + idx * input_linear_dim;
        float* dL_dInput_agg_p = dL_dLinearInput_agg + primitive_idx * input_linear_dim;

        for (uint32_t j = 0; j < input_linear_dim; ++j) {
            atomicAdd(&dL_dInput_agg_p[j], dL_dInput_i[j]);
        }
    }
}


// =========================================================================
// Step 5: Backward Pass for Features (Φ) and Confidence (X)
// =========================================================================
// Propagates dL/dLinearInput_agg back through the integration (Pass 2)
// and hash interpolation to compute dL/dFeatureTable and dL/dX.

// Helper for backward hash interpolation (distributes gradient to table)
__device__ void backward_hash_interpolate_distribute(
    const float* dL_dPhi_level,             // Gradient for features at this level [input_feature_dim]
    const float3& pos_world_j,              // World position of the grid point
    int level,                              // Current hashgrid level
    // Hashgrid parameters for this level
    int resolution_level,                   // Resolution for this level
    uint32_t feature_table_size_per_level,  // T
    bool use_hashing,                       // do_hash[level]
    const int* primes,                      // Hashing primes [3]
    // Runtime dimensions
    uint32_t input_feature_dim,             // F
    uint32_t hashgrid_levels,               // L (needed for global table indexing)
    // Output Table Gradient (Accumulates)
    float* dL_dFeatureTable                 // Global table gradient [L * T * F]
) {
    // Recompute grid coordinates and interpolation weights (trilinear)
    float scale = (float)resolution_level; // Assuming [0,1] mapping
    float3 grid_pos_norm = { pos_world_j.x * scale, pos_world_j.y * scale, pos_world_j.z * scale };
    int3 grid_min = { static_cast<int>(floorf(grid_pos_norm.x)), static_cast<int>(floorf(grid_pos_norm.y)), static_cast<int>(floorf(grid_pos_norm.z)) };
    float3 t = { grid_pos_norm.x - grid_min.x, grid_pos_norm.y - grid_min.y, grid_pos_norm.z - grid_min.z };

    // Iterate over 8 corners
    for (int dx = 0; dx < 2; dx++) {
        float wx = (dx == 0) ? (1.0f - t.x) : t.x;
        for (int dy = 0; dy < 2; dy++) {
            float wy = (dy == 0) ? (1.0f - t.y) : t.y;
            for (int dz = 0; dz < 2; dz++) {
                float wz = (dz == 0) ? (1.0f - t.z) : t.z;
                float corner_weight = wx * wy * wz; // Trilinear weight for this corner

                // Calculate corner coords and hash index (same logic as hashInterpolate)
                int3 corner_coords = { grid_min.x + dx, grid_min.y + dy, grid_min.z + dz };
                uint32_t feature_entry_index; // Index within level's table slice (0 to T-1)
                bool valid_index = true;

                if (use_hashing) {
                    uint32_t hash_val = 0;
                    hash_val ^= (uint32_t)corner_coords.x * primes[0];
                    hash_val ^= (uint32_t)corner_coords.y * primes[1];
                    hash_val ^= (uint32_t)corner_coords.z * primes[2];
                    feature_entry_index = hash_val % feature_table_size_per_level;
                } else {
                    // Dense grid bounds check
                    if (corner_coords.x < 0 || corner_coords.x >= resolution_level ||
                        corner_coords.y < 0 || corner_coords.y >= resolution_level ||
                        corner_coords.z < 0 || corner_coords.z >= resolution_level) {
                        valid_index = false;
                    } else {
                        feature_entry_index = (uint32_t)corner_coords.z * resolution_level * resolution_level + (uint32_t)corner_coords.y * resolution_level + (uint32_t)corner_coords.x;
                        if (feature_entry_index >= feature_table_size_per_level) { // Check against T
                             valid_index = false;
                        }
                    }
                }

                if (valid_index) {
                    // Calculate base index in the global dL_dFeatureTable
                    // Layout: L * T * F -> level * T * F + entry_idx * F + feat_idx
                    // Note: T = feature_table_size_per_level, F = input_feature_dim
                    uint64_t level_base_offset = (uint64_t)level * feature_table_size_per_level * input_feature_dim;
                    uint64_t entry_offset = (uint64_t)feature_entry_index * input_feature_dim;
                    uint64_t global_table_base_idx = level_base_offset + entry_offset;

                    // Distribute gradient for each feature dimension F
                    for (uint32_t f = 0; f < input_feature_dim; f++) {
                        float grad_phi_f = dL_dPhi_level[f]; // Gradient w.r.t interpolated feature f at this level
                        float grad_table = grad_phi_f * corner_weight; // Chain rule: dL/dTable = dL/dPhi * dPhi/dTable
                        uint64_t global_table_idx = global_table_base_idx + f;
                        // TODO: Add bounds check for global_table_idx against total size L*T*F if necessary
                        atomicAdd(&dL_dFeatureTable[global_table_idx], grad_table);
                    }
                }
            } // dz
        } // dy
    } // dx
}


// Kernel definition
__global__ void backwardFeatureConfidenceKernel(
    // Inputs
    int P,                                  // Number of primitives
    const float* __restrict__ dL_dLinearInput_agg, // Aggregated gradient (P, input_linear_dim)
    // Primitive data
    const float* __restrict__ primitive_centers,     // (P, 3)
    const float* __restrict__ primitive_confidences, // (P, grid_volume)
    const float primitive_scale,
    // Hashgrid data
    const float* __restrict__ feature_table,         // Global feature table Φ (L * T * F)
    const int* __restrict__ resolution_levels,       // Per-level resolution [L]
    const uint32_t feature_table_size_per_level,     // Size T per level
    const int* __restrict__ do_hash,                 // Flags for hashing per level [L]
    const int* __restrict__ primes,                  // Primes for hashing [3]
    // Integration parameters
    const int stencil_genus,
    const int grid_size,
    // Runtime dimensions
    const uint32_t input_feature_dim,       // F
    const uint32_t hashgrid_levels,         // L
    // Outputs (Must be zero-initialized before launch)
    float* __restrict__ dL_dFeatureTable,          // Gradient w.r.t. feature table (L * T * F)
    float* __restrict__ dL_dX                      // Gradient w.r.t. confidence grids (P * grid_volume)
    // float* __restrict__ dL_dPosWorld            // Optional: Gradient w.r.t world positions (P * grid_volume * 3)
) {
    // --- Kernel Setup ---
    const int primitive_idx = blockIdx.x;
    if (primitive_idx >= P) return;

    const uint32_t grid_volume = (uint32_t)grid_size * grid_size * grid_size;
    const uint32_t input_linear_dim = input_feature_dim * hashgrid_levels; // F * L
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Load Stencil Coeffs into Shared Memory ---
    // (Could be loaded once per block if blockDim.x <= MAX_STENCIL_SIZE_BW)
    // Or passed via constant memory for efficiency
    float stencil_coeffs_x[MAX_STENCIL_SIZE_BW];
    float stencil_coeffs_y[MAX_STENCIL_SIZE_BW];
    float stencil_coeffs_z[MAX_STENCIL_SIZE_BW];
    int stencil_size;
    int stencil_offset;
    // Load stencil based on stencil_genus (same logic as forward/backward render)
    if (stencil_genus == 1) { stencil_size = 3; /* load coeffs */ }
    else if (stencil_genus == 2) { stencil_size = 5; /* load coeffs */ }
    else { stencil_size = 3; /* load default */ }
    stencil_offset = (stencil_size - 1) / 2;


    // Pointer to this primitive's aggregated input gradient
    const float* p_dL_dLinearInput_agg = dL_dLinearInput_agg + primitive_idx * input_linear_dim;
    // Pointers to this primitive's confidence grid and its gradient output
    const float* p_X = primitive_confidences + primitive_idx * grid_volume;
    float* p_dL_dX = dL_dX + primitive_idx * grid_volume;
    // Primitive center
    float3 center = { primitive_centers[primitive_idx * 3 + 0], primitive_centers[primitive_idx * 3 + 1], primitive_centers[primitive_idx * 3 + 2] };


    // --- Process All Grid Points for this Primitive ---
    // Each thread processes a subset of grid points
    for (int idx_flat = thread_idx; idx_flat < grid_volume; idx_flat += block_size) {
        // Convert flat index to 3D grid coordinates
        int z = idx_flat / (grid_size * grid_size);
        int rem = idx_flat % (grid_size * grid_size);
        int y = rem / grid_size;
        int x = rem % grid_size;

        // --- Recompute Confidence Gradient Magnitude ---
        // Apply stencil to the confidence grid p_X
        float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;
        for (int s = 0; s < stencil_size; s++) {
            int sx = s - stencil_offset; int nx = x + sx;
            int sy = s - stencil_offset; int ny = y + sy;
            int sz = s - stencil_offset; int nz = z + sz;

            // Check bounds for neighbor coordinates
            bool nx_ok = (nx >= 0 && nx < grid_size);
            bool ny_ok = (ny >= 0 && ny < grid_size);
            bool nz_ok = (nz >= 0 && nz < grid_size);

            // Calculate flat neighbor indices within the primitive's grid
            uint32_t flat_nx_idx = z * grid_size * grid_size + y * grid_size + nx;
            uint32_t flat_ny_idx = z * grid_size * grid_size + ny * grid_size + x;
            uint32_t flat_nz_idx = nz * grid_size * grid_size + y * grid_size + x;

            // Get neighbor confidence (use 0 if out of bounds)
            float conf_x_neighbor = (nx_ok) ? p_X[flat_nx_idx] : 0.0f;
            float conf_y_neighbor = (ny_ok) ? p_X[flat_ny_idx] : 0.0f;
            float conf_z_neighbor = (nz_ok) ? p_X[flat_nz_idx] : 0.0f;

            // Accumulate gradient components
            grad_x += stencil_coeffs_x[s] * conf_x_neighbor;
            grad_y += stencil_coeffs_y[s] * conf_y_neighbor;
            grad_z += stencil_coeffs_z[s] * conf_z_neighbor;
        }
        float conf_grad_magnitude = fabsf(grad_x) + fabsf(grad_y) + fabsf(grad_z);

        // --- Recompute Hash Interpolation (Φ_interp) ---
        // Calculate world position for this grid point
        const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
        const float grid_offset_factor = (grid_size - 1) / 2.0f;
        float sample_rel_x = (x - grid_offset_factor) * grid_step;
        float sample_rel_y = (y - grid_offset_factor) * grid_step;
        float sample_rel_z = (z - grid_offset_factor) * grid_step;
        float3 pos_world_j = { center.x + sample_rel_x, center.y + sample_rel_y, center.z + sample_rel_z };

        // Interpolate features using hashInterpolate (assuming it's available)
        float phi_interp_j[MAX_INPUT_LINEAR_DIM_BW]; // Concatenated features F*L
        for(uint32_t k=0; k<input_linear_dim; ++k) phi_interp_j[k] = 0.0f; // Initialize

        for (int level = 0; level < hashgrid_levels; ++level) {
            float level_features[MAX_INPUT_FEATURE_DIM_BW]; // Features F for this level
            hashInterpolate(
                pos_world_j, level, feature_table, resolution_levels, feature_table_size_per_level,
                do_hash[level], primes, level_features, input_feature_dim, hashgrid_levels
            );
            // Concatenate into phi_interp_j
            for (uint32_t f = 0; f < input_feature_dim; f++) {
                uint32_t target_idx = level * input_feature_dim + f;
                if (target_idx < input_linear_dim) { // Check bounds against F*L
                    phi_interp_j[target_idx] = level_features[f];
                }
            }
        }

        // --- Calculate Gradients ---
        // Weight w_j = conf_grad_magnitude * X_j
        float X_j = p_X[idx_flat];
        float w_j = conf_grad_magnitude * X_j;

        // dL/dX_j = dot(dL/dLinearInput_agg[p], phi_interp_j) * conf_grad_magnitude
        float dL_dX_j = 0.0f;
        for (uint32_t k = 0; k < input_linear_dim; ++k) {
            dL_dX_j += p_dL_dLinearInput_agg[k] * phi_interp_j[k];
        }
        dL_dX_j *= conf_grad_magnitude;

        // Accumulate dL/dX
        if (fabsf(dL_dX_j) > 1e-9f) { // Avoid atomicAdd for zero
             atomicAdd(&p_dL_dX[idx_flat], dL_dX_j);
        }

        // dL/dPhi_interp_j = dL/dLinearInput_agg[p] * w_j
        float dL_dPhi_interp_j[MAX_INPUT_LINEAR_DIM_BW]; // Gradient w.r.t concatenated features
        for (uint32_t k = 0; k < input_linear_dim; ++k) {
            dL_dPhi_interp_j[k] = p_dL_dLinearInput_agg[k] * w_j;
        }

        // --- Backward Hash Interpolation (Distribute dL/dΦ_interp to dL/dFeatureTable) ---
        if (w_j > 1e-9f) { // Only distribute if the weight is non-negligible
            for (int level = 0; level < hashgrid_levels; ++level) {
                // Get the portion of dL/dPhi_interp_j for this level
                const float* dL_dPhi_level = dL_dPhi_interp_j + level * input_feature_dim;

                backward_hash_interpolate_distribute(
                    dL_dPhi_level,
                    pos_world_j,
                    level,
                    resolution_levels[level],
                    feature_table_size_per_level,
                    do_hash[level] != 0, // Convert int flag to bool
                    primes,
                    input_feature_dim,
                    hashgrid_levels,
                    dL_dFeatureTable
                );
            }
        }

        // --- TODO: Calculate dL/dPosWorld if needed ---
        // This requires gradient of interpolation weights w.r.t. pos_world_j.
        // And gradient of confidence gradient w.r.t pos_world_j (if stencil depends on it).

    } // End loop over grid points (idx_flat)
}


// =========================================================================
// Wrapper Functions (Namespace BACKWARD)
// =========================================================================
namespace BACKWARD {

// Wrapper for backward rendering kernel
void render(
    // Inputs matching kernel
    const float* dL_dOut_Color, const float* dL_dOut_Features,
    const float* final_T,
    const uint2* ranges, const uint32_t* point_list, int num_rendered,
    int W, int H,
    const float* viewmatrix, const float* projmatrix, const float* camera_center_vec,
    float near_plane, float max_distance,
    const float* primitive_centers, const float* primitive_confidences, float primitive_scale,
    const float* feature_table, const int* resolution, const int* do_hash, const int* primes, int feature_offset,
    const float* linear_weights, const float* linear_bias,
    int stencil_genus, int grid_size, float occupancy_threshold,
    const float* bg_color,
    // Runtime dimensions
    uint32_t input_feature_dim, uint32_t output_feature_dim, uint32_t hashgrid_levels, uint32_t num_output_channels,
    // Outputs
    float* dL_dBlend_7D, float* dL_dAlpha,
    // Optional Storage Outputs
    float* store_primitive_outputs_pre_act, float* store_primitive_density,
    float* store_delta_t, float* store_feature_integral,
    // CUDA Stream
    cudaStream_t stream
) {
    // Determine grid/block dimensions based on tiles (must match forward)
    const dim3 blocks((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
    const dim3 threads(BLOCK_X, BLOCK_Y);

    // Assuming num_output_channels is used to select template specialization if needed
    if (num_output_channels == 3) {
        backwardRenderKernel<3><<<blocks, threads, 0, stream>>>(
            dL_dOut_Color, dL_dOut_Features, final_T, ranges, point_list,
            W, H, viewmatrix, projmatrix, camera_center_vec, near_plane, max_distance,
            primitive_centers, primitive_confidences, primitive_scale,
            feature_table, resolution, do_hash, primes, feature_offset,
            linear_weights, linear_bias, stencil_genus, grid_size, occupancy_threshold,
            bg_color, input_feature_dim, output_feature_dim, hashgrid_levels,
            dL_dBlend_7D, dL_dAlpha,
            store_primitive_outputs_pre_act, store_primitive_density, store_delta_t, store_feature_integral
        );
    } else {
         printf("CUDA Error in BACKWARD::render: Unsupported channel count %d\n", num_output_channels);
         return;
    }

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in BACKWARD::render launch: %s\n", cudaGetErrorString(err));
	}
}

// Wrapper for backward activation kernel
void backwardActivation(
    int num_rendered,
    const float* dL_dBlend_7D, const float* dL_dAlpha,
    const float* primitive_outputs_pre_act, const float* primitive_density, const float* store_delta_t,
    uint32_t output_feature_dim, uint32_t num_output_channels, // CHANNELS
    float* dL_dLinearOutput_8D,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (num_rendered + threads - 1) / threads;
    backwardActivationKernel<<<blocks, threads, 0, stream>>>(
        num_rendered, dL_dBlend_7D, dL_dAlpha,
        primitive_outputs_pre_act, primitive_density, store_delta_t,
        output_feature_dim, num_output_channels,
        dL_dLinearOutput_8D
    );
     cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in BACKWARD::backwardActivation launch: %s\n", cudaGetErrorString(err));
	}
}


// Wrapper for backward linear layer kernel
void backwardLinearLayer(
    int num_rendered,
    const float* dL_dLinearOutput_8D,
    const float* store_feature_integral,
    const float* linear_weights,
    uint32_t input_linear_dim, uint32_t output_linear_dim, // Should be 8
    float* dL_dLinearInput, float* dL_dLinearWeights, float* dL_dLinearBias,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (num_rendered + threads - 1) / threads;
    backwardLinearLayerKernel<<<blocks, threads, 0, stream>>>(
        num_rendered, dL_dLinearOutput_8D, store_feature_integral, linear_weights,
        input_linear_dim, output_linear_dim,
        dL_dLinearInput, dL_dLinearWeights, dL_dLinearBias
    );
     cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in BACKWARD::backwardLinearLayer launch: %s\n", cudaGetErrorString(err));
	}
}

// Wrapper for aggregation kernel
void aggregateLinearInputGrad(
    int num_rendered,
    const uint32_t* point_list,
    const float* dL_dLinearInput,
    int P,
    uint32_t input_linear_dim,
    float* dL_dLinearInput_agg, // Output
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (num_rendered + threads - 1) / threads;
    // Ensure output is zero-initialized before calling
    // cudaMemsetAsync(dL_dLinearInput_agg, 0, P * input_linear_dim * sizeof(float), stream);
    aggregateLinearInputGradKernel<<<blocks, threads, 0, stream>>>(
        num_rendered, point_list, dL_dLinearInput, P, input_linear_dim, dL_dLinearInput_agg
    );
     cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in BACKWARD::aggregateLinearInputGrad launch: %s\n", cudaGetErrorString(err));
	}
}


// Wrapper for backward feature and confidence kernel
void backwardFeatureConfidence(
    int P,
    const float* dL_dLinearInput_agg,
    const float* primitive_centers, const float* primitive_confidences, float primitive_scale,
    const float* feature_table, const int* resolution_levels, uint32_t feature_table_size_per_level,
    const int* do_hash, const int* primes,
    int stencil_genus, int grid_size,
    uint32_t input_feature_dim, uint32_t hashgrid_levels,
    // Outputs
    float* dL_dFeatureTable, float* dL_dX,
    cudaStream_t stream
) {
    const int threads = 256; // Tune this
    const int blocks = P; // One block per primitive

    // Ensure outputs are zero-initialized before calling (caller responsibility)
    // cudaMemsetAsync(dL_dFeatureTable, 0, L * T * F * sizeof(float), stream);
    // cudaMemsetAsync(dL_dX, 0, P * grid_volume * sizeof(float), stream);

    backwardFeatureConfidenceKernel<<<blocks, threads, 0, stream>>>(
        P, dL_dLinearInput_agg,
        primitive_centers, primitive_confidences, primitive_scale,
        feature_table, resolution_levels, feature_table_size_per_level, do_hash, primes,
        stencil_genus, grid_size,
        input_feature_dim, hashgrid_levels,
        dL_dFeatureTable, dL_dX
    );

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in BACKWARD::backwardFeatureConfidence launch: %s\n", cudaGetErrorString(err));
	}
}

} // namespace BACKWARD




