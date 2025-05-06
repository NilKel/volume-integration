#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Define constants needed for template parameters if not defined elsewhere
// These should match the definitions in common headers or the .cu files
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif
// MAX_PRIMITIVES_PER_RAY is now passed as an argument 'max_primitives_per_ray'

namespace BACKWARD {

    // Unified Backward Pass Kernel Launcher
    // Recomputes forward pass intermediates and calculates gradients for
    // MLP weights/bias, feature table, and confidence grids in a single kernel.
    void compute_gradients( // Renamed from 'render' for clarity
        // <<< Kernel Launch Config >>>
        const dim3 grid, dim3 block,
        // --- Input Gradients (Pixel-based) ---
        const float* dL_dOut_Color,    // (CHANNELS, H, W)
        const float* dL_dOut_Features, // (output_feature_dim, H, W) - Optional
        // --- Forward Pass State/Outputs (Pixel-based & Contribution-based) ---
        const float* final_T,                 // Final transmittance per pixel (H, W)
        const int* num_contrib_per_pixel,     // Contribution count per pixel (H, W)
        const int* pixel_contributing_indices,// GLOBAL index i from point_list per pixel contribution (H, W, max_primitives_per_ray)
        const float* store_delta_t,           // Stored delta_t per contribution (num_rendered) - CRITICAL
        // --- Data from Sorting ---
        const uint32_t* point_list,           // Sorted list of primitive indices (num_rendered) - Maps global index i to primitive_idx
        int P,                                // Total number of unique primitives
        int num_rendered,                     // Total number of contributions in point_list
        // --- Image Dimensions ---
        int W, int H,
        // --- Camera Parameters ---
        const float* viewmatrix,
        const float* projmatrix,
        const float* camera_center_vec, // float3 passed as float*
        float near_plane,
        float max_distance,
        // --- Primitive Data (Needed for recomputation) ---
        const float* primitive_centers,     // (P, 3)
        const float* primitive_confidences, // (P, grid_volume)
        float primitive_scale,
        // --- Hashgrid Data ---
        const float* feature_table,         // (L * T * F) or similar layout
        const int* resolution,              // Array of resolutions per level (L)
        const int* do_hash,                 // Array of hash flags per level (L)
        const int* primes,                  // Hashing primes (3)
        int feature_offset,                 // Size T of hash table per level
        uint32_t feature_table_size,        // Total size of feature table (L*T*F)
        // --- MLP Data ---
        const float* linear_weights,        // (input_linear_dim, output_linear_dim)
        const float* linear_bias,           // (output_linear_dim)
        // --- Integration Parameters ---
        int stencil_genus,
        int grid_size,
        int max_primitives_per_ray,         // Max contributions considered per pixel
        float occupancy_threshold,
        // --- Background ---
        const float* bg_color,              // (CHANNELS)
        // --- Runtime Dimensions ---
        uint32_t input_feature_dim,         // F
        uint32_t output_feature_dim,        // F_out
        uint32_t hashgrid_levels,           // L
        uint32_t num_output_channels,       // CHANNELS (used for template selection)
        // --- FINAL Output Gradients (Accumulated atomically, MUST be zero-initialized before launch) ---
        float* dL_dLinearWeights,           // (input_linear_dim, output_linear_dim)
        float* dL_dLinearBias,              // (output_linear_dim)
        float* dL_dFeatureTable,            // (L * T * F) - Size matches feature_table_size_total
        float* dL_dprimitive_confidences    // (P * grid_volume) - Renamed from dL_dX
    );

} // namespace BACKWARD
