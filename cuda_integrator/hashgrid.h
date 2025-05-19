#include "config.h" // For HASHGRID constants if needed
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio> // For printf
#include <cmath> // <<< Add for floorf, powf >>>
#include <vector_types.h> // <<< Add for float3, int3 >>>
#include "auxiliary.h" // <<< ADDED: Include for transformPoint4x4 if used here >>>

__device__ inline void hashComputeGradient(
    const float3& pos_world,
    const int level,
    const int* __restrict__ resolution_table, // Table of resolutions per level
    const bool use_hash,
    const int* __restrict__ primes,     // Hashing primes {p1, p2, p3}
    const float* __restrict__ dL_dPhi_level, // Input: Gradient w.r.t interpolated features for this level (size: F)
    const uint32_t F,                   // Feature dimension (features_per_level)
    const uint32_t L,                   // Number of levels (total_hashgrid_levels)
    const int feature_offset,           // Max entries per level (T parameter) - Used for hashing/dense grid size check
    const uint64_t feature_table_size,  // Total size of the feature table buffer (use uint64_t for large tables)
    float* __restrict__ dL_dfeature_table // Output: Global gradient buffer for features (atomicAdd target)
) {
    // --- Calculate grid coordinates and weights (Must match hashInterpolate) ---
    int res = resolution_table[level];
    float level_scale = powf(2.0f, (float)level); // MUST BE CONSISTENT with hashInterpolate
    float3 pos_level = make_float3(pos_world.x * level_scale, pos_world.y * level_scale, pos_world.z * level_scale);
    float3 floor_val = make_float3(floorf(pos_level.x), floorf(pos_level.y), floorf(pos_level.z));
    int3 grid_coords_base = make_int3((int)floor_val.x, (int)floor_val.y, (int)floor_val.z);
    float3 weights = make_float3(pos_level.x - floor_val.x, pos_level.y - floor_val.y, pos_level.z - floor_val.z);

    // --- Iterate over 8 corners ---
    for (int dz = 0; dz < 2; ++dz) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                float weight_corner = 1.0f;
                weight_corner *= (dx == 1) ? weights.x : (1.0f - weights.x);
                weight_corner *= (dy == 1) ? weights.y : (1.0f - weights.y);
                weight_corner *= (dz == 1) ? weights.z : (1.0f - weights.z);

                int3 corner_grid_coords = make_int3(grid_coords_base.x + dx, grid_coords_base.y + dy, grid_coords_base.z + dz);

                // --- Calculate feature table index (Must match hashInterpolate) ---
                uint32_t feature_entry_index;
                bool valid_index = true;

                if (use_hash) {
                    uint32_t hash_val = 0;
                    hash_val ^= (uint32_t)corner_grid_coords.x * primes[0];
                    hash_val ^= (uint32_t)corner_grid_coords.y * primes[1];
                    hash_val ^= (uint32_t)corner_grid_coords.z * primes[2];
                    feature_entry_index = hash_val % (uint32_t)feature_offset;
                } else {
                    if (corner_grid_coords.x < 0 || corner_grid_coords.x >= res ||
                        corner_grid_coords.y < 0 || corner_grid_coords.y >= res ||
                        corner_grid_coords.z < 0 || corner_grid_coords.z >= res) {
                        valid_index = false;
                    } else {
                        feature_entry_index = (uint32_t)corner_grid_coords.z * res * res + (uint32_t)corner_grid_coords.y * res + (uint32_t)corner_grid_coords.x;
                        if (feature_entry_index >= (uint32_t)feature_offset) {
                            valid_index = false;
                        }
                    }
                }

                // --- Distribute gradient if index is valid ---
                if (valid_index) {
                    // NEW Layout: [Entry][Level][Feature]
                    // Index: (entry_idx * L + level_idx) * F + feature_idx
                    uint64_t base_idx_for_entry_and_level = 
                        ((uint64_t)feature_entry_index * L + level) * F;

                    for (uint32_t f = 0; f < F; ++f) { // f is feature_idx_in_level
                        float grad_contribution = dL_dPhi_level[f] * weight_corner;
                        uint64_t write_idx = base_idx_for_entry_and_level + f;

                        // Bounds check before atomicAdd
                        if (write_idx < feature_table_size) {
                            atomicAdd(&dL_dfeature_table[write_idx], grad_contribution);
                        }
                    }
                }
            } // dx
        } // dy
    } // dz
}


// Hash interpolation function
// Takes runtime dimensions as arguments.
// NEW Assumed feature_table layout: [Entry][Level][Feature]
// (entry_idx * L + level_idx) * F + feature_idx
__forceinline__ __device__ void hashInterpolate(
    const float3& pos,
    const int level, // current_level
    const float* __restrict__ feature_table, // Base pointer for the entire table
    const int* __restrict__ resolutions, // Array of resolutions per level
    const int feature_offset, // Max entries per level (T parameter)
    const int do_hash, // Flag indicating if hashing is used for this level
    const int* __restrict__ primes, // Hashing primes {p1, p2, p3}
    float* __restrict__ output_features, // Output array (size input_feature_dim)
    // Runtime dimensions
    const uint32_t input_feature_dim,   // F (features_per_level)
    const uint32_t hashgrid_levels      // L (total_hashgrid_levels)
) {
    // Get resolution for this level
    int res = resolutions[level];

    // --- Calculate grid coordinates and weights (Must match hashComputeGradient) ---
    float level_scale = powf(2.0f, (float)level); 
    float3 pos_level = make_float3(pos.x * level_scale, pos.y * level_scale, pos.z * level_scale);
    float3 floor_val = make_float3(floorf(pos_level.x), floorf(pos_level.y), floorf(pos_level.z));
    int3 grid_min = make_int3((int)floor_val.x, (int)floor_val.y, (int)floor_val.z);
    float3 t = make_float3(pos_level.x - floor_val.x, pos_level.y - floor_val.y, pos_level.z - floor_val.z);

    // Initialize output features for this level to zero
    for (uint32_t f = 0; f < input_feature_dim; ++f) {
        output_features[f] = 0.0f;
    }

    // Trilinear interpolation over the 8 corners of the voxel
    for (int dx = 0; dx < 2; dx++) {
        float wx = (dx == 0) ? (1.0f - t.x) : t.x;
        for (int dy = 0; dy < 2; dy++) {
            float wy = (dy == 0) ? (1.0f - t.y) : t.y;
            for (int dz = 0; dz < 2; dz++) {
                float wz = (dz == 0) ? (1.0f - t.z) : t.z;
                float w = wx * wy * wz; // Combined weight for the corner

                int3 corner_idx = { grid_min.x + dx, grid_min.y + dy, grid_min.z + dz };

                uint32_t feature_entry_index;
                bool valid_index = true;

                if (do_hash) {
                    uint32_t hash_val = 0;
                    hash_val ^= (uint32_t)corner_idx.x * primes[0];
                    hash_val ^= (uint32_t)corner_idx.y * primes[1];
                    hash_val ^= (uint32_t)corner_idx.z * primes[2];
                    feature_entry_index = hash_val % feature_offset;
                } else {
                    if (corner_idx.x < 0 || corner_idx.x >= res ||
                        corner_idx.y < 0 || corner_idx.y >= res ||
                        corner_idx.z < 0 || corner_idx.z >= res) {
                        valid_index = false;
                    } else {
                        feature_entry_index = (uint32_t)corner_idx.z * res * res + (uint32_t)corner_idx.y * res + (uint32_t)corner_idx.x;
                        if (feature_entry_index >= (uint32_t)feature_offset) {
                            valid_index = false;
                        }
                    }
                }

                if (valid_index) {
                    // NEW Layout: [Entry][Level][Feature]
                    // Index: (entry_idx * L + level_idx) * F + feature_idx
                    uint64_t base_idx_for_entry_and_level = 
                        ((uint64_t)feature_entry_index * hashgrid_levels + level) * input_feature_dim;
                        
                    for (uint32_t f = 0; f < input_feature_dim; f++) { // f is feature_idx_in_level
                        uint64_t table_read_idx = base_idx_for_entry_and_level + f;
                        // Add bounds check if feature_table_size is available and if table_read_idx could exceed it.
                        // For now, assuming valid_index and feature_entry_index are within T,
                        // and F*L*T fits in memory.
                        output_features[f] += feature_table[table_read_idx] * w;
                    }
                }
            }
        }
    }
}

// Hash interpolation function
// MODIFIED: Writes directly to a target slice in a larger feature buffer (e.g., shared memory).
// NEW Assumed feature_table layout: [Entry][Level][Feature]
// (entry_idx * L + level_idx) * F + feature_idx
__forceinline__ __device__ void hashInterpolateShared(
    const float3& pos,
    const int current_processing_level, // Renamed from 'level' for clarity
    const float* __restrict__ feature_table, // Base pointer for the entire table
    const int* __restrict__ resolutions, // Array of resolutions per level
    const int feature_offset_T, // Max entries per level (T parameter)
    const int use_hash_for_this_level, // Flag indicating if hashing is used for this level
    const int* __restrict__ primes, // Hashing primes {p1, p2, p3}
    // --- Target for writing output ---
    float* __restrict__ sh_target_slice_for_grid_point, // Base pointer to where the current grid point's concatenated features start
    // --- Runtime dimensions for indexing and bounds ---
    const uint32_t features_per_level,   // F (runtime, features for a single level)
    const uint32_t total_hashgrid_levels, // L (runtime, total number of levels)
    // --- Bounds for writing into the target slice ---
    const uint32_t max_concatenated_features_in_slice, // Compile-time max size of the target slice (e.g., MAX_INPUT_LINEAR_DIM)
    const uint32_t runtime_concatenated_features_to_fill // Runtime total features to fill in the slice (e.g., input_linear_dim)
) {
    int res = resolutions[current_processing_level];
    float level_scale_factor = powf(2.0f, (float)current_processing_level);
    float3 pos_scaled_to_level = make_float3(pos.x * level_scale_factor, pos.y * level_scale_factor, pos.z * level_scale_factor);
    float3 floor_val = make_float3(floorf(pos_scaled_to_level.x), floorf(pos_scaled_to_level.y), floorf(pos_scaled_to_level.z));
    int3 grid_min_coords = make_int3((int)floor_val.x, (int)floor_val.y, (int)floor_val.z);
    float3 interp_weights_t = make_float3(pos_scaled_to_level.x - floor_val.x, pos_scaled_to_level.y - floor_val.y, pos_scaled_to_level.z - floor_val.z);

    for (uint32_t f_idx_in_level = 0; f_idx_in_level < features_per_level; ++f_idx_in_level) {
        float interpolated_value_for_feature_f = 0.0f;

        for (int dx_offset = 0; dx_offset < 2; dx_offset++) {
            float wx = (dx_offset == 0) ? (1.0f - interp_weights_t.x) : interp_weights_t.x;
            for (int dy_offset = 0; dy_offset < 2; dy_offset++) {
                float wy = (dy_offset == 0) ? (1.0f - interp_weights_t.y) : interp_weights_t.y;
                for (int dz_offset = 0; dz_offset < 2; dz_offset++) {
                    float wz = (dz_offset == 0) ? (1.0f - interp_weights_t.z) : interp_weights_t.z;
                    float corner_weight = wx * wy * wz;
                    int3 corner_coords = { grid_min_coords.x + dx_offset, grid_min_coords.y + dy_offset, grid_min_coords.z + dz_offset };
                    uint32_t feature_table_entry_idx;
                    bool is_valid_table_idx = true;

                    if (use_hash_for_this_level) {
                        uint32_t hash_val = 0;
                        hash_val ^= (uint32_t)corner_coords.x * primes[0];
                        hash_val ^= (uint32_t)corner_coords.y * primes[1];
                        hash_val ^= (uint32_t)corner_coords.z * primes[2];
                        feature_table_entry_idx = hash_val % feature_offset_T;
                    } else {
                        if (corner_coords.x < 0 || corner_coords.x >= res ||
                            corner_coords.y < 0 || corner_coords.y >= res ||
                            corner_coords.z < 0 || corner_coords.z >= res) {
                            is_valid_table_idx = false;
                        } else {
                            feature_table_entry_idx = (uint32_t)corner_coords.z * res * res + (uint32_t)corner_coords.y * res + (uint32_t)corner_coords.x;
                            if (feature_table_entry_idx >= (uint32_t)feature_offset_T) {
                                is_valid_table_idx = false;
                            }
                        }
                    }

                    if (is_valid_table_idx) {
                        // NEW Layout: [Entry][Level][Feature]
                        // Index: (entry_idx * L + level_idx) * F + feature_idx
                        uint64_t base_idx_for_entry_and_level = 
                            ((uint64_t)feature_table_entry_idx * total_hashgrid_levels + current_processing_level) * features_per_level;
                        
                        uint64_t table_read_idx = base_idx_for_entry_and_level + f_idx_in_level;
                        // Add bounds check if feature_table_size is available and if table_read_idx could exceed it.
                        // For now, assuming valid_index and feature_table_entry_idx are within T,
                        // and F*L*T fits in memory.
                        interpolated_value_for_feature_f += feature_table[table_read_idx] * corner_weight;
                    }
                } 
            } 
        } 

        uint32_t write_offset_in_concat_slice = (uint32_t)current_processing_level * features_per_level + f_idx_in_level;
        if (write_offset_in_concat_slice < runtime_concatenated_features_to_fill &&
            write_offset_in_concat_slice < max_concatenated_features_in_slice) {
            sh_target_slice_for_grid_point[write_offset_in_concat_slice] = interpolated_value_for_feature_f;
        }
    } 
}

// Projects a 3D world point to 2D screen coordinates (floating point)
// and checks depth and w component validity.
__forceinline__ __device__ float2 projectPointToScreenAndCheckDepth(
    const float3& point_world,
    const float* __restrict__ viewmatrix, // 4x4 view matrix
    const float* __restrict__ projmatrix, // 4x4 projection matrix (world to clip)
    const int W, const int H,
    const float near_plane, // Near plane distance (positive value)
    const float max_distance, // Far plane distance (positive value)
    bool& is_valid_projection // Output: true if point passes depth/w checks
) {
    is_valid_projection = false; // Assume invalid initially

    // 1. Transform point to View Space
    float4 point_view_h = transformPoint4x4(point_world, viewmatrix);

    // 2. Check Near and Far Planes (in View Space Z)
    float view_depth = -point_view_h.z;
    if (view_depth < near_plane || view_depth > max_distance) {
        return make_float2(-1.0f, -1.0f); // Invalid, return sentinel for screen coords
    }

    // 3. Transform point to Homogeneous Clip Space (World -> Clip)
    float4 point_clip = transformPoint4x4(point_world, projmatrix);

    // 4. Check if behind camera (W <= 0)
    if (point_clip.w <= 1e-6f) {
        return make_float2(-1.0f, -1.0f); // Invalid, return sentinel for screen coords
    }

    // 5. Perspective Divide to Normalized Device Coordinates (NDC) [-1, 1]
    float inv_w = 1.0f / point_clip.w;
    float ndc_x = point_clip.x * inv_w;
    float ndc_y = point_clip.y * inv_w;

    // 6. Convert NDC to Pixel Coordinates (Float) [0, W] x [0, H]
    float pix_fx = (ndc_x * 0.5f + 0.5f) * W;
    float pix_fy = (1.0f - (ndc_y * 0.5f + 0.5f)) * H; // Y flipped

    is_valid_projection = true; // All checks passed
    return make_float2(pix_fx, pix_fy);
}

// Check if a 3D point (world space) is within the pyramidal frustum of a specific pixel.
// This function uses projectPointToScreenAndCheckDepth and then does the 2D bounds check.
__forceinline__ __device__ bool isPointInPyramidalFrustum(
    const float3& point_world,
    const float* __restrict__ viewmatrix, 
    const float* __restrict__ projmatrix, 
    const int W, const int H,
    const uint2& pixel_coord,  // Integer coordinates of the pixel (x, y)
    const float near_plane, 
    const float max_distance)
{
    bool is_valid_proj; // This will be set by projectPointToScreenAndCheckDepth
    float2 pix_f = projectPointToScreenAndCheckDepth(
        point_world, viewmatrix, projmatrix, W, H, near_plane, max_distance, is_valid_proj
    );

    if (!is_valid_proj) { // If projection itself was invalid (e.g. out of depth, w<=0)
        return false;
    }

    // Check if the floating-point pixel coordinate falls within the boundaries of the target integer pixel
    bool within_x = (pix_f.x >= pixel_coord.x && pix_f.x < (pixel_coord.x + 1.0f));
    bool within_y = (pix_f.y >= pixel_coord.y && pix_f.y < (pixel_coord.y + 1.0f));

    return within_x && within_y;
}

// <<< NEW HELPER FUNCTIONS END >>> // (Keep this marker if it helps organization)
