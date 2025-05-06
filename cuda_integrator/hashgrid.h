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
    const uint32_t F,                   // Feature dimension
    const uint32_t L,                   // Number of levels
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
                    // Layout: [Entry * F * L + Feature * L + Level]
                    uint64_t base_idx = (uint64_t)feature_entry_index * F * L;

                    for (uint32_t f = 0; f < F; ++f) {
                        float grad_contribution = dL_dPhi_level[f] * weight_corner;
                        uint64_t write_idx = base_idx + (uint64_t)f * L + level;

                        // Bounds check before atomicAdd (already present, good)
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
// Takes runtime dimensions as arguments. Assumes feature_table layout: entry * F * L + f * L + level
__forceinline__ __device__ void hashInterpolate(
    const float3& pos,
    const int level,
    const float* __restrict__ feature_table, // Base pointer for the entire table
    const int* __restrict__ resolutions, // Array of resolutions per level
    const int feature_offset, // Max entries per level (T parameter)
    const int do_hash, // Flag indicating if hashing is used for this level
    const int* __restrict__ primes, // Hashing primes {p1, p2, p3}
    float* __restrict__ output_features, // Output array (size input_feature_dim)
    // Runtime dimensions
    const uint32_t input_feature_dim,   // F
    const uint32_t hashgrid_levels
) {
    // Get resolution for this level
    int res = resolutions[level];

    // --- Calculate grid coordinates and weights (Must match hashComputeGradient) ---
    float level_scale = powf(2.0f, (float)level); // MUST BE CONSISTENT with hashComputeGradient
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
                float w = wx * wy * wz;

                int3 corner_idx = { grid_min.x + dx, grid_min.y + dy, grid_min.z + dz };

                // Get feature index for this corner
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
                    // Accumulate weighted features
                    // Indexing: entry_index*F*L + f*L + level
                    uint64_t base_idx = (uint64_t)feature_entry_index * input_feature_dim * hashgrid_levels;
                    for (uint32_t f = 0; f < input_feature_dim; f++) {
                        uint64_t feature_idx = base_idx + (uint64_t)f * hashgrid_levels + level;
                        
                    }
                }
            }
        }
    }
}


// Check if a 3D point (world space) is within the pyramidal frustum of a specific pixel.
__forceinline__ __device__ bool isPointInPyramidalFrustum(
    const float3& point_world,
    const float* __restrict__ viewmatrix, // Should be 4x4 view matrix
    const float* __restrict__ projmatrix, // 4x4 projection matrix
    const int W, const int H,
    const uint2& pixel_coord, // Integer coordinates of the pixel (x, y)
    const float near_plane, // Near plane distance (positive value)
    const float max_distance) // Far plane distance (positive value)
{
    // 1. Transform point to View Space using 4x4 matrix
    // Ensure viewmatrix is indeed 4x4 (passed from camera object?)
    float4 point_view_h = transformPoint4x4(point_world, viewmatrix);

    // 2. Check Near and Far Planes (in View Space Z)
    // View space Z is typically negative. We use -Z for distance.
    float view_depth = -point_view_h.z; // Use z from homogeneous coordinates before perspective divide
    if (view_depth < near_plane || view_depth > max_distance) {
        return false;
    }

    // 3. Transform point to Homogeneous Clip Space (World -> Clip directly)
    float4 point_clip = transformPoint4x4(point_world, projmatrix); // Assumes projmatrix includes view transform

    // 4. Check if behind camera (W <= 0)
    if (point_clip.w <= 1e-6f) {
        return false;
    }

    // 5. Perspective Divide to Normalized Device Coordinates (NDC) [-1, 1]
    float inv_w = 1.0f / point_clip.w;
    float ndc_x = point_clip.x * inv_w;
    float ndc_y = point_clip.y * inv_w;

    // 6. Convert NDC to Pixel Coordinates (Float) [0, W] x [0, H]
    float pix_fx = (ndc_x * 0.5f + 0.5f) * W;
    float pix_fy = (1.0f - (ndc_y * 0.5f + 0.5f)) * H; // Y flipped

    // 7. Check if the floating-point pixel coordinate falls within the boundaries of the target integer pixel
    bool within_x = (pix_fx >= pixel_coord.x && pix_fx < (pixel_coord.x + 1.0f));
    bool within_y = (pix_fy >= pixel_coord.y && pix_fy < (pixel_coord.y + 1.0f));

    return within_x && within_y;
}

// <<< NEW HELPER FUNCTIONS END >>> // (Keep this marker if it helps organization)
