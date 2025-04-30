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
namespace cg = cooperative_groups;

// Commenting out defines related to rendering kernel for now
// #define MAX_INPUT_LINEAR_DIM 256 // Placeholder, adjust if needed
// #define MAX_INPUT_FEATURE_DIM 32 // Placeholder, adjust if needed


// <<< NEW HELPER FUNCTIONS START >>> (Commented out as they are for rendering)

// Hash interpolation function (from reference code)
// Takes runtime dimensions as arguments. Assumes feature_table layout: entry * F * L + f * L + level
__device__ void hashInterpolate(
    const float3& pos,
    const int level,
    const float* __restrict__ feature_table, // Base pointer for the entire table
    const int* __restrict__ resolutions, // Array of resolutions per level
    const int feature_offset, // Max entries per level (T parameter)
    const int do_hash, // Flag indicating if hashing is used for this level
    const int* __restrict__ primes, // Hashing primes {p1, p2, p3}
    float* __restrict__ output_features, // Output array (size input_feature_dim)
    // Runtime dimensions
    const uint32_t input_feature_dim,
    const uint32_t hashgrid_levels) {

    // Get resolution for this level
    int res = resolutions[level];

    // Calculate grid position relative to this level's grid (assuming pos is in [0,1]^3)
    // If pos is in world coords, it needs normalization first based on scene bounds.
    // Assuming pos is already normalized to [0, 1]^3 for hashgrid lookup.
    float3 grid_pos_norm = { pos.x * res, pos.y * res, pos.z * res };

    // Get integer grid coordinates of the bottom-left-back corner
    int3 grid_min = {
        static_cast<int>(floorf(grid_pos_norm.x)),
        static_cast<int>(floorf(grid_pos_norm.y)),
        static_cast<int>(floorf(grid_pos_norm.z))
    };

    // Calculate interpolation weights (distances from grid_min)
    float3 t = {
        grid_pos_norm.x - grid_min.x,
        grid_pos_norm.y - grid_min.y,
        grid_pos_norm.z - grid_min.z
    };

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
                float w = wx * wy * wz; // Interpolation weight for this corner

                // Calculate integer grid coordinates of the corner
                int3 corner_idx = { grid_min.x + dx, grid_min.y + dy, grid_min.z + dz };

                // Get feature index for this corner
                uint32_t feature_entry_index; // Index within this level's feature entries

                if (do_hash) {
                    // Hash-based indexing
                    uint32_t hash_val = 0;
                    hash_val ^= (uint32_t)corner_idx.x * primes[0];
                    hash_val ^= (uint32_t)corner_idx.y * primes[1];
                    hash_val ^= (uint32_t)corner_idx.z * primes[2];
                    feature_entry_index = hash_val % feature_offset; // feature_offset is T (hash table size)
                } else {
                    // Direct indexing (dense grid)
                    // Check bounds: corner coords must be within [0, res-1]
                    if (corner_idx.x < 0 || corner_idx.x >= res ||
                        corner_idx.y < 0 || corner_idx.y >= res ||
                        corner_idx.z < 0 || corner_idx.z >= res) {
                        // If out of bounds for dense grid, contribute zero
                        // (Or could implement clamping/wrapping if desired)
                        continue;
                    }
                    feature_entry_index = (uint32_t)corner_idx.z * res * res + (uint32_t)corner_idx.y * res + (uint32_t)corner_idx.x;
                     // Ensure index is within the expected dense grid size (res^3)
                    if (feature_entry_index >= (uint32_t)res*res*res) continue; // Should not happen if bounds check passes
                }

                // Accumulate weighted features
                // Indexing: entry_index*F*L + f*L + level
                // Assumes feature_table stores features grouped by entry, then feature dim, then level
                uint32_t base_idx = feature_entry_index * input_feature_dim * hashgrid_levels;
                for (uint32_t f = 0; f < input_feature_dim; f++) {
                     // Check bounds implicitly via loop condition
                     // Calculate the final index for the feature
                     uint32_t feature_idx = base_idx + f * hashgrid_levels + level;
                     // TODO: Add bounds check for feature_idx against total size of feature_table if necessary
                     output_features[f] += w * feature_table[feature_idx];
                }
            }
        }
    }
}


// Check if a 3D point (world space) is within the pyramidal frustum of a specific pixel.
// This is a simplified check using NDC coordinates. More robust checks might involve plane equations.
__device__ bool isPointInPyramidalFrustum(
    const float3& point_world,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const int W, const int H,
    const uint2& pixel_coord, // Integer coordinates of the pixel (x, y)
    const float near_plane, // Near plane distance (positive value)
    const float max_distance) // Far plane distance (positive value)
{
    // 1. Transform point to View Space
    float4 point_view_h = transformPoint4x4(point_world, viewmatrix);
    float3 point_view = { point_view_h.x, point_view_h.y, point_view_h.z };

    // 2. Check Near and Far Planes (in View Space Z)
    // View space Z is typically negative. We use -Z for distance.
    float view_depth = -point_view.z;
    if (view_depth < near_plane || view_depth > max_distance) {
        return false;
    }

    // 3. Transform point to Homogeneous Clip Space
    float4 point_clip = transformPoint4x4(point_world, projmatrix);

    // 4. Check if behind camera (W <= 0) - Optional but good practice
    if (point_clip.w <= 1e-6f) {
        return false;
    }

    // 5. Perspective Divide to Normalized Device Coordinates (NDC) [-1, 1]
    float inv_w = 1.0f / point_clip.w;
    float ndc_x = point_clip.x * inv_w;
    float ndc_y = point_clip.y * inv_w;
    // float ndc_z = point_clip.z * inv_w; // Z check already done via view_depth

    // 6. Convert NDC to Pixel Coordinates (Float) [0, W] x [0, H]
    // Assumes standard NDC to Pixel mapping (adjust if different)
    // Y is often flipped depending on API/convention (here assuming Y=0 is top)
    float pix_fx = (ndc_x * 0.5f + 0.5f) * W;
    float pix_fy = (1.0f - (ndc_y * 0.5f + 0.5f)) * H; // Y flipped

    // 7. Check if the floating-point pixel coordinate falls within the boundaries of the target integer pixel
    // Check if pix_f is within [pixel_coord.x, pixel_coord.x + 1) and [pixel_coord.y, pixel_coord.y + 1)
    // Add a small tolerance (epsilon) if needed for edge cases, but strict check is usually fine.
    bool within_x = (pix_fx >= pixel_coord.x && pix_fx < (pixel_coord.x + 1.0f));
    bool within_y = (pix_fy >= pixel_coord.y && pix_fy < (pixel_coord.y + 1.0f));

    return within_x && within_y;
}

// <<< NEW HELPER FUNCTIONS END >>>

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
    // Runtime dimensions
    const uint32_t input_feature_dim,       // Feature dimension F in hashgrid
    const uint32_t output_feature_dim,      // Feature dimension requested in output (excluding RGB, density)
    const uint32_t hashgrid_levels          // Number of hashgrid levels L
)
{
    // <<< Read camera center from pointer inside kernel >>>
    const float3 camera_center = {camera_center_vec[0], camera_center_vec[1], camera_center_vec[2]};

    // --- Define Maximum Compile-Time Dimensions ---
    // These MUST be >= the largest values passed as runtime arguments and match the wrapper
    // Adjust these based on your expected maximums to avoid stack overflow / excessive shared memory
    const uint32_t MAX_INPUT_FEATURE_DIM = 4;  // F_max: Max feature dim from hashgrid
    const uint32_t MAX_OUTPUT_FEATURE_DIM = 4; // F_out_max: Max requested output feature dim (excluding RGB, density)
    const uint32_t MAX_HASHGRID_LEVELS = 8;    // L_max: Max hashgrid levels
    const uint32_t MAX_GRID_SIZE = 5;           // Max dimension of confidence grid (e.g., 5 for 5x5x5) - ADJUST AS NEEDED
    const uint32_t MAX_GRID_VOLUME = MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE; // Max total elements in confidence grid

    // Max dimension of the MLP output (RGB + Density + Output Features)
    const uint32_t MAX_OUTPUT_LINEAR_DIM = CHANNELS + MAX_OUTPUT_FEATURE_DIM + 1;
    // Max dimension of the MLP input (Concatenated features from all levels)
    const uint32_t MAX_INPUT_LINEAR_DIM = MAX_INPUT_FEATURE_DIM * MAX_HASHGRID_LEVELS;

    // Max size for the stencil coefficients array (e.g., 5 for genus 2)
    const uint32_t MAX_STENCIL_SIZE = 5;

    // --- Runtime Dimension Calculations ---
    // Total number of elements in the confidence grid (e.g., 3*3*3 = 27)
    const uint32_t grid_volume = (uint32_t)grid_size * grid_size * grid_size;
    // Dimension of the MLP input layer (concatenated features)
    const uint32_t input_linear_dim = input_feature_dim * hashgrid_levels;
    // Dimension of the MLP output layer (RGB + Density + Output Features)
    const uint32_t output_linear_dim = CHANNELS + output_feature_dim + 1;

    // --- Shared Memory Allocation ---
    // Stencil coefficients (max size based on MAX_STENCIL_SIZE)
    __shared__ float stencil_coeffs_x[MAX_STENCIL_SIZE];
    __shared__ float stencil_coeffs_y[MAX_STENCIL_SIZE];
    __shared__ float stencil_coeffs_z[MAX_STENCIL_SIZE];
    __shared__ int shared_stencil_size; // Actual stencil size used (3 or 5)

    // MLP Weights and Biases (using MAX dimensions for allocation)
    // Weights: (input_linear_dim, output_linear_dim) -> flat size = MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM
    // Bias: (output_linear_dim) -> flat size = MAX_OUTPUT_LINEAR_DIM
    __shared__ float shared_linear_weights[MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM];
    __shared__ float shared_linear_bias[MAX_OUTPUT_LINEAR_DIM];

    // Confidence grid for the current primitive (max size based on MAX_GRID_VOLUME)
    __attribute__((shared)) float sh_confidence[MAX_GRID_VOLUME];

    // Occupancy grid derived from confidence (0 or 1 per grid point)
    // Needs to be accessible by all threads for gradient calculation
    __attribute__((shared)) float sh_pixel_occupancy[MAX_GRID_VOLUME];

    // --- Thread/Block Identification ---
    auto block = cg::this_thread_block();
    uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x; // Global tile index
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x; // Global pixel index (row-major)

    // Check if this thread is associated with a valid pixel
    bool inside = pix.x < W && pix.y < H;

    // --- Load Shared Data ---
    // Thread 0 of the block loads stencil coefficients
    if (block.thread_rank() == 0) {
        int stencil_size_local;
        // Determine stencil coefficients based on genus
        if (stencil_genus == 1) { // [-1/2, 0, 1/2]
            stencil_size_local = 3;
            // Check bounds before writing (using MAX_STENCIL_SIZE)
            if (stencil_size_local <= MAX_STENCIL_SIZE) {
                stencil_coeffs_x[0] = -0.5f; stencil_coeffs_x[1] = 0.0f; stencil_coeffs_x[2] = 0.5f;
                stencil_coeffs_y[0] = -0.5f; stencil_coeffs_y[1] = 0.0f; stencil_coeffs_y[2] = 0.5f;
                stencil_coeffs_z[0] = -0.5f; stencil_coeffs_z[1] = 0.0f; stencil_coeffs_z[2] = 0.5f;
            } else { // Handle error or default // stencil_size_local = 0; } // Avoid overflow
            } // Closing brace for if (stencil_size_local <= MAX_STENCIL_SIZE)
        } else if (stencil_genus == 2) { // [1/12, -2/3, 0, 2/3, -1/12]
            stencil_size_local = 5;
             // Check bounds before writing (using MAX_STENCIL_SIZE)
            if (stencil_size_local <= MAX_STENCIL_SIZE) {
                stencil_coeffs_x[0] = 1.0f/12.0f; stencil_coeffs_x[1] = -2.0f/3.0f; stencil_coeffs_x[2] = 0.0f; stencil_coeffs_x[3] = 2.0f/3.0f; stencil_coeffs_x[4] = -1.0f/12.0f;
                stencil_coeffs_y[0] = 1.0f/12.0f; stencil_coeffs_y[1] = -2.0f/3.0f; stencil_coeffs_y[2] = 0.0f; stencil_coeffs_y[3] = 2.0f/3.0f; stencil_coeffs_y[4] = -1.0f/12.0f;
                stencil_coeffs_z[0] = 1.0f/12.0f; stencil_coeffs_z[1] = -2.0f/3.0f; stencil_coeffs_z[2] = 0.0f; stencil_coeffs_z[3] = 2.0f/3.0f; stencil_coeffs_z[4] = -1.0f/12.0f;
            } else { // Handle error or default // stencil_size_local = 0; } // Avoid overflow
            } // Closing brace for else
        } else { // Default to genus 1 if invalid input
            stencil_size_local = 3;
             // Check bounds before writing (using MAX_STENCIL_SIZE)
            if (stencil_size_local <= MAX_STENCIL_SIZE) {
                stencil_coeffs_x[0] = -0.5f; stencil_coeffs_x[1] = 0.0f; stencil_coeffs_x[2] = 0.5f;
                stencil_coeffs_y[0] = -0.5f; stencil_coeffs_y[1] = 0.0f; stencil_coeffs_y[2] = 0.5f;
                stencil_coeffs_z[0] = -0.5f; stencil_coeffs_z[1] = 0.0f; stencil_coeffs_z[2] = 0.5f;
            } else { // Handle error or default // stencil_size_local = 0; } // Avoid overflow
            } // Closing brace for if (stencil_size_local <= MAX_STENCIL_SIZE)
        }
        shared_stencil_size = stencil_size_local; // Store the actual size used
    }

    // Cooperatively load linear layer weights and biases
    // Total elements based on RUNTIME dimensions
    int total_weights = input_linear_dim * output_linear_dim;
    for (int i = block.thread_rank(); i < total_weights; i += block.size()) {
        // Indexing into shared memory uses MAX dimensions layout for safety
        // Assumes linear_weights is ordered correctly (output_dim major, input_dim minor)
        // shared_linear_weights[output_idx * MAX_IN + input_idx] = linear_weights[output_idx * input_linear_dim + input_idx]
        // Simplified loop assumes flat layout matches shared layout up to total_weights
        // Bounds check against the statically allocated size
        if (i < MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM) {
             shared_linear_weights[i] = linear_weights[i];
        }
    }
    int total_biases = output_linear_dim;
     for (int i = block.thread_rank(); i < total_biases; i += block.size()) {
        // Indexing into shared memory uses MAX dimensions layout
        // Bounds check against the statically allocated size
         if (i < MAX_OUTPUT_LINEAR_DIM) {
            shared_linear_bias[i] = linear_bias[i];
         }
    }

    block.sync(); // Ensure all shared data (stencil, weights, bias) is loaded

    // Get stencil size from shared memory (now available to all threads)
    const int stencil_size = shared_stencil_size; // Use the value loaded by thread 0
    const int stencil_offset = (stencil_size - 1) / 2; // e.g., 1 for size 3, 2 for size 5

    // --- Per-Pixel Initialization ---
    float T = 1.0f; // Transmittance accumulator
    float C[CHANNELS]; // Color accumulator (template param size)
    // Feature accumulator - Use MAX size for stack allocation
    float F[MAX_OUTPUT_FEATURE_DIM]; // Accumulated features

    // Initialize accumulators
    #pragma unroll
    for(int i=0; i<CHANNELS; ++i) C[i] = 0.0f;
    // Loop uses runtime output_feature_dim for actual features needed
    for(uint32_t i=0; i<output_feature_dim; ++i) F[i] = 0.0f;

    uint32_t contributor_count = 0;
    float last_view_depth = near_plane; // Initialize depth for delta_t calculation

    // --- Process Primitives for this Tile ---
    uint2 range = ranges[tile_idx]; // Get the start/end indices for this tile from the sorted list

    for (int i = range.x; i < range.y; ++i) {

        // --- Early Exit Conditions ---
        if (!inside || contributor_count >= max_primitives_per_ray || T < 1e-4f) {
            // Stop processing if pixel is outside image bounds,
            // max contributors reached, or ray is already opaque.
            break;
        }

        // --- Get Primitive Data ---
        uint32_t primitive_idx = point_list[i]; // Get the original index of the primitive
        float3 center = { // Get world-space center
            primitive_centers[primitive_idx * 3 + 0],
            primitive_centers[primitive_idx * 3 + 1],
            primitive_centers[primitive_idx * 3 + 2]
        };

        // --- Calculate Depth and Step Size (delta_t) ---
        // Transform center to view space to get depth
        float3 p_view = transformPoint4x3(center, viewmatrix);
        // View space depth is typically negative Z
        float view_depth = p_view.z;

        // // Check near/far plane culling based on view depth
        // if (view_depth < near_plane || view_depth > max_distance) {
        //     continue; // Skip primitive if its center is outside the view depth range
        // }

        // Calculate step size along the ray based on view depth difference
        float delta_t_orig = max(0.0f, view_depth - last_view_depth);

        // <<< ADD DEBUG OFFSET >>>
        float delta_t = delta_t_orig;

        // --- Initialize Primitive Feature Integral ---
        // Accumulator for integrated features over the primitive's volume
        // Use MAX size for stack allocation
        float primitive_feature_integral[MAX_INPUT_LINEAR_DIM];
        // Initialize using runtime dimension
        for(uint32_t k=0; k<input_linear_dim; ++k) primitive_feature_integral[k] = 0.0f;

        // --- Integrate Over Primitive's Confidence Grid (Two-Pass Method) ---
        // Step size between grid points
        const float grid_step = (grid_size > 1) ? primitive_scale / (float)(grid_size - 1) : primitive_scale;
        // Offset to center the grid sampling around the primitive center
        const float grid_offset_factor = (grid_size - 1) / 2.0f;
        // Base index for global confidence data
        const uint64_t base_conf_idx_global = (uint64_t)primitive_idx * grid_volume; // Use grid_volume

        // == Cooperative Loading of Confidence Grid ==
        // Loop iterates up to the actual grid_volume needed for this primitive
        for (uint32_t j = block.thread_rank(); j < grid_volume; j += block.size()) {
            // Bounds check shared memory write against MAX size
            if (j < MAX_GRID_VOLUME) {
                // Read from global memory. Assume primitive_confidences is large enough
                // for base_conf_idx_global + j based on primitive_idx.
                sh_confidence[j] = primitive_confidences[base_conf_idx_global + j];
            }
        }
        block.sync(); // Ensure all confidence values are loaded

        // == Pass 1: Calculate Per-Pixel Occupancy ==
        // Loop iterates up to the actual grid_volume
        for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
            // Bounds check shared memory access against MAX size
            if (flat_idx >= MAX_GRID_VOLUME) continue;

            // Convert flat index to 3D grid coordinates
            int z = flat_idx / (grid_size * grid_size);
            int rem = flat_idx % (grid_size * grid_size);
            int y = rem / grid_size;
            int x = rem % grid_size;

            char occupancy = 0; // Default to non-occupied
            float confidence = sh_confidence[flat_idx]; // Read from shared mem (already bounds checked)

            if (confidence > occupancy_threshold) {
                // Calculate world position only if confidence is high enough
                float sample_rel_x = (x - grid_offset_factor) * grid_step;
                float sample_rel_y = (y - grid_offset_factor) * grid_step;
                float sample_rel_z = (z - grid_offset_factor) * grid_step;
                float3 sample_pos_world = {
                    center.x + sample_rel_x,
                    center.y + sample_rel_y,
                    center.z + sample_rel_z
                };

                // Check if the world sample point lies within the current pixel's frustum
                if (isPointInPyramidalFrustum(sample_pos_world, viewmatrix, projmatrix, W, H, pix, near_plane, max_distance)) {
                    occupancy = 1;
                }
            }
            // Write to shared memory (already bounds checked)
            sh_pixel_occupancy[flat_idx] = occupancy;
        }
        block.sync(); // Ensure all occupancy values are calculated and stored

        // == Pass 2: Calculate Gradient & Accumulate Features ==
        // Loop iterates up to the actual grid_volume
        for (uint32_t flat_idx = block.thread_rank(); flat_idx < grid_volume; flat_idx += block.size()) {
             // Bounds check shared memory access against MAX size
            if (flat_idx >= MAX_GRID_VOLUME) continue;

            // Convert flat index to 3D grid coordinates
            int z = flat_idx / (grid_size * grid_size);
            int rem = flat_idx % (grid_size * grid_size);
            int y = rem / grid_size;
            int x = rem % grid_size;

            // --- Calculate Occupancy Gradient (using Stencil on Shared Occupancy) ---
            float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

            // Apply stencil using sh_pixel_occupancy
            for (int s = 0; s < stencil_size; s++) {
                int sx = s - stencil_offset;
                int sy = s - stencil_offset;
                int sz = s - stencil_offset;

                // Neighbor coordinates
                int nx = x + sx;
                int ny = y + sy;
                int nz = z + sz;

                // Calculate flat neighbor indices for shared memory access
                uint32_t flat_nx_idx = z * grid_size * grid_size + y * grid_size + nx;
                uint32_t flat_ny_idx = z * grid_size * grid_size + ny * grid_size + x;
                uint32_t flat_nz_idx = nz * grid_size * grid_size + y * grid_size + x;

                // Check bounds and get neighbor occupancy from shared memory
                // Also check shared memory bounds against MAX_GRID_VOLUME
                char occ_x_neighbor = (nx >= 0 && nx < grid_size && flat_nx_idx < MAX_GRID_VOLUME) ? sh_pixel_occupancy[flat_nx_idx] : 0;
                char occ_y_neighbor = (ny >= 0 && ny < grid_size && flat_ny_idx < MAX_GRID_VOLUME) ? sh_pixel_occupancy[flat_ny_idx] : 0;
                char occ_z_neighbor = (nz >= 0 && nz < grid_size && flat_nz_idx < MAX_GRID_VOLUME) ? sh_pixel_occupancy[flat_nz_idx] : 0;


                // Accumulate gradient components using shared stencil coefficients
                grad_x += stencil_coeffs_x[s] * (float)occ_x_neighbor;
                grad_y += stencil_coeffs_y[s] * (float)occ_y_neighbor;
                grad_z += stencil_coeffs_z[s] * (float)occ_z_neighbor;
            }

            // Calculate total gradient magnitude (L1 norm)
            float occupancy_grad_magnitude = fabsf(grad_x) + fabsf(grad_y) + fabsf(grad_z);

            // Skip if occupancy gradient is negligible
            if (occupancy_grad_magnitude < 1e-6f) {
                continue;
            }
            
            // --- Fetch Features from Hashgrid (Φ) ---
            // Use MAX size for stack allocation
            float sample_features[MAX_INPUT_LINEAR_DIM]; // Combined features from all levels
            // Initialize using runtime dimension
            for(uint32_t f=0; f<input_linear_dim; ++f) sample_features[f] = 0.0f;

            // Calculate world position (needed for hash lookup)
            float sample_rel_x = (x - grid_offset_factor) * grid_step;
            float sample_rel_y = (y - grid_offset_factor) * grid_step;
            float sample_rel_z = (z - grid_offset_factor) * grid_step;
            float3 sample_pos_world = {
                center.x + sample_rel_x,
                center.y + sample_rel_y,
                center.z + sample_rel_z
            };

            
            // Loop through hashgrid levels
            for (int level = 0; level < hashgrid_levels; ++level) {
                // Use MAX size for stack allocation
                float level_features[MAX_INPUT_FEATURE_DIM]; // Features for this level

                // Call hashInterpolate with WORLD coordinates (assuming hashInterpolate handles this)
                hashInterpolate(
                    sample_pos_world,   // Use world position directly
                    level,
                    feature_table,
                    resolution,
                    feature_offset,
                    do_hash[level],
                    primes,
                    level_features,
                    input_feature_dim,
                    hashgrid_levels
                );

                // Concatenate features
                for (uint32_t f = 0; f < input_feature_dim; f++) {
                    uint32_t target_idx = level * input_feature_dim + f;
                    if (target_idx < input_linear_dim) {
                        sample_features[target_idx] = level_features[f];
                    }
                }
            } // End hashgrid level loop

            // --- Accumulate Weighted Features ---
            // Read confidence from shared memory (already bounds checked)
            float current_confidence = sh_confidence[flat_idx];
            float weight = occupancy_grad_magnitude * current_confidence;

            // Accumulate features weighted by the sample weight
            for (uint32_t f = 0; f < input_linear_dim; ++f) {
                // Accumulation is thread-local, no sync needed inside this loop
                primitive_feature_integral[f] += sample_features[f] * weight;
            }

        } // End Pass 2 Loop (flat_idx)
        // Note: No block sync needed here as primitive_feature_integral is thread-local


        // --- Apply Linear Layer (MLP) ---
        // Use MAX size for stack allocation
        
        float primitive_outputs[MAX_OUTPUT_LINEAR_DIM]; // RGB/Color + Features + Density
        // Initialize using runtime dimension
        for(uint32_t k=0; k<output_linear_dim; ++k) primitive_outputs[k] = 0.0f;

        // Matrix multiplication: outputs = weights * integrated_features + bias
        // Loop uses runtime dimensions
        for (uint32_t out_idx = 0; out_idx < output_linear_dim; out_idx++) {
            float dot_prod = 0.0f;
            for (uint32_t in_idx = 0; in_idx < input_linear_dim; in_idx++) {
                // Indexing into shared weights uses MAX dimensions layout
                // Ensure weight layout matches: shared_weights[out * MAX_IN + in]
                // Bounds check shared mem access against static allocation size
                if (out_idx * MAX_INPUT_LINEAR_DIM + in_idx < MAX_INPUT_LINEAR_DIM * MAX_OUTPUT_LINEAR_DIM) {
                     dot_prod += shared_linear_weights[out_idx * MAX_INPUT_LINEAR_DIM + in_idx] * primitive_feature_integral[in_idx];
                }
            }
            // Indexing into shared bias uses MAX dimensions layout
            // Bounds check shared mem access against static allocation size
            if (out_idx < MAX_OUTPUT_LINEAR_DIM) {
                primitive_outputs[out_idx] = dot_prod + shared_linear_bias[out_idx];
            } else {
                 primitive_outputs[out_idx] = dot_prod; // Or handle error if bias is expected
            }
        }

        // --- Apply Activations ---
        // Sigmoid for Color (first CHANNELS elements) and Features (next output_feature_dim elements)
        // Loop combined for color and features, up to CHANNELS + output_feature_dim
        for (uint32_t k = 0; k < CHANNELS + output_feature_dim; ++k) {
             // Check bounds against runtime output dimension
             if (k < output_linear_dim) {
                primitive_outputs[k] = (1.0f / (1.0f + expf(-primitive_outputs[k])));
             }
        }
        // Exp for density (last element)
        float primitive_density = 0.0f;
        uint32_t density_idx = CHANNELS + output_feature_dim;
        if (density_idx < output_linear_dim) { // Check if density channel exists
            primitive_density = expf(primitive_outputs[density_idx]);
            primitive_density = max(0.0f, primitive_density); // Ensure non-negative density
        }


        // --- Extract Activated Outputs ---
        float primitive_rgb[CHANNELS]; // Color (template param size)
        // Use MAX size for stack allocation
        float primitive_features[MAX_OUTPUT_FEATURE_DIM]; // Features

        #pragma unroll
        for(int c=0; c<CHANNELS; ++c) {
            primitive_rgb[c] = primitive_outputs[c];
        }
        // Loop uses runtime output_feature_dim
        for(uint32_t f=0; f<output_feature_dim; ++f) {
            uint32_t source_idx = CHANNELS + f;
            if (source_idx < output_linear_dim) { // Check bounds
                primitive_features[f] = primitive_outputs[source_idx];
            } else {
                primitive_features[f] = 0.0f; // Default if out of bounds
            }
        }

        // --- Calculate Alpha and Composite ---
        // Use the modified delta_t
        float alpha = 1.0f - expf(-primitive_density * delta_t);
        alpha = min(max(alpha, 0.0f), 1.0f);
        float weight = T * alpha;

        // Accumulate Color
        #pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
            C[c] += weight * primitive_rgb[c];
        }

        // Accumulate Features
        // Loop uses runtime output_feature_dim
        for (uint32_t f = 0; f < output_feature_dim; f++) {
            F[f] += weight * primitive_features[f];
        }

        // Update Transmittance: T_new = T_old * (1 - alpha)
        T *= (1.0f - alpha);

        // --- Update State for Next Primitive ---
        last_view_depth = view_depth; // Still update with the original depth
        contributor_count++;
    } // --- End Loop Over Primitives (i) ---

    // --- Finalize Pixel Color and Features ---
    if (inside) {
        // Add background contribution based on final transmittance
        #pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
            // Handle potential case where bg_color might be NULL
            float bg = (bg_color != nullptr) ? bg_color[c] : 0.0f;
            C[c] += T * bg;
            // Write final color (Output is CHW: channel * H * W + pixel_id)
            out_color[c * H * W + pix_id] = C[c];
        }

        // Write accumulated features (Output is FHW: feature * H * W + pixel_id)
        if (out_features != nullptr) {
            // Loop uses runtime output_feature_dim
            for (uint32_t f = 0; f < output_feature_dim; f++) {
                // Background features are assumed to be 0
                out_features[f * H * W + pix_id] = F[f];
            }
        }

        // Write visibility info (e.g., 1 - T)
        if (visibility_info != nullptr) {
            visibility_info[pix_id] = 1.0f - T;
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
	const float* camera_center_vec, // Keep as float*
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
    const float occupancy_threshold, // NEW Parameter
	const float* bg_color,
	float* out_color,
	float* out_features,
	float* visibility_info,
	const uint32_t input_feature_dim,
	const uint32_t output_feature_dim,
	const uint32_t hashgrid_levels,
	const uint32_t num_output_channels // Matches header
)
{
    printf("[FORWARD::render] About to launch renderCUDA kernel...\n"); // <<< Should be reached now >>>
    fflush(stdout); // Ensure the print buffer is flushed

	// Determine which kernel template specialization to launch based on num_output_channels
	if (num_output_channels == 3) {
		renderCUDA<3> <<<grid, block>>> (
			ranges,
			point_list,
			W, H,
			viewmatrix,
			projmatrix,
			camera_center_vec, // <<< PASS float* pointer directly >>>
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

    // Optional: Synchronize here if you want to ensure kernel completion before this function returns
    // cudaDeviceSynchronize();
    // printf("[FORWARD::render] renderCUDA kernel finished execution (if synchronized).\n");
    // fflush(stdout);


	// CHECK_CUDA(,debug); // Removed this problematic check
}

// ===================================================================================
// == End of Rendering Code                                                        ==
// ===================================================================================

// Wrapper for the preprocess kernel launch. Updated signature.
void FORWARD::preprocess(int P, // Removed D, M
	const float* means3D,
	// Removed scales, rotations, opacities, shs, clamped, cov3D_precomp, colors_precomp, cam_pos
	const float primitive_scale,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	// Removed cov3Ds, rgb, conic_opacity
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > ( // Removed template arg <NUM_CHANNELS>
		P, // Removed D, M
		means3D,
		primitive_scale,
		// Removed scales, scale_modifier, rotations, opacities, shs, clamped,
		// cov3D_precomp, colors_precomp, cam_pos
		viewmatrix,
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		// Removed cov3Ds, rgb, conic_opacity
		grid,
		tiles_touched,
		prefiltered
		);
}

// Removed placeholder comments for volume integration kernel
// <<< VOLUME INTEGRATION KERNEL START >>>
// <<< VOLUME INTEGRATION KERNEL END >>>

// ... rest of the file (preprocessCUDA kernel, FORWARD::preprocess wrapper, FORWARD::render wrapper) ...