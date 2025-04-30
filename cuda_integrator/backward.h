#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace BACKWARD {

    // Step 1: Backward Pass for Rendering (Alpha Blending)
    // Calculates dL/dFeatures_7D and dL/dAlphas from dL/dpixels
    // Uses tile ranges and point lists, similar to forward.
    // BLEND_DIM is assumed to be 7 based on dL_dFeatures_7D output.
    template <const int BLEND_DIM>
    void render(
        // --- Grid/Block/Tile Info ---
        const uint2* ranges,          // Tile ranges (start, end) in point_list (from forward sort)
        const uint32_t* point_list,   // Sorted list of primitive indices per tile (from forward sort)
        const int W, const int H,                 // Image dimensions
        const float* bg_color,        // Background color (size BLEND_DIM, assumed 7)

        // --- Per-Primitive Inputs (Saved/Recomputed from Forward) ---
        const float* primitive_features_input_7D, // Features (C) for each primitive ( P * 7 )
        const float* primitive_alphas_input,      // Alpha for each primitive ( P )

        // --- Forward Pass Outputs needed for Backward ---
        const float* final_Ts,        // Final transmittance T per pixel ( W * H )
        // const uint32_t* n_contrib,  // Often not strictly needed for backward, depends on implementation

        // --- Input Gradients ---
        const float* dL_dpixels_7D,   // Gradient of Loss w.r.t. output pixels ( W * H * 7 )

        // --- Output Gradients (to be computed) ---
        float* dL_dFeatures_7D,       // Gradient of Loss w.r.t. primitive features C ( P * 7 )
        float* dL_dAlphas,
        cudaStream_t stream = 0
    );


    // Step 2: Backward Pass for Alpha Activation (Sigmoid)
    // Calculates dL/dDensityFeature from dL/dAlphas assuming alpha = sigmoid(density_feature)
    void backwardAlphaActivation(
        int P, // Number of primitives
        const float* dL_dAlphas,             // Input: Gradient w.r.t. alpha (P)
        const float* primitive_alphas_input, // Input: Alpha values from forward (P)
        // Output
        float* dL_dDensityFeature,          // Output: Gradient w.r.t. density feature (P)
        cudaStream_t stream = 0
    );


    // Step 3: Construct 8D Gradient for Linear Layer Input
    // Combines dL/dFeatures_7D and dL_dDensityFeature
    void construct8DGradient(
        int P, // Number of primitives
        const float* dL_dFeatures_7D,    // Input: Gradient w.r.t. 7D features (P * 7)
        const float* dL_dDensityFeature, // Input: Gradient w.r.t. density feature (P)
        // Output
        float* dL_dOutput_8D,            // Output: Gradient w.r.t. linear layer output (P * 8)
        cudaStream_t stream = 0
    );


    // Step 4: Backward Pass for Linear Layer
    // Calculates dL/dInput (dL_dPrimitiveFeatures_D), dL/dWeights, dL/dBias
    // D is now a runtime argument
    void linearLayer(
        int P, // Number of primitives (Batch dimension)
        int D, // Input dimension (aggregated features) - NOW AN ARGUMENT
        const float* dL_dOutput_8D,             // Input: Gradient w.r.t. layer output (P * 8)
        const float* forward_primitive_features_D, // Input: Layer input from forward pass (P * D)
        const float* weights,                   // Input: Layer weights (D * 8)
        // Outputs
        float* dL_dPrimitiveFeatures_D,         // Output: Gradient w.r.t. layer input (P * D)
        float* dL_dWeights,                     // Output: Gradient w.r.t. weights (D * 8)
        float* dL_dBias,                        // Output: Gradient w.r.t. bias (8)
        cudaStream_t stream = 0
    );


    // Step 5: Backward Pass for Features (Φ) and Confidence (X) via Product Rule
    // Calculates dL/dFeatureTable, dL/dX, and dL/dPosWorld from dL/dPrimitiveFeatures_D.
    // Passes hashgrid parameters individually.
    // Dimensions D, L, D_per_level, GRID_SIZE_X/Y/Z are now runtime arguments
    void backwardFeatureAndConfidence(
        int P,
        int D,
        int L,
        int D_per_level,
        int GRID_SIZE_X,
        int GRID_SIZE_Y,
        int GRID_SIZE_Z,
        const float* dL_dPrimitiveFeatures_D,
        const float* confidenceGrids,
        const float* feature_table,
        // --- Individual Hashgrid Parameters ---
        const int* resolution_levels,
        const uint32_t feature_table_size_per_level,
        const int* do_hash,
        const int* primes,
        // --- End Hashgrid Parameters ---
        // Outputs
        float* dL_dFeatureTable,
        float* dL_dX,
        float* dL_dPosWorld,
        cudaStream_t stream = 0
    );

} // namespace BACKWARD
