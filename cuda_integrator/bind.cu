// Assuming pybind11 or similar

// --- Binding for Forward ---
m.def("integrate_primitives", [](
    // <<< Keep existing arguments, including buffer tensors >>>
    torch::Tensor geomBufferTensor,
    torch::Tensor binningBufferTensor,
    torch::Tensor imgBufferTensor,
    // ... other existing forward arguments ...
    // Primitive Data
    const torch::Tensor& means3D,
    const torch::Tensor& primitive_confidences_flat,
    float primitive_scale,
    // Camera Parameters
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& cam_pos,
    float tanfovx, float tanfovy,
    int image_height, int image_width,
    float near_plane, float max_distance,
    // Hashgrid Data
    const torch::Tensor& feature_table,
    const torch::Tensor& resolution,
    const torch::Tensor& do_hash,
    const torch::Tensor& primes,
    int feature_offset,
    // MLP Data
    const torch::Tensor& linear_weights,
    const torch::Tensor& linear_bias,
    // Integration Parameters
    int stencil_genus, int grid_size, int max_primitives_per_ray, float occupancy_threshold,
    // Background Color
    const torch::Tensor& bg_color,
    // Runtime Dimensions
    uint32_t input_feature_dim, uint32_t output_feature_dim,
    uint32_t hashgrid_levels, uint32_t num_output_channels,
    // Misc
    bool debug
) {
    // --- Create resizeFunctional lambdas (assuming helper exists) ---
    // This part remains necessary for the forward call itself
    auto resize_fn = [](torch::Tensor& tensor_ref) { /* ... definition from previous example ... */ };
    auto geomFunc = resize_fn(geomBufferTensor);
    auto binningFunc = resize_fn(binningBufferTensor);
    auto imgFunc = resize_fn(imgBufferTensor);

    // --- Prepare Output Buffers (allocated based on dimensions) ---
    int P_input = means3D.size(0); // Get P from input tensor
    int num_pixels = image_height * image_width;
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(means3D.device());
    torch::Tensor out_color = torch::zeros({num_pixels, (long)num_output_channels}, options_float);
    torch::Tensor out_features = torch::zeros({num_pixels, (long)output_feature_dim}, options_float);
    torch::Tensor visibility_info = torch::zeros({image_height, image_width}, options_float); // Assuming HxW

    // --- Call C++ Forward ---
    // Pass the lambdas and other arguments as expected by Integrator::forward
    int num_rendered = CudaIntegrator::Integrator::forward(
        geomFunc, binningFunc, imgFunc,
        P_input, image_width, image_height,
        means3D.data_ptr<float>(), primitive_scale,
        viewmatrix.data_ptr<float>(), projmatrix.data_ptr<float>(), cam_pos.data_ptr<float>(),
        near_plane, max_distance, tanfovx, tanfovy,
        primitive_confidences_flat.data_ptr<float>(),
        feature_table.data_ptr<float>(),
        resolution.data_ptr<int>(), do_hash.data_ptr<int>(), primes.data_ptr<int>(),
        feature_offset,
        linear_weights.data_ptr<float>(), linear_bias.data_ptr<float>(),
        stencil_genus, grid_size, max_primitives_per_ray, occupancy_threshold,
        bg_color.data_ptr<float>(),
        out_color.data_ptr<float>(), out_features.data_ptr<float>(), visibility_info.data_ptr<float>(),
        input_feature_dim, output_feature_dim, hashgrid_levels, num_output_channels,
        nullptr, // radii_override
        debug,
        at::cuda::getCurrentCUDAStream()
    );

    // --- <<< Get final pointers AFTER forward call >>> ---
    uintptr_t geom_ptr_int = reinterpret_cast<uintptr_t>(geomFunc(0));
    uintptr_t binning_ptr_int = reinterpret_cast<uintptr_t>(binningFunc(0));
    uintptr_t image_ptr_int = reinterpret_cast<uintptr_t>(imgFunc(0));

    // --- Return num_rendered, outputs, and pointers ---
    // Note: We also need to return the output tensors allocated here
    return std::make_tuple(num_rendered, out_color, out_features, visibility_info,
                           geom_ptr_int, binning_ptr_int, image_ptr_int);

}, /* pybind11::arg(...) definitions matching the lambda inputs */);


// --- Binding for Backward ---
m.def("integrate_primitives_backward", [](
    // <<< Accept pointers as uintptr_t >>>
    uintptr_t geom_ptr_int,
    uintptr_t binning_ptr_int,
    uintptr_t image_ptr_int,
    // <<< Keep other existing backward arguments >>>
    // Input Gradients
    const torch::Tensor& dL_dout_color, // Make const ref
    const torch::Tensor& dL_dout_features, // Make const ref
    // Dimensions and Saved State (passed from Python ctx)
    int P, int num_rendered, int width, int height,
    const torch::Tensor& means3D, float primitive_scale,
    const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix, const torch::Tensor& camera_center_vec,
    float near_plane, float max_distance, float tan_fovx, float tan_fovy,
    const torch::Tensor& primitive_confidences, // Flat
    const torch::Tensor& feature_table,
    const torch::Tensor& resolution, const torch::Tensor& do_hash, const torch::Tensor& primes,
    int feature_offset,
    const torch::Tensor& linear_weights, const torch::Tensor& linear_bias,
    int stencil_genus, int grid_size, int max_primitives_per_ray, float occupancy_threshold,
    const torch::Tensor& bg_color,
    uint32_t input_feature_dim, uint32_t output_feature_dim,
    uint32_t hashgrid_levels, uint32_t num_output_channels,
    uint32_t feature_table_size,
    bool debug
    // <<< Output Gradient Tensors (passed from Python) >>>
    // These tensors are allocated in Python and passed here to be filled
    // torch::Tensor& dL_dmeans3D, // Pass by reference if modified in-place
    // torch::Tensor& dL_dprimitive_confidences,
    // torch::Tensor& dL_dfeature_table,
    // torch::Tensor& dL_dlinear_weights,
    // torch::Tensor& dL_dlinear_bias
) {
    // --- <<< Create dummy std::functions >>> ---
    auto make_dummy_func = [](uintptr_t ptr_int) { /* ... definition from previous example ... */ };
    auto dummyGeomFunc = make_dummy_func(geom_ptr_int);
    auto dummyBinningFunc = make_dummy_func(binning_ptr_int);
    auto dummyImgFunc = make_dummy_func(image_ptr_int);

    // --- Prepare Output Gradient Buffers (Allocate tensors to be filled) ---
    // The C++ backward function expects pointers to memory it can write to.
    // We allocate tensors here based on the shapes of the corresponding forward inputs.
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(means3D.device());
    auto options_like_means = means3D.options(); // Use options from a relevant input tensor
    auto options_like_conf = primitive_confidences.options();
    auto options_like_feat = feature_table.options();
    auto options_like_weights = linear_weights.options();
    auto options_like_bias = linear_bias.options();

    torch::Tensor dL_dmeans3D_out = torch::zeros_like(means3D, options_like_means);
    torch::Tensor dL_dprimitive_confidences_out = torch::zeros_like(primitive_confidences, options_like_conf); // Use flat shape
    torch::Tensor dL_dfeature_table_out = torch::zeros_like(feature_table, options_like_feat);
    torch::Tensor dL_dlinear_weights_out = torch.zeros_like(linear_weights, options_like_weights);
    torch::Tensor dL_dlinear_bias_out = torch.zeros_like(linear_bias, options_like_bias);


    // --- Call C++ Backward with dummy functions ---
    // Pass pointers to the data of the newly allocated output tensors
    CudaIntegrator::Integrator::backward(
        dummyGeomFunc, dummyBinningFunc, dummyImgFunc, // Pass dummy functions
        // Input Gradients
        dL_dout_color.data_ptr<float>(),
        dL_dout_features.data_ptr<float>(),
        // Dimensions and Saved State
        P, width, height,
        means3D.data_ptr<float>(), primitive_scale,
        viewmatrix.data_ptr<float>(), projmatrix.data_ptr<float>(), camera_center_vec.data_ptr<float>(),
        near_plane, max_distance, tan_fovx, tan_fovy,
        primitive_confidences.data_ptr<float>(), // Flat
        feature_table.data_ptr<float>(),
        resolution.data_ptr<int>(), do_hash.data_ptr<int>(), primes.data_ptr<int>(),
        feature_offset,
        linear_weights.data_ptr<float>(), linear_bias.data_ptr<float>(),
        stencil_genus, grid_size, max_primitives_per_ray, occupancy_threshold,
        bg_color.data_ptr<float>(),
        input_feature_dim, output_feature_dim, hashgrid_levels, num_output_channels,
        feature_table_size, // Pass calculated size
        num_rendered, // Pass num_rendered
        // Output Gradient Pointers (to the allocated tensors)
        dL_dmeans3D_out.data_ptr<float>(),
        dL_dprimitive_confidences_out.data_ptr<float>(),
        dL_dfeature_table_out.data_ptr<float>(),
        dL_dlinear_weights_out.data_ptr<float>(),
        dL_dlinear_bias_out.data_ptr<float>(),
        // Misc
        debug,
        at::cuda::getCurrentCUDAStream()
    );

    // --- Return the computed gradient tensors ---
    return std::make_tuple(
        dL_dmeans3D_out,
        dL_dprimitive_confidences_out,
        dL_dfeature_table_out,
        dL_dlinear_weights_out,
        dL_dlinear_bias_out
    );

}, /* pybind11::arg(...) definitions matching the lambda inputs */); 