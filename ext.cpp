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

#include <torch/extension.h>
// Updated include
#include "integrate_primitives.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Updated function names and pointers
  m.def("integrate_primitives", &IntegratePrimitivesCUDA, "Integrate Primitives CUDA forward");
  m.def("integrate_primitives_backward", &IntegratePrimitivesBackwardCUDA, "Integrate Primitives CUDA backward");
  m.def("mark_visible", &markVisible, "Mark Visible Primitives CUDA");
}