#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
# from . import _C # This line was likely intended if _C was the name
import os
# import cuda_integrator # Remove this incorrect import
from . import _C as cuda_integrator # Import the compiled extension as _C and alias it
from torch.autograd import Function
import numpy as np
from dataclasses import dataclass, field

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

@dataclass
class IntegrationSettings:
    # Camera parameters
    viewmatrix: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    projmatrix: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    cam_pos: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    tanfovx: float = 1.0
    tanfovy: float = 1.0
    image_height: int = 512
    image_width: int = 512
    near_plane: float = 0.1
    max_distance: float = 100.0
    # Primitive parameters
    primitive_scale: float = 0.01
    # Hashgrid parameters
    feature_table: torch.Tensor = field(default_factory=lambda: torch.empty(0)) # Placeholder
    resolution: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32)) # Placeholder
    do_hash: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32)) # Placeholder
    primes: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32)) # Placeholder
    feature_offset: int = 0 # T
    # MLP parameters
    linear_weights: torch.Tensor = field(default_factory=lambda: torch.empty(0)) # Placeholder
    linear_bias: torch.Tensor = field(default_factory=lambda: torch.empty(0)) # Placeholder
    # Integration parameters
    stencil_genus: int = 1
    grid_size: int = 3
    max_primitives_per_ray: int = 64
    occupancy_threshold: float = 0.5
    # Background color
    bg_color: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    # Runtime dimensions (deduced or set explicitly)
    input_feature_dim: int = 2 # F
    output_feature_dim: int = 8 # Output feature dim (excluding RGB, density)
    hashgrid_levels: int = 16 # L
    num_output_channels: int = 3 # RGB
    # Misc
    debug: bool = False

class _IntegratePrimitives(Function):
    @staticmethod
    def forward(
        ctx,
        # --- Inputs requiring gradients (potentially) ---
        means3D,               # (P, 3)
        primitive_confidences, # (P, G, G, G) or (P, G^3)
        feature_table,         # (T * L * F) - Assuming flat layout
        linear_weights,        # (input_linear_dim, output_linear_dim)
        linear_bias,           # (output_linear_dim)
        # --- Other inputs (packed in settings) ---
        integration_settings,
    ):
        # Ensure inputs are contiguous and correct type
        means3D = means3D.contiguous().float()
        primitive_confidences = primitive_confidences.contiguous().float()
        feature_table = feature_table.contiguous().float()
        linear_weights = linear_weights.contiguous().float()
        linear_bias = linear_bias.contiguous().float()

        # --- Prepare arguments for C++ call ---
        # Flatten confidence grids if necessary and save original shape
        P_input = means3D.shape[0]
        G = integration_settings.grid_size
        original_confidences_shape = None # Store original shape if 4D
        if primitive_confidences.dim() == 4:
            original_confidences_shape = primitive_confidences.shape
            P_conf, G1, G2, G3 = primitive_confidences.shape
            assert P_conf == P_input, f"Primitive count mismatch: means3D ({P_input}) vs confidences ({P_conf})"
            assert G1 == G2 == G3 == G, f"Confidence grid dimensions mismatch: expected {G}, got ({G1}, {G2}, {G3})"
            primitive_confidences_flat = primitive_confidences.view(P_conf, -1).contiguous()
        elif primitive_confidences.dim() == 2:
            P_conf, G_flat = primitive_confidences.shape
            assert P_conf == P_input, f"Primitive count mismatch: means3D ({P_input}) vs confidences ({P_conf})"
            assert G_flat == G**3, f"Confidence grid flat dimension mismatch: expected {G**3}, got {G_flat}"
            primitive_confidences_flat = primitive_confidences # Assume already flat (P, G^3)
        else:
            raise ValueError(f"primitive_confidences has unexpected shape: {primitive_confidences.shape}")


        # Ensure hashgrid params are contiguous int tensors
        resolution = integration_settings.resolution.contiguous().int()
        do_hash = integration_settings.do_hash.contiguous().int()
        primes = integration_settings.primes.contiguous().int()

        # Ensure background color is contiguous float tensor
        bg_color = integration_settings.bg_color.contiguous().float()

        # Ensure camera matrices are contiguous float tensors
        viewmatrix = integration_settings.viewmatrix.contiguous().float()
        projmatrix = integration_settings.projmatrix.contiguous().float()
        cam_pos = integration_settings.cam_pos.contiguous().float()

        # <<< ADD PYTHON CHECKS for cam_pos >>>
        if not isinstance(cam_pos, torch.Tensor):
            raise TypeError(f"cam_pos must be a torch.Tensor, got {type(cam_pos)}")
        if not cam_pos.is_cuda:
            raise TypeError(f"cam_pos must be a CUDA tensor, but device is {cam_pos.device}")
        if cam_pos.numel() < 3:
             raise ValueError(f"cam_pos must have at least 3 elements, but has shape {cam_pos.shape}")
        if cam_pos.dtype != torch.float32:
             print(f"Warning: cam_pos dtype is {cam_pos.dtype}, ensuring float32.") # Or raise error
             cam_pos = cam_pos.float()
        cam_pos = cam_pos.contiguous() # Ensure contiguous
        print(f"Python check: cam_pos is valid CUDA tensor: shape={cam_pos.shape}, device={cam_pos.device}")
        # <<< END PYTHON CHECKS >>>

        # --- Argument List for C++ Forward Function ---
        # Order must match integrate_primitives.h forward
        args = (
            # Primitive Data
            means3D,
            primitive_confidences_flat,
            integration_settings.primitive_scale,
            # Camera Parameters
            viewmatrix,
            projmatrix,
            cam_pos,
            integration_settings.tanfovx,
            integration_settings.tanfovy,
            integration_settings.image_height,
            integration_settings.image_width,
            integration_settings.near_plane,
            integration_settings.max_distance,
            # Hashgrid Data
            feature_table, # Assuming already flat (T*L, F) or similar
            resolution,
            do_hash,
            primes,
            integration_settings.feature_offset, # T
            # MLP Data
            linear_weights,
            linear_bias,
            # Integration Parameters
            integration_settings.stencil_genus,
            integration_settings.grid_size,
            integration_settings.max_primitives_per_ray,
            integration_settings.occupancy_threshold,
            # Background Color
            bg_color,
            # Runtime Dimensions
            integration_settings.input_feature_dim,
            integration_settings.output_feature_dim,
            integration_settings.hashgrid_levels,
            integration_settings.num_output_channels,
            # Misc
            integration_settings.debug
        )

        # Invoke C++/CUDA integrator forward pass
        if cuda_integrator is None:
            raise ImportError("CUDA extension not loaded.")

        # The C++ function now returns:
        # (num_rendered, out_color, out_features, visibility_info, geomBuffer, binningBuffer, imgBuffer)
        num_rendered, out_color, out_features, visibility_info, \
        geomBuffer, binningBuffer, imgBuffer = cuda_integrator.integrate_primitives(*args)

        # --- Save tensors and context for backward pass ---
        ctx.save_for_backward(
            # Original Inputs needed for backward
            means3D, primitive_confidences_flat, feature_table, linear_weights, linear_bias,
            viewmatrix, projmatrix, cam_pos, resolution, do_hash, primes, bg_color,
            # State Buffers from forward
            geomBuffer, binningBuffer, imgBuffer
        )
        # Store non-tensor settings and intermediate info in ctx
        ctx.integration_settings = integration_settings
        ctx.num_rendered = num_rendered # Store this potentially useful info
        ctx.original_confidences_shape = original_confidences_shape # Store shape for grad reshaping

        # Return the rendered outputs relevant to the graph
        return out_color, out_features, visibility_info

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_features, grad_out_visibility):
        # --- Retrieve saved tensors and settings ---
        ( means3D, primitive_confidences_flat, feature_table, linear_weights, linear_bias,
          viewmatrix, projmatrix, cam_pos, resolution, do_hash, primes, bg_color,
          geomBuffer, binningBuffer, imgBuffer ) = ctx.saved_tensors
        settings = ctx.integration_settings
        original_confidences_shape = ctx.original_confidences_shape

        # --- Check if backward pass is implemented ---
        if not hasattr(cuda_integrator, "integrate_primitives_backward"):
             raise NotImplementedError("Backward pass for volume integration is not implemented in the CUDA extension.")

        # Ensure input gradients are contiguous
        grad_out_color = grad_out_color.contiguous()
        grad_out_features = grad_out_features.contiguous()
        # grad_out_visibility is often not needed or None, handle appropriately if used

        # --- Prepare arguments for C++ backward call ---
        # Order must match integrate_primitives.h backward
        grad_args = (
            # Input Gradients
            grad_out_color,
            grad_out_features,
            # Saved State Buffers
            geomBuffer,
            binningBuffer,
            imgBuffer,
            # Saved Original Inputs
            means3D, primitive_confidences_flat, settings.primitive_scale,
            viewmatrix, projmatrix, cam_pos,
            settings.tanfovx, settings.tanfovy,
            settings.image_height, settings.image_width,
            settings.near_plane, settings.max_distance,
            feature_table, resolution, do_hash, primes, settings.feature_offset,
            linear_weights, linear_bias,
            settings.stencil_genus, settings.grid_size, settings.max_primitives_per_ray,
            settings.occupancy_threshold, bg_color,
            # Runtime Dimensions
            settings.input_feature_dim, settings.output_feature_dim,
            settings.hashgrid_levels, settings.num_output_channels,
            feature_table.numel(), # Pass total size
            # Misc
            settings.debug
        )

        # --- Call C++ backward function ---
        # Returns grads for: means3D, primitive_confidences, feature_table, linear_weights, linear_bias
        dL_dmeans3D, dL_dprimitive_confidences_flat, dL_dfeature_table, \
        dL_dlinear_weights, dL_dlinear_bias = cuda_integrator.integrate_primitives_backward(*grad_args)

        # --- Reshape confidence gradient if necessary ---
        if original_confidences_shape is not None:
            dL_dprimitive_confidences = dL_dprimitive_confidences_flat.view(original_confidences_shape)
        else:
            dL_dprimitive_confidences = dL_dprimitive_confidences_flat

        # --- Return gradients ---
        # Must return gradients for each input of the forward function in the same order
        # Return None for inputs that don't require gradients (like integration_settings)
        return (
            dL_dmeans3D,
            dL_dprimitive_confidences,
            dL_dfeature_table,
            dL_dlinear_weights,
            dL_dlinear_bias,
            None # Gradient for integration_settings is None
        )

class VolumeIntegrator(nn.Module):
    def __init__(self, integration_settings: IntegrationSettings):
        super().__init__()
        self.integration_settings = integration_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        if cuda_integrator is None:
            raise ImportError("CUDA extension not loaded.")
        with torch.no_grad():
            settings = self.integration_settings
            # Call the C++ function using the alias
            visible = cuda_integrator.mark_visible( # Use the alias
                positions.contiguous().float(), # Ensure contiguous float
                settings.viewmatrix.contiguous().float(),
                settings.projmatrix.contiguous().float())

        return visible

    def forward(self,
                means3D,               # (P, 3)
                primitive_confidences, # (P, G, G, G) or (P, G^3)
                # feature_table,         # Passed via settings
                # linear_weights,        # Passed via settings
                # linear_bias            # Passed via settings
               ):

        # Retrieve tensors from settings that might require gradients
        # (or are large and shouldn't be copied repeatedly)
        feature_table = self.integration_settings.feature_table
        linear_weights = self.integration_settings.linear_weights
        linear_bias = self.integration_settings.linear_bias

        # Invoke the autograd Function
        # Pass all tensors that might require gradients, plus the settings object
        return _IntegratePrimitives.apply(
            means3D,
            primitive_confidences,
            feature_table,
            linear_weights,
            linear_bias,
            self.integration_settings # Pass the settings object
        )

