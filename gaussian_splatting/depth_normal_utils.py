#
# Depth-Normal rendering utilities for Gaussian Splatting
# Based on SuGaR implementation
#

import torch
import math


def depths_to_points(view, depthmap):
    """Comes from 2DGS.

    Args:
        view (_type_): view camera
        depthmap (_type_): depth map tensor

    Returns:
        _type_: 3D points
    """
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth2normal_2dgs(view, depth):
    """Comes from 2DGS.
    
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depth_normal_consistency_loss(
    depth: torch.Tensor, 
    normal: torch.Tensor,
    camera,
    scale_rendered_normals=False,
    return_normal_maps=False
):
    """
    Computes a loss that enforces the consistency between the depth map and the normal map.

    Args:
        depth (torch.Tensor): Has shape (1, height, width).
        normal (torch.tensor): Has shape (3, height, width). Should be in view space.
        camera: Camera object

    Returns:
        Depth-normal consistency loss
    """
    
    # Compute the normals from the depth map in world space.
    normal_from_depth = depth2normal_2dgs(camera, depth)
    
    # Transform the normals from the depth map to the view space (COLMAP convention).
    normal_from_depth = (normal_from_depth @ camera.world_view_transform[:3,:3]).permute(2, 0, 1)
    
    if scale_rendered_normals:
        # We normalize the normals to have the same scale as the normals from the depth map.
        normal_view = ((normal - normal.mean()) / normal.std()) * normal_from_depth.std() + normal_from_depth.mean()
    else:
        normal_view = normal

    # Compute the error between the normals from the depth map and the rendered normals.    
    normal_error = (1 - (normal_view * normal_from_depth).sum(dim=0))
    
    if return_normal_maps:
        return normal_error, normal_view, normal_from_depth    
    return normal_error.mean()


def estimate_depth_from_gaussians(viewpoint_camera, gaussians):
    """
    Estimate depth map from Gaussian positions.
    This is a simplified approach that projects Gaussian centers to depth.
    """
    # Transform Gaussian centers to view space
    view_positions = gaussians.get_xyz() @ viewpoint_camera.world_view_transform[:3,:3].T + viewpoint_camera.world_view_transform[:3,3]
    depths = view_positions[:, 2]  # Z component is depth
    
    # Project to screen space to get pixel coordinates
    ndc_positions = view_positions @ viewpoint_camera.full_proj_transform[:3,:3].T + viewpoint_camera.full_proj_transform[:3,3]
    screen_positions = ndc_positions[:, :2] / ndc_positions[:, 2:3]
    
    # Convert to pixel coordinates
    pixel_x = (screen_positions[:, 0] + 1) * 0.5 * viewpoint_camera.image_width
    pixel_y = (screen_positions[:, 1] + 1) * 0.5 * viewpoint_camera.image_height
    
    # Create depth map by splatting depths to pixels
    depth_map = torch.zeros(viewpoint_camera.image_height, viewpoint_camera.image_width, device=depths.device)
    
    # Simple splatting - this is a basic approximation
    valid_mask = (pixel_x >= 0) & (pixel_x < viewpoint_camera.image_width) & \
                 (pixel_y >= 0) & (pixel_y < viewpoint_camera.image_height) & \
                 (depths > 0)
    
    if valid_mask.sum() > 0:
        valid_x = pixel_x[valid_mask].long()
        valid_y = pixel_y[valid_mask].long()
        valid_depths = depths[valid_mask]
        
        # Simple assignment - could be improved with proper splatting
        depth_map[valid_y, valid_x] = valid_depths
    
    return depth_map


def estimate_normals_from_gaussians(viewpoint_camera, gaussians):
    """
    Estimate normal map from Gaussian orientations.
    This uses the rotation quaternions to derive surface normals.
    """
    # Get rotation quaternions
    quaternions = gaussians.get_rotation()
    
    # Convert quaternions to rotation matrices
    # Simplified normal estimation from primary axis of rotation
    # In practice, you'd want a more sophisticated approach
    
    # Use the first axis of the rotation as the normal direction
    # This is a simplification - proper implementation would use surface derivatives
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Rotation matrix first column (simplified normal direction)
    normals = torch.stack([
        1 - 2*(y*y + z*z),
        2*(x*y + w*z),
        2*(x*z - w*y)
    ], dim=1)
    
    # Normalize
    normals = torch.nn.functional.normalize(normals, dim=1)
    
    # Transform to view space
    view_normals = normals @ viewpoint_camera.world_view_transform[:3,:3].T
    
    # Project to screen space similar to depth estimation
    view_positions = gaussians.get_xyz() @ viewpoint_camera.world_view_transform[:3,:3].T + viewpoint_camera.world_view_transform[:3,3]
    ndc_positions = view_positions @ viewpoint_camera.full_proj_transform[:3,:3].T + viewpoint_camera.full_proj_transform[:3,3]
    screen_positions = ndc_positions[:, :2] / ndc_positions[:, 2:3]
    
    # Convert to pixel coordinates
    pixel_x = (screen_positions[:, 0] + 1) * 0.5 * viewpoint_camera.image_width
    pixel_y = (screen_positions[:, 1] + 1) * 0.5 * viewpoint_camera.image_height
    
    # Create normal map
    normal_map = torch.zeros(viewpoint_camera.image_height, viewpoint_camera.image_width, 3, device=normals.device)
    
    # Simple splatting
    valid_mask = (pixel_x >= 0) & (pixel_x < viewpoint_camera.image_width) & \
                 (pixel_y >= 0) & (pixel_y < viewpoint_camera.image_height)
    
    if valid_mask.sum() > 0:
        valid_x = pixel_x[valid_mask].long()
        valid_y = pixel_y[valid_mask].long()
        valid_normals = view_normals[valid_mask]
        
        # Simple assignment
        normal_map[valid_y, valid_x] = valid_normals
    
    return normal_map


def get_normals_from_smallest_axis(gaussians):
    """
    Extract normals from Gaussians using the smallest axis approach (similar to SuGaR).
    
    Args:
        gaussians: Gaussian model
        
    Returns:
        normals: Tensor of shape [N, 3] containing normal vectors
    """
    # Get rotation quaternions and scaling
    rotations = gaussians.get_rotation  # [N, 4] - quaternions (w, x, y, z)
    scales = gaussians.get_scaling      # [N, 3] - scaling factors
    
    # Convert quaternions to rotation matrices
    # Quaternion format: [w, x, y, z]
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    
    # Build rotation matrix (3x3 for each Gaussian)
    rotation_matrices = torch.stack([
        torch.stack([1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
        torch.stack([2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)], dim=1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)], dim=1)
    ], dim=2)  # [N, 3, 3]
    
    # Find the axis corresponding to the smallest scaling factor
    smallest_axis_idx = scales.min(dim=-1)[1]  # [N] - indices of smallest scale
    
    # Extract the corresponding column from rotation matrix (the smallest axis)
    normals = rotation_matrices[torch.arange(rotation_matrices.shape[0]), :, smallest_axis_idx]
    
    return normals


def render_depth_and_normal(viewpoint_camera, gaussians, pipe, bg_color, scaling_modifier=1.0):
    """
    Render depth and normal maps using Gaussian rasterizer (similar to SuGaR approach).
    
    This function uses the rasterizer to blend/project depth and normals properly,
    rather than simple screen-space estimation.
    
    Args:
        viewpoint_camera: Camera object
        gaussians: Gaussian model
        pipe: Pipeline parameters
        bg_color: Background color tensor
        scaling_modifier: Scaling modifier
        
    Returns:
        depth_img: Depth image [H, W]
        normal_img: Normal image [H, W, 3]
    """
    from gaussian_renderer import render
    
    # Get Gaussian parameters
    means3D = gaussians.get_xyz
    
    # Get normals from smallest axis (similar to SuGaR's approach)
    world_normals = get_normals_from_smallest_axis(gaussians)
    
    # Get camera center and flip normals to face camera
    camera_center = viewpoint_camera.camera_center
    world_normals = world_normals * torch.sign(
        (world_normals * (camera_center - means3D)).sum(dim=-1, keepdim=True)
    )
    
    # Transform points to view space for depth
    view_positions = means3D @ viewpoint_camera.world_view_transform[:3,:3].T + viewpoint_camera.world_view_transform[:3,3]
    depth_values = view_positions[:, 2:3]  # Z component is depth
    
    # Transform normals to view space
    view_normals = world_normals @ viewpoint_camera.world_view_transform[:3,:3].T
    
    # Create depth and normal features to render (depth + normal_x + normal_y)
    dn_features = torch.cat([
        depth_values,                    # [N, 1] - depth 
        view_normals[:, :2],            # [N, 2] - normal x,y components
    ], dim=-1)  # [N, 3]
    
    # Create a temporary Gaussian model with DN features as colors
    # We'll use the existing render function but with custom colors
    temp_features_dc = dn_features[:, None, :]  # [N, 1, 3] to match expected format
    temp_features_rest = torch.zeros((gaussians.get_xyz.shape[0], 0, 3), device=gaussians.get_xyz.device)
    
    # Store original features
    orig_features_dc = gaussians._features_dc
    orig_features_rest = gaussians._features_rest
    orig_active_sh_degree = gaussians.active_sh_degree
    
    try:
        # Temporarily replace features with DN features
        gaussians._features_dc = temp_features_dc
        gaussians._features_rest = temp_features_rest
        gaussians.active_sh_degree = 0  # Use only DC component
        
        # Render with DN features
        render_pkg = render(viewpoint_camera, gaussians, pipe, bg_color, scaling_modifier, override_color=dn_features)
        dn_img = render_pkg["render"]  # [3, H, W]
        
        # Convert to [H, W, 3] format
        dn_img = dn_img.permute(1, 2, 0)  # [H, W, 3]
        
        # Extract depth
        depth_img = dn_img[..., 0]  # [H, W]
        
        # Extract normals and compute z component
        # We flip the x and y components to match the OpenGL convention,
        # And we compute the z component from the x and y components
        normal_xy = dn_img[..., 1:3]  # [H, W, 2]
        normal_z = -torch.sqrt(1 - torch.sum(normal_xy**2, dim=-1, keepdim=True).clamp_max(1.0))  # [H, W, 1]
        normal_img = torch.cat([
            -normal_xy,  # Flip x,y to match OpenGL convention
            normal_z
        ], dim=-1)  # [H, W, 3]
        
        return depth_img, normal_img
        
    finally:
        # Restore original features
        gaussians._features_dc = orig_features_dc
        gaussians._features_rest = orig_features_rest
        gaussians.active_sh_degree = orig_active_sh_degree