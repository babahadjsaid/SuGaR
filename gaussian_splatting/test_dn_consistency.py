#!/usr/bin/env python3
"""
Test script for depth-normal consistency loss integration
"""
import torch
import numpy as np
import math
from depth_normal_utils import depth_normal_consistency_loss

# Mock camera class for testing
class MockCamera:
    def __init__(self):
        self.image_width = 512
        self.image_height = 512
        
        # Create a proper camera transformation
        # View matrix (camera at z=10, looking down negative z)
        view_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, -10],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device='cuda')
        self.world_view_transform = view_matrix
        
        # Simple perspective projection matrix
        fov = math.pi / 4  # 45 degrees
        aspect = self.image_width / self.image_height
        near, far = 0.1, 100.0
        f = 1.0 / math.tan(fov / 2)
        
        proj_matrix = torch.tensor([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32, device='cuda')
        self.full_proj_transform = proj_matrix
        
        # Camera center (position in world space)
        self.camera_center = torch.tensor([0.0, 0.0, 10.0], device='cuda')
        
        # Field of view attributes (needed for some functions)
        self.FoVx = fov
        self.FoVy = fov

def test_depth_normal_consistency():
    """Test the depth-normal consistency loss computation"""
    print("Testing depth-normal consistency loss...")
    
    # Create mock data
    height, width = 512, 512
    
    # Create a simple depth map (plane at z=5)
    depth = torch.ones((1, height, width), device='cuda') * 5.0
    
    # Create a simple normal map (pointing towards camera)
    normal = torch.zeros((3, height, width), device='cuda')
    normal[2, :, :] = -1.0  # Z component pointing towards camera
    
    # Create mock camera
    camera = MockCamera()
    
    try:
        # Compute loss
        loss = depth_normal_consistency_loss(
            depth=depth,
            normal=normal,
            camera=camera,
            scale_rendered_normals=False,
            return_normal_maps=False,
        )
        
        print(f"✓ Depth-normal consistency loss computed successfully: {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"✗ Error computing depth-normal consistency loss: {e}")
        return False


def test_full_pipeline():
    """Test the full depth-normal consistency pipeline with rendered depth and normals"""
    from depth_normal_utils import render_depth_and_normal
    
    print("Testing full DN consistency pipeline...")
    
    # Mock Gaussian model that matches the real GaussianModel interface
    class MockGaussians:
        def __init__(self):
            n_gaussians = 1000
            self._xyz = torch.randn((n_gaussians, 3), device='cuda') * 2  # Scale up for visibility
            self._rotation = torch.randn((n_gaussians, 4), device='cuda')
            self._rotation = torch.nn.functional.normalize(self._rotation, dim=1)
            self._scaling = torch.ones((n_gaussians, 3), device='cuda') * 0.1  # Small scaling
            self._opacity = torch.ones((n_gaussians, 1), device='cuda') * 0.5
            self._features_dc = torch.ones((n_gaussians, 1, 3), device='cuda') * 0.5
            self._features_rest = torch.zeros((n_gaussians, 0, 3), device='cuda')
            self.active_sh_degree = 0
            
        @property
        def get_xyz(self):
            return self._xyz
            
        @property
        def get_rotation(self):
            return torch.nn.functional.normalize(self._rotation, dim=1)
            
        @property
        def get_scaling(self):
            return torch.exp(self._scaling)  # Assuming log scaling like in 3DGS
            
        @property
        def get_opacity(self):
            return torch.sigmoid(self._opacity)
    
    # Mock pipeline
    class MockPipe:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    # Create test objects
    camera = MockCamera()
    gaussians = MockGaussians()
    pipe = MockPipe()
    bg_color = torch.zeros(3, device='cuda')
    
    try:
        # Render depth and normals
        depth_img, normal_img = render_depth_and_normal(camera, gaussians, pipe, bg_color)
        
        # Prepare for DN consistency loss
        depth_tensor = depth_img[None]  # Shape: (1, height, width)
        normal_tensor = normal_img.permute(2, 0, 1)  # Shape: (3, height, width)
        
        # Compute depth-normal consistency loss
        dn_loss = depth_normal_consistency_loss(
            depth=depth_tensor,
            normal=normal_tensor,
            camera=camera,
            scale_rendered_normals=False,
            return_normal_maps=False,
        )
        
        print(f"✓ Full pipeline DN consistency loss: {dn_loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"✗ Error in full pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_render_depth_and_normal():
    """Test the proper render_depth_and_normal function"""    
    from depth_normal_utils import render_depth_and_normal, get_normals_from_smallest_axis
    
    print("Testing render_depth_and_normal...")
    
    # Mock Gaussian model that matches the real GaussianModel interface
    class MockGaussians:
        def __init__(self):
            n_gaussians = 1000
            self._xyz = torch.randn((n_gaussians, 3), device='cuda') * 2  # Scale up for visibility
            self._rotation = torch.randn((n_gaussians, 4), device='cuda')
            self._rotation = torch.nn.functional.normalize(self._rotation, dim=1)
            self._scaling = torch.ones((n_gaussians, 3), device='cuda') * 0.1  # Small scaling
            self._opacity = torch.ones((n_gaussians, 1), device='cuda') * 0.5
            self._features_dc = torch.ones((n_gaussians, 1, 3), device='cuda') * 0.5
            self._features_rest = torch.zeros((n_gaussians, 0, 3), device='cuda')
            self.active_sh_degree = 0
            
        @property
        def get_xyz(self):
            return self._xyz
            
        @property
        def get_rotation(self):
            return torch.nn.functional.normalize(self._rotation, dim=1)
            
        @property
        def get_scaling(self):
            return torch.exp(self._scaling)  # Assuming log scaling like in 3DGS
            
        @property
        def get_opacity(self):
            return torch.sigmoid(self._opacity)
    
    # Mock pipeline
    class MockPipe:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    # Create test objects
    camera = MockCamera()
    gaussians = MockGaussians()
    pipe = MockPipe()
    bg_color = torch.zeros(3, device='cuda')
    
    try:
        # Test normals extraction first
        normals = get_normals_from_smallest_axis(gaussians)
        print(f"✓ Normal extraction: {normals.shape}, norm range=[{normals.norm(dim=-1).min():.3f}, {normals.norm(dim=-1).max():.3f}]")
        
        # Test render_depth_and_normal function
        depth_img, normal_img = render_depth_and_normal(camera, gaussians, pipe, bg_color)
        print(f"✓ Render depth: {depth_img.shape}, range=[{depth_img.min():.3f}, {depth_img.max():.3f}]")
        print(f"✓ Render normals: {normal_img.shape}, norm range=[{normal_img.norm(dim=-1).min():.3f}, {normal_img.norm(dim=-1).max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in render_depth_and_normal: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing depth-normal consistency integration...\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Tests require GPU.")
        exit(1)
    
    # Run tests
    success = True
    success &= test_depth_normal_consistency()
    success &= test_render_depth_and_normal()
    success &= test_full_pipeline()
    
    if success:
        print("\n✅ All tests passed! Depth-normal consistency integration is working.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")