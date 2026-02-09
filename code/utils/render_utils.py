import torch
import math
import torch.nn.functional as F

# Try to import standard 3DGS rasterizers
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("Warning: diff_gaussian_rasterization not found.")
    class GaussianRasterizer:
        def __init__(self, settings): pass
        def __call__(self, *args, **kwargs): return torch.zeros((3, 100, 100)), torch.zeros((1, 100, 100))
    class GaussianRasterizationSettings:
        def __init__(self, **kwargs): pass

def render(viewpoint_camera, model, t_value, bg_color, override_opacity=None):
    """
    Renders the scene at time t.
    override_opacity: Used for Hard Depth Guidance (Eq 4)
    """
    xyz, scaling, rotation, opacity, sh_dc, sh_rest = model.get_deformed_gaussians(t_value)
    
    # Activation
    opacity_activation = torch.sigmoid(opacity)
    
    # Hard Depth Guidance Logic: Force high opacity
    if override_opacity is not None:
        opacity_activation = torch.ones_like(opacity_activation) * override_opacity
        
    # Create rasterizer
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings)
    
    # Rasterize
    shs = torch.cat([sh_dc, sh_rest], dim=1)
    
    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=torch.zeros_like(xyz),
        shs=shs,
        colors_precomp=None,
        opacities=opacity_activation,
        scales=torch.exp(scaling),
        rotations=F.normalize(rotation),
        cov3D_precomp=None
    )
    
    # Simple depth rendering implementation:
    # Project centers to camera space
    view_matrix = viewpoint_camera.world_view_transform
    means3D_cam = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device='cuda')], dim=1) @ view_matrix
    depths = means3D_cam[:, 2:3] # Z-depth
    
    # Rasterize Depth (reuse rasterizer geometry, swap colors for depth)
    rendered_depth, _ = rasterizer(
        means3D=xyz,
        means2D=torch.zeros_like(xyz),
        shs=None,
        colors_precomp=depths.repeat(1, 3), # Broadcast depth to RGB
        opacities=opacity_activation,
        scales=torch.exp(scaling),
        rotations=F.normalize(rotation),
        cov3D_precomp=None
    )
    
    return {
        "render": rendered_image,
        "depth": rendered_depth[0:1, :, :], # Take one channel
        "viewspace_points": torch.zeros_like(xyz), # Placeholder
        "visibility_filter": radii > 0,
        "radii": radii
    }