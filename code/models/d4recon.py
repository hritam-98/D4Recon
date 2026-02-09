import torch
import torch.nn as nn
import numpy as np
from .hexplane import HexPlaneField
from .deformation import DeformationNetwork

# Import KNN for initialization
try:
    from simple_knn._C import distCUDA2
except ImportError:
    print("Warning: simple_knn not found. Initialization may fail.")
    def distCUDA2(x): return torch.zeros((x.shape[0],))

class DReconModel(nn.Module):
    def __init__(self, sh_degree=3):
        super().__init__()
        self.sh_degree = sh_degree
        
        # --- Canonical Gaussians ---
        self._xyz = nn.Parameter(torch.empty(0))
        self._features_dc = nn.Parameter(torch.empty(0))
        self._features_rest = nn.Parameter(torch.empty(0))
        self._scaling = nn.Parameter(torch.empty(0))
        self._rotation = nn.Parameter(torch.empty(0))
        self._opacity = nn.Parameter(torch.empty(0))
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        
        # --- Deformation Modules (Dual-Stage) ---
        # 1. Spatial Deformation (corrects multiview inconsistency)
        # 2. Temporal Deformation (models dynamics)
        
        self.spatial_bounds = nn.Parameter(torch.tensor([[-10, 10]]*3), requires_grad=False) 
        self.time_bounds = nn.Parameter(torch.tensor([[0, 1]]), requires_grad=False)
        
        self.hexplane_spatial = None 
        self.hexplane_temporal = None
        self.mlp_spatial = None
        self.mlp_temporal = None
        
        self.setup_deformation_modules()

    def setup_deformation_modules(self):
        # 6 planes * 32 dim = 192 input dim
        self.hexplane_spatial = HexPlaneField(torch.cat([self.spatial_bounds, torch.zeros(1, 2).to(self.spatial_bounds)], dim=0))
        self.mlp_spatial = DeformationNetwork(192)
        
        self.hexplane_temporal = HexPlaneField(torch.cat([self.spatial_bounds, self.time_bounds], dim=0))
        self.mlp_temporal = DeformationNetwork(192)

    def create_from_pcd(self, pcd, cam_infos, time_duration):
        """
        Initialization logic (Eq 2)
        pcd: Initial point cloud object (expects .points and .colors attributes)
        """
        fused_point_cloud = pcd.points
        fused_colors = pcd.colors
        
        print(f"Number of points at initialization: {fused_point_cloud.shape[0]}")

        fused_point_cloud = torch.tensor(fused_point_cloud).float().cuda()
        fused_colors = torch.tensor(fused_colors).float().cuda()
        
        # Initialize attributes
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        # SH Colors
        features = torch.zeros((fused_colors.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_colors
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        # Scaling (log space)
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        
        # Rotation (quaternions)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        # Opacity (logit space, init to 0.1)
        opacities = torch.logit(0.1 * torch.ones((fused_point_cloud.shape[0], 1), device="cuda"))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # Update bounds for HexPlanes
        min_xyz = fused_point_cloud.min(dim=0)[0] - 1.0
        max_xyz = fused_point_cloud.max(dim=0)[0] + 1.0
        self.spatial_bounds = nn.Parameter(torch.stack([min_xyz, max_xyz], dim=1), requires_grad=False)
        self.time_bounds = nn.Parameter(torch.tensor([[0.0, time_duration]]).cuda(), requires_grad=False)
        
        # Re-init deformation with correct bounds
        self.setup_deformation_modules()
        self.hexplane_spatial.to("cuda")
        self.mlp_spatial.to("cuda")
        self.hexplane_temporal.to("cuda")
        self.mlp_temporal.to("cuda")

    def get_deformed_gaussians(self, t_value):
        """
        Eq 3: G_Dyn = G_t + Delta G_t
        Returns deformed properties for current time t
        """
        N = self._xyz.shape[0]
        xyz = self._xyz
        t_vec = torch.ones((N, 1), device=xyz.device) * t_value
        
        coords_st = torch.cat([xyz, t_vec], dim=1)
        
        # 1. Spatial Deformation (Static correction)
        # Pass 0 for t to spatial encoder to enforce static nature relative to time
        coords_s = torch.cat([xyz, torch.zeros_like(t_vec)], dim=1)
        feat_s = self.hexplane_spatial(coords_s)
        delta_s = self.mlp_spatial(feat_s)
        
        # 2. Temporal Deformation
        feat_t = self.hexplane_temporal(coords_st)
        delta_t = self.mlp_temporal(feat_t)
        
        # Combine
        d_xyz = self._xyz + delta_s['d_xyz'] + delta_t['d_xyz']
        d_scaling = self._scaling + delta_s['d_scale'] + delta_t['d_scale']
        d_rotation = self._rotation + delta_s['d_rot'] + delta_t['d_rot']
        d_opacity = self._opacity + delta_s['d_opacity'] + delta_t['d_opacity']
        
        # For color, we add offset to DC
        d_features_dc = self._features_dc + (delta_s['d_color'] + delta_t['d_color']).unsqueeze(1)
        
        return d_xyz, d_scaling, d_rotation, d_opacity, d_features_dc, self._features_rest