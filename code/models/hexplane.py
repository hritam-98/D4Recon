import torch
import torch.nn as nn
import torch.nn.functional as F

class HexPlaneField(nn.Module):
    """
    Spatiotemporal feature encoding using HexPlanes (6 2D planes).
    Planes: xy, xz, yz, xt, yt, zt
    """
    def __init__(self, bounds, resolution=[64, 64, 64, 20], feat_dim=32):
        super().__init__()
        self.bounds = bounds # [[min_x, max_x], ..., [min_t, max_t]]
        self.feat_dim = feat_dim
        
        # 6 Planes: XY, XZ, YZ, XT, YT, ZT
        self.planes = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.randn(1, feat_dim, resolution[0], resolution[1]) * 0.1)]) for _ in range(6)
        ])
        
    def normalize_coords(self, coords, bounds):
        # Normalize coordinates to [-1, 1] for grid_sample
        # coords: [N, 3] or [N, 4]
        min_b = bounds[:, 0]
        max_b = bounds[:, 1]
        norm = 2 * (coords - min_b) / (max_b - min_b) - 1
        return norm

    def forward(self, coords_st):
        """
        coords_st: [N, 4] (x, y, z, t)
        """
        N = coords_st.shape[0]
        norm_coords = self.normalize_coords(coords_st, self.bounds)
        
        # Extract pairs
        x, y, z, t = norm_coords[:, 0], norm_coords[:, 1], norm_coords[:, 2], norm_coords[:, 3]
        
        # Construct grid coordinates for sampling: [1, 1, N, 2]
        # Note: grid_sample expects (x, y) order
        pairs = [
            torch.stack([x, y], dim=-1), # XY
            torch.stack([x, z], dim=-1), # XZ
            torch.stack([y, z], dim=-1), # YZ
            torch.stack([x, t], dim=-1), # XT
            torch.stack([y, t], dim=-1), # YT
            torch.stack([z, t], dim=-1)  # ZT
        ]
        
        feats = []
        for i, plane_param in enumerate(self.planes):
            grid = pairs[i].view(1, 1, -1, 2)
            # Sample: [1, C, 1, N] -> [N, C]
            f = F.grid_sample(plane_param[0], grid, align_corners=True).view(self.feat_dim, -1).T
            feats.append(f)
            
        # Combine features (concatenation)
        combined_feat = torch.cat(feats, dim=1) 
        return combined_feat