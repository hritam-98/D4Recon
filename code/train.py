import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.drecon_model import DReconModel
from utils.render_utils import render
from utils.loss_utils import dual_scale_depth_loss, SDSLoss

def train_drecon(dataset, iterations=3000):
    # Setup
    drecon = DReconModel().cuda()
    print("Model initialized.")
    
    # Mock initialization logic here
    # drecon.create_from_pcd(...)

    # Optimizers
    # Separate params for alternating optimization
    params_base = [drecon._xyz, drecon._opacity, drecon._scaling, drecon._rotation, drecon._features_dc, drecon._features_rest]
    params_spatial = list(drecon.hexplane_spatial.parameters()) + list(drecon.mlp_spatial.parameters())
    params_temporal = list(drecon.hexplane_temporal.parameters()) + list(drecon.mlp_temporal.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': params_base, 'lr': 0.00016},
        {'params': params_spatial, 'lr': 0.001, 'name': 'spatial'},
        {'params': params_temporal, 'lr': 0.001, 'name': 'temporal'}
    ])
    
    sds_loss_fn = SDSLoss()
    
    # Loop
    progress_bar = tqdm(range(iterations))
    for i in progress_bar:
        # 1. Sample Batch (Mock Data)
        t = (i % 100) / 100.0
        gt_image = torch.rand((3, 512, 512)).cuda()
        gt_depth = torch.rand((1, 512, 512)).cuda()
        
        class MockCam:
            image_height=512
            image_width=512
            FoVx=1.0
            FoVy=1.0
            world_view_transform=torch.eye(4).cuda()
            full_proj_transform=torch.eye(4).cuda()
            camera_center=torch.zeros(3).cuda()
        view = MockCam()
        
        # 2. Alternating Optimization Strategy (Section 2.4)
        # "spatial and temporal fields are optimized alternately"
        optimize_spatial = (i % 2 == 0)
        
        # Zero grad
        optimizer.zero_grad()
        
        # 3. Render
        bg = torch.zeros(3).cuda()
        render_pkg = render(view, drecon, t, bg)
        image = render_pkg["render"]
        
        # 4. Losses
        
        # Photometric
        l1_loss = F.l1_loss(image, gt_image)
        
        # Dual-Scale Depth Guidance
        ddg_loss = dual_scale_depth_loss(render_pkg, gt_depth, drecon, view, t)
        
        # SDS Losses (Spatial or Temporal)
        sds_term = 0.0
        if optimize_spatial:
            # L_SDS-S
            sds_term = sds_loss_fn(image, t, view) 
        else:
            # L_SDS-T
            sds_term = sds_loss_fn(image, t, view)
            
        total_loss = l1_loss + 0.1 * ddg_loss + 0.01 * sds_term
        
        # 5. Backward & Step
        total_loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            progress_bar.set_description(f"Loss: {total_loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    print("Starting D'Recon Test Run...")
    try:
        train_drecon(None, iterations=10)
    except Exception as e:
        print(f"Run stopped (likely due to missing data/cuda): {e}")