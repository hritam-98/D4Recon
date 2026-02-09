import torch
import torch.nn as nn
import torch.nn.functional as F
from .render_utils import render

# Try importing diffusers for real SDS
try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: 'diffusers' library not found. SDSLoss will run in mock mode.")
    print("To enable full functionality: pip install diffusers transformers accelerate")

def dual_scale_depth_loss(model_render_dict, gt_depth_map, model, camera, t_value):
    """
    Computes L_DDG = ||D_HDG - D|| + ||D_SDG - D||
    """
    # 1. Soft Depth (D_SDG)
    soft_depth = model_render_dict['depth']
    
    # 2. Hard Depth (D_HDG)
    # Render with fixed high opacity (beta = 0.95 from paper text)
    hard_render_dict = render(camera, model, t_value, bg_color=torch.zeros(3).cuda(), override_opacity=0.95)
    hard_depth = hard_render_dict['depth']
    
    # Loss
    l_sdg = F.mse_loss(soft_depth, gt_depth_map)
    l_hdg = F.mse_loss(hard_depth, gt_depth_map)
    
    return l_sdg + l_hdg

class SDSLoss(nn.Module):
    """
    Real implementation of Score Distillation Sampling (SDS) Loss.
    Uses a pre-trained Diffusion model (surrogate for ArSDM) to guide the optimization.
    """
    def __init__(self, device='cuda', model_key="stabilityai/stable-diffusion-2-1-base"):
        super().__init__()
        self.device = device
        self.valid = False
        
        if DIFFUSERS_AVAILABLE:
            try:
                print(f"Loading SDS model: {model_key}...")
                # We use Stable Diffusion as a robust backbone. 
                # Note: The paper uses ArSDM. To replicate exactly, replace this pipeline 
                # with the specific ArSDM checkpoint/architecture if available.
                pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to(device)
                
                self.vae = pipe.vae
                self.unet = pipe.unet
                self.tokenizer = pipe.tokenizer
                self.text_encoder = pipe.text_encoder
                self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
                
                # Freeze parameters to save memory/compute
                self.vae.requires_grad_(False)
                self.unet.requires_grad_(False)
                self.text_encoder.requires_grad_(False)
                
                # Pre-compute text embedding. 
                # Paper mentions "semantic priors". We use a relevant medical prompt.
                # If ArSDM is unconditional or uses different conditioning, adjust here.
                prompt = "endoscopic view of biological tissue, high fidelity, medical imaging"
                text_input = self.tokenizer(
                    [prompt], 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                )
                with torch.no_grad():
                    self.text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
                    
                print("SDS Model loaded successfully.")
                self.valid = True
                del pipe # Cleanup shell wrapper
                
            except Exception as e:
                print(f"Failed to load Diffusion model components: {e}")
                self.valid = False
        else:
            self.valid = False

    def forward(self, image, t_scene, camera_condition):
        """
        Eq 7: grad = w(t) * (epsilon_phi - epsilon) * dI/dtheta
        Returns the gradient norm for logging, while populating gradients in the computation graph.
        """
        # Fallback if diffusers not installed or failed to load
        if not self.valid:
            grad = torch.randn_like(image) * 1e-4 
            image.backward(gradient=grad, retain_graph=True)
            return grad.norm()

        # 1. Prepare Image for VAE (512x512, Normalized [-1, 1])
        # Input image: [C, H, W] -> unsqueeze -> [1, C, H, W]
        if image.dim() == 3:
            img_batch = image.unsqueeze(0)
        else:
            img_batch = image
            
        # Interpolate to 512x512 (Standard SD resolution)
        img_resized = F.interpolate(img_batch, (512, 512), mode='bilinear', align_corners=False)
        
        # Normalize: 3DGS outputs [0, 1], VAE expects [-1, 1]
        img_norm = img_resized * 2.0 - 1.0

        # 2. Encode to Latents
        # We need gradients to flow back to the image, so we encode with autograd enabled
        # Note: VAE encode is memory intensive. 
        posterior = self.vae.encode(img_norm.half()).latent_dist
        latents = posterior.sample() * 0.18215 # Scaling factor for SD
        
        # 3. Sample Timestep t ~ U(min, max)
        # We sample a random timestep for the diffusion process
        t_diffusion = torch.randint(
            int(self.scheduler.config.num_train_timesteps * 0.02), 
            int(self.scheduler.config.num_train_timesteps * 0.98), 
            (1,), 
            device=self.device
        ).long()
        
        # 4. Add Noise to Latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t_diffusion)
        
        # 5. Predict Noise (Denoising Step)
        # In the paper, this is epsilon_phi(z_t; y, P_hat, t)
        # Standard SD doesn't take camera pose P_hat without adapters. 
        # We condition on text embeddings as a proxy for semantic consistency.
        with torch.no_grad():
            noise_pred = self.unet(
                noisy_latents, 
                t_diffusion, 
                encoder_hidden_states=self.text_embeddings
            ).sample

        # 6. Compute SDS Gradient
        # w(t) is a weighting function. For simplicity (and like DreamFusion), we set w(t) = 1
        # The gradient direction is (predicted_noise - added_noise)
        w = 1.0
        grad_direction = w * (noise_pred - noise)
        
        # 7. Backpropagate
        # We manually apply the gradient to the latents to backprop through the VAE encoder
        # to the rendered image, and subsequently to the 3DGS parameters.
        latents.backward(gradient=grad_direction, retain_graph=True)
        
        return grad_direction.norm()