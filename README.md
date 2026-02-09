# D4Recon accepted at MICCAI 2025
Code will be updated soon. 
Please contact hbasak@cs.stonybrook.edu for questions


D4Recon: Dual-stage Deformation and Dual-scale Depth Guidance for Endoscopic Reconstruction

Implementation of D4Recon, a dynamic 3D Gaussian Splatting framework designed for high-fidelity endoscopic reconstruction. This method addresses the challenges of irreversible tissue deformations and sparse viewpoints in surgical scenes by introducing Dual-stage Spatiotemporal Deformation modeling and Dual-scale Depth Guidance.

📄 Abstract

Deformable tissue reconstruction in endoscopy is vital for surgery, yet current methods struggle with high-fidelity reconstruction of irreversible tissue deformations. D4Recon proposes a dynamic 3D Gaussian Splatting paradigm with two core innovations:

Dual-stage Deformation Modeling: Separates spatial deformations (to correct static multiview inconsistencies) from temporal deformations (to model dynamic tissue interactions).

Dual-scale Depth Guidance: Enforces global consistency via "Hard Depth" constraints while refining local details using "Soft Depth" gradients.

📂 Project Structure

D-Recon/
├── models/
│   ├── drecon_model.py   # Main class coordinating deformation and 3DGS
│   ├── hexplane.py       # HexPlane feature encoding (Spatial & Temporal)
│   └── deformation.py    # MLP for decoding features into Gaussian offsets
├── utils/
│   ├── render_utils.py   # Rasterization wrappers and depth rendering logic
│   └── loss_utils.py     # Dual-scale Depth Loss and SDS Loss wrappers
├── train.py              # Main training loop with alternating optimization
├── requirements.txt      # Python dependencies
└── README.md


🛠️ Installation

Prerequisites

OS: Linux (Recommended) or Windows

GPU: NVIDIA GPU with CUDA support (Tested on RTX 4090)

CUDA: 11.8 or higher

Python: 3.8+

1. Install Dependencies

First, install the Python packages listed in requirements.txt:

pip install -r requirements.txt


2. Install 3DGS Rasterizers (CUDA)

This project relies on the standard diff-gaussian-rasterization kernels. You must compile and install them:

# Clone the original Gaussian Splatting repository submodules
git clone --recursive [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
pip install ./gaussian-splatting/submodules/diff-gaussian-rasterization
pip install ./gaussian-splatting/submodules/simple-knn


Note: If you skip this step, the code includes mock classes for debugging, but training will not produce actual renderings.

🚀 Usage

Training

To train the model on an endoscopic dataset (e.g., EndoNeRF or StereoMIS), run:

python train.py


Note: The current train.py contains a mock data loader for demonstration. You will need to plug in your specific dataset loader (parsing Colmap/SfM data) in the train_drecon function.

Configuration

Key hyperparameters mentioned in the paper are set as defaults in train.py:

Iterations: 3000 (standard for fast endoscopic reconstruction)

Learning Rates:

Spatial Deformation: 1e-3

Temporal Deformation: 1e-3

Gaussians: 1.6e-4

Loss Weights:

Depth Guidance: 0.1

SDS Loss: 0.01

🧠 Method Details

1. HexPlane Encoding (models/hexplane.py)

We use a 4D HexPlane representation to encode spatiotemporal features efficiently.

Spatial Planes: $XY, XZ, YZ$

Spatiotemporal Planes: $XT, YT, ZT$

2. Dual-Scale Depth Guidance (utils/loss_utils.py)

To prevent the "hollow structure" artifacts common in endoscopy:

Hard Depth ($D_{HDG}$): Rendered with forced high opacity ($\alpha \approx 0.95$) to anchor global geometry.

Soft Depth ($D_{SDG}$): Rendered with learned opacity to refine local surface details.

Loss: $\mathcal{L}_{DDG} = ||D_{HDG} - D_{GT}||_2 + ||D_{SDG} - D_{GT}||_2$

🔗 Citation

If you use this code, please cite the original MICCAI paper:

@inproceedings{basak2024drecon,
  title={D4Recon: Dual-stage Deformation and Dual-scale Depth Guidance for Endoscopic Reconstruction},
  author={Basak, Hritam and Yin, Zhaozheng},
  booktitle={MICCAI},
  year={2024}
}
