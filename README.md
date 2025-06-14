# Endoscopy-Image-Enhancement-GAN
Project focused on enhancing the quality of low-fidelity endoscopy images using Generative Adversarial Networks (GANs) implemented in PyTorch. The primary goal is to demonstrate the potential of deep learning to improve the visual clarity of medical images, which can aid in more accurate diagnostics.

Project Overview
Low-quality endoscopy images, often plagued by noise, blur, and inconsistent lighting, can hinder the accurate detection of abnormalities and lesions, potentially impacting patient outcomes. This project addresses this challenge by developing a GAN-based solution that learns to transform degraded endoscopy images into high-quality, enhanced versions.

Key Features
GAN-Based Image-to-Image Translation: Utilizes a Pix2Pix-like Conditional GAN architecture for supervised image enhancement.
U-Net Generator: Employs a robust U-Net as the generator, incorporating skip connections to effectively capture and reconstruct intricate image details.
PatchGAN Discriminator: Features a PatchGAN discriminator that evaluates the realism of image patches, encouraging the generator to produce high-fidelity local details.
Synthetic Data Degradation: Implements on-the-fly synthetic noise injection (Gaussian and Salt-and-Pepper), Gaussian blur, and color jitter to create paired low-quality inputs from pristine high-quality images. This approach simulates real-world imperfections and builds a robust enhancement model.
Hybrid Loss Function: Combines:
Adversarial Loss: Drives the generator to produce realistic images that can fool the discriminator.
L1 Reconstruction Loss: Ensures pixel-wise similarity between generated and ground-truth images.
Perceptual Loss (VGG-based): Leverages a pre-trained VGG-19 network to compare high-level feature representations, resulting in perceptually more realistic and visually pleasing enhancements.
Balanced Training Strategy: Implements a rebalanced training scheme with a lower discriminator learning rate and multiple generator updates per discriminator step to stabilize GAN convergence.
Dynamic Learning Rate Scheduling: Incorporates a linear learning rate decay after a specified epoch to optimize training and convergence.
Quantitative Evaluation: Measures performance using standard image quality metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
Visual Output & Checkpointing: Periodically saves sample enhanced images and model checkpoints during training for qualitative assessment and continuation.
Dataset
The project utilizes images from the Kvasir-SEG dataset, a publicly available collection of high-quality endoscopy images from the gastrointestinal (GI) tract. For training purposes, synthetic low-quality versions of these images are generated on-the-fly, creating the necessary paired data for the Pix2Pix model.

Installation and Usage
Clone the repository:
Bash

```git clone https://github.com/01AbhiSingh/Endoscopy-Image-Enhancement-GAN.git```
cd Endoscopy-Image-Enhancement-GAN
Download Kvasir-SEG Dataset:
Download the Kvasir-SEG dataset.
Extract the Kvasir-SEG folder.
Place the Kvasir-SEG folder inside a new directory named kvasirseg at the root of your project, so the image files are accessible at kvasirseg/Kvasir-SEG/images/.
Install Dependencies:
Bash

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # or 'cpu' if no GPU
pip install Pillow numpy scikit-image matplotlib
```
(You might run this in a Kaggle Notebook or similar environment, ensuring the DATA_ROOT and OUTPUT_DIR paths in Config.py are correctly set for your environment.)
Future Work
Experiment with additional degradation types (e.g., lens flare, compression artifacts).
Investigate more advanced GAN architectures (e.g., StyleGAN, ESRGAN variants).
Incorporate real-world low-quality endoscopy data for more robust model training.
Explore real-time inference capabilities for live endoscopy procedures.
Integrate medical professional feedback for clinical validation.
