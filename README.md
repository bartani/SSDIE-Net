# SSDIE-Net

## Semi-Supervised Sand-Dust Image Enhancement via Attention-Driven Multi-Scale Feature Fusion

The lack of paired training data poses a significant challenge for the data-driven sand-dust image enhancement model. Supervised methods rely on simulated data, but the domain gap between simulated and real-world scenarios limits their generalizability. Unsupervised methods, while avoiding this issue, often have complex architectures and fail to restore fine details. Motivated by these, we propose a semi-supervised sand-dust image enhancement method, SSDR-Net, which integrates the strengths of supervised and unsupervised learning within a unified framework. SSDR-Net is trained on simulated data using supervised reconstruction loss functions in the supervised branch. It integrates classical image restoration techniques with conditional adversarial networks to generate highly realistic sand-dust images. Moreover, SSDR-Net adopts consistency regularization, dark channel priors-based regression minimization, Retinex-based pseudo-labeling, and adversarial learning to translate sand-dust images to clean ones in the unsupervised branch. Additionally, we designed an attention-based multi-scale feature fusion network in which feature map extraction from different scales facilitates improved local-to-global learning. Unlike previous methods that focus on extracting local features, SSDR-Net learns long-range dependencies, which are essential for understanding the overall scene structure. Extensive experiments show that SSDR-Net outperforms state-of-the-art supervised, unsupervised, and classical methods, producing dust-free images with enhanced details and better generalization to real-world scenarios.

## News

  -Paper Download (coming soon)
  
  -Training Code (coming soon)
  
  -Pre-trained Models (coming soon)
