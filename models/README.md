# Models Directory

This directory contains trained model files.

## Files
- `model_VAE_50_latent.pt`: Pre-trained VAE model for RBP encoding
- `best_ret_checkpoint.pt`: Best performing GreyHound model checkpoint
- `GH_default_R0.62.pt`: Default GreyHound model with R=0.62 performance

## Download Instructions

Due to file size limitations, model files are not included in the repository. To download:

1. Download from [model repository link]
2. Place files in this directory
3. Update paths in notebooks if needed

## Model Information

- **VAE Model**: 50-dimensional latent space, trained on RBP expression profiles
- **GreyHound Model**: CNN + VAE fusion model for RNA stability prediction
- **Performance**: Test correlation ~0.62
