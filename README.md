# GreyHound: RNA Stability Prediction Model

A deep learning model for predicting RNA stability using genomic sequences and RNA-binding protein (RBP) expression profiles.

## Overview

GreyHound is a multi-modal deep learning model that combines:
- **CNN backbone**: 1D convolutions for processing DNA/RNA sequences
- **VAE integration**: Pre-trained variational autoencoder for RBP expression profiles
- **Fusion architecture**: Concatenates sequence features with RBP latent representations

The model predicts RNA decay rates (stability) in breast cancer cell lines, achieving correlation of ~0.62 on test data.

## Installation

```bash
git clone https://github.com/yourusername/greyhound.git
cd greyhound
pip install -r requirements.txt
```

## Project Structure
```
GreyHound/
├── src/ # Core model implementation
│ ├── greyhound.py # Main model architecture
│ └── rbp_vae.py # VAE for RBP encoding
├── notebooks/ # Jupyter notebooks
│ ├── 01_main_workflow.ipynb
│ ├── 02_comparison_analysis.ipynb
│ └── 03_generate_figures.ipynb
├── data/ # Data files
│ ├── raw/ # Original data files
│ └── processed/ # Processed data files
├── models/ # Trained model files
└── results/ # Output files and visualizations
```

## Usage

1. **Data Preparation**: Run `notebooks/01_main_workflow.ipynb`
2. **Model Training**: Follow the workflow in the main notebook
3. **Analysis**: Run `notebooks/02_comparison_analysis.ipynb`
4. **Visualization**: Run `notebooks/03_generate_figures.ipynb`

## Model Architecture

### GreyHound Model
- **Input**: DNA/RNA sequences (4,096 bp) + RBP expression profiles (1,378 dimensions)
- **CNN layers**: 2 conv blocks + 3 dilated conv layers
- **VAE integration**: 50-dimensional latent space
- **Output**: RNA decay rate prediction

### RBP VAE
- **Input**: RBP expression profiles
- **Encoder**: 1,378 → 500 → 50 dimensions
- **Decoder**: 50 → 500 → 1,378 dimensions
- **Purpose**: Dimensionality reduction and feature learning

## Results

The model achieves:
- **Test correlation**: ~0.62
- **Key findings**: RBMS3 identified as important RNA stability regulator
- **Comparison**: Outperforms other EMT-related RBPs

## License

MIT License - see LICENSE file for details.
