# Unsupervised Anomaly Detection with Explainable VAE-LSTM

## Project Overview
This project implements a data-driven approach for **Endpoint Detection (EPD)** and anomaly detection in plasma etching processes using Optical Emission Spectroscopy (OES) data.

## Methodology
We utilize a hybrid unsupervised machine learning architecture:

### Variational Autoencoder (VAE)
- **Role:** Dimensionality Reduction & Feature Extraction
- **Explainability:** Analyzes the latent space to understand what physical features (e.g., specific chemical species) the model is tracking

### Long Short-Term Memory (LSTM)
- **Role:** Temporal Forecasting
- **Anomaly Detection:** Predicts the next state of the process. Large deviations between predicted and actual state indicate anomalies (e.g., process shifts, endpoint)

## Workflow
1. **Data Preprocessing:** Load and normalize OES spectra
2. **Latent Dimension Analysis:** Determine the optimal number of latent features using Keras Tuner
3. **VAE Training:** Train the VAE to compress spectra into a low-dimensional latent space
4. **Explainable AI (XAI):** Visualize the latent dimensions to interpret their physical meaning
5. **LSTM Forecasting:** Train an LSTM to learn the temporal dynamics of the latent features
6. **Anomaly Detection:** Combine VAE reconstruction error and LSTM prediction error to identify process changes

## Key Features
- **Automated Hyperparameter Optimization** using Keras Tuner (Bayesian Optimization)
- **Explainable AI visualizations** including:
  - Enhanced Latent Space Traversal with Spectral Analysis
  - Latent Dimension Peak Mapping
  - Latent Feature Correlation and Statistics
  - Temporal Evolution Heatmaps
  - Reconstruction Contribution Analysis
- **Physical wavelength mapping** (195.0 - 1104.8 nm) for interpretable results
- **Dual anomaly detection** combining spatial (VAE reconstruction error) and temporal (LSTM prediction error) signals

## Requirements
```
numpy
pandas
matplotlib
tensorflow
keras-tuner
scikit-learn
seaborn
ruptures
```

## Usage
Run the Jupyter notebook `autoencoder test.ipynb` sequentially from Cell 1 onwards. The notebook is self-contained and includes detailed explanations for each step.

## Data Format
The expected input data format is a pickled pandas DataFrame where:
- Index: Wavelength values (nm)
- Columns: Time steps
- Values: Normalized intensity values

## Results
The model provides:
- Identification of dominant wavelengths controlled by each latent dimension
- Temporal evolution patterns of latent features
- Anomaly scores combining reconstruction and prediction errors
- Physical interpretation of each latent dimension mapped to emission lines

## License
This project is for research and educational purposes.

## Author
Data-driven EPD Team
