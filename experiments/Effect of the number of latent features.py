# --- Importing necessary libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = 'D:/James_archive/OneDrive/On_Going/VASSCAA_submission/Paper draft/data copy/Comp_1.pkl'
oes_data = pd.read_pickle(file_path)

# Separate the wavelength (first column) and intensity values (other columns)
wavelengths = oes_data.iloc[:, 0]  # Assuming first column is wavelength
intensities = oes_data.iloc[:, 1:]  # The remaining columns are intensity values at different time steps

# Normalize the intensity values between 0 and 1
scaler = MinMaxScaler()
normalized_intensities = scaler.fit_transform(intensities)

# Convert the normalized intensities back to a DataFrame for easier processing
normalized_data = pd.DataFrame(normalized_intensities, columns=intensities.columns)

# Print the first few rows of the normalized data
normalized_data.head()

## Data Transformation 


transposed_data = normalized_data.T
transposed_data_np = transposed_data.to_numpy()

# --- VAE Model Definition ---

# Sampling function for VAE
def sampling(z_mean, z_log_var):
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Build VAE Encoder
def build_encoder(input_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)  # Latent mean
    z_log_var = layers.Dense(latent_dim)(x)  # Latent log variance
    return models.Model(inputs, [z_mean, z_log_var], name='encoder')

# Build VAE Decoder
def build_decoder(latent_dim, output_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(output_dim, activation='sigmoid')(x)
    return models.Model(latent_inputs, outputs, name='decoder')

# Define VAE Model
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling(z_mean, z_log_var)
        return self.decoder(z)
    
# --- Build and Train VAE ---

# Set dimensions for VAE
input_dim = transposed_data_np.shape[1]  # 4550 wavelengths (rows)
# Range of latent dimensions to test (expanded for more values)
latent_dims = [ld for ld in range(5,101,5) ]
results = []

for latent_dim in latent_dims:
    # Build VAE for the given latent dimension
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam', loss='mse')
    
    # Train the VAE
    vae.fit(transposed_data_np, transposed_data_np, epochs=50, batch_size=32, verbose=0)
    # --- Extract Latent Features from VAE Encoder ---

    # Use the trained VAE encoder to get the latent space representation
    latent_features = vae.encoder(transposed_data_np)[0].numpy()  # The [0] selects the z_mean

    # The shape of `latent_features` is (num_time_steps, latent_dim)
    print(f"Shape of latent features: {latent_features.shape}")
    
    # --- Extract Latent Features from VAE Encoder ---

    # Use the trained VAE encoder to get the latent space representation
    latent_features = vae.encoder(transposed_data_np)[0].numpy()  # The [0] selects the z_mean

    
    # Evaluate on reconstruction error (MSE)
    reconstructed_data = vae(transposed_data_np)
    reconstruction_error = tf.reduce_mean(tf.square(transposed_data_np - reconstructed_data)).numpy()

    results.append({'latent_dim': latent_dim, 'reconstruction_error': reconstruction_error})


# Analyze results to find optimal latent_dim based on reconstruction error
latent_dims = [result['latent_dim'] for result in results]
reconstruction_errors = [result['reconstruction_error'] for result in results]


# Improved visualization
plt.figure(figsize=(10, 6))
plt.plot(latent_dims, reconstruction_errors, marker='o', linestyle='-', color='b', markersize=8)
plt.xlabel('Latent Dimension', fontsize=12)
plt.ylabel('Reconstruction Error (MSE)', fontsize=12)
plt.title('Reconstruction Error vs. Latent Dimension', fontsize=14)
plt.grid(True)  # Add gridlines for better readability
plt.xticks(latent_dims)  # Ensure all latent dims are shown on the x-axis
plt.tight_layout()  # Ensure no clipping of labels
plt.show()

