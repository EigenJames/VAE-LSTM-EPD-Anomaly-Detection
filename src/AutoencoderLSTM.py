# --- Importing necessary libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


import pandas as pd
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
latent_dim = 19  # Latent space dimension

# Build encoder and decoder
encoder = build_encoder(input_dim, latent_dim)
decoder = build_decoder(latent_dim, input_dim)

# Create VAE model
vae = VAE(encoder, decoder)

# Compile the VAE
vae.compile(optimizer='adam', loss='mse')

# Train the VAE
vae.fit(transposed_data_np, transposed_data_np, epochs=70, batch_size=32)

# --- Extract Latent Features from VAE Encoder ---

# Use the trained VAE encoder to get the latent space representation
latent_features = vae.encoder(transposed_data_np)[0].numpy()  # The [0] selects the z_mean

# The shape of `latent_features` is (num_time_steps, latent_dim)
print(f"Shape of latent features: {latent_features.shape}")

# --- LSTM Model Definition and Training ---

# Assuming latent_features is your input data
print("Shape of latent_features:", latent_features.shape)

# Ensure latent_features is a 3D array (samples, time steps, features)
# No need to add an extra dimension multiple times, just ensure it's (num_samples, time_steps, latent_dim)
if len(latent_features.shape) == 2:
    latent_features = np.expand_dims(latent_features, axis=1)
    print("Expanded shape of latent_features:", latent_features.shape)

# Use latent_features directly for X_train (should be 3D)
X_train = latent_features[:-1]  # Remove the last time step
y_train = np.zeros(len(X_train))  # Create dummy labels (unsupervised)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

# Build LSTM model for temporal pattern detection
lstm_model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(64, activation='relu', return_sequences=True),
    layers.LSTM(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.summary()

lstm_model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the LSTM model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# --- Evaluate VAE Reconstruction Error and LSTM Output ---

# Get reconstruction from the VAE
reconstructed_data = vae(transposed_data_np)

# Compute VAE reconstruction error (MSE for each time step)
reconstruction_errors = tf.reduce_mean(tf.square(transposed_data_np - reconstructed_data), axis=1).numpy()

# Get LSTM predictions (anomaly score) - no need to add extra dimension
lstm_predictions = lstm_model.predict(latent_features)

# Combine the VAE reconstruction error and LSTM predictions
combined_signal = reconstruction_errors + lstm_predictions.flatten()

import numpy as np
import tensorflow as tf

# Convert TensorFlow tensors to NumPy arrays if necessary and concatenate them
latent_features_full = np.concatenate(
    [tensor.numpy() if isinstance(tensor, tf.Tensor) else tensor for tensor in latent_features],
    axis=0
)

# Check the shape of the final reconstructed data
print("Shape of latent_features_full:", latent_features_full.shape)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a larger figure and 3D axis with adjusted margins
fig = plt.figure(figsize=(6, 9))
ax = fig.add_subplot(111, projection='3d')

# Prepare the data for plotting
time = np.arange(latent_features_full.shape[0])
features = np.arange(latent_features_full.shape[1])

# Plot with better separation and slightly transparent lines
for feature in features:
    ax.plot(time, np.full_like(time, feature), latent_features_full[:, feature], alpha=0.7)

# Set labels
ax.set_xlabel('Time', labelpad=20)  # Adding padding to the labels
ax.set_ylabel('Latent Feature', labelpad=20)
ax.set_zlabel('Value', labelpad=20)

# Adjusting the view angle and margins
ax.view_init(elev=30, azim=240)  # Adjust the viewing angle to make axes labels more visible
fig.subplots_adjust(left=0.1, right=0.85, top=0.85, bottom=0.1)  # Adjust the margins

# Set axis limits to better visualize the data
ax.set_xlim([0, latent_features_full.shape[0]])
ax.set_ylim([0, latent_features_full.shape[1] - 1])
ax.set_zlim([latent_features_full.min(), latent_features_full.max()])

# Display the plot
plt.show()

import matplotlib.pyplot as plt

# Assuming lstm_predictions is a numpy array or a similar data structure
plt.figure(figsize=(10, 6))
plt.plot(combined_signal, label='VAE and LSTM combined signal')
plt.xlabel('Time Steps')
plt.ylabel('Deviation from Predicted Values')
plt.legend()
plt.show()


# Define the wavelength range
wavelengths = np.linspace(200, 1200, input_dim)  # Spectral range from 200 nm to 1200 nm

# Get the decoder's weights (this shows how latent features are mapped to the original input)
decoder_weights = decoder.get_weights()[0]  # The first weight matrix corresponds to the mapping from latent space to input

# Shape of decoder_weights should be (input_dim, latent_dim)
# input_dim = 4550 (wavelengths), latent_dim = e.g. 19 (latent features)

# Let's say you want to see how latent feature 'latent_dim_to_inspect' influences the input wavelengths
latent_dim_to_inspect = 5  # Replace with the index of the latent dimension you're interested in

# Ensure decoder_weights has the correct shape (input_dim, latent_dim)
print(f"Decoder weights shape: {decoder_weights.shape}")

# Plot the contribution of the selected latent feature to each wavelength
plt.figure(figsize=(8, 6))

# Plot the absolute contribution of the selected latent feature (latent_dim_to_inspect) to all wavelengths
# We're accessing the entire column for this latent feature
# Plot the contribution of the selected latent feature to each wavelength
plt.plot(wavelengths, np.abs(decoder_weights[:, latent_dim_to_inspect]), label=f'Latent Feature {latent_dim_to_inspect}', color='green')


# Add labels and title
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Contribution Magnitude', fontsize=12)
plt.title(f'Latent Feature {latent_dim_to_inspect} Contribution to Wavelengths', fontsize=14)
plt.grid(True)

# Show the plot
plt.show()


