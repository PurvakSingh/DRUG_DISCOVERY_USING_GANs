import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model

# Load your dataset
dataset_path = '/content/Dataset1.csv'  # Path to your dataset file
df = pd.read_csv(dataset_path)
smiles = df['smile'].values  # Extract the SMILES strings

df.head(5)

# Function to convert SMILES strings to molecular fingerprints
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule object
    if mol is not None:
        # Generate a Morgan fingerprint (circular fingerprint)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    else:
        # Return a zero vector if the SMILES string is invalid
        return np.zeros((nBits,))

# Convert all SMILES strings to fingerprints
fingerprints = np.array([smiles_to_fingerprint(smile) for smile in smiles])
output_dim = fingerprints.shape[1]  # Dimension of the output molecule representation

# Function to build the generator model
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))  # First dense layer with 128 units
    model.add(LeakyReLU(alpha=0.2))  # LeakyReLU activation function
    model.add(BatchNormalization(momentum=0.8))  # Batch normalization

    model.add(Dense(256))  # Second dense layer with 256 units
    model.add(LeakyReLU(alpha=0.2))  # LeakyReLU activation function
    model.add(BatchNormalization(momentum=0.8))  # Batch normalization

    model.add(Dense(512))  # Third dense layer with 512 units
    model.add(LeakyReLU(alpha=0.2))  # LeakyReLU activation function
    model.add(BatchNormalization(momentum=0.8))  # Batch normalization

    model.add(Dense(output_dim, activation='tanh'))  # Output layer with 'tanh' activation
    model.add(Reshape((output_dim,)))  # Reshape the output to the correct dimensions

    noise = Input(shape=(latent_dim,))  # Input tensor for the generator
    generated_molecule = model(noise)  # The output of the generator

    return Model(noise, generated_molecule)  # Return the generator model

# Function to build the discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))  # First dense layer with 512 units
    model.add(LeakyReLU(alpha=0.2))  # LeakyReLU activation function

    model.add(Dense(256))  # Second dense layer with 256 units
    model.add(LeakyReLU(alpha=0.2))  # LeakyReLU activation function

    model.add(Dense(1, activation='sigmoid'))  # Output layer with 'sigmoid' activation

    fingerprint = Input(shape=(input_dim,))  # Input tensor for the discriminator
    validity = model(fingerprint)  # The output of the discriminator

    return Model(fingerprint, validity)  # Return the discriminator model

# Function to compile the GAN model
def compile_gan(generator, discriminator, latent_dim):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False  # Freeze discriminator's weights when training the GAN

    noise = Input(shape=(latent_dim,))  # Input tensor for the GAN
    generated_molecule = generator(noise)  # Generate a molecule
    validity = discriminator(generated_molecule)  # Determine if the generated molecule is real

    combined = Model(noise, validity)  # Combine the generator and discriminator into a GAN
    combined.compile(loss='binary_crossentropy', optimizer='adam')  # Compile the GAN

    return combined

# Define parameters
latent_dim = 100  # Size of the noise vector

# Create the models
generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)
gan = compile_gan(generator, discriminator, latent_dim)

# Print model summaries
generator.summary()
discriminator.summary()
gan.summary()

# Function to train the GAN
def train_gan(generator, discriminator, gan, fingerprints, latent_dim, epochs, batch_size=128):
    valid = np.ones((batch_size, 1))  # Labels for real data (all 1s)
    fake = np.zeros((batch_size, 1))  # Labels for fake data (all 0s)

    for epoch in range(epochs):
        # Train Discriminator

        # Select a random batch of real fingerprints
        idx = np.random.randint(0, fingerprints.shape[0], batch_size)
        real_fingerprints = fingerprints[idx]

        # Generate a batch of fake fingerprints
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_fingerprints = generator.predict(noise)

        # Train the discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_fingerprints, valid)
        d_loss_fake = discriminator.train_on_batch(generated_fingerprints, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator

        # Generate noise and train the generator to produce fingerprints that the discriminator labels as real
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Print the progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100}%] [G loss: {g_loss}]")

# Set training parameters
epochs = 200 # Number of training epochs
batch_size = 32  # Batch size

# Train the GAN
train_gan(generator, discriminator, gan, fingerprints, latent_dim, epochs,batch_size)

import matplotlib.pyplot as plt
import numpy as np

def train_gan(generator, discriminator, gan, fingerprints, latent_dim, epochs, batch_size=128, save_interval=50):
    valid = np.ones((batch_size, 1))  # Labels for real data (all 1s)
    fake = np.zeros((batch_size, 1))  # Labels for fake data (all 0s)

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        # Train Discriminator

        # Select a random batch of real fingerprints
        idx = np.random.randint(0, fingerprints.shape[0], batch_size)
        real_fingerprints = fingerprints[idx]

        # Generate a batch of fake fingerprints
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_fingerprints = generator.predict(noise)

        # Train the discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_fingerprints, valid)
        d_loss_fake = discriminator.train_on_batch(generated_fingerprints, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator

        # Generate noise and train the generator to produce fingerprints that the discriminator labels as real
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Save the losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        # Print the progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]*100}%] [G loss: {g_loss}]")

        # If at save interval, save generated fingerprint samples
        if epoch % save_interval == 0:
            save_generated_fingerprints(generator, latent_dim, epoch)

    # Plot the losses
    plot_losses(d_losses, g_losses)

def save_generated_fingerprints(generator, latent_dim, epoch, n=10):
    noise = np.random.normal(0, 1, (n, latent_dim))
    generated_fingerprints = generator.predict(noise)
    # Implement any saving logic here, for example saving to a file or plotting
    for i, fingerprint in enumerate(generated_fingerprints):
        plt.figure()
        plt.imshow(fingerprint.reshape(32, 64), cmap='gray')  # Adjust the shape as necessary
        plt.title(f"Generated Fingerprint at Epoch {epoch}")
        plt.savefig(f"generated_fingerprint_{epoch}_{i}.png")
        plt.close()

def plot_losses(d_losses, g_losses):
    d_losses = np.array(d_losses)
    g_losses = np.array(g_losses)
    plt.figure(figsize=(3, 3))
    plt.plot(d_losses[:, 0], label="Discriminator Loss")
    plt.plot(d_losses[:, 1], label="Discriminator Accuracy")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("GAN Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

# Example usage:
# Set training parameters
epochs = 200 # Number of training epochs
batch_size = 32  # Batch size

# Train the GAN
train_gan(generator, discriminator, gan, fingerprints, latent_dim, epochs, batch_size)

import matplotlib.pyplot as plt

# Sample data for demonstration purposes
# Replace these lists with your actual GAN training data
epochs = list(range(1, 101))
generator_accuracy = [0.5 + 0.02 * i for i in range(100)]  # Example data
discriminator_accuracy = [0.9 - 0.01 * i for i in range(100)]  # Example data

# Create the plot
plt.figure(figsize=(3, 3))
plt.plot(epochs, generator_accuracy, label='Generator Accuracy', color='blue')
plt.plot(epochs, discriminator_accuracy, label='Discriminator Accuracy', color='red')

# Add titles and labels
plt.title('GAN Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

