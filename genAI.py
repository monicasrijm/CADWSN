import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout

# Generate synthetic attack and normal data
def generate_data(num_samples):
    normal_data = np.random.rand(num_samples, 20)
    attack_data = np.random.rand(num_samples, 20) + 1  # Shifted to distinguish
    return normal_data, attack_data

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(20, activation='tanh'))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(1024, input_dim=20))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Training the GAN
def train_gan(generator, discriminator, gan, epochs, batch_size, data):
    for epoch in range(epochs):
        normal_data, attack_data = data
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_data = generator.predict(noise)

        real_data = np.concatenate((normal_data, attack_data))
        labels_real = np.ones((batch_size * 2, 1))
        labels_fake = np.zeros((batch_size, 1))

        discriminator.train_on_batch(real_data, labels_real)
        discriminator.train_on_batch(generated_data, labels_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        labels_gan = np.ones((batch_size, 1))
        gan.train_on_batch(noise, labels_gan)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")

# Main function
if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    normal_data, attack_data = generate_data(1000)
    data = (normal_data, attack_data)
    train_gan(generator, discriminator, gan, epochs=10000, batch_size=64, data=data)
