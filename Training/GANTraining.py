import torch
import torch.nn as nn
import torch.optim as optim

from Models.Generator import Generator
from Models.Discriminator import Discriminator


FAKE_LABEL = 0
REAL_LABEL = 1


def train_gan(training_data: torch.Tensor,
              generator: Generator, discriminator: Discriminator,
              epochs: int = 100, batch_size: int = 32):

    noise_size = (batch_size, *generator.input_shape)
    real_tensor = torch.full([batch_size], REAL_LABEL)

    discriminator_train_labels = torch.cat((torch.full([batch_size], REAL_LABEL), torch.full([batch_size], FAKE_LABEL)))

    criterion = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters())
    discriminator_optimizer = optim.Adam(discriminator.parameters())

    num_batches = (len(training_data) // batch_size) - 1
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for batch_index in range(num_batches):
            print(f"\tBatch {batch_index}")

            # Within each epoch: do one training iteration with generated data, and perform updates only
            # on the generator (with inverted labels - need to optimize the generator to produce things
            # that convince the discriminator that the generated sequences are real).
            # After a loop through the generator, generate more data and do an update on the discriminator
            # using the newly generated data and a batch from the training data. This process will
            # train both the generator and the discriminator together.

            # Discriminator pass
            discriminator_optimizer.zero_grad()

            # Grab a batch from the training data
            real_data = training_data[batch_index*batch_size:(batch_index + 1) + batch_size]
            # Generate new fake data for training, detached as only the discriminator should be trained now.
            fake_data = generator(torch.rand(noise_size)).detach()

            # Concatenate the real and fake data to be passed through the discriminator
            batch = torch.cat(real_data, fake_data)
            classifications = discriminator(batch)

            # Calculate loss and step parameters
            discriminator_loss = criterion(classifications, discriminator_train_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Generator steps happen every five steps of the discriminator, as per Gulrajani et al, 2017
            # "Improved Training of Wasserstein GANs"
            if batch_index % 5 == 0 and batch_index != 0:
                # Generator pass
                generator_optimizer.zero_grad()

                # Create noise for an input
                input_noise = torch.rand(noise_size)

                # Generate sequences and classify through discriminator
                generated_sequences = generator(input_noise)
                classifications = discriminator(generated_sequences)

                # Calculate loss and step parameters
                generator_loss = criterion(classifications, real_tensor)
                generator_loss.backward()
                generator_optimizer.step()
