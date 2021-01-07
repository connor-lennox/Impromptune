import random

import numpy as np
import torch


all_events = list(range(333))


def generate_sequence(model, stubs, generation_length, stochastic=False, temperature=1.0):
    """model is a generative model. stubs is a 2d tensor of shape (batch, seq)
    It is permitted for stubs to be "empty": however since the model needs something to work off of
    a single velocity/timestep event should be present (as neither of these introduce any real notes
    to the sequence).
    """
    model.eval()
    for _ in range(generation_length):
        # Generate predictions of shape (batch, event): currently not normalized
        predictions = model(stubs)

        # Two approaches here: either only the top probability event is taken, or we take a stochastic sample
        # from the post-softmax probability space (with optional temperature control)
        if stochastic:
            # Apply temperature control to affect how flat/peaked the distribution should be
            predictions *= temperature

            # Convert probabilities to a python list so we can do weighted random choices
            probs = predictions.detach().numpy().tolist()

            # Randomly sample over each probability-space list
            choices = [random.choices(population=all_events, weights=probs[i], k=1) for i in range(len(probs))]

            # Recombine choices into a tensor of shape (batch)
            continuation = torch.tensor(choices)

        # Non-stochastic case: just take the most likely options (temperature has no impact)
        else:
            continuation = torch.argmax(predictions, dim=1)

        # Concatenate our choices onto the stubs tensors
        stubs = torch.cat([stubs, continuation], dim=1)

    # Once all the events have been generated, return the tensor of shape (batch, new_seq_len)
    return stubs


def events_to_piano_roll(events):
    """events is a list of event numbers - 1d list of ints"""
    # Prep the "default state" of the piano roll
    current_state = [0] * 88
    current_velocity = 127

    # Store each time-step state as an element in this list
    roll = []

    for event in events:
        if event < 88:
            current_state[event] = current_velocity
        elif event < 176:
            current_state[event-88] = 0
        elif event < 301:
            # 8 ms sampling rate: 1 time-step = 1 state
            for _ in range(event-176+1):
                roll.append(np.array(current_state))
        elif event < 333:
            current_velocity = (event-301)*4
        else:
            raise ValueError(f"Invalid event {event} passed")

    # Convert the roll into a 2d numpy array
    roll = np.array(roll)
    # Pad the matrix to return it to shape (seq_len, 128)
    roll = np.pad(roll, ((0, 0), (21, 19)))
    return roll


def seqs_to_rolls(seqs):
    return [events_to_piano_roll(seq) for seq in seqs]


if __name__ == '__main__':
    print(events_to_piano_roll([332, 53, 179, 53+88, 57, 179, 57+88]))