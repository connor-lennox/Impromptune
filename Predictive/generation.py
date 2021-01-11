import random
import os
import datetime

import numpy as np
import torch
import pretty_midi

from Data import event_loader
from Predictive.Training import training_util
from Predictive.Models import model_persistence


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
            predictions = torch.softmax(predictions, dim=1)

            # Convert probabilities to a python list so we can do weighted random choices
            probs = predictions.detach().numpy().tolist()

            # Randomly sample over each probability-space list
            choices = [random.choices(population=all_events, weights=probs[i], k=1) for i in range(len(probs))]

            # Recombine choices into a tensor of shape (batch)
            continuation = torch.tensor(choices)

        # Non-stochastic case: just take the most likely options (temperature has no impact)
        else:
            continuation = torch.argmax(predictions, dim=1, keepdim=True)

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
    # Transpose the matrix so the first axis is the note axis (for conversion to PrettyMIDI objects)
    return roll.T


def seqs_to_rolls(seqs):
    return [events_to_piano_roll(seq) for seq in seqs]


# PrettyMIDI demo script showing conversion of piano roll to PrettyMIDI object
# https://github.com/craffel/pretty-midi/blob/master/examples/reverse_pianoroll.py
def piano_roll_to_pretty_midi(piano_roll, fs=125, program=0):
    """Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    """
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


if __name__ == '__main__':
    data = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_SMALL_DENSE)
    _, _, samples, _ = training_util.create_train_test(data, given=256)
    test_stubs = torch.vstack([samples[0], samples[4000], samples[8000], samples[12000]])
    # test_stubs = torch.tensor([[332], [332], [332], [332], [332]])
    model_to_load = "1610256542_pram_k64_v256_e256_r1024_attn1.pram"
    generator_model = model_persistence.load_model(model_to_load)
    test_generated_seqs = generate_sequence(generator_model, test_stubs, 1100, stochastic=True, temperature=.9)
    test_roll = seqs_to_rolls(test_generated_seqs)
    output_folder = datetime.datetime.now().strftime('%d%m%y-%H%M%S')
    os.mkdir("OutputMIDI/" + output_folder)
    with open(os.path.join("OutputMIDI", output_folder, "output.txt"), 'w+') as outtext:
        outtext.write(f"Output generated {datetime.datetime.now().strftime('%d%m%y-%H%M%S')} by model {model_to_load}")
    for i in range(len(test_roll)):
        test_pm = piano_roll_to_pretty_midi(test_roll[i])
        test_pm.write(f"OutputMIDI/{output_folder}/output_{i}.mid")
