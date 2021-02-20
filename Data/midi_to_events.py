import random

import numpy as np
from pretty_midi import PrettyMIDI


NOTE_ON = 0
NOTE_OFF = 1
TIME_STEP = 2
VELOCITY = 3


# Piano roll to event processor as defined in "Learning Expressive Music Performance", Oore et. al. 2018
def piano_roll_to_events(piano_roll):
    current_velocity = -1
    timestep_counter = 0

    events = []
    prev_state = np.zeros(88)

    for state in piano_roll.T:
        diff = state - prev_state
        # If anything is going to happen this step, we first have to move the "cursor"
        if not (diff == 0).all():
            events.append(_event_to_number(TIME_STEP, timestep_counter))
            timestep_counter = 0
            # Iterate through this state to figure out what's different
            for note, elem in enumerate(diff):
                # For notes where the current velocity is greater than the previous one, something must have changed.
                if elem != 0:
                    # There are 128 velocity values in the midi spec. Down-sample this to 32:
                    # Special consideration for velocities from 1-3 as these would become 0.
                    vel = int(state[note]) // 4 if state[note] >= 4 else int(state[note])
                    if vel == 0:
                        events.append(_event_to_number(NOTE_OFF, note))
                    else:
                        if vel != current_velocity:
                            events.append(_event_to_number(VELOCITY, vel))
                            current_velocity = vel
                        events.append(_event_to_number(NOTE_ON, note))
        prev_state = state

        timestep_counter += 1

        # The maximum length time step is .248 second,
        # so we need to consider that and break the gap into multiple events.
        if timestep_counter == 32:
            events.append(_event_to_number(TIME_STEP, timestep_counter))
            timestep_counter = 0

    return events


def _event_to_number(event_type, arg):
    result = 0
    if event_type == NOTE_ON:
        result = arg
    elif event_type == NOTE_OFF:
        result = arg + 88
    elif event_type == TIME_STEP:
        result = arg + 176 - 1
    elif event_type == VELOCITY:
        result = min(arg + 208, 239)

    if result >= 240:
        print(f"invalid event generated: {event_type}, {arg}")

    return result


def read_midi(midi_file, segment_length=30, sample_size=None):
    print(midi_file)
    segment_resolution = int(segment_length * 125)
    midi = PrettyMIDI(midi_file)
    piano_roll = midi.get_piano_roll(fs=125)
    # Restrict piano roll to the actual notes on the piano
    piano_roll = piano_roll[21:109, :]
    roll_segments = [piano_roll[:, x:x+segment_resolution]
                     for x in range(0, piano_roll.shape[1]-segment_resolution, segment_resolution)]
    if sample_size is not None:
        roll_segments = random.sample(roll_segments, sample_size)
    event_segments = [piano_roll_to_events(segment) for segment in roll_segments]
    return event_segments


if __name__ == '__main__':
    test_midi = r"C:\Users\Connor\Documents\Research\Impromptune\Data\Datasets\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi"
    test_segments = read_midi(test_midi)
    print(test_segments)
