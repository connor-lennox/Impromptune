import numpy as np


NOTE_ON = 0
NOTE_OFF = 1
TIME_STEP = 2
VELOCITY = 3


# Piano roll to event processor as defined in "Learning Expressive Music Performance", Oore et. al. 2018
def piano_roll_to_events(piano_roll):
    current_velocity = -1
    timestep_counter = 0

    events = []
    prev_state = np.zeros(128)

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

        # The maximum length time step is 1 second, so we need to consider that and break the gap into multiple events.
        if timestep_counter == 125:
            events.append(_event_to_number(TIME_STEP, timestep_counter))
            timestep_counter = 0

    return events


def _event_to_number(event_type, arg):
    if event_type == NOTE_ON:
        return arg
    elif event_type == NOTE_OFF:
        return arg + 128
    elif event_type == TIME_STEP:
        return arg + 256 - 1
    elif event_type == VELOCITY:
        return arg + 381


if __name__ == '__main__':
    from pretty_midi import PrettyMIDI
    test_midi = r"C:\Users\Connor\Documents\Research\Impromptune\Data\Datasets\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi"
    pm = PrettyMIDI(test_midi)
    # Get first 30 seconds of the piece:
    roll = pm.get_piano_roll(125)[:3125]
    test_events = piano_roll_to_events(roll)
    print(test_events[:100])