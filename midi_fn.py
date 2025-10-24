import pretty_midi
from vocab_constants import int_to_events, events_to_int, timeshift_event, TIMESHIFT_RES, SEQ_LEN


NOTE_RANGE = range(21, 109)   # piano notes from A0 to C8


def prepare_seed(event_seq, seq_len = SEQ_LEN):
    if len(event_seq) < seq_len:
        padding = [events_to_int[timeshift_event]] * (seq_len - len(event_seq))
        event_seq = padding + event_seq   # pad with silence at the start
    else:
        event_seq = event_seq[-seq_len:]

    return [event_seq]  # shape for model input (1, seq_len)

def midi_to_events(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    events = []  # shape (time, event_name)

    for instrument in pm.instruments:
        # skip drums
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            # skip super low/high notes
            if note.pitch not in NOTE_RANGE:
                continue
            # skip very short notes < 50ms
            if note.end - note.start < 0.05:
                continue
            events.append((note.start, f'note_on_{note.pitch}'))
            events.append((note.end, f'note_off_{note.pitch}'))

    event_sequence = []
    events.sort(key = lambda x: x[0])  # sort by time
    prev_time = 0

    for event_time, event_name in events:
        time_diff = event_time - prev_time
        time_diff_ms = int(time_diff * 1000)
        num_timeshifts = min(time_diff_ms // TIMESHIFT_RES, 500)  # dont allow more than 5sec of silence
        event_sequence.extend([events_to_int[timeshift_event]] * num_timeshifts)
        event_sequence.append(events_to_int[event_name])
        prev_time = event_time

    return event_sequence

def generated_to_midi(generated, output_midi):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    curr_time = 0
    active_notes = dict()  # key: pitch, value: start_time

    for event_int in generated:
        event_name = int_to_events[event_int]

        if event_name.startswith('note_on'):
            pitch = int(event_name.split('_')[-1])
            active_notes[pitch] = curr_time
        elif event_name.startswith('note_off'):
            pitch = int(event_name.split('_')[-1])
            if pitch in active_notes:
                start_time = active_notes[pitch]
                end_time = curr_time
                if end_time > start_time:
                    note = pretty_midi.Note(velocity=75, pitch=pitch, start=start_time, end=end_time)
                    instrument.notes.append(note)
                del active_notes[pitch]
        else:
            # timeshift event
            curr_time += (TIMESHIFT_RES / 1000)

    # end any remaining notes
    for pitch, start_time in active_notes.items():
        end_time = curr_time
        if end_time > start_time:
            note = pretty_midi.Note(velocity=75, pitch=pitch, start=start_time, end=end_time)
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_midi)