import numpy as np
import pretty_midi
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tempfile
import tensorflow as tf
from keras.models import load_model
import io
import time

app = Flask(__name__)
CORS(app)

vocab = np.load('vocabulary.npz', allow_pickle=True)
int_to_events = vocab['int_to_events'].item()
events_to_int = vocab['events_to_int'].item()
timeshift_event = 'timeshift_10'

NOTE_RANGE = range(21, 109)   # piano notes from A0 to C8
TIMESHIFT_RES = vocab['timeshift_res'].item()  # 10ms
SEQ_LEN = 255
GENERATION_DURATION = 10  # seconds
TEMP = 0.97
TOP_K = 40


model = load_model('best_model_keras.keras', compile = False)

@tf.function(reduce_retracing=True)
def fast_predict(input_seq):
    return model(input_seq, training = False)
sample_input = tf.constant(np.zeros((1, SEQ_LEN), dtype = np.int32))
_ = fast_predict(sample_input)  # warm-up

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

def topk_with_temperature(last_pos_probs, temperature = TEMP, top_k = TOP_K):
    # Apply temperature
    adjusted_logits = np.log(last_pos_probs + 1e-10) / temperature
    exp_logits = np.exp(adjusted_logits)
    adjusted_probs = exp_logits / np.sum(exp_logits)

    # top-k filtering
    top_k_indices = np.argsort(adjusted_probs)[-top_k:]
    top_k_probs = adjusted_probs[top_k_indices]
    top_k_probs /= np.sum(top_k_probs)  # re-normalize

    return np.random.choice(top_k_indices, p = top_k_probs)

@app.route('/generate', methods = ['POST'])
def generate_midi():
    start_time = time.time()

    if 'midi' not in request.files:
        return jsonify({'error': 'No MIDI file provided'}), 400
    file = request.files['midi']

    with tempfile.NamedTemporaryFile(suffix = '.mid') as temp_in, \
        tempfile.NamedTemporaryFile(suffix = '.mid') as temp_out:

        file.save(temp_in.name)
        event_sequence = midi_to_events(temp_in.name)
        seed = prepare_seed(event_sequence)
        curr_time = 0

        while curr_time < GENERATION_DURATION:
            model_input = np.array(seed)
            input_tensor = tf.constant(model_input, dtype = tf.int32)
            predictions = fast_predict(input_tensor).numpy()[0]  # shape(seq_len, vocab_size)
            last_prediction = predictions[-1]  # shape(vocab_size,)
            next_event_int = topk_with_temperature(last_prediction)

            if int_to_events[next_event_int] == timeshift_event:
                curr_time += (TIMESHIFT_RES / 1000)   # convert ms to seconds

            # sliding window
            seed[0].append(next_event_int)
            seed[0] = seed[0][1:]

            # extend user sequence with generated
            event_sequence.append(next_event_int)

        generated_to_midi(event_sequence, temp_out.name)
        temp_out.seek(0)
        midi_bytes = temp_out.read()

        print(f"Generation time: {time.time() - start_time} seconds")

        return send_file(
            io.BytesIO(midi_bytes),
            mimetype='audio/midi',
            as_attachment=False,
            download_name='generated.mid'
        )




@app.route('/', methods = ['GET'])
def welcome():
    return "Welcome to the LSTM Music Generation API! Use the /generate endpoint to generate music."


# comment out for deployment
if __name__ == '__main__':
    app.run()