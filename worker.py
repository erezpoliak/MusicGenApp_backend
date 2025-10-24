import redis
from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import json
import tempfile
import time
import io
import redis_constants
from midi_fn import midi_to_events, prepare_seed, generated_to_midi
from vocab_constants import events_to_int, timeshift_event, TIMESHIFT_RES, SEQ_LEN
from flask import Flask
import threading

app = Flask(__name__)

load_dotenv()
redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url, decode_responses=False)

GENERATION_DURATION = 10  # seconds
TEMP = 0.97
TOP_K = 40

model = load_model('best_model_keras.keras', compile = False)

@tf.function(reduce_retracing=True)
def fast_predict(input_seq):
    return model(input_seq, training = False)

sample_input = tf.constant(np.zeros((1, SEQ_LEN), dtype = np.int32))
_ = fast_predict(sample_input)  # warm-up

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

def generate_music(midi_file):
    event_sequence = midi_to_events(midi_file)
    seed = prepare_seed(event_sequence)
    curr_time = 0

    while curr_time < GENERATION_DURATION:
        model_input = np.array(seed)
        input_tensor = tf.constant(model_input, dtype = tf.int32)
        predictions = fast_predict(input_tensor).numpy()[0]  # shape(seq_len, vocab_size)
        last_prediction = predictions[-1]  # shape(vocab_size,)
        next_event_int = topk_with_temperature(last_prediction)

        if next_event_int == events_to_int[timeshift_event]:
            curr_time += (TIMESHIFT_RES / 1000)  # convert ms to seconds

        # sliding window
        seed[0].append(next_event_int)
        seed[0] = seed[0][1:]

        # extend user input with generated
        event_sequence.append(next_event_int)

    return event_sequence

def run():
    print('Worker started, waiting for tasks...')
    try:
        while True:
            _, task_data_json = redis_client.blpop([redis_constants.TASK_QUEUE], timeout=0)
            task_data = json.loads(task_data_json)
            task_id = task_data['task_id']

            midi_key = redis_constants.MIDI_KEY_FMT.format(task_id=task_id)
            result_key = redis_constants.RESULT_KEY_FMT.format(task_id=task_id)
            status_key = redis_constants.STATUS_KEY_FMT.format(task_id=task_id)

            midi_data = redis_client.get(midi_key)

            start_time = time.time()
            event_seq = generate_music(io.BytesIO(midi_data))

            with tempfile.NamedTemporaryFile(suffix='.mid') as temp_out:
                generated_to_midi(event_seq, temp_out.name)
                with open(temp_out.name, 'rb') as f:
                    generated_midi_data = f.read()

            redis_client.setex(result_key, redis_constants.TASK_EXPIRY, generated_midi_data)
            redis_client.setex(status_key, redis_constants.TASK_EXPIRY, 'completed')

            print(f'Task {task_id} completed in {time.time() - start_time:.2f} seconds.')

    except Exception as err:
        print(f'Worker encountered an error: {err}')

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

if __name__ == '__main__':
    worker_thread = threading.Thread(target=run, daemon=True)
    worker_thread.start()
