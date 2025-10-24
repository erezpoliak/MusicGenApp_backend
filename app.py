import redis
import uuid
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import io
import redis_constants

app = Flask(__name__)
CORS(app, origins = ['https://poli-musicgen.netlify.app', 'http://localhost:5173'])

load_dotenv()
redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url, decode_responses=False)

TASK_QUEUE = redis_constants.TASK_QUEUE
TASK_EXPIRY = redis_constants.TASK_EXPIRY
MIDI_KEY_FMT = redis_constants.MIDI_KEY_FMT
RESULT_KEY_FMT = redis_constants.RESULT_KEY_FMT
STATUS_KEY_FMT = redis_constants.STATUS_KEY_FMT

@app.route('/generate', methods = ['POST'])
def generate_midi():
    if 'midi' not in request.files:
        return jsonify({'error': 'No MIDI file provided'}), 400

    try:
        file = request.files['midi']
        midi_data = file.read()

        task_id = str(uuid.uuid4())
        midi_key = MIDI_KEY_FMT.format(task_id=task_id)
        status_key = STATUS_KEY_FMT.format(task_id=task_id)

        redis_client.setex(midi_key, TASK_EXPIRY, midi_data)

        task_data = {'task_id': task_id}

        redis_client.rpush(TASK_QUEUE, json.dumps(task_data))
        redis_client.expire(TASK_QUEUE, TASK_EXPIRY)

        redis_client.setex(status_key, TASK_EXPIRY, 'pending')

        return jsonify({'task_id': task_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result/<task_id>', methods = ['GET'])
def get_result(task_id):
    try:
        status_key = STATUS_KEY_FMT.format(task_id=task_id)
        status = redis_client.get(status_key)

        if status == b'pending':
            return jsonify({'status': 'pending'}), 202

        result_key = RESULT_KEY_FMT.format(task_id=task_id)
        result = redis_client.get(result_key)

        return send_file(
            io.BytesIO(result),
            mimetype='audio/midi',
            as_attachment=False,
            download_name='generated.mid',
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# comment for deployment
if __name__ == '__main__':
    app.run()