"""Local Flask demo. Only explicitly registered checkpoints are available."""
import io
import json
import os
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from predict import Predictor

BASE = Path(__file__).resolve().parent


def create_app(registry_path=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
    registry_path = Path(registry_path or os.environ.get('MODEL_REGISTRY', BASE/'demo_models.json')).resolve()
    registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {}
    lock = threading.Lock()

    @lru_cache(maxsize=2)
    def predictor(key):
        entry = registry[key]
        return Predictor(registry_path.parent / entry['checkpoint'])

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/models')
    def models():
        return jsonify([{'id': key, 'name': value.get('name', key), 'dataset': value['dataset']}
                        for key, value in registry.items()])

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='Choose an image smaller than 5 MB.'), 413

    @app.post('/predict')
    def predict():
        key = request.form.get('model')
        if key not in registry: return jsonify(error='Select an available model.'), 400
        if 'file' not in request.files: return jsonify(error='Choose an image.'), 400
        try:
            raw = request.files['file'].read()
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.format not in ('JPEG', 'PNG', 'WEBP'):
                        return jsonify(error='Use a JPEG, PNG or WebP image.'), 400
                    if opened.width * opened.height > 20_000_000:
                        return jsonify(error='Choose an image smaller than 20 megapixels.'), 400
                    opened.load(); image = opened.copy()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return jsonify(error='This file could not be read as an image.'), 400
        try:
            with lock:
                engine = predictor(key)
                if engine.cfg.dataset != registry[key]['dataset']:
                    raise ValueError('Registry dataset does not match checkpoint')
                result = engine.predict(image)
            return jsonify(result)
        except (OSError, ValueError, RuntimeError, KeyError):
            app.logger.exception('Checkpoint inference failed')
            return jsonify(error='The model could not be loaded. Check its checkpoint and registry settings.'), 503

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000, debug=False)
