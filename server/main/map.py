import os, json
from flask import Blueprint, current_app, request, jsonify, redirect, url_for

bp = Blueprint('map', __name__, url_prefix='/map')

@bp.get('/get/<int:z>')
def get_geojson_map(z):
    path = os.path.join(current_app.root_path, 'base-map')
    if z in [int(os.path.splitext(file)[0]) for file in os.listdir(path)]:
        with open(os.path.join(path, f"{z}.geojson")) as f:
            return f.read()
    else:
        return jsonify({'error': 'geojson map does not exist'}), 404

@bp.post('/post/<int:z>')
def post_geojson_map(z):
    path = os.path.join(current_app.root_path, 'base-map')
    if z not in [int(os.path.splitext(file)[0]) for file in os.listdir(path)]:
        z_map = request.json
        with open(os.path.join(current_app.root_path, 'base-map', f'{z}.geojson'), 'w') as f:
            json.dump(z_map, f)
        return redirect(url_for('map.get_geojson_map', z=z))
    else:
        return jsonify({'error': 'geojson map already exists'}), 409

@bp.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    return response
