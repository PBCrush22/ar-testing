from flask import Blueprint, request, redirect, url_for, jsonify
from main.db import get_db

bp = Blueprint('localization', __name__, url_prefix='/localization')

@bp.get('/poi/get')
def get_pois():
    pois = get_db().execute("SELECT * FROM poi").fetchall()
    return jsonify([{"id": poi['poi_id'], "name": poi['poi_name'], "lat": poi['latitude'], "long": poi['longitude']} for poi in pois])

@bp.get('/poi/get/<int:poi_id>')
def get_poi(poi_id):
    poi = get_db().execute(f"SELECT * FROM poi WHERE poi_id = {poi_id}").fetchone()
    return jsonify({"id": poi['poi_id'], "name": poi['poi_name'], "lat": poi['latitude'], "long": poi['longitude']})

@bp.delete('/poi/delete/<int:poi_id>')
def delete_poi(poi_id):
    db = get_db()
    db.execute(f"DELETE FROM poi WHERE poi_id = {poi_id}")
    db.commit()
    return redirect(url_for('localization.get_pois'))

@bp.post('/poi/post')
def post_poi():
    db = get_db()
    poi = request.json
    id, name, lat, long = poi['id'], poi['name'], poi['lat'], poi['long']
    db.execute(f"INSERT INTO poi (poi_id, poi_name, latitude, longitude) VALUES ({id}, '{name}', {lat}, {long})")
    db.commit()
    return redirect(url_for('localization.get_poi', poi_id=id))

