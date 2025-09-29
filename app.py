from flask import Flask, request, jsonify
from db import init_db
from face_utils import enroll_user, _embedding_from_image, recognize, capture_image_from_webcam
import numpy as np
import cv2
import tempfile
import os

app = Flask(__name__)
init_db()

@app.route("/status")
def status():
    return jsonify({"status":"ok"})

@app.route("/enroll_file", methods=["POST"])
def enroll_file():
    """
    Recebe form-data 'file' (imagem), 'name', 'level'
    """
    if 'file' not in request.files:
        return jsonify({"error":"Nenhum arquivo enviado"}), 400
    f = request.files['file']
    name = request.form.get("name")
    level = int(request.form.get("level", 1))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    f.save(tmp.name)
    img = cv2.imread(tmp.name)
    os.unlink(tmp.name)
    user = enroll_user(name, level, img=img)
    return jsonify({"id": user.id, "name": user.name, "level": user.role_level})

@app.route("/authenticate_file", methods=["POST"])
def authenticate_file():
    if 'file' not in request.files:
        return jsonify({"error":"Nenhum arquivo enviado"}), 400
    f = request.files['file']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    f.save(tmp.name)
    img = cv2.imread(tmp.name)
    os.unlink(tmp.name)
    user, dist, err = recognize(img=img)
    if user:
        return jsonify({"id": user.id, "name": user.name, "level": user.role_level, "distance": float(dist)})
    else:
        return jsonify({"error": err, "distance": float(dist) if dist is not None else None}), 401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
