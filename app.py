from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta

app = Flask(__name__)

# ---------------- MODEL ----------------

model = tf.keras.models.load_model(
    "model/freshness_model.h5"
)

# ---------------- UPLOAD FOLDER ----------------

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------

conn = sqlite3.connect(
    "database/freshness.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    result TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # Image preprocessing
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    score = float(prediction[0][0])

    if score > 0.5:

        result = "Spoiled"
        confidence = score * 100

        shelf_life = 0

        advice = """
Food appears spoiled.
Discard immediately.
"""

    else:

        result = "Fresh"
        confidence = (1 - score) * 100

        shelf_life = 5

        advice = """
Safe to consume.
Store properly for maximum freshness.
"""

    expiry_date = (
        datetime.now()
        + timedelta(days=shelf_life)
    ).strftime("%d-%m-%Y")

    # Save Prediction
    cursor.execute("""
    INSERT INTO predictions
    (image_name, result, confidence)
    VALUES (?, ?, ?)
    """, (
        file.filename,
        result,
        confidence
    ))

    conn.commit()

    return render_template(
        "result.html",
        image_path=filepath,
        result=result,
        confidence=round(confidence, 2),
        shelf_life=shelf_life,
        expiry_date=expiry_date,
        advice=advice
    )

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    cursor.execute("""
    SELECT *
    FROM predictions
    ORDER BY id DESC
    """)
    data = cursor.fetchall()

    cursor.execute("""
    SELECT COUNT(*)
    FROM predictions
    """)
    total = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*)
    FROM predictions
    WHERE result='Fresh'
    """)
    fresh = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*)
    FROM predictions
    WHERE result='Spoiled'
    """)
    spoiled = cursor.fetchone()[0]

    return render_template(
        "dashboard.html",
        data=data,
        total=total,
        fresh=fresh,
        spoiled=spoiled
    )

# ---------------- ADMIN ----------------

@app.route("/admin")
def admin():
    return render_template("admin.html")

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)