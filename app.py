import os
import json
import threading
from flask import Flask, render_template, jsonify, request
from generator import process_property, save_results

app = Flask(__name__)

# In-memory store: list of result dicts, keyed by session property name
_results: list[dict] = []
_running = False
_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Accepts a single property object from the form and processes it.
    Returns the result immediately (blocking call — typically ~5-10s).
    """
    prop = request.json
    if not prop or not prop.get("name"):
        return jsonify({"error": "Invalid property data"}), 400

    # Normalise numeric fields
    prop["bedrooms"] = int(prop.get("bedrooms", 0))
    prop["bathrooms"] = int(prop.get("bathrooms", 1))
    prop["price"] = float(prop.get("price", 0))
    prop["area_sqft"] = int(prop.get("area_sqft", 0))
    features_raw = prop.get("features", "")
    if isinstance(features_raw, str):
        prop["features"] = [f.strip() for f in features_raw.split(",") if f.strip()]

    try:
        result = process_property(prop)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with _lock:
        # Replace if same property name already exists, otherwise append
        idx = next((i for i, r in enumerate(_results) if r["property"] == result["property"]), None)
        if idx is not None:
            _results[idx] = result
        else:
            _results.append(result)

    return jsonify(result)


@app.route("/api/results")
def get_results():
    return jsonify(_results)


@app.route("/api/results/save", methods=["POST"])
def save():
    if not _results:
        return jsonify({"error": "No results to save"}), 400
    json_path, csv_path = save_results(_results)
    return jsonify({"json": json_path, "csv": csv_path})


@app.route("/api/results/clear", methods=["POST"])
def clear():
    with _lock:
        _results.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐  http://localhost:{port}")
    app.run(debug=True, port=port)
