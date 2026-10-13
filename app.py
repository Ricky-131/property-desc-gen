import os
import json
import uuid
import threading
from flask import Flask, render_template, jsonify, request
from generator import process_property, save_results

app = Flask(__name__)

# ── Job store ──────────────────────────────────────────────
# { job_id: { "status": "running"|"done"|"error", "result": {...}, "error": "..." } }
_jobs: dict[str, dict] = {}
_results: list[dict] = []
_lock = threading.Lock()


# ── Pages ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Generate (non-blocking) ────────────────────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    prop = request.json
    if not prop or not prop.get("name"):
        return jsonify({"error": "Invalid property data"}), 400

    # Normalise numeric fields
    prop["bedrooms"]  = int(prop.get("bedrooms", 0))
    prop["bathrooms"] = int(prop.get("bathrooms", 1))
    prop["price"]     = float(prop.get("price", 0))
    prop["area_sqft"] = int(prop.get("area_sqft", 0))
    features_raw = prop.get("features", "")
    if isinstance(features_raw, str):
        prop["features"] = [f.strip() for f in features_raw.split(",") if f.strip()]

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "running", "result": None, "error": None}

    def do_work():
        try:
            result = process_property(prop)
            with _lock:
                idx = next((i for i, r in enumerate(_results) if r["property"] == result["property"]), None)
                if idx is not None:
                    _results[idx] = result
                else:
                    _results.append(result)
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
        except Exception as e:
            with _lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)

    threading.Thread(target=do_work, daemon=True).start()

    # Return immediately — frontend will poll /api/job/<job_id>
    return jsonify({"job_id": job_id}), 202


# ── Poll job status ────────────────────────────────────────
@app.route("/api/job/<job_id>")
def job_status(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ── All results ────────────────────────────────────────────
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
        _jobs.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐  http://localhost:{port}")
    app.run(debug=True, port=port)
