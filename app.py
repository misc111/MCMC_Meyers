import json
import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, List, Optional
import numpy as np

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from lcl_mcmc import (
    LCLData,
    LCLMCMCRunner,
    LCLPosteriorSamples,
    build_sample_dataset,
    simulate_triangles_from_samples,
)


app = Flask(__name__)


class AppState:
    def __init__(self):
        self.data: LCLData = build_sample_dataset()
        self.sse_queue: "Queue[str]" = Queue()
        self.mcmc_thread: Optional[threading.Thread] = None
        self.mcmc_runner: Optional[LCLMCMCRunner] = None
        self.samples: Optional[LCLPosteriorSamples] = None
        self.simulated: Optional[Dict[str, Any]] = None

    def reset_stream(self):
        # Drain queue
        try:
            while True:
                self.sse_queue.get_nowait()
        except Empty:
            pass


STATE = AppState()


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/data")
def api_data():
    # Sanitize NaNs for JSON (replace with null)
    C_arr = STATE.data.C
    W, D = STATE.data.W, STATE.data.D
    C_clean: List[List[Optional[float]]] = []
    for w in range(W):
        row: List[Optional[float]] = []
        for d in range(D):
            v = C_arr[w, d]
            try:
                if not np.isfinite(v):
                    row.append(None)
                else:
                    row.append(float(v))
            except Exception:
                row.append(None)
        C_clean.append(row)
    premiums = STATE.data.premiums.tolist()
    return jsonify(dict(W=W, D=D, C=C_clean, premiums=premiums))


@app.post("/api/use_sample")
def api_use_sample():
    STATE.data = build_sample_dataset()
    return jsonify({"ok": True})


@app.post("/api/start_mcmc")
def api_start_mcmc():
    payload = request.get_json(force=True)
    iters = int(payload.get("iterations", 5000))
    burn_in = int(payload.get("burn_in", iters // 2))
    thin = int(payload.get("thin", 5))
    adapt = int(payload.get("adapt", max(100, iters // 10)))
    chains = int(payload.get("chains", 1))
    trace_param = payload.get("trace_param", "logELR")
    update_every = int(payload.get("update_every", 10))

    # Optional tuning overrides
    tau = payload.get("tau", {}) or {}

    # Stop any existing run
    if STATE.mcmc_runner is not None:
        STATE.mcmc_runner.request_stop()
        time.sleep(0.05)

    STATE.reset_stream()
    STATE.samples = None

    def on_update(info: Dict[str, Any]):
        try:
            STATE.sse_queue.put_nowait(json.dumps(info))
        except Exception:
            pass

    runner = LCLMCMCRunner(
        data=STATE.data,
        iterations=iters,
        burn_in=burn_in,
        thin=thin,
        adapt=adapt,
        chains=chains,
        update_every=update_every,
        trace_param=trace_param,
        tau_overrides=tau,
        on_update=on_update,
    )

    def run_and_store():
        samples = runner.run()
        STATE.samples = samples
        STATE.mcmc_runner = None
        # Signal completion once more (in case the stream consumer missed the last message)
        try:
            STATE.sse_queue.put_nowait(json.dumps({"type": "done"}))
        except Exception:
            pass

    t = threading.Thread(target=run_and_store, daemon=True)
    STATE.mcmc_runner = runner
    STATE.mcmc_thread = t
    t.start()

    return jsonify({"ok": True})


@app.post("/api/stop_mcmc")
def api_stop_mcmc():
    if STATE.mcmc_runner is not None:
        STATE.mcmc_runner.request_stop()
        return jsonify({"ok": True, "stopped": True})
    return jsonify({"ok": True, "stopped": False})


@app.get("/api/stream/mcmc")
def api_stream_mcmc():
    def gen():
        # Send an initial hello
        yield "data: {}\n\n"
        while True:
            try:
                msg = STATE.sse_queue.get(timeout=1.0)
                yield f"data: {msg}\n\n"
                if msg:
                    try:
                        j = json.loads(msg)
                        if j.get("type") == "done":
                            break
                    except Exception:
                        pass
            except Empty:
                # Keep-alive ping
                yield "data: {\"type\":\"ping\"}\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.post("/api/simulate")
def api_simulate():
    if STATE.samples is None or STATE.samples.num_draws == 0:
        return jsonify({"ok": False, "error": "No MCMC samples available. Run MCMC first."}), 400

    payload = request.get_json(force=True)
    use_draws = int(payload.get("use_draws", min(1000, STATE.samples.num_draws)))
    # Select evenly spaced indices across samples
    idxs = list(range(0, STATE.samples.num_draws, max(1, STATE.samples.num_draws // use_draws)))[:use_draws]

    sim = simulate_triangles_from_samples(STATE.data, STATE.samples, idxs)
    STATE.simulated = sim

    # Return summary and first sample index
    return jsonify({
        "ok": True,
        "num_samples": len(idxs),
        "indices": idxs,
        "hist": sim["hist"],
        "summary": sim["summary"],
        "first_index": idxs[0] if idxs else None,
    })


@app.get("/api/sim_sample")
def api_sim_sample():
    if STATE.simulated is None:
        return jsonify({"ok": False, "error": "No simulations yet."}), 400
    idx = int(request.args.get("i", 0))
    tri_map: Dict[str, List[List[float]]] = STATE.simulated["triangles"]
    key = str(idx)
    if key not in tri_map:
        return jsonify({"ok": False, "error": f"Sample {idx} not available."}), 404
    return jsonify({"ok": True, "triangle": tri_map[key]})


def main():
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
