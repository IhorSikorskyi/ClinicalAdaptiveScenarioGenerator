import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from deep_translator import GoogleTranslator

from Markov import load_artifacts
from PatientModel import PatientModel
from SmartVirtualPatient import SmartVirtualPatient
from VirtualPatient import VirtualPatient
from VirtualPatientEngine import VirtualPatientEngine
from AdaptiveQuestionSelector import AdaptiveQuestionSelector
from BaseScoringLogic import get_tau, compute_probabilities, shannon_entropy
from UtilsBert import embed_action
from Evaluation import EvaluationEngine

root_dir = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(root_dir))
SAVE_DIR = root_dir / "DataSaves"

app = Flask(__name__)
app.secret_key = "dc_train_secret_2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

class AppState:
    def __init__(self):
        self.virtual_patient = None
        self.patient_model   = None
        self.arts            = None
        self.engine          = None
        self.selector        = None
        self.complaint_vec   = None
        self.session_log     = []
        self.step            = 1
        self.h_prev          = 0.0
        self.initialized     = False
        self.loading         = False
        self.lang            = 'uk'

state = AppState()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("init_session")
def handle_init(data):
    mode = data.get("mode", "smart")
    lang = data.get("lang", "uk")
    sid  = request.sid

    def bg():
        socketio.emit("load_progress", {"msg": "Завантаження артефактів..."}, room=sid)
        arts = load_artifacts()

        socketio.emit("load_progress", {"msg": "Ініціалізація моделі пацієнта..."}, room=sid)
        pm = PatientModel(
            arts["diagnoses"], arts["symptoms"],
            arts["adj_matrix"], arts["diag_vectors"], arts["symp_vectors"],
            beta=0.3, decay=0.96, top_k=3,
        )

        socketio.emit("load_progress", {"msg": "Завантаження графа знань..."}, room=sid)
        with open(os.path.join(SAVE_DIR, "knowledge_graph.pkl"), "rb") as f:
            G = pickle.load(f)

        socketio.emit("load_progress", {"msg": "Генерація пацієнта..."}, room=sid)
        if mode == "smart":
            gemma = VirtualPatientEngine(model_name="gemma3:12b")
            patient_row = arts["df"].sample(1).iloc[0]
            vp = SmartVirtualPatient(patient_row, G, arts["symptom_names_dict"], gemma, lang=lang)
        else:
            vp = VirtualPatient(
                arts["df"], arts["diagnoses"], arts["symptoms"],
                symptom_names_dict=arts["symptom_names_dict"],
                similarity_threshold=0.80,
                lang=lang,
            )

        socketio.emit("load_progress", {"msg": "Підготовка двигуна оцінювання..."}, room=sid)
        engine = EvaluationEngine()

        selector = AdaptiveQuestionSelector(
            pm,
            get_tau,
            procedures=arts["procedures"],
            candidate_questions=(
                list(arts["symptom_questions_dict"].values())
                if arts["symptom_questions_dict"] else None
            ),
        )

        state.virtual_patient = vp
        state.patient_model   = pm
        state.arts            = arts
        state.engine          = engine
        state.selector        = selector
        state.session_log     = []
        state.step            = 1
        state.initialized     = True
        state.lang            = lang

        tau0 = get_tau(0)
        h0   = shannon_entropy(compute_probabilities(pm.get_diagnosis_activations(), tau0))
        state.h_prev = h0

        status    = vp.get_status()
        complaint = vp.get_initial_complaint()
        state.complaint_vec = embed_action(complaint)

        def to_py(obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return obj

        age_val = status.get("age", "?")

        procedures_raw = [str(p) for p in arts["procedures"]]
        procedures_display = list(procedures_raw)
        if lang == 'uk':
            translated_procedures = []
            for p in procedures_raw:
                try:
                    tr = GoogleTranslator(source='en', target='uk').translate(p)
                    translated_procedures.append(tr if tr and tr.strip() else p)
                except Exception:
                    translated_procedures.append(p)
            procedures_display = translated_procedures

        socketio.emit("session_ready", {
            "mode":           mode,
            "age":            to_py(age_val) if age_val != "?" else "?",
            "sex":            "M" if str(status.get("sex", "")).upper() == "M" else "F",
            "total_symptoms": to_py(status.get("total_symptoms", 0)),
            "complaint":      complaint,
            "h0":             round(float(h0), 4),
            "diagnoses":      [str(d) for d in arts["diagnoses"]],
            "procedures":     procedures_display,
            "procedures_raw": procedures_raw,
        }, room=sid)

    socketio.start_background_task(bg)

@socketio.on("ask_question")
def handle_question(data):
    if not state.initialized:
        return

    text   = data.get("text", "").strip()
    q_type = data.get("type", "question")
    sid    = request.sid

    if not text:
        return

    def bg():
        vp     = state.virtual_patient
        pm     = state.patient_model
        arts   = state.arts
        engine = state.engine

        if q_type == "test":
            answer         = vp.perform_test(text, arts["procedures"], arts["diagnosis_tests"])
            newly_revealed = []
        else:
            answer, newly_revealed = vp.answer_question(text)

        a_vec          = embed_action(text)
        revealed_names = [s for s, _ in newly_revealed]
        pm.update_state(a_vec, revealed_symptoms=revealed_names)

        tau      = get_tau(state.step)
        diag_act = pm.get_diagnosis_activations()
        probs    = compute_probabilities(diag_act, tau)
        h_new    = shannon_entropy(probs)
        ig       = state.h_prev - h_new
        state.h_prev = h_new

        step_metrics = engine.calculate_step_metrics(
            action_text=text,
            true_diagnosis=vp.get_status()["true_diagnosis"],
            delta_h=ig,
        )

        state.session_log.append({
            "step":    state.step,
            "type":    q_type,
            "query":   text,
            "ig":      ig,
            "metrics": step_metrics,
        })

        top_idx   = np.argsort(probs)[::-1][:8]
        top_diags = [
            {"name": arts["diagnoses"][i], "prob": float(probs[i])}
            for i in top_idx
        ]

        socketio.emit("answer", {
            "step":           state.step,
            "type":           q_type,
            "answer":         answer,
            "newly_revealed": [(s, float(sim)) for s, sim in newly_revealed],
            "h_new":          round(float(h_new), 4),
            "ig":             round(float(ig), 4),
            "top_diagnoses":  top_diags,
            "metrics":        step_metrics,
        }, room=sid)

        state.step += 1

    socketio.start_background_task(bg)

@socketio.on("finalize")
def handle_finalize(data):
    if not state.initialized:
        return

    sid       = request.sid
    user_diag = data.get("diagnosis", "").strip()
    true_diag = state.virtual_patient.get_status()["true_diagnosis"]
    log       = state.session_log

    is_correct  = user_diag.strip().lower() == true_diag.strip().lower()
    total_score = sum(e["metrics"]["total_step"] for e in log)
    total_ig    = sum(e["ig"] for e in log)
    all_comms   = [e["metrics"]["comm_score"] for e in log]
    avg_comm    = sum(all_comms) / max(len(all_comms), 1)
    crit_errors = sum(1 for e in log if e["metrics"]["penalty"] > 0)

    steps_data = [{
        "step":       e["step"],
        "query":      e["query"],
        "ig":         round(e["ig"], 4),
        "penalty":    e["metrics"]["penalty"],
        "total_step": e["metrics"]["total_step"],
        "comm_score": e["metrics"]["comm_score"],
    } for e in log]

    socketio.emit("report", {
        "correct":         is_correct,
        "user_diagnosis":  user_diag,
        "true_diagnosis":  true_diag,
        "total_score":     round(total_score, 4),
        "total_ig":        round(total_ig,    4),
        "avg_comm":        round(avg_comm,    4),
        "critical_errors": crit_errors,
        "steps":           steps_data,
    }, room=sid)

@socketio.on("get_hints")
def handle_get_hints(data):
    if not state.initialized or state.selector is None:
        return

    sid     = request.sid
    n_hints = int(data.get("n", 3))

    def bg():
        selector      = state.selector
        pm            = state.patient_model
        step          = state.step
        complaint_vec = state.complaint_vec

        original_asked = set(selector.asked)
        snap           = pm.snapshot()

        hints      = []
        temp_asked = set(original_asked)

        for _ in range(n_hints):
            selector.asked = temp_asked

            action, ig, a_type = selector.select_best(
                step, initial_complaint_vec=complaint_vec
            )

            selector.asked = original_asked
            pm.restore(snap)

            if action is None:
                break

            hints.append({
                "text": action,
                "type": a_type,
                "ig":   round(float(ig), 4),
            })
            temp_asked.add(action)

            if state.lang == 'uk':
                for h in hints:
                    try:
                        h["text"] = GoogleTranslator(
                            source='en', target='uk'
                        ).translate(h["text"])
                    except Exception:
                        pass

        socketio.emit("hints_ready", {"hints": hints, "step": step}, room=sid)

    socketio.start_background_task(bg)

if __name__ == "__main__":
    print("=" * 60)
    print("  DS·Train — Graphical Web Interface")
    print("  Відкрийте http://localhost:5000 у браузері")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)