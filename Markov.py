import os
import json
import pickle
import random
import numpy as np
import pandas as pd

from BaseScoringLogic import compute_probabilities, shannon_entropy, get_tau
from VirtualPatient import VirtualPatient
from PatientModel import PatientModel
from AdaptiveQuestionSelector import AdaptiveQuestionSelector
from UtilsBert import embed_action
from Evaluation import EvaluationEngine
from SmartVirtualPatient import SmartVirtualPatient
from VirtualPatientEngine import VirtualPatientEngine

DATA_DIR = "Dataset/"
SAVE_DIR = "DataSaves/"
N_STEPS = 12


def load_artifacts() -> dict:
    def _path(name):
        return os.path.join(SAVE_DIR, name)

    with open(_path("graph_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(_path("diagnosis_tests.json"), "r", encoding="utf-8") as f:
        diagnosis_tests = json.load(f)

    with open(_path("symptom_names.json"), "r", encoding="utf-8") as f:
        symptom_names_dict = json.load(f)

    questions_path = _path("symptom_questions.json")
    if os.path.exists(questions_path):
        with open(questions_path, "r", encoding="utf-8") as f:
            symptom_questions_dict = json.load(f)
    else:
        symptom_questions_dict = {}
        print("  УВАГА: symptom_questions.json не знайдено. "
              "Запустіть Preprocessing.py для генерації.")

    adj_matrix   = np.load(_path("adjacency_matrix.npy"))
    diag_vectors = np.load(_path("diagnosis_vectors.npy"))
    symp_vectors = np.load(_path("symptom_name_vectors.npy"))

    train_path = os.path.join(DATA_DIR, "release_train_patients")
    csv_path   = train_path + ".csv" if os.path.exists(train_path + ".csv") else train_path
    df = pd.read_csv(csv_path)

    diagnoses  = metadata["diagnoses"]
    symptoms   = metadata["symptoms"]
    procedures = metadata.get("procedures", [])

    print(f"Артефакти завантажено з {SAVE_DIR}")
    print(f"  Діагнозів: {len(diagnoses)}  Симптомів: {len(symptoms)}  "
          f"Процедур: {len(procedures)}  Пацієнтів: {len(df):,}")

    return dict(
        diagnoses=diagnoses,
        symptoms=symptoms,
        procedures=procedures,
        adj_matrix=adj_matrix,
        diag_vectors=diag_vectors,
        symptom_names_dict=symptom_names_dict,
        symptom_questions_dict=symptom_questions_dict,
        symp_vectors=symp_vectors,
        diagnosis_tests=diagnosis_tests,
        df=df,
    )


def run_test_session_default(virtual_patient, patient_model, selector, diagnoses, arts, n_steps=N_STEPS):
    engine = EvaluationEngine()

    status_init = virtual_patient.get_status()
    tau_0 = get_tau(0)
    probs_0 = compute_probabilities(patient_model.get_diagnosis_activations(), tau_0)
    H_0 = shannon_entropy(probs_0)

    print("\n" + "=" * 65)
    print("НОВИЙ ПАЦІЄНТ ЗГЕНЕРОВАНИЙ (АВТО-ТЕСТ З ОЦІНЮВАННЯМ)")
    print("=" * 65)
    print(f"  Справжній діагноз: {status_init['true_diagnosis']}")
    print(f"  Початкова скарга: \"{virtual_patient.get_initial_complaint()}\"")
    print(f"  Початкова ентропія H(S_0): {H_0:.4f}")
    print("-" * 65)

    H_prev = H_0
    true_idx = (diagnoses.index(status_init["true_diagnosis"])
                if status_init["true_diagnosis"] in diagnoses else -1)

    history = []
    complaint_text = virtual_patient.get_initial_complaint()
    complaint_vec = embed_action(complaint_text)

    for step in range(1, n_steps + 1):
        best_act, pred_ig, act_type = selector.select_best(step, initial_complaint_vec=complaint_vec)
        if best_act is None:
            break

        if act_type == "test":
            answer = virtual_patient.perform_test(best_act, arts["procedures"], arts["diagnosis_tests"])
            newly_revealed = []
        else:
            answer, newly_revealed = virtual_patient.answer_question(best_act)

        a_vec = embed_action(best_act)
        revealed_names = [s for s, _ in newly_revealed]
        patient_model.update_state(a_vec, revealed_symptoms=revealed_names)

        tau = get_tau(step)
        diag_act = patient_model.get_diagnosis_activations()
        probs = compute_probabilities(diag_act, tau=tau)
        H_new = shannon_entropy(probs)
        dH = H_prev - H_new

        step_metrics = engine.calculate_step_metrics(
            action_text=best_act,
            true_diagnosis=status_init["true_diagnosis"],
            delta_h=dH
        )

        true_rank = (np.argsort(probs)[::-1].tolist().index(true_idx) + 1 if true_idx >= 0 else -1)

        history.append({
            "step": step,
            "query": best_act,
            "action": best_act,
            "type": act_type,
            "ig": dH,
            "rank": true_rank,
            "p": probs[true_idx] if true_idx >= 0 else 0,
            "metrics": step_metrics
        })

        print(f"Step {step} [{act_type}]: {best_act}")
        print(f"  ΔH: {dH:+.4f} | R-score: {step_metrics['r_score']:.3f} | Score: {step_metrics['total_step']:.2f}")

        H_prev = H_new

    final_top_diag = diagnoses[np.argsort(probs)[::-1][0]]

    engine.generate_session_report(
        session_log=history,
        true_diagnosis=status_init['true_diagnosis'],
        user_diagnosis=final_top_diag
    )

def run_test_session(virtual_patient, patient_model, selector, diagnoses, arts, n_steps=N_STEPS):
    engine = EvaluationEngine()
    status_init = virtual_patient.get_status()

    for s_name in virtual_patient.revealed_history:
        selector.asked.add(s_name)

    tau_0 = get_tau(0)
    probs_0 = compute_probabilities(patient_model.get_diagnosis_activations(), tau_0)
    H_0 = shannon_entropy(probs_0)

    print("\n" + "=" * 65)
    print("НОВИЙ ПАЦІЄНТ ЗГЕНЕРОВАНИЙ (АВТО-ТЕСТ З ОЦІНЮВАННЯМ)")
    print("=" * 65)
    print(f"  Справжній діагноз: {status_init['true_diagnosis']}")
    print(f"  Початкова скарга: \"{virtual_patient.get_initial_complaint()}\"")
    print(f"  Початкова ентропія H(S_0): {H_0:.4f}")
    print("-" * 65)

    H_prev = H_0
    true_idx = (diagnoses.index(status_init["true_diagnosis"])
                if status_init["true_diagnosis"] in diagnoses else -1)

    history = []
    complaint_text = virtual_patient.get_initial_complaint()
    complaint_vec = embed_action(complaint_text)

    for step in range(1, n_steps + 1):
        best_act, pred_ig, act_type = selector.select_best(step, initial_complaint_vec=complaint_vec)
        if best_act is None:
            break

        if act_type == "test":
            answer = virtual_patient.perform_test(best_act, arts["procedures"], arts["diagnosis_tests"])
            newly_revealed = []
        else:
            answer, newly_revealed = virtual_patient.answer_question(best_act)

        a_vec = embed_action(best_act)
        revealed_names = [s for s, _ in newly_revealed]
        patient_model.update_state(a_vec, revealed_symptoms=revealed_names)

        tau = get_tau(step)
        diag_act = patient_model.get_diagnosis_activations()
        probs = compute_probabilities(diag_act, tau=tau)
        H_new = shannon_entropy(probs)
        dH = H_prev - H_new

        step_metrics = engine.calculate_step_metrics(
            action_text=best_act,
            true_diagnosis=status_init["true_diagnosis"],
            delta_h=dH
        )

        true_rank = (np.argsort(probs)[::-1].tolist().index(true_idx) + 1 if true_idx >= 0 else -1)

        history.append({
            "step": step,
            "query": best_act,
            "action": best_act,
            "type": act_type,
            "ig": dH,
            "rank": true_rank,
            "p": probs[true_idx] if true_idx >= 0 else 0,
            "metrics": step_metrics
        })

        print(f"Step {step} [{act_type}]: {best_act}")
        print(f"  ΔH: {dH:+.4f} | R-score: {step_metrics['r_score']:.3f} | Score: {step_metrics['total_step']:.2f}")

        H_prev = H_new

    final_top_diag = diagnoses[np.argsort(probs)[::-1][0]]

    engine.generate_session_report(
        session_log=history,
        true_diagnosis=status_init['true_diagnosis'],
        user_diagnosis=final_top_diag
    )

def run_session(virtual_patient, patient_model, diagnoses, arts):
    engine = EvaluationEngine()
    status_init = virtual_patient.get_status()
    session_log = []

    tau_0 = get_tau(0)
    h_prev = shannon_entropy(compute_probabilities(patient_model.get_diagnosis_activations(), tau_0))

    print("\n" + "═" * 65)
    print("РАБОТА З ПАЦІЄНТОМ (ЕКЗАМЕН З ОЦІНЮВАННЯМ)")
    print(f"Скарги пацієнта: {virtual_patient.get_initial_complaint()}")
    print("═" * 65)

    step = 1
    while True:
        user_input = input(f"\n[{step}] Ваша дія (або DIAGNOSIS): ").strip()
        if not user_input: continue
        if user_input.upper() == "DIAGNOSIS": break

        is_test = user_input.upper().startswith("TEST")
        action_query = user_input[5:].strip() if is_test else user_input
        action_type = "test" if is_test else "question"

        if is_test:
            answer = virtual_patient.perform_test(action_query, arts["procedures"], arts["diagnosis_tests"])
            newly_revealed = []
            print(f"Лабораторія: {answer}")
        else:
            answer, newly_revealed = virtual_patient.answer_question(action_query)
            print(f"Пацієнт: {answer}")

        a_vec = embed_action(user_input)
        revealed_names = [s for s, _ in newly_revealed]

        patient_model.update_state(a_vec, revealed_symptoms=revealed_names)

        tau = get_tau(step)
        h_new = shannon_entropy(compute_probabilities(patient_model.get_diagnosis_activations(), tau))
        ig = h_prev - h_new
        h_prev = h_new

        session_log.append({
            "step": step,
            "type": action_type,
            "query": user_input,
            "ig": ig
        })
        step += 1

    user_diag = input("\nВаш остаточний діагноз: ").strip()

    engine.generate_session_report(
        session_log=session_log,
        true_diagnosis=status_init['true_diagnosis'],
        user_diagnosis=user_diag
    )

def main():
    print("=" * 65)
    print("STEP 3 — Автоматична діагностика (Питання + Тести)")
    print("=" * 65)

    arts = load_artifacts()
    random.seed(42)

    llama = VirtualPatientEngine(model_name="llama3.1")

    pm = PatientModel(
        arts["diagnoses"],
        arts["symptoms"],
        arts["adj_matrix"],
        arts["diag_vectors"],
        arts["symp_vectors"],
        beta=0.3,
        decay=0.96,
        top_k=3
    )

    with open(os.path.join(SAVE_DIR, "knowledge_graph.pkl"), "rb") as f:
        G = pickle.load(f)

    vp = VirtualPatient(
        arts["df"],
        arts["diagnoses"],
        arts["symptoms"],
        symptom_names_dict=arts["symptom_names_dict"],
        similarity_threshold=0.80
    )

    patient_row = arts["df"].sample(1).iloc[0]
    svp = SmartVirtualPatient(
        patient_row,
        G,
        arts["symptom_names_dict"],
        llama
    )

    selector = AdaptiveQuestionSelector(
        pm,
        get_tau,
        procedures=arts["procedures"],
        candidate_questions=list(arts["symptom_questions_dict"].values())
    )

    run_test_session(svp, pm, selector, arts["diagnoses"], arts, n_steps=N_STEPS)

    # run_test_session_default(vp, pm, selector, arts["diagnoses"], arts, n_steps=N_STEPS)

    # run_session(svp, pm, arts["diagnoses"], arts)

    with open(os.path.join(SAVE_DIR, "patient_model_class.pkl"), "wb") as f:
        pickle.dump(pm, f)

if __name__ == "__main__":
    main()