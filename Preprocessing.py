import json
import os
import pickle
from collections import defaultdict
import ast
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = "Dataset/"
SAVE_DIR = "DataSaves/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"   Створено директорію: {SAVE_DIR}")

PROCEDURE_KEYWORDS = [
    " x-ray", "scan", "mri", "ct ", "blood test", "blood count",
    "ecg", "ekg", "ultrasound", "biopsy", " culture", "spirometry",
    "oxygen", "spo2", "saturation", "troponin", "d-dimer", "crp",
    "measure", "level", "rate", "pressure", "temperature", "pulse",
    "swab", "urine", "stool", "smear", "angiography",
]

NOT_SYMPTOMS = {
    "asthma", "diabetes", "hypertension", "epilepsy", "heart failure",
    "copd", "cancer", "hiv", "tuberculosis", "hepatitis",
}

def load_raw_data():
    with open(f"{DATA_DIR}release_conditions.json", "r", encoding="utf-8") as f:
        conditions = json.load(f)
    with open(f"{DATA_DIR}release_evidences.json", "r", encoding="utf-8") as f:
        evidences = json.load(f)

    train_path = f"{DATA_DIR}release_train_patients"
    df = pd.read_csv(train_path + ".csv" if os.path.exists(train_path + ".csv") else train_path)

    print(f"   Діагнозів: {len(conditions)}")
    print(f"   Симптомів/антецедентів: {len(evidences)}")
    print(f"   Пацієнтів: {len(df):,}")
    return conditions, evidences, df

def _clean_symptom_label(raw: str) -> str | None:
    PREFIXES = [
        "Do you have ",
        "Have you had ",
        "Have you been ",
        "Have you ",
        "Has your ",
        "Are you currently ",
        "Are you ",
        "Did you have ",
        "Did you ",
        "Is there ",
        "Is your ",
        "Do you ",
    ]
    label = raw.strip()
    for prefix in PREFIXES:
        if label.lower().startswith(prefix.lower()):
            label = label[len(prefix):]
            break
    label = label.replace("?", "").replace(":", "").replace("_", " ").strip()
    if any(w in label.lower() for w in NOT_SYMPTOMS):
        return None

    return label[:1].upper() + label[1:] if label else raw


def build_symptom_names(evidences):
    symptom_names = {}
    symptom_questions = {}
    procedure_codes = set()

    for code, info in evidences.items():
        raw_label = info.get("question_en") or info.get("name") or code

        if "?" in raw_label:
            symptom_questions[code] = raw_label.strip()

        label = _clean_symptom_label(raw_label)

        if label is None:
            procedure_codes.add(code)
            symptom_names[code] = raw_label
            continue

        if len(label) > 60:
            label = label[:57] + "..."

        symptom_names[code] = label

        search_text = (info.get("question_en", "") + " " + info.get("name", "")).lower()
        if any(kw in search_text for kw in PROCEDURE_KEYWORDS):
            procedure_codes.add(code)

    print(f"   Словник побудовано: {len(symptom_names)} записів")
    print(f"   Питань для селектора: {len(symptom_questions)}")
    return symptom_names, symptom_questions, procedure_codes

def build_graph(conditions, symptom_names, procedure_codes):
    G = nx.Graph()

    for condition_name, cond_data in conditions.items():
        G.add_node(condition_name, node_type="diagnosis",
                   severity=cond_data.get("severity", 0))

        for symptom_code in cond_data.get("symptoms", {}).keys():
            label = symptom_names.get(symptom_code, symptom_code)
            ntype = "procedure" if symptom_code in procedure_codes else "symptom"
            G.add_node(label, node_type=ntype, code=symptom_code)
            G.add_edge(condition_name, label, weight=1.0, edge_type=ntype)

        for ant_code in cond_data.get("antecedents", {}).keys():
            label = symptom_names.get(ant_code, ant_code)
            ntype = "procedure" if ant_code in procedure_codes else "antecedent"
            G.add_node(label, node_type=ntype, code=ant_code)
            G.add_edge(condition_name, label, weight=0.5, edge_type=ntype)

    print(f"   Граф побудовано: {G.number_of_nodes()} вузлів, {G.number_of_edges()} ребер")
    return G

def update_graph_weights(G, df, symptom_names, conditions):
    co_occurrence = defaultdict(lambda: defaultdict(int))
    diagnosis_count = defaultdict(int)

    print("Розрахунок частот...")
    for _, row in df.iterrows():
        diagnosis = row["PATHOLOGY"]
        diagnosis_count[diagnosis] += 1
        try:
            evid_list = ast.literal_eval(row["EVIDENCES"])
        except Exception:
            continue
        seen = set()
        for ev_code in evid_list:
            seen.add(ev_code.split("_@_")[0])
        for base_code in seen:
            ev_name = symptom_names.get(base_code, base_code)
            co_occurrence[diagnosis][ev_name] += 1

    for diagnosis, symp_freq in co_occurrence.items():
        total = diagnosis_count[diagnosis]
        for symptom, count in symp_freq.items():
            alpha_ij = count / total
            if G.has_edge(diagnosis, symptom):
                G[diagnosis][symptom]["weight"] = alpha_ij
            elif G.has_node(diagnosis):
                G.add_edge(diagnosis, symptom, weight=alpha_ij, edge_type="frequency")

    first = list(conditions.keys())[0]
    top5 = sorted(G[first].items(), key=lambda x: x[1]["weight"], reverse=True)[:5]
    print(f"\nТоп-5 симптомів для '{first}':")
    for symp, data in top5:
        print(f"  α={data['weight']:.3f}  →  {symp}")

    return G

def build_matrices_and_metadata(G, conditions):
    diagnoses  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "diagnosis"]
    symptoms   = [n for n, d in G.nodes(data=True)
                  if d.get("node_type") in ("symptom", "antecedent", "procedure")]
    procedures = [n for n, d in G.nodes(data=True) if d.get("node_type") == "procedure"]

    adj_matrix = np.zeros((len(diagnoses), len(symptoms)))
    for i, diag in enumerate(diagnoses):
        for j, symp in enumerate(symptoms):
            if G.has_edge(diag, symp):
                adj_matrix[i][j] = G[diag][symp]["weight"]

    diagnosis_tests = {}
    for diag in diagnoses:
        tests = [(nbr, G[diag][nbr]["weight"])
                 for nbr in G.neighbors(diag)
                 if G.nodes[nbr].get("node_type") == "procedure"]
        tests.sort(key=lambda x: x[1], reverse=True)
        diagnosis_tests[diag] = [t[0] for t in tests[:5]]

    return diagnoses, symptoms, procedures, adj_matrix, diagnosis_tests

def save_artifacts(G, diagnoses, symptoms, procedures, adj_matrix,
                   diagnosis_tests, symptom_names, symptom_questions, procedure_codes):

    np.save(os.path.join(SAVE_DIR, "adjacency_matrix.npy"), adj_matrix)

    with open(os.path.join(SAVE_DIR, "symptom_names.json"), "w", encoding="utf-8") as f:
        json.dump(symptom_names, f, ensure_ascii=False, indent=2)

    with open(os.path.join(SAVE_DIR, "symptom_questions.json"), "w", encoding="utf-8") as f:
        json.dump(symptom_questions, f, ensure_ascii=False, indent=2)

    with open(os.path.join(SAVE_DIR, "graph_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "diagnoses":      diagnoses,
                "symptoms":       symptoms,
                "procedures":     procedures,
                "procedure_codes": list(procedure_codes),
            },
            f, ensure_ascii=False, indent=2,
        )

    with open(os.path.join(SAVE_DIR, "diagnosis_tests.json"), "w", encoding="utf-8") as f:
        json.dump(diagnosis_tests, f, ensure_ascii=False, indent=2)

    with open(os.path.join(SAVE_DIR, "knowledge_graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    print(f"\n   --- Артефакти збережено в {SAVE_DIR} ---")
    print(f"   adjacency_matrix.npy, symptom_names.json, symptom_questions.json,")
    print(f"   graph_metadata.json, diagnosis_tests.json, knowledge_graph.pkl,")

def visualize_sample(G, conditions):
    diag_list = list(conditions.keys())

    best_trio = diag_list[:3]
    best_shared = -1
    for i in range(min(len(diag_list), 15)):
        for j in range(i + 1, min(len(diag_list), 15)):
            for k in range(j + 1, min(len(diag_list), 15)):
                a, b, c = diag_list[i], diag_list[j], diag_list[k]
                sa = set(G.neighbors(a))
                sb = set(G.neighbors(b))
                sc = set(G.neighbors(c))
                shared = len(sa & sb) + len(sa & sc) + len(sb & sc)
                if shared > best_shared:
                    best_shared, best_trio = shared, [a, b, c]

    chosen_diags = best_trio

    sym_nodes = set()
    for diag in chosen_diags:
        candidates = [
            (n, d["weight"])
            for n, d in G[diag].items()
            if G.nodes[n].get("node_type") == "symptom"
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        n_cand = len(candidates)
        if n_cand == 0:
            continue
        top_s = candidates[:3]
        mid_s = candidates[n_cand // 3 : n_cand // 3 + 3]
        low_s = candidates[max(n_cand - 3, 0):]
        picked = {n for n, _ in top_s + mid_s + low_s}
        sym_nodes.update(picked)

    all_nodes = set(chosen_diags) | sym_nodes
    sub = G.subgraph(all_nodes)

    color_map = {"diagnosis": "#e74c3c", "symptom": "#3498db",
                 "antecedent": "#2ecc71", "procedure": "#f39c12"}
    colors = [color_map.get(G.nodes[n].get("node_type"), "#95a5a6") for n in sub.nodes()]
    sizes  = [2200 if G.nodes[n].get("node_type") == "diagnosis" else 800
              for n in sub.nodes()]
    edge_widths = [sub[u][v]["weight"] * 5 for u, v in sub.edges()]

    fig, ax = plt.subplots(figsize=(20, 13))
    pos = nx.spring_layout(sub, seed=42, k=2.5)

    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors,
                           node_size=sizes, alpha=0.92)
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(sub, pos, ax=ax, width=edge_widths,
                           alpha=0.55, edge_color="#555555")

    edge_labels = {(u, v): f"{sub[u][v]['weight']:.2f}" for u, v in sub.edges()}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=6, font_color="#c0392b",
                                 bbox=dict(boxstyle="round,pad=0.15",
                                           fc="white", alpha=0.7, ec="none"))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Діагноз"),
        Patch(facecolor="#3498db", label="Симптом"),
        Patch(facecolor="#2ecc71", label="Антецедент"),
        Patch(facecolor="#f39c12", label="Процедура"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.85)

    title_names = ", ".join(f"«{d}»" for d in chosen_diags)
    ax.set_title(f"Граф знань: {title_names}\n"
                 f"(спільних сусідів між парами: {best_shared})",
                 fontsize=11)
    ax.axis("off")

    save_path = os.path.join(SAVE_DIR, "graph_sample.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   graph_sample.png")

def main():
    print("=" * 60)
    print("Обробка даних та побудова графа знань")
    print("=" * 60)

    conditions, evidences, df = load_raw_data()
    symptom_names, symptom_questions, procedure_codes = build_symptom_names(evidences)
    G = build_graph(conditions, symptom_names, procedure_codes)
    G = update_graph_weights(G, df, symptom_names, conditions)
    diagnoses, symptoms, procedures, adj_matrix, diagnosis_tests = \
        build_matrices_and_metadata(G, conditions)

    save_artifacts(G, diagnoses, symptoms, procedures, adj_matrix,
                   diagnosis_tests, symptom_names, symptom_questions, procedure_codes)

    visualize_sample(G, conditions)

if __name__ == "__main__":
    main()