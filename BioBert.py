import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── Конфігурація ──────────────────────────────────────────────────────────────
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
DATA_DIR   = "Dataset/"
SAVE_DIR   = "DataSaves/"  # Папка з артефактами зі Step 1

# Перевірка наявності папки
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ── 1. Завантаження артефактів зі Step 1 ─────────────────────────────────────
def load_artifacts():
    # Читаємо файли з SAVE_DIR
    with open(os.path.join(SAVE_DIR, "knowledge_graph.pkl"), "rb") as f:
        G = pickle.load(f)
    with open(os.path.join(SAVE_DIR, "graph_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(SAVE_DIR, "symptom_names.json"), "r", encoding="utf-8") as f:
        symptom_names = json.load(f)
    with open(os.path.join(SAVE_DIR, "diagnosis_tests.json"), "r", encoding="utf-8") as f:
        diagnosis_tests = json.load(f)

    # Датасет залишається в DATA_DIR
    train_path = os.path.join(DATA_DIR, "release_train_patients")
    df = pd.read_csv(train_path + ".csv" if os.path.exists(train_path + ".csv") else train_path)

    diagnoses      = metadata["diagnoses"]
    symptoms       = metadata["symptoms"]
    procedures     = metadata["procedures"]
    procedure_codes = set(metadata.get("procedure_codes", []))

    print(f"   Граф завантажено з {SAVE_DIR}: {G.number_of_nodes()} вузлів")
    print(f"   Діагнозів: {len(diagnoses)}  Симптомів: {len(symptoms)}")
    return (G, metadata, diagnosis_tests, symptom_names, procedure_codes,
            df, diagnoses, symptoms, procedures)

# ── 2. Ініціалізація BioBERT ──────────────────────────────────────────────────
def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"   \nBioBERT завантажено на: {device}")
    return tokenizer, model, device

# ── 3. Допоміжні функції ──────────────────────────────────────────────────────
def make_encoder(tokenizer, model, device):
    def encode_text(text: str) -> np.ndarray:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=128, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    return encode_text


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def translate_to_english(text: str) -> str:
    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        return result if result and result.strip() else text
    except Exception as e:
        return text

# ── 4. Векторизація всіх вузлів ───────────────────────────────────────────────
def vectorize_nodes(diagnoses, symptoms, encode_text):
    all_nodes = diagnoses + symptoms
    node_vectors = {}
    print(f"   Векторизація {len(all_nodes)} вузлів...")
    for node_name in tqdm(all_nodes):
        node_vectors[node_name] = encode_text(node_name)
    return node_vectors

# ── 5. Збереження векторних артефактів (в SAVE_DIR) ───────────────────────────
def save_vector_artifacts(G, node_vectors, diagnoses, symptoms, procedures):
    diagnosis_matrix = normalize(np.stack([node_vectors[d] for d in diagnoses]))
    symptom_matrix   = normalize(np.stack([node_vectors[s] for s in symptoms]))

    # Збереження матриць
    np.save(os.path.join(SAVE_DIR, "diagnosis_vectors.npy"), diagnosis_matrix)
    np.save(os.path.join(SAVE_DIR, "symptom_vectors.npy"), symptom_matrix)

    if procedures:
        procedure_matrix = normalize(np.stack([node_vectors[p] for p in procedures]))
        np.save(os.path.join(SAVE_DIR, "procedure_vectors.npy"), procedure_matrix)
        np.save(os.path.join(SAVE_DIR, "procedure_name_vectors.npy"), procedure_matrix)

    symptom_name_matrix = normalize(np.stack([node_vectors[s] for s in symptoms]))
    np.save(os.path.join(SAVE_DIR, "symptom_name_vectors.npy"), symptom_name_matrix)

    with open(os.path.join(SAVE_DIR, "node_vectors.pkl"), "wb") as f:
        pickle.dump(node_vectors, f)

    # Оновлення та збереження матриці суміжності
    adj_matrix = np.zeros((len(diagnoses), len(symptoms)))
    for i, diag in enumerate(diagnoses):
        for j, symp in enumerate(symptoms):
            if G.has_edge(diag, symp):
                adj_matrix[i][j] = G[diag][symp]["weight"]
    np.save(os.path.join(SAVE_DIR, "adjacency_matrix.npy"), adj_matrix)

    # Оновлення самого графа (з новими вагами, якщо вони мінялися)
    with open(os.path.join(SAVE_DIR, "knowledge_graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    print(f"\n   --- Векторні артефакти оновлено в {SAVE_DIR} ---")
    return symptom_name_matrix


# ── 6. Тест пошуку подібних вузлів ───────────────────────────────────────────
def test_search(encode_text, symptom_name_matrix, symptoms):
    def find_similar_nodes(query: str, top_k: int = 5) -> list:
        eng = translate_to_english(query)
        vec = normalize(encode_text(eng).reshape(1, -1))
        sims = cosine_similarity(vec, symptom_name_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(symptoms[i], float(sims[i])) for i in top_idx]

    test_queries = ["задишка", "біль у грудях", "кашель"]
    print("\n   Перевірка пошуку:")
    for q in test_queries:
        results = find_similar_nodes(q, top_k=2)
        print(f"   '{q}' -> найближчий: {results[0][0]} (score: {results[0][1]:.3f})")

# ── Точка входу ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 2 — Векторизація вузлів графа (BioBERT)")
    print("=" * 60)

    (G, metadata, diagnosis_tests, symptom_names, procedure_codes,
     df, diagnoses, symptoms, procedures) = load_artifacts()

    tokenizer, model, device = load_biobert()
    encode_text = make_encoder(tokenizer, model, device)

    node_vectors = vectorize_nodes(diagnoses, symptoms, encode_text)

    symptom_name_matrix = save_vector_artifacts(
        G, node_vectors, diagnoses, symptoms, procedures
    )

    test_search(encode_text, symptom_name_matrix, symptoms)

    print(f"\nStep 2 завершено. Всі результати в {SAVE_DIR}")

if __name__ == "__main__":
    main()