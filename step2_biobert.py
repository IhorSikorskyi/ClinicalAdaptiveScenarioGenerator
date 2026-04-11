# КЛІТИНКА 1: Імпорти та завантаження артефактів з Кроку 1
import torch
import pickle
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from tqdm import tqdm
from collections import defaultdict
import ast

with open("knowledge_graph.pkl", "rb") as f:
    G = pickle.load(f)

with open("graph_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("Dataset/release_evidences.json", "r", encoding="utf-8") as f:
    evidences = json.load(f)

with open("Dataset/release_conditions.json", "r", encoding="utf-8") as f:
    conditions = json.load(f)

import pandas as pd
df = pd.read_csv("Dataset/release_train_patients")

diagnoses = metadata["diagnoses"]
symptoms  = metadata["symptoms"]

print(f"✅ Граф: {G.number_of_nodes()} вузлів, {G.number_of_edges()} ребер")
print(f"✅ Діагнозів: {len(diagnoses)}")
print(f"✅ Симптомів: {len(symptoms)}")
print(f"✅ Пацієнтів: {len(df):,}")

# КЛІТИНКА 2: Завантаження BioBERT
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

print(f"✅ BioBERT завантажено. Пристрій: {device}")

# КЛІТИНКА 3: Базові функції
def encode_text(text: str) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_vector = outputs.last_hidden_state[:, 0, :]
    return cls_vector.cpu().numpy().squeeze()


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

print("✅ Функції визначено")

# КЛІТИНКА 4: Словник кодів → назви (з symptom_names)
symptom_names = {}
for code, info in evidences.items():
    question = info.get("question_en", "")
    if question:
        label = (question
                 .replace("Do you have ", "")
                 .replace("Have you had ", "")
                 .replace("Are you ", "")
                 .replace("Did you ", "")
                 .replace("Is there ", "")
                 .replace("Do you ", "")
                 .replace("Have you ", "")
                 .rstrip("?")
                 .strip())
        if len(label) > 50:
            label = label[:47] + "..."
        symptom_names[code] = label
    else:
        symptom_names[code] = code

print(f"✅ symptom_names: {len(symptom_names)} записів")

# КЛІТИНКА 5: Виправлений розрахунок α_ij (унікальні симптоми на пацієнта)
co_occurrence  = defaultdict(lambda: defaultdict(int))
diagnosis_count = defaultdict(int)

print("Розрахунок частот (виправлена версія)...")

for _, row in df.iterrows():
    diagnosis = row["PATHOLOGY"]
    diagnosis_count[diagnosis] += 1

    try:
        evid_list = ast.literal_eval(row["EVIDENCES"])
    except:
        continue

    # Унікальні базові коди для цього пацієнта
    seen_base_codes = set()
    for ev_code in evid_list:
        base_code = ev_code.split("_@_")[0]
        seen_base_codes.add(base_code)

    for base_code in seen_base_codes:
        ev_name = symptom_names.get(base_code, base_code)
        co_occurrence[diagnosis][ev_name] += 1

# Оновлюємо ваги в графі
for diagnosis, symp_freq in co_occurrence.items():
    total = diagnosis_count[diagnosis]
    for symptom, count in symp_freq.items():
        alpha_ij = count / total
        if G.has_edge(diagnosis, symptom):
            G[diagnosis][symptom]["weight"] = alpha_ij
        elif G.has_node(diagnosis):
            G.add_edge(diagnosis, symptom, weight=alpha_ij, edge_type="frequency")

# Перевірка
example = list(conditions.keys())[0]
top5 = sorted(G[example].items(), key=lambda x: x[1]["weight"], reverse=True)[:5]
print(f"\nТоп-5 для '{example}':")
for symp, data in top5:
    print(f"  α={data['weight']:.3f}  →  {symp}")

max_w = max(data["weight"] for _, _, data in G.edges(data=True))
print(f"\nМакс. вага в графі: {max_w:.3f}  (має бути ≤ 1.0)")

# КЛІТИНКА 6: Векторизація всіх вузлів графа
all_nodes    = diagnoses + symptoms
node_vectors = {}

print(f"Векторизація {len(all_nodes)} вузлів...")

for node_name in tqdm(all_nodes):
    node_vectors[node_name] = encode_text(node_name)

print(f"✅ Векторизовано: {len(node_vectors)} вузлів")

# КЛІТИНКА 7: Збереження матриць + нормалізація
diagnosis_matrix = normalize(np.stack([node_vectors[d] for d in diagnoses]))
symptom_matrix   = normalize(np.stack([node_vectors[s] for s in symptoms]))

np.save("diagnosis_vectors.npy", diagnosis_matrix)
np.save("symptom_vectors.npy",   symptom_matrix)

with open("node_vectors.pkl", "wb") as f:
    pickle.dump(node_vectors, f)

print(f"✅ diagnosis_vectors.npy  →  shape: {diagnosis_matrix.shape}")
print(f"✅ symptom_vectors.npy    →  shape: {symptom_matrix.shape}")
print(f"✅ node_vectors.pkl збережено")

# КЛІТИНКА 8: Виправлений пошук — за назвами вузлів
print("Будуємо індекс назв вузлів...")

symptom_name_vectors = {}
for symp in symptoms:
    symptom_name_vectors[symp] = encode_text(symp)

symptom_name_matrix = normalize(
    np.stack([symptom_name_vectors[s] for s in symptoms])
)

np.save("symptom_name_vectors.npy", symptom_name_matrix)
print(f"✅ symptom_name_vectors.npy  →  shape: {symptom_name_matrix.shape}")

# КЛІТИНКА 9: Виправлена матриця суміжності + збереження графа
adj_matrix = np.zeros((len(diagnoses), len(symptoms)))
for i, diag in enumerate(diagnoses):
    for j, symp in enumerate(symptoms):
        if G.has_edge(diag, symp):
            adj_matrix[i][j] = G[diag][symp]["weight"]

np.save("adjacency_matrix.npy", adj_matrix)

with open("knowledge_graph.pkl", "wb") as f:
    pickle.dump(G, f)

print(f"✅ adjacency_matrix.npy  →  shape: {adj_matrix.shape}")
print(f"   max α = {adj_matrix.max():.3f}  (має бути ≤ 1.0)")
print(f"✅ knowledge_graph.pkl оновлено")

# КЛІТИНКА 10: Функція пошуку + фінальний тест
def find_similar_nodes(query: str, top_k: int = 5) -> list:
    eng = translate_to_english(query)
    vec = normalize(encode_text(eng).reshape(1, -1))
    sims = cosine_similarity(vec, symptom_name_matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(symptoms[i], float(sims[i])) for i in top_idx]


test_queries = [
    "задишка",
    "біль у грудях",
    "кашель",
    "fever and chills",
    "chest pain and shortness of breath",
]

print("Фінальний тест пошуку:")
print("=" * 60)
for q in test_queries:
    results = find_similar_nodes(q, top_k=3)
    eng = translate_to_english(q)
    print(f"\n'{q}' → '{eng}'")
    for node, score in results:
        bar = "█" * int(score * 20)
        print(f"  {score:.3f} {bar}  →  {node}")

print("\n🎉 step2_biobert.py повністю завершено!")