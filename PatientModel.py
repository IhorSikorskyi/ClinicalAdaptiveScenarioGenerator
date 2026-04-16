import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PatientModel:
    def __init__(self, diagnoses, symptoms, adj_matrix, diag_vectors, symp_vectors, beta=0.4, decay=0.85, top_k=5):
        self.diagnoses = diagnoses
        self.symptoms = symptoms
        self.diag_vectors = diag_vectors / np.linalg.norm(diag_vectors, axis=1, keepdims=True)
        self.symp_vectors = symp_vectors / np.linalg.norm(symp_vectors, axis=1, keepdims=True)
        self.beta = beta
        self.decay = decay
        self.top_k = top_k

        # TF-IDF normalization
        N = len(diagnoses)
        df_ = np.sum(adj_matrix > 0, axis=0)
        df_[df_ == 0] = 1
        idf = np.log(N / df_)
        adj_tfidf = adj_matrix * idf[np.newaxis, :]
        self.adj_matrix = adj_tfidf / np.linalg.norm(adj_tfidf, axis=1, keepdims=True)

        self.diag_scores = np.zeros(len(diagnoses))
        self.symp_scores = np.zeros(len(symptoms))

    def snapshot(self): return {"diag": self.diag_scores.copy(), "symp": self.symp_scores.copy()}

    def restore(self, snap):
        self.diag_scores, self.symp_scores = snap["diag"].copy(), snap["symp"].copy()

    def update_state(self, action_vector, revealed_symptoms=None):
        """
        Оновлення стану моделі з посиленим акцентом на специфічність симптомів.
        """
        # 1. Розраховуємо схожість питання з усіма симптомами
        symp_sim = cosine_similarity(action_vector.reshape(1, -1), self.symp_vectors)[0]

        # 2. Створюємо розріджений вектор активації (Top-K)
        sparse_symp = np.zeros_like(symp_sim)
        top_k_idx = np.argsort(symp_sim)[::-1][:self.top_k]
        for idx in top_k_idx:
            if symp_sim[idx] > 0.7:
                sparse_symp[idx] = symp_sim[idx]

        # 3. Якщо пацієнт реально підтвердив симптоми, даємо їм пріоритетну вагу
        if revealed_symptoms:
            for name in revealed_symptoms:
                if name in self.symptoms:
                    idx = self.symptoms.index(name)
                    # Використовуємо 1.5, щоб реальні докази домінували над прогнозом BioBERT
                    sparse_symp[idx] = 1.5

        # 4. Визначаємо силу впливу (multiplier)
        # Якщо симптомів НЕ знайдено (відповідь "Ні"), multiplier = 0.3 (м'яке заперечення)
        multiplier = 1.0 if (revealed_symptoms and len(revealed_symptoms) > 0) else 0.3

        # 5. ОНОВЛЮЄМО СИМПТОМ-СКОРИ (тільки один раз!)
        self.symp_scores = self.decay * self.symp_scores + (self.beta * sparse_symp * multiplier)

        # 6. Обчислення активації діагнозів
        # Вплив через матрицю зв'язків (80%) + Прямий контекст BioBERT (20%)
        diagnosis_from_symptoms = self.adj_matrix @ self.symp_scores
        direct_diagnosis_sim = cosine_similarity(action_vector.reshape(1, -1), self.diag_vectors)[0]

        self.diag_scores = (self.decay * self.diag_scores +
                            0.8 * diagnosis_from_symptoms +
                            0.2 * self.beta * direct_diagnosis_sim)

        return np.concatenate([self.diag_scores, self.symp_scores])

    def get_diagnosis_activations(self): return self.diag_scores