import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PatientModel:
    def __init__(self, diagnoses, symptoms, adj_matrix,
                 diag_vectors, symp_vectors,
                 beta: float = 0.4,
                 decay: float = 0.85,
                 top_k: int = 5,
                 eta: float = 0.15,
                 w_clip: float = 3.0,
                 adaptive_adj: bool = True):

        self.diagnoses = diagnoses
        self.symptoms  = symptoms
        self.beta  = beta
        self.decay = decay
        self.top_k = top_k
        self.eta   = eta
        self.w_clip = w_clip
        self.adaptive_adj = adaptive_adj

        self.diag_vectors = diag_vectors / np.linalg.norm(diag_vectors, axis=1, keepdims=True)
        self.symp_vectors = symp_vectors / np.linalg.norm(symp_vectors, axis=1, keepdims=True)

        N    = len(diagnoses)
        df_  = np.sum(adj_matrix > 0, axis=0)
        df_[df_ == 0] = 1
        idf  = np.log(N / df_)
        adj_tfidf = adj_matrix * idf[np.newaxis, :]

        row_norms = np.linalg.norm(adj_tfidf, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        self._adj_base = adj_tfidf / row_norms
        self.adj_matrix = self._adj_base.copy()

        self.diag_scores = np.zeros(len(diagnoses))
        self.symp_scores = np.zeros(len(symptoms))

        self.step = 0

    def snapshot(self) -> dict:
        return {
            "diag": self.diag_scores.copy(),
            "symp": self.symp_scores.copy(),
            "adj":  self.adj_matrix.copy(),
            "step": self.step,
        }

    def restore(self, snap: dict) -> None:
        self.diag_scores = snap["diag"].copy()
        self.symp_scores = snap["symp"].copy()
        self.adj_matrix  = snap["adj"].copy()
        self.step        = snap["step"]

    def _update_adj(self, confirmed_symp_indices: list[int]) -> None:
        exp_d = np.exp(self.diag_scores - self.diag_scores.max())
        p_diag = exp_d / (exp_d.sum() + 1e-12)

        if confirmed_symp_indices:
            for j in confirmed_symp_indices:
                self.adj_matrix[:, j] += self.eta * p_diag

            if self.adaptive_adj:
                not_confirmed = np.ones(len(self.symptoms), dtype=bool)
                not_confirmed[confirmed_symp_indices] = False
                self.adj_matrix[:, not_confirmed] = (
                        (1 - self.eta * self.decay) * self.adj_matrix[:, not_confirmed]
                        + self.eta * self.decay * self._adj_base[:, not_confirmed]
                )
        else:
            if self.adaptive_adj:
                soft_signal = self.symp_scores.clip(0)
                soft_signal = soft_signal / (soft_signal.sum() + 1e-12)
                self.adj_matrix += self.eta * 0.05 * np.outer(p_diag, soft_signal)
                self.adj_matrix = (
                        (1 - self.eta * 0.02) * self.adj_matrix
                        + self.eta * 0.02 * self._adj_base
                )

        np.clip(self.adj_matrix, 0.0, self.w_clip, out=self.adj_matrix)
        row_norms = np.linalg.norm(self.adj_matrix, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1.0)
        self.adj_matrix /= row_norms

    def update_state(self, action_vector: np.ndarray,
                     revealed_symptoms: list[str] | None = None) -> np.ndarray:
        self.step += 1

        symp_sim = cosine_similarity(action_vector.reshape(1, -1), self.symp_vectors)[0]

        sparse_symp = np.zeros_like(symp_sim)
        top_k_idx = np.argsort(symp_sim)[::-1][:self.top_k]
        for idx in top_k_idx:
            if symp_sim[idx] > 0.7:
                sparse_symp[idx] = symp_sim[idx]

        confirmed_indices = []
        if revealed_symptoms:
            for name in revealed_symptoms:
                if name in self.symptoms:
                    idx = self.symptoms.index(name)
                    sparse_symp[idx] = 1.5
                    confirmed_indices.append(idx)

        multiplier = 1.0 if confirmed_indices else 0.3

        raw_symp = self.decay * self.symp_scores + (self.beta * sparse_symp * multiplier)
        self.symp_scores = np.tanh(raw_symp)

        self._update_adj(confirmed_indices)

        diagnosis_from_symptoms = self.adj_matrix @ self.symp_scores
        direct_diagnosis_sim = cosine_similarity(action_vector.reshape(1, -1), self.diag_vectors)[0]

        raw_diag = (self.decay * self.diag_scores
                    + 0.8 * diagnosis_from_symptoms
                    + 0.2 * self.beta * direct_diagnosis_sim)
        self.diag_scores = np.tanh(raw_diag)

        return np.concatenate([self.diag_scores, self.symp_scores])

    def get_diagnosis_activations(self) -> np.ndarray:
        return self.diag_scores

    def get_matrix_drift(self) -> float:
        return float(np.linalg.norm(self.adj_matrix - self._adj_base, 'fro')
                     / (self._adj_base.shape[0] * self._adj_base.shape[1]))

    def reset_adj(self) -> None:
        self.adj_matrix = self._adj_base.copy()
        self.step = 0