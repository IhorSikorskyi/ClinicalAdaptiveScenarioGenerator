import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from UtilsBert import embed_action, encode_text, normalize_vec, translate_to_english
from BaseScoringLogic import shannon_entropy, compute_probabilities

_CACHE_DIR = "DataSaves/"

class AdaptiveQuestionSelector:
    _FALLBACK_QUESTIONS = [
        'Do you have chest pain?',
        'Do you have shortness of breath or dyspnea?',
        'Do you have a cough?',
        'Do you have fever?',
        'Do you have fatigue or weakness?',
        'Do you have abdominal pain?',
        'Do you have headache?',
        'Do you have nausea or vomiting?',
        'Do you have joint pain or swelling?',
        'Are you taking any medications?',
    ]

    def __init__(self, patient_model, tau_fn, procedures=None, candidate_questions=None,
                 cache_dir: str = _CACHE_DIR):
        self.model = patient_model
        self.tau_fn = tau_fn
        self.asked = set()
        self.procedures = procedures if procedures else []

        if candidate_questions:
            self.CANDIDATE_QUESTIONS = list(candidate_questions)
        else:
            self.CANDIDATE_QUESTIONS = self._FALLBACK_QUESTIONS
            print("[AdaptiveQuestionSelector] УВАГА: використовуються fallback-питання. "
                  "Передайте candidate_questions з symptom_questions.json")

        self._cache_dir = cache_dir
        self._cache_path = os.path.join(cache_dir, "question_embeddings_cache.npz")
        self._embed_cache: dict[str, np.ndarray] = {}
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            data = np.load(self._cache_path, allow_pickle=True)
            self._embed_cache = {str(k): data[k] for k in data.files}
            print(f"[AdaptiveQuestionSelector] Завантажено {len(self._embed_cache)} "
                  f"векторів із дискового кешу ({self._cache_path})")
        except Exception as e:
            print(f"[AdaptiveQuestionSelector] Не вдалося завантажити кеш: {e}")

    def _save_disk_cache(self) -> None:
        if not self._embed_cache:
            return
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            np.savez(self._cache_path, **self._embed_cache)
        except Exception as e:
            print(f"[AdaptiveQuestionSelector] Не вдалося зберегти кеш: {e}")

    def _get_emb(self, q: str) -> np.ndarray:
        if q not in self._embed_cache:
            self._embed_cache[q] = normalize_vec(encode_text(translate_to_english(q)))
            self._save_disk_cache()
        return self._embed_cache[q]

    def precompute_all(self) -> None:
        all_items = self.CANDIDATE_QUESTIONS + self.procedures
        new_count = sum(1 for q in all_items if q not in self._embed_cache)
        if new_count == 0:
            print("[AdaptiveQuestionSelector] Всі вектори вже в кеші, пропускаємо.")
            return
        print(f"[AdaptiveQuestionSelector] Попередня векторизація {new_count} нових питань...")
        for q in all_items:
            self._get_emb(q)
        print(f"[AdaptiveQuestionSelector] Готово. Кеш збережено: {self._cache_path}")

    def select_best(self, step: int, initial_complaint_vec=None) -> tuple:
        tau = self.tau_fn(step)
        current_activations = self.model.get_diagnosis_activations()
        H_before = shannon_entropy(compute_probabilities(current_activations, tau))

        best_action = None
        best_ig = -1e10
        best_type = "question"

        snap = self.model.snapshot()

        pool = [(q, "question") for q in self.CANDIDATE_QUESTIONS if q not in self.asked]
        pool += [(p, "test") for p in self.procedures if p not in self.asked]

        if not pool:
            return None, 0.0, None

        for action, a_type in pool:
            a_vec = self._get_emb(action) if a_type == "question" else embed_action(action)

            self.model.update_state(a_vec, revealed_symptoms=None)
            H_after = shannon_entropy(compute_probabilities(self.model.get_diagnosis_activations(), tau))
            ig = H_before - H_after

            self.model.restore(snap)

            if a_type == "test":
                ig -= 0.05

            if initial_complaint_vec is not None:
                relevance = float(cosine_similarity(a_vec.reshape(1, -1),
                                                    initial_complaint_vec.reshape(1, -1))[0][0])
                ig += (relevance * 0.1)

            if ig > best_ig:
                best_ig = ig
                best_action = action
                best_type = a_type

        if best_action is None and pool:
            best_action, best_type = pool[0]
            best_ig = 0.0

        if best_action:
            self.asked.add(best_action)

        return best_action, best_ig, best_type