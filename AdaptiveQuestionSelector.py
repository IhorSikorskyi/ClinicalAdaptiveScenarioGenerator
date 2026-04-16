import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from UtilsBert import embed_action, encode_text, normalize_vec, translate_to_english
from BaseScoringLogic import shannon_entropy, compute_probabilities


class AdaptiveQuestionSelector:
    CANDIDATE_QUESTIONS = [
        # -- Кардiо / Пульмо -------------------------------------------
        'Do you have chest pain?',
        'Do you have shortness of breath or dyspnea?',
        'Do you have a cough?',
        'Is your cough productive or dry?',
        'Do you cough up blood?',
        'Do you hear wheezing when you breathe?',
        'Do you have rapid or labored breathing?',
        'Are breath sounds decreased on one side?',
        'Do you have palpitations or irregular heartbeat?',
        'Do you have swelling in your legs or ankles?',
        'Did the chest pain radiate to your arm or jaw?',
        'Is the pain worse when breathing deeply?',
        'Did the pain appear suddenly?',
        'Is the pain sharp or dull?',
        'Do you have high blood pressure?',
        # -- Загальнi симптоми -----------------------------------------
        'Do you have fever?',
        'Do you have night sweats?',
        'Do you have fatigue or weakness?',
        'Have you lost weight recently?',
        'Do you have chills?',
        # -- ШКТ --------------------------------------------------------
        'Do you have abdominal pain?',
        'Do you have nausea or vomiting?',
        'Do you have diarrhea?',
        'Do you have blood in your stool?',
        'Do you have jaundice?',
        # -- Неврологiчнi -----------------------------------------------
        'Do you have headache?',
        'Do you have dizziness or loss of balance?',
        'Have you had a seizure?',
        'Do you have tingling or numbness in your limbs?',
        # -- Урологiчнi -------------------------------------------------
        'Do you have pain when urinating?',
        'Do you have frequent urination?',
        # -- Musculoskeletal --------------------------------------------
        'Do you have joint pain or swelling?',
        'Do you have back pain?',
        # -- Анамнез ----------------------------------------------------
        'Have you ever had this condition before?',
        'Are you taking any medications?',
        'Have you recently travelled abroad?',
        'Do you smoke?',
        'Do you consume alcohol regularly?',
        'Do you have any known allergies?',
    ]

    def __init__(self, patient_model, tau_fn, procedures=None):
        self.model = patient_model
        self.tau_fn = tau_fn
        self.asked = set()
        self.procedures = procedures if procedures else []
        self._embed_cache = {}

    def _get_emb(self, q):
        if q not in self._embed_cache:
            self._embed_cache[q] = normalize_vec(encode_text(translate_to_english(q)))
        return self._embed_cache[q]

    def select_best(self, step: int, initial_complaint_vec=None) -> tuple:
        """
        Обирає найкращу дію (питання або тест) на основі максимізації Information Gain.
        initial_complaint_vec: вектор початкової скарги для підсилення медичної релевантності.
        """
        tau = self.tau_fn(step)
        current_activations = self.model.get_diagnosis_activations()
        H_before = shannon_entropy(compute_probabilities(current_activations, tau))

        best_action = None
        best_ig = -1e10
        best_type = "question"

        snap = self.model.snapshot()

        # Створюємо пул доступних дій
        pool = [(q, "question") for q in self.CANDIDATE_QUESTIONS if q not in self.asked]
        pool += [(p, "test") for p in self.procedures if p not in self.asked]

        if not pool:
            return None, 0.0, None

        for action, a_type in pool:
            # Отримуємо вектор дії
            a_vec = embed_action(action)

            # Симулюємо оновлення стану для розрахунку IG
            self.model.update_state(a_vec, revealed_symptoms=None)
            H_after = shannon_entropy(compute_probabilities(self.model.get_diagnosis_activations(), tau))
            ig = H_before - H_after

            # Повертаємо модель до початкового стану кроку
            self.model.restore(snap)

            # --- МЕДИЧНА ПРІОРИТЕТИЗАЦІЯ ---

            # 1. Штраф за тип дії (тести дорожчі за питання)
            if a_type == "test":
                ig -= 0.05  # Тести вимагають значно більшого IG, щоб бути обраними

            # 2. Бонус за релевантність до скарги (Focus Bonus)
            if initial_complaint_vec is not None:
                # Чим ближче питання до скарги тематично, тим вищий пріоритет
                relevance = float(cosine_similarity(a_vec.reshape(1, -1),
                                                    initial_complaint_vec.reshape(1, -1))[0][0])
                # Додаємо до 10% бонусу від максимально можливої релевантності
                ig += (relevance * 0.1)

            if ig > best_ig:
                best_ig = ig
                best_action = action
                best_type = a_type

        # Запасний варіант, якщо IG скрізь нульовий
        if best_action is None and pool:
            best_action, best_type = pool[0]
            best_ig = 0.0

        if best_action:
            self.asked.add(best_action)

        return best_action, best_ig, best_type