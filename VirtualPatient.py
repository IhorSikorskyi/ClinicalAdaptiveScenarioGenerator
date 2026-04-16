import random
import ast
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from UtilsBert import encode_text, normalize_vec, translate_to_english


class VirtualPatient:
    _DEMO_WORDS = {
        "age", "old", "young", "year", "male", "female",
        "man", "woman", "gender", "sex", "boy", "girl",
        "adult", "child", "elderly", "patient"
    }

    def __init__(self, df, diagnoses, symptoms, symptom_names_dict,
                 reveal_ratio: float = 0.3,
                 similarity_threshold: float = 0.90,
                 max_reveal_per_question: int = 3):
        self.diagnoses = diagnoses
        self.symptoms = symptoms
        self.sim_threshold = similarity_threshold
        self.max_reveal = max_reveal_per_question

        row = df.sample(1).iloc[0]
        self.true_diagnosis = row["PATHOLOGY"]
        self.age = int(row["AGE"]) if "AGE" in row.index and pd.notna(row["AGE"]) else None
        self.sex = str(row["SEX"]).strip() if "SEX" in row.index and pd.notna(row["SEX"]) else None

        evid_list = ast.literal_eval(row["EVIDENCES"])
        base_codes = list({ev.split("_@_")[0] for ev in evid_list})

        self.patient_symptoms = {}
        for code in base_codes:
            name = symptom_names_dict.get(code, code)
            self.patient_symptoms[name] = False

        all_s = list(self.patient_symptoms.keys())
        n_reveal = max(2, int(len(all_s) * reveal_ratio))
        for s in random.sample(all_s, n_reveal):
            self.patient_symptoms[s] = True

        self.revealed_symptoms = [s for s, v in self.patient_symptoms.items() if v]
        self.hidden_symptoms = [s for s, v in self.patient_symptoms.items() if not v]

    def _is_demographic(self, question: str) -> bool:
        return bool(set(question.lower().split()) & self._DEMO_WORDS)

    def _answer_demographic(self, question: str) -> tuple:
        q = question.lower()
        if self.age is not None and any(w in q for w in ("age", "old", "year")):
            return f"I am {self.age} years old.", []
        if self.sex is not None and any(
                w in q for w in ("male", "female", "man", "woman", "gender", "sex")):
            sex_str = "male" if str(self.sex).upper() in ("M", "MALE") else "female"
            return f"I am {sex_str}.", []
        return "I cannot provide that information.", []

    def answer_question(self, question: str) -> tuple:
        if question is None or not str(question).strip():
            return "I have nothing more to say.", []

        if self._is_demographic(question):
            return self._answer_demographic(question)

        q_vec = normalize_vec(encode_text(translate_to_english(question)))

        confirmed = []
        for symp in self.revealed_symptoms:
            s_vec = normalize_vec(encode_text(symp))
            sim = float(cosine_similarity(q_vec.reshape(1, -1), s_vec.reshape(1, -1))[0][0])
            if sim >= 0.94:
                confirmed.append(symp)

        if confirmed:
            return f'Yes, we already discussed that: {", ".join(confirmed)}', []

        candidates = []
        for symp in list(self.hidden_symptoms):
            s_vec = normalize_vec(encode_text(symp))
            sim = float(cosine_similarity(q_vec.reshape(1, -1), s_vec.reshape(1, -1))[0][0])
            if sim >= self.sim_threshold:
                candidates.append((symp, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        newly_revealed = []

        for symp, sim in candidates[:self.max_reveal]:
            self.patient_symptoms[symp] = True
            self.revealed_symptoms.append(symp)
            self.hidden_symptoms.remove(symp)
            newly_revealed.append((symp, sim))

        if newly_revealed:
            return f'Yes: {", ".join(s for s, _ in newly_revealed)}', newly_revealed

        return "No, I don't think I have that.", []

    def get_initial_complaint(self) -> str:
        if self.revealed_symptoms:
            complaints = ", ".join(self.revealed_symptoms)
            return f"Patient complains of: {complaints}"
        return "Patient came in for a check-up"

    def get_status(self) -> dict:
        return {
            "true_diagnosis": self.true_diagnosis,
            "age": self.age,
            "sex": self.sex,
            "total_symptoms": len(self.patient_symptoms),
            "revealed": len(self.revealed_symptoms),
            "hidden": len(self.hidden_symptoms),
            "revealed_list": self.revealed_symptoms,
        }

    def perform_test(self, test_query: str, procedures_list: list, diagnosis_tests: dict) -> str:
        if not test_query:
            return "Please specify which test you want to perform."

        q_vec = normalize_vec(encode_text(translate_to_english(test_query)))
        best_proc, max_sim = None, -1

        for proc in procedures_list:
            p_vec = normalize_vec(encode_text(proc))
            sim = float(cosine_similarity(q_vec.reshape(1, -1), p_vec.reshape(1, -1))[0][0])
            if sim > max_sim:
                max_sim, best_proc = sim, proc

        if max_sim < self.sim_threshold:
            return f"I'm not sure what '{test_query}' is."

        relevant_tests = diagnosis_tests.get(self.true_diagnosis, [])
        if best_proc in relevant_tests:
            return f"Results for '{best_proc}': Abnormal findings consistent with {self.true_diagnosis}."
        else:
            return f"Results for '{best_proc}': Findings are within normal range."