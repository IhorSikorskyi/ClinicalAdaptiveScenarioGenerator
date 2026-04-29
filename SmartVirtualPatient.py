import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
from UtilsBert import encode_text, normalize_vec, translate_to_english

# ── Ключові слова, що вказують на НЕ-симптомне питання ──────────────────────
_TEMPORAL_WORDS = {
    "long", "when", "start", "began", "since", "ago", "duration",
    "how long", "давно", "коли", "початок", "починалось", "тривалість",
    "скільки", "з якого",
}
_CONTEXTUAL_WORDS = {
    "where", "who", "contact", "travel", "work", "family", "history",
    "allerg", "medication", "smoke", "alcohol", "diet", "stress",
    "де", "хто", "контакт", "подорож", "робота", "сім", "ліки",
    "алергія", "куріння", "алкоголь",
}
_SEVERITY_WORDS = {
    "severe", "mild", "rate", "scale", "worse", "better", "pain level",
    "наскільки", "сильно", "шкала", "гірше", "краще", "рівень болю",
}
SYMPTOM_THRESH = 0.80


class SmartVirtualPatient:
    def __init__(self, patient_row, G, symptom_names, llm_connector, lang: str = 'en'):
        self.llm = llm_connector
        self.G = G
        self.symptom_names = symptom_names
        self.lang = lang

        self.true_diagnosis = patient_row["PATHOLOGY"]
        self.age = patient_row.get("AGE", "unknown")
        self.sex = "male" if str(patient_row.get("SEX")).upper() == "M" else "female"

        evid_list = ast.literal_eval(patient_row["EVIDENCES"])
        self.actual_symptoms: dict[str, dict] = {}
        for ev in evid_list:
            code = ev.split("_@_")[0]
            name = symptom_names.get(code, code)
            weight = (
                G[self.true_diagnosis][name]["weight"]
                if G.has_edge(self.true_diagnosis, name)
                else 0.1
            )
            self.actual_symptoms[name] = {"weight": weight, "revealed": False}

        self.revealed_history: list[str] = []

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _classify_question(self, question: str) -> str:
        low = question.lower()
        if any(w in low for w in _TEMPORAL_WORDS):
            return "temporal"
        if any(w in low for w in _SEVERITY_WORDS):
            return "severity"
        if any(w in low for w in _CONTEXTUAL_WORDS):
            return "contextual"
        return "symptom"

    def _best_symptom_match(self, q_vec) -> tuple[str, float]:
        best_sim, best_name = -1.0, ""
        for s_name in self.actual_symptoms:
            s_vec = normalize_vec(encode_text(s_name))
            sim = float(
                cosine_similarity(q_vec.reshape(1, -1), s_vec.reshape(1, -1))[0][0]
            )
            if sim > best_sim:
                best_sim, best_name = sim, s_name
        return best_name, best_sim

    def _system_msg(self) -> str:
        lang_rule = (
            "IMPORTANT: You MUST respond ONLY in Ukrainian (Українська). Never write English words."
            if self.lang == 'uk'
            else "Always respond in English only."
        )
        return (
            f"You are a {self.age}-year-old {self.sex} patient. "
            f"{lang_rule} "
            f"Be brief (1–2 sentences) and stay in character as a patient."
        )

    def _lang_rule(self) -> str:
        return (
            "Reply ONLY in Ukrainian language."
            if self.lang == 'uk'
            else "Reply in English."
        )

    @staticmethod
    def _clean_symptom_name(name: str) -> str:
        cleaned = (name
                   .replace("Do you have ", "")
                   .replace("Have you had ", "")
                   .replace("Are you ", "")
                   .replace("Did you ", "")
                   .replace("Is there ", "")
                   .replace("Do you ", "")
                   .replace("Have you ", "")
                   .replace("Has your ", "")
                   .replace("?", "")
                   .replace(":", "")
                   .strip())
        return cleaned[:1].upper() + cleaned[1:] if cleaned else name

    def _translate_symptom(self, name: str) -> str:
        clean = self._clean_symptom_name(name)
        if self.lang == 'en':
            return clean
        try:
            from deep_translator import GoogleTranslator
            result = GoogleTranslator(source='en', target='uk').translate(clean)
            return result if result and result.strip() else clean
        except Exception:
            return clean

    def answer_question(self, question: str) -> tuple[str, list]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_en = translate_to_english(question)
        q_vec = normalize_vec(encode_text(q_en))

        target_symptom, best_sim = self._best_symptom_match(q_vec)
        q_type = self._classify_question(q_en)

        is_symptom_q = (best_sim >= SYMPTOM_THRESH) and (q_type == "symptom")

        system_msg = self._system_msg()

        # ── ФІЛЬТРАЦІЯ КОНТЕКСТУ ──
        all_known = [s for s in self.actual_symptoms]
        filtered_symps = []

        q_vec_reshaped = q_vec.reshape(1, -1)
        for s_name in all_known:
            s_vec = normalize_vec(encode_text(s_name)).reshape(1, -1)
            sim = float(cosine_similarity(q_vec_reshaped, s_vec)[0][0])

            if sim > 0.45:
                filtered_symps.append(s_name)

        context_symps = filtered_symps if filtered_symps else all_known[:2]
        symps_str = ", ".join(context_symps)
        revealed_str = ", ".join(self.revealed_history) if self.revealed_history else "none yet"

        # Гілка 1: Темпоральне питання
        if q_type == "temporal":
            prompt = (
                f'The doctor asks: "{question}"\n\n'
                f'You are a patient with these relevant symptoms: {symps_str}.\n'
                f'Already discussed: {revealed_str}.\n\n'
                f'Answer ONLY about timing/duration. Do NOT switch to other pain sites. '
                f'{self._lang_rule()} 1–2 sentences.'
            )

        # Гілка 2: Інтенсивність
        elif q_type == "severity":
            prompt = (
                f'The doctor asks: "{question}"\n\n'
                f'Relevant context: {symps_str}.\n'
                f'Answer about severity or progression. {self._lang_rule()}'
            )

        # Гілка 3: Контекстне питання
        elif q_type == "contextual":
            all_known = [s for s in self.actual_symptoms]
            filtered_symps = []
            for s_name in all_known:
                s_vec = normalize_vec(encode_text(s_name)).reshape(1, -1)
                sim = float(cosine_similarity(q_vec_reshaped, s_vec)[0][0])
                if sim > 0.45:
                    filtered_symps.append(s_name)

            if not filtered_symps:
                context_symps = self.revealed_history if self.revealed_history else all_known[:2]
            else:
                context_symps = filtered_symps

            prompt = (
                f'The doctor asks: "{question}"\n\n'
                f'Your diagnosis is {self.true_diagnosis}. '
                f'Pertinent history: {context_symps}.\n'
                f'Answer this background/history question naturally. '
                f'Do NOT mention unrelated conditions like blood pressure unless specifically asked. '
                f'{self._lang_rule()}'
            )

        # Гілка 4: Питання про конкретний симптом
        elif is_symptom_q:
            has_symptom = self.actual_symptoms[target_symptom].get("revealed", False) or best_sim >= SYMPTOM_THRESH
            if has_symptom:
                prompt = (
                    f'The doctor asks: "{question}"\n\n'
                    f'FACT: You DO have "{target_symptom}".\n'
                    f'Confirm naturally in 1–2 sentences. Do NOT add unrelated symptoms. '
                    f'{self._lang_rule()}'
                )
            else:
                prompt = (
                    f'The doctor asks: "{question}"\n\n'
                    f'FACT: You do NOT have "{target_symptom}". Deny naturally. '
                    f'{self._lang_rule()}'
                )

        # Гілка 5: Низька схожість
        else:
            prompt = (
                f'The doctor asks: "{question}"\n\n'
                f'Your symptoms: {symps_str}. Answer naturally. {self._lang_rule()}'
            )

        response = self.llm.generate_response(prompt, system_prompt=system_msg)

        # Оновлення стану виявлених симптомів
        newly_revealed: list[tuple[str, float]] = []
        if is_symptom_q and best_sim >= SYMPTOM_THRESH:
            self.actual_symptoms[target_symptom]["revealed"] = True
            if target_symptom not in self.revealed_history:
                self.revealed_history.append(target_symptom)
                translated = self._translate_symptom(target_symptom)
                newly_revealed.append((translated, best_sim))

        return response, newly_revealed

    # ── Початкова скарга ─────────────────────────────────────────────────────

    def get_initial_complaint(self) -> str:
        top_s = (
            self.revealed_history[:2]
            if self.revealed_history
            else list(self.actual_symptoms.keys())[:2]
        )
        lang_rule = (
            "Respond ONLY in Ukrainian."
            if self.lang == 'uk'
            else "Respond in English only."
        )
        prompt = (
            f"You are a patient. Your ONLY symptoms right now are: {', '.join(top_s)}.\n"
            f"STRICT RULES:\n"
            f"1. {lang_rule}\n"
            f"2. Mention ONLY the listed symptoms — nothing else.\n"
            f"3. Describe how you feel in 1–2 natural sentences as if talking to a doctor."
        )
        return self.llm.generate_response(prompt)

    # ── Статус та тести ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "true_diagnosis": self.true_diagnosis,
            "age": self.age,
            "sex": self.sex,
            "total_symptoms": len(self.actual_symptoms),
            "revealed": len(self.revealed_history),
            "hidden": len(self.actual_symptoms) - len(self.revealed_history),
            "revealed_list": self.revealed_history,
        }

    def perform_test(self, test_query: str, procedures_list: list, diagnosis_tests: dict) -> str:
        if not test_query:
            return "Please specify which test you want to perform."

        q_vec = normalize_vec(encode_text(translate_to_english(test_query)))
        best_proc, max_sim = None, -1.0

        for proc in procedures_list:
            p_vec = normalize_vec(encode_text(proc))
            sim = float(
                cosine_similarity(q_vec.reshape(1, -1), p_vec.reshape(1, -1))[0][0]
            )
            if sim > max_sim:
                max_sim, best_proc = sim, proc

        if max_sim < 0.80:
            return f"I'm not sure what '{test_query}' refers to."

        relevant_tests = diagnosis_tests.get(self.true_diagnosis, [])
        if best_proc in relevant_tests:
            return (
                f"Results for '{best_proc}': "
                f"Abnormal findings consistent with {self.true_diagnosis}."
            )
        return f"Results for '{best_proc}': Findings are within normal range."