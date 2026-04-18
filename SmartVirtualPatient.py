import ast
from sklearn.metrics.pairwise import cosine_similarity
from UtilsBert import encode_text, normalize_vec, translate_to_english

class SmartVirtualPatient:
    def __init__(self, patient_row, G, symptom_names, llm_connector):
        self.llm = llm_connector
        self.G = G
        self.symptom_names = symptom_names

        self.true_diagnosis = patient_row["PATHOLOGY"]
        self.age = patient_row.get("AGE", "unknown")
        self.sex = "male" if str(patient_row.get("SEX")).upper() == "M" else "female"

        evid_list = ast.literal_eval(patient_row["EVIDENCES"])
        self.actual_symptoms = {}
        for ev in evid_list:
            code = ev.split("_@_")[0]
            name = symptom_names.get(code, code)
            weight = G[self.true_diagnosis][name]['weight'] if G.has_edge(self.true_diagnosis, name) else 0.1
            self.actual_symptoms[name] = {"weight": weight, "revealed": False}

        self.revealed_history = []

    def answer_question(self, question):
        q_vec = normalize_vec(encode_text(translate_to_english(question)))

        best_sim, target_symptom = -1, None
        for s_name in self.actual_symptoms.keys():
            s_vec = normalize_vec(encode_text(s_name))
            sim = float(cosine_similarity(q_vec.reshape(1, -1), s_vec.reshape(1, -1))[0][0])
            if sim > best_sim:
                best_sim, target_symptom = sim, s_name

        is_confirmed = best_sim > 0.85
        system_msg = f"You are a {self.age}yo {self.sex} patient with {self.true_diagnosis}. Be natural and brief."

        if is_confirmed:
            prompt = f"""
            Doctor asks: "{question}"
            Medical Fact: Patient HAS the symptom '{target_symptom}'.
            Confidence: {best_sim:.2f}.
            History: Already discussed {', '.join(self.revealed_history)}.
            Instruction: Confirm the symptom naturally. Be brief.
            """
        else:
            prompt = f"""
            Doctor asks: "{question}"
            Medical Fact: Patient DOES NOT have the symptom the doctor is asking about.
            Instruction: Deny the symptom naturally. 
            Example: "No, I haven't noticed any {target_symptom}." 
            """

        response = self.llm.generate_response(prompt, system_prompt=system_msg)

        newly_revealed = []
        if is_confirmed:
            self.actual_symptoms[target_symptom]['revealed'] = True
            if target_symptom not in self.revealed_history:
                self.revealed_history.append(target_symptom)
                newly_revealed.append((target_symptom, best_sim))

        return response, newly_revealed

    def get_initial_complaint(self) -> str:
        top_s = self.revealed_history[:3] if self.revealed_history else list(self.actual_symptoms.keys())[:2]

        prompt = f"""
        You are a patient with {self.true_diagnosis}. 
        YOUR ONLY SYMPTOMS ARE: {', '.join(top_s)}.
        STRICT RULE: Do not mention any other body parts or pains (like joints, head, etc.) if they are not in the list.
        Describe your condition to the doctor in 1-2 natural sentences.
        """
        return self.llm.generate_response(prompt)

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

    def perform_test(self, test_query, procedures_list, diagnosis_tests):
        if not test_query:
            return "Please specify which test you want to perform."

        q_vec = normalize_vec(encode_text(translate_to_english(test_query)))
        best_proc, max_sim = None, -1

        for proc in procedures_list:
            p_vec = normalize_vec(encode_text(proc))
            sim = float(cosine_similarity(q_vec.reshape(1, -1), p_vec.reshape(1, -1))[0][0])
            if sim > max_sim:
                max_sim, best_proc = sim, proc

        if max_sim < 0.85:
            return f"I'm not sure what '{test_query}' is."

        relevant_tests = diagnosis_tests.get(self.true_diagnosis, [])
        if best_proc in relevant_tests:
            return f"Results for '{best_proc}': Abnormal findings consistent with {self.true_diagnosis}."
        else:
            return f"Results for '{best_proc}': Findings are within normal range."