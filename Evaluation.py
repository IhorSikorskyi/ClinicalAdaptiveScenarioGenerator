import json
from sklearn.metrics.pairwise import cosine_similarity

from UtilsBert import encode_text

class EvaluationEngine:
    DEFAULT_PROTOCOLS = {
        "Localized edema": {
            "gold_standard": [
                "Apply cold compress and elevate the limb",
                "Prescribe non-steroidal anti-inflammatory drugs",
                "Perform lymphatic drainage massage",
            ],
            "red_flags": [
                "Apply intense heat to the swelling",
                "Administer aggressive deep tissue massage",
                "Prescribe medication that increases fluid retention",
            ],
        },
        "Spontaneous pneumothorax": {
            "gold_standard": [
                "Administer supplemental oxygen",
                "Perform needle decompression if tension is present",
                "Prepare for chest tube insertion",
            ],
            "red_flags": [
                "Encourage deep forceful coughing",
                "Perform chest physiotherapy with heavy percussion",
            ],
        }
    }

    def __init__(self, protocols: dict = None,
                 w1: float = 0.6, w2: float = 0.4,
                 w_side: float = 1.5, penalty_threshold: float = 0.82):
        self.w1 = w1
        self.w2 = w2
        self.w_side = w_side
        self.penalty_threshold = penalty_threshold

        raw_protocols = protocols if protocols is not None else self.DEFAULT_PROTOCOLS
        self.eval_data = self._vectorize_protocols(raw_protocols)
        print(f"EvaluationEngine: векторизовано протоколи для {len(self.eval_data)} діагнозів")

    def _vectorize_protocols(self, protocols: dict) -> dict:
        result = {}
        for diagnosis, data in protocols.items():
            result[diagnosis] = {
                "gold_standard": [encode_text(t) for t in data.get("gold_standard", [])],
                "red_flags": [encode_text(t) for t in data.get("red_flags", [])],
            }
        return result

    def calculate_step_metrics(self, action_text: str,
                               true_diagnosis: str,
                               delta_h: float) -> dict:
        v_at = encode_text(action_text)
        protocol = self.eval_data.get(true_diagnosis, {"gold_standard": [], "red_flags": []})

        r_score = 0.0
        if protocol["gold_standard"]:
            r_score = max(
                cosine_similarity(v_at.reshape(1, -1), v.reshape(1, -1))[0][0]
                for v in protocol["gold_standard"]
            )

        penalty = 0.0
        if protocol["red_flags"]:
            max_p = max(
                cosine_similarity(v_at.reshape(1, -1), v.reshape(1, -1))[0][0]
                for v in protocol["red_flags"]
            )
            if max_p > self.penalty_threshold:
                penalty = self.w_side * max_p

        step_score = self.w1 * delta_h + self.w2 * r_score - penalty

        return {
            "r_score": round(float(r_score), 4),
            "penalty": round(float(penalty), 4),
            "total_step": round(float(step_score), 4),
        }

    def generate_session_report(self, session_log: list, true_diagnosis: str, user_diagnosis: str):
        print("\n" + "═" * 70)
        print(f"{'ФІНАЛЬНИЙ ЗВІТ ПРО КОМПЕТЕНЦІЇ':^70}")
        print("═" * 70)

        is_correct = user_diagnosis.strip().lower() == true_diagnosis.strip().lower()
        status = "ПРАВИЛЬНО" if is_correct else "НЕПРАВИЛЬНО"

        print(f"Справжній діагноз: {true_diagnosis}")
        print(f"Ваш діагноз:       {user_diagnosis} ({status})")
        print("-" * 70)

        total_score = 0.0
        total_ig = 0.0
        critical_errors = 0

        for entry in session_log:
            m = self.calculate_step_metrics(entry['query'], true_diagnosis, entry.get('ig', 0))

            total_score += m['total_step']
            total_ig += entry.get('ig', 0)
            if m['penalty'] > 0:
                critical_errors += 1

            type_label = "ТЕСТ" if entry.get('type') == 'test' else "ПИТАННЯ"
            print(f"Крок {entry['step']} [{type_label}]: '{entry['query']}'")
            print(f"   IG: {entry.get('ig', 0):+.4f} | Бал: {m['total_step']:.2f} (Penalty: {m['penalty']:.2f})")

        print("-" * 70)
        print(f"ПІДСУМКОВИЙ БАЛ КОМПЕТЕНЦІЙ: {round(total_score, 2)}")
        print(f"Загальний інф. приріст (IG): {round(total_ig, 4)}")

        if critical_errors > 0:
            print(f"УВАГА: Виявлено {critical_errors} критичних помилок (Red Flags)!")

        if not is_correct:
            print("\nПорада: Спробуйте більше фокусуватися на диференційній діагностиці.")
        print("═" * 70 + "\n")

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "EvaluationEngine":
        with open(path, "r", encoding="utf-8") as f:
            protocols = json.load(f)
        return cls(protocols=protocols, **kwargs)