import json
import re
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

    _POLITE_PATTERNS = [
        r"\bplease\b", r"\bkindly\b", r"\bwould you\b", r"\bcould you\b",
        r"\bthank(s| you)\b", r"\bi('d| would) like\b", r"\bmay i\b",
        r"\bif you don'?t mind\b", r"\bexcuse me\b", r"\bi appreciate\b",
        r"\bsorry to\b", r"\bi understand\b", r"\bi see\b",
        r"\bwould it be possible\b", r"\bare you comfortable\b",
        r"\bfeel free\b", r"\btake your time\b",
    ]
    _RUDE_PATTERNS = [
        r"\bstupid\b", r"\bidiot\b", r"\bdumb\b", r"\bjust (do|tell|answer)\b",
        r"\bshut up\b", r"\bwhatever\b", r"\bi don'?t care\b",
        r"\bhurry up\b", r"\byou must\b", r"\byou have to\b",
        r"\bdo it now\b", r"\bnow!\b", r"\bimmediately\b",
        r"!{2,}", r"[A-Z]{5,}",
    ]
    _EMPATHY_PATTERNS = [
        r"\bi know (this|it|that)\b", r"\bdon'?t worry\b",
        r"\beverything (will|is)\b", r"\bwe'?re here\b",
        r"\byou'?re (doing|feeling)\b", r"\bi'?m (here|listening)\b",
        r"\blet me (explain|help|clarify)\b", r"\bi'?ll make sure\b",
        r"\bwe can\b", r"\btell me more\b",
    ]

    def score_communication(self, text: str) -> dict:
        low = text.lower()
        pol_hits   = sum(1 for p in self._POLITE_PATTERNS  if re.search(p, low))
        emp_hits   = sum(1 for p in self._EMPATHY_PATTERNS if re.search(p, low))
        rude_flags = [p for p in self._RUDE_PATTERNS       if re.search(p, low)]
        rude_hits  = len(rude_flags)
        words      = max(len(text.split()), 1)
        politeness = min(pol_hits / max(words / 8, 1), 1.0)
        empathy    = min(emp_hits / max(words / 8, 1), 1.0)
        rudeness   = min(rude_hits / 2, 1.0)
        raw        = (politeness + empathy) / 2
        comm_score = max(0.0, raw * (1 - rudeness) - rudeness * 0.15)
        if pol_hits == 0 and emp_hits == 0 and rude_hits == 0:
            comm_score = 0.5
        return {
            "politeness": round(politeness,  3),
            "empathy":    round(empathy,     3),
            "rudeness":   round(rudeness,    3),
            "comm_score": round(comm_score,  3),
            "flags":      rude_flags,
        }

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

        comm = self.score_communication(action_text)
        comm_bonus = 0.05 * comm["comm_score"] - 0.10 * comm["rudeness"]
        step_score = self.w1 * delta_h + self.w2 * r_score - penalty + comm_bonus

        return {
            "r_score":    round(float(r_score), 4),
            "penalty":    round(float(penalty), 4),
            "comm_score": comm["comm_score"],
            "politeness": comm["politeness"],
            "empathy":    comm["empathy"],
            "rudeness":   comm["rudeness"],
            "rude_flags": comm["flags"],
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
            comm_bar = "\u2605" * round(m['comm_score'] * 5) + "\u2606" * (5 - round(m['comm_score'] * 5))
            rude_warn = f"  \u26a0 {m['rude_flags']}" if m['rude_flags'] else ""
            print(f"   IG: {entry.get('ig', 0):+.4f} | Бал: {m['total_step']:.2f} "
                  f"(Penalty: {m['penalty']:.2f}) | "
                  f"Комунікація [{comm_bar}] {m['comm_score']:.2f}{rude_warn}")

        print("-" * 70)
        print(f"ПІДСУМКОВИЙ БАЛ КОМПЕТЕНЦІЙ: {round(total_score, 2)}")
        print(f"Загальний інф. приріст (IG): {round(total_ig, 4)}")

        all_m = [self.calculate_step_metrics(e['query'], true_diagnosis, e.get('ig', 0))
                 for e in session_log]
        avg_comm  = sum(m['comm_score'] for m in all_m) / max(len(all_m), 1)
        avg_pol   = sum(m['politeness'] for m in all_m) / max(len(all_m), 1)
        avg_emp   = sum(m['empathy']    for m in all_m) / max(len(all_m), 1)
        n_rude    = sum(1 for m in all_m if m['rudeness'] > 0)
        cbar = "\u2605" * round(avg_comm * 5) + "\u2606" * (5 - round(avg_comm * 5))
        print(f"\nКОМУНІКАЦІЯ [{cbar}] {avg_comm:.2f}  "
              f"(ввічливість: {avg_pol:.2f} | емпатія: {avg_emp:.2f})")
        if n_rude:
            print(f"  \u26a0  Грубих реплік: {n_rude} \u2014 знижує загальну оцінку!")

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