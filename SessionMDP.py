from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from BaseScoringLogic import shannon_entropy, compute_probabilities
from UtilsBert import embed_action, encode_text, normalize_vec, translate_to_english

@dataclass
class MDPStep:
    step: int
    action_text: str
    action_type: str

    H_before: float = 0.0
    H_after: float = 0.0
    delta_h: float = 0.0
    r_protocol: float = 0.0
    penalty: float = 0.0
    reward: float = 0.0

    optimal_action: str = ""
    optimal_reward: float = 0.0
    q_student: float = 0.0
    q_optimal: float = 0.0
    regret: float = 0.0


@dataclass
class SessionReport:
    steps: list[MDPStep] = field(default_factory=list)
    total_reward_student: float = 0.0
    total_reward_optimal: float = 0.0
    total_regret: float = 0.0
    student_diagnosis: str = ""
    true_diagnosis: str = ""
    diagnosis_correct: bool = False
    terminal_bonus: float = 0.0   # бонус за правильний кінцевий діагноз
    final_score: float = 0.0

def compute_reward(
    delta_h: float,
    r_protocol: float,
    penalty: float,
    action_type: str,
    *,
    w1: float = 0.6,
    w2: float = 0.4,
    test_cost: float = 0.05,
) -> float:
    cost = test_cost if action_type == "test" else 0.0
    return w1 * delta_h + w2 * r_protocol - penalty - cost

class SessionMDP:
    def __init__(
        self,
        patient_model,
        tau_fn,
        candidate_questions: list[str],
        procedures: list[str],
        evaluation_engine,
        gamma: float = 0.85,
        terminal_bonus: float = 2.0,
        reward_kwargs: dict | None = None,
    ):
        self.model = patient_model
        self.tau_fn = tau_fn
        self.candidate_questions = list(candidate_questions)
        self.procedures = list(procedures)
        self.engine = evaluation_engine
        self.gamma = gamma
        self.terminal_bonus = terminal_bonus
        self.reward_kwargs = reward_kwargs or {}

        self._steps: list[MDPStep] = []
        self._snapshots: list[dict] = []   # стан моделі ДО кожного кроку студента

    def record_student_action(
        self,
        step: int,
        action_text: str,
        action_type: str,
        delta_h: float,
        true_diagnosis: str,
        H_before: float,
        H_after: float,
        snapshot_before: dict,
    ) -> MDPStep:
        metrics = self.engine.calculate_step_metrics(
            action_text, true_diagnosis, delta_h
        )

        reward = compute_reward(
            delta_h,
            metrics["r_score"],
            metrics["penalty"],
            action_type,
            **self.reward_kwargs,
        )

        mdp_step = MDPStep(
            step=step,
            action_text=action_text,
            action_type=action_type,
            H_before=H_before,
            H_after=H_after,
            delta_h=delta_h,
            r_protocol=metrics["r_score"],
            penalty=metrics["penalty"],
            reward=reward,
        )

        self._steps.append(mdp_step)
        self._snapshots.append(snapshot_before)
        return mdp_step

    def _get_action_vec(self, action: str, a_type: str) -> np.ndarray:
        if a_type == "test":
            return embed_action(action)
        return normalize_vec(encode_text(translate_to_english(action)))

    def _simulate_one_step(
        self,
        action: str,
        a_type: str,
        true_diagnosis: str,
        step: int,
        snap: dict,
    ) -> tuple[float, float, float]:
        self.model.restore(snap)

        tau = self.tau_fn(step)
        H_before = shannon_entropy(
            compute_probabilities(self.model.get_diagnosis_activations(), tau)
        )

        a_vec = self._get_action_vec(action, a_type)
        self.model.update_state(a_vec, revealed_symptoms=None)

        H_after = shannon_entropy(
            compute_probabilities(self.model.get_diagnosis_activations(), tau)
        )
        delta_h = H_before - H_after

        metrics = self.engine.calculate_step_metrics(action, true_diagnosis, delta_h)
        reward = compute_reward(
            delta_h,
            metrics["r_score"],
            metrics["penalty"],
            a_type,
            **self.reward_kwargs,
        )

        return reward, H_before, H_after

    def _find_optimal_action(
        self,
        snap: dict,
        step: int,
        asked: set[str],
        true_diagnosis: str,
    ) -> tuple[str, str, float]:
        pool = (
            [(q, "question") for q in self.candidate_questions if q not in asked]
            + [(p, "test") for p in self.procedures if p not in asked]
        )

        if not pool:
            return "", "question", 0.0

        best_action, best_type, best_r = "", "question", -np.inf

        for action, a_type in pool:
            reward, _, _ = self._simulate_one_step(
                action, a_type, true_diagnosis, step, snap
            )
            self.model.restore(snap)

            if reward > best_r:
                best_r = reward
                best_action = action
                best_type = a_type

        return best_action, best_type, best_r

    def _compute_discounted_return(
        self, start_idx: int
    ) -> float:
        total = 0.0
        for k, step in enumerate(self._steps[start_idx:]):
            total += (self.gamma ** k) * step.reward
        return total

    def finalize(
        self,
        true_diagnosis: str,
        student_diagnosis: str,
    ) -> SessionReport:
        is_correct = (
            student_diagnosis.strip().lower() == true_diagnosis.strip().lower()
        )
        t_bonus = self.terminal_bonus if is_correct else 0.0

        asked_so_far: set[str] = set()

        for i, mdp_step in enumerate(self._steps):
            mdp_step.q_student = self._compute_discounted_return(i)

            snap = self._snapshots[i]
            opt_action, opt_type, opt_r_immediate = self._find_optimal_action(
                snap, mdp_step.step, asked_so_far, true_diagnosis
            )

            mdp_step.optimal_action = opt_action
            mdp_step.optimal_reward = opt_r_immediate
            mdp_step.q_optimal = opt_r_immediate
            mdp_step.regret = max(0.0, opt_r_immediate - mdp_step.reward)

            asked_so_far.add(mdp_step.action_text)

        total_student = sum(s.reward for s in self._steps) + t_bonus
        total_optimal = sum(s.q_optimal for s in self._steps) + t_bonus
        total_regret = sum(s.regret for s in self._steps)

        return SessionReport(
            steps=self._steps,
            total_reward_student=round(total_student, 4),
            total_reward_optimal=round(total_optimal, 4),
            total_regret=round(total_regret, 4),
            student_diagnosis=student_diagnosis,
            true_diagnosis=true_diagnosis,
            diagnosis_correct=is_correct,
            terminal_bonus=t_bonus,
            final_score=round(total_student, 4),
        )

    @staticmethod
    def print_comparison_report(report: SessionReport) -> None:
        W = 72
        print("\n" + "═" * W)
        print(f"{'MDP АНАЛІЗ СЕСІЇ':^{W}}")
        print("═" * W)

        diag_status = "ПРАВИЛЬНО" if report.diagnosis_correct else "НЕПРАВИЛЬНО"
        print(f"Справжній діагноз : {report.true_diagnosis}")
        print(f"Діагноз студента  : {report.student_diagnosis}  [{diag_status}]")
        if report.terminal_bonus > 0:
            print(f"Бонус за діагноз  : +{report.terminal_bonus:.2f}")
        print("-" * W)

        print(f"{'Крок':<5} {'Дія студента':<32} {'R студ':>8} {'R опт':>8} {'Regret':>8}")
        print("-" * W)

        for s in report.steps:
            student_label = s.action_text[:30] + ".." if len(s.action_text) > 30 else s.action_text
            flag = "  ⚠ RED FLAG" if s.penalty > 0 else ""
            print(
                f"{s.step:<5} {student_label:<32} "
                f"{s.reward:>+8.3f} {s.optimal_reward:>+8.3f} {s.regret:>8.3f}"
                f"{flag}"
            )

            if s.regret > 0.05 and s.optimal_action:
                opt_label = s.optimal_action[:50]
                print(f"      ↳ Оптимально: «{opt_label}»")

        print("-" * W)
        efficiency = (
            report.total_reward_student / report.total_reward_optimal * 100
            if report.total_reward_optimal > 0 else 0.0
        )
        print(f"Сумарна нагорода студента : {report.total_reward_student:+.4f}")
        print(f"Сумарна нагорода оптимуму : {report.total_reward_optimal:+.4f}")
        print(f"Загальний regret          : {report.total_regret:.4f}")
        print(f"Ефективність стратегії    : {efficiency:.1f}%")
        print("═" * W + "\n")