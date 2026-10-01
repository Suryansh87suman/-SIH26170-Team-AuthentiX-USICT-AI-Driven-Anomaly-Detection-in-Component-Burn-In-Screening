from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import PARAMETERS
from multi_inference import score_all_parameters, supported_parameters
from risk_engine import explain_row, recommendation_for_row
from storage import load_feedback, load_history


INTENT_EXAMPLES = {
    "batch_summary": [
        "summarize this batch", "give me batch summary", "how is this batch looking",
        "how many components are risky", "overall status", "batch overview",
        "how many normal warning critical", "what is happening in this batch",
    ],
    "highest_risk": [
        "which components are most risky", "show highest risk components", "top risky components",
        "what should i inspect first", "worst components", "most critical components",
        "show top anomalies", "which parts should engineer review first",
    ],
    "data_quality": [
        "is the data good", "data quality", "are there missing values", "is csv valid",
        "any duplicate rows", "can i trust this dataset", "problems in uploaded data",
    ],
    "model_performance": [
        "how accurate is the model", "model performance", "what model is used",
        "what is mae", "false negatives", "recall precision f2", "which regressor won",
        "which anomaly detector is selected",
    ],
    "combined_risk": [
        "combined risk", "all parameters together", "overall component risk",
        "combine leakage iddq delay", "multi parameter risk", "which parameter dominates",
    ],
    "prediction_range": [
        "what is 90 percent range", "explain prediction interval", "confidence range",
        "why is there lower and upper prediction", "prediction uncertainty",
    ],
    "risk_meaning": [
        "what does risk score mean", "what does critical mean", "what is warning",
        "how risk is calculated", "why early reject", "what is early rejection",
    ],
    "next_action": [
        "what should i do next", "recommend next step", "what should engineer do",
        "which parts need review", "give action plan", "what action should be taken",
    ],
    "history": [
        "show previous batches", "batch history", "historical batches", "past batch risk",
        "what did previous batches look like", "saved history",
    ],
    "feedback": [
        "engineer feedback", "human feedback", "how many false alarms", "review outcomes",
        "what engineers marked", "human in the loop results",
    ],
    "parameter_info": [
        "what parameter am i viewing", "what is leakage", "what is iddq",
        "what is propagation delay", "what limit is used", "what safety slope is used",
    ],
    "help": [
        "help", "what can i ask", "chatbot commands", "what can you do", "examples",
    ],
}


def _detect_intent(question: str) -> tuple[str, float]:
    q = (question or "").strip().lower()
    examples, labels = [], []
    for label, samples in INTENT_EXAMPLES.items():
        examples.extend(samples)
        labels.extend([label] * len(samples))
    corpus = examples + [q]
    vect = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vect.fit_transform(corpus)
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    if not len(sims):
        return "help", 0.0
    idx = int(np.argmax(sims))
    return labels[idx], float(sims[idx])


def _component_match(question: str, component_ids: Iterable[str]) -> str | None:
    q = (question or "").lower()
    ids = [str(x) for x in component_ids]
    # Prefer a literal ID present in the question.
    for cid in ids:
        if cid.lower() in q:
            return cid
    # Common synthetic ID style: CMP-00042 / C003 etc.
    candidates = re.findall(r"\b(?:cmp[-_ ]?\d+|c\d{2,})\b", q, flags=re.I)
    normalized = {re.sub(r"[-_ ]", "", x.lower()): x for x in ids}
    for c in candidates:
        key = re.sub(r"[-_ ]", "", c.lower())
        if key in normalized:
            return normalized[key]
    return None


def _fmt(v, digits=2):
    try:
        if pd.isna(v):
            return "N/A"
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _component_answer(row: pd.Series, parameter: str) -> str:
    cfg = PARAMETERS[parameter]
    reasons = explain_row(row, parameter)
    bullets = "\n".join(f"- {r}" for r in reasons)
    early = "Yes — engineering review is recommended" if bool(row.get("early_reject", False)) else "No — continue testing under the prototype rule"
    return (
        f"### {row['component_id']} — {cfg.label}\n"
        f"- **Lot:** {row.get('lot_id', 'N/A')}\n"
        f"- **0h:** {_fmt(row.get(f'{parameter}_0h'))} {cfg.unit}\n"
        f"- **24h:** {_fmt(row.get(f'{parameter}_24h'))} {cfg.unit}\n"
        f"- **Predicted 168h:** **{_fmt(row.get('predicted_168h'))} {cfg.unit}**\n"
        f"- **Estimated 90% range:** {_fmt(row.get('prediction_lower_90'))}–{_fmt(row.get('prediction_upper_90'))} {cfg.unit}\n"
        f"- **Risk:** **{_fmt(row.get('risk_score'), 1)}/100 ({row.get('risk_level', 'N/A')})**\n"
        f"- **Anomaly percentile:** {_fmt(float(row.get('anomaly_percentile', 0))*100, 1)}%\n"
        f"- **Early reject/review:** {early}\n"
        f"- **Recommended action:** {recommendation_for_row(row)}\n\n"
        f"**Why the model says this:**\n{bullets}\n\n"
        f"_Note: limits and safety slopes are prototype assumptions, not official ISRO/device acceptance limits._"
    )


def answer_question(
    question: str,
    df: pd.DataFrame,
    scored: pd.DataFrame,
    parameter: str,
    quality: dict,
    meta: dict,
    model_dir: str | Path,
    db_path: str | Path,
) -> str:
    """Return a grounded conversational answer using current batch/model outputs.

    This intentionally avoids external LLM/API dependencies. It uses TF-IDF intent
    matching plus deterministic analytics so answers remain tied to the current data.
    """
    q = (question or "").strip()
    if not q:
        return "Ask me about this batch, a component ID, model performance, data quality, combined risk, or what to inspect next."

    cid = _component_match(q, scored["component_id"].astype(str).tolist())
    if cid is not None:
        row = scored.loc[scored["component_id"].astype(str) == cid].iloc[0]
        return _component_answer(row, parameter)

    ql = q.lower()
    # A few high-value deterministic rules before semantic intent matching.
    if any(x in ql for x in ["why critical", "why risky", "why anomaly", "explain component", "explain this component"]):
        return "Tell me the component ID too — for example: **Why is CMP-00042 critical?**"

    intent, confidence = _detect_intent(q)
    cfg = PARAMETERS[parameter]

    if intent == "batch_summary":
        counts = scored["risk_level"].value_counts().to_dict()
        top = scored.sort_values("risk_score", ascending=False).iloc[0]
        return (
            f"### Current batch summary — {cfg.label}\n"
            f"- **Components analysed:** {len(scored):,}\n"
            f"- **Normal:** {int(counts.get('Normal', 0))}\n"
            f"- **Warning:** {int(counts.get('Warning', 0))}\n"
            f"- **Critical:** {int(counts.get('Critical', 0))}\n"
            f"- **Early-review flags:** {int(scored['early_reject'].sum())}\n"
            f"- **Detected anomalies:** {int(scored['is_anomaly'].sum())}\n"
            f"- **Mean risk:** {_fmt(scored['risk_score'].mean(), 1)}/100\n"
            f"- **Highest-risk component:** **{top['component_id']}** at {_fmt(top['risk_score'], 1)}/100\n"
            f"- **Data-quality score:** {quality.get('score', 'N/A')}/100\n\n"
            f"The fastest next check is to inspect **{top['component_id']}** and then the other highest-risk components."
        )

    if intent == "highest_risk":
        top = scored.sort_values("risk_score", ascending=False).head(5)
        lines = ["### Components to inspect first"]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            lines.append(
                f"{i}. **{r['component_id']}** — {r['risk_score']:.1f}/100 ({r['risk_level']}), "
                f"predicted 168h {_fmt(r['predicted_168h'])} {cfg.unit}, action: {recommendation_for_row(r)}"
            )
        lines.append("\nAsk **Why is <component ID> critical?** for a detailed explanation.")
        return "\n".join(lines)

    if intent == "data_quality":
        issues = quality.get("issues", [])
        if not issues:
            issue_text = "No issues were detected by the prototype quality checks."
        else:
            issue_text = "\n".join(
                f"- **{i.get('check', 'Issue')} ({i.get('severity', 'Info')}):** {i.get('details', i)}" for i in issues[:8]
            )
        return (
            f"### Data-quality check\n"
            f"- **Score:** {quality.get('score', 'N/A')}/100\n"
            f"- **Rows:** {quality.get('rows', len(df))}\n"
            f"- **Issues found:** {quality.get('issue_count', len(issues))}\n\n"
            f"{issue_text}\n\n"
            "The checker looks for duplicate IDs, missing/non-numeric readings, negative measurements, and suspiciously large early jumps."
        )

    if intent == "model_performance":
        am = meta.get("anomaly_metrics", {})
        return (
            f"### Current model setup — {cfg.label}\n"
            f"- **Selected 168h regressor:** {meta.get('selected_regressor', 'N/A').replace('_', ' ').title()}\n"
            f"- **Selected anomaly detector:** {meta.get('selected_anomaly_model', 'N/A').replace('_', ' ').title()}\n"
            f"- **Validation recall:** {_fmt(am.get('recall', np.nan), 3)}\n"
            f"- **Validation precision:** {_fmt(am.get('precision', np.nan), 3)}\n"
            f"- **Validation F2:** {_fmt(am.get('f2', np.nan), 3)}\n"
            f"- **False negatives:** {am.get('false_negatives', 'N/A')}\n"
            f"- **Selected regressor validation MAE:** {_fmt(meta.get('regression_models', {}).get(meta.get('selected_regressor', ''), {}).get('mae', np.nan), 3)} {cfg.unit}\n\n"
            "F2 is emphasized because this prototype gives extra importance to recall — missing a defective component is more serious than reviewing an extra suspicious one."
        )

    if intent == "combined_risk":
        params = supported_parameters(df)
        if len(params) < 2:
            return "This upload currently has only one complete supported parameter set (0h + 24h). Upload data containing at least two of Leakage, IDDQ, and Propagation Delay to calculate a meaningful combined risk."
        combined, _ = score_all_parameters(df, model_dir)
        top = combined.sort_values("combined_risk_score", ascending=False).head(5)
        lines = [f"### Combined risk across {', '.join(p.upper() for p in params)}"]
        lines.append(f"- **Critical:** {int((combined['combined_risk_level'] == 'Critical').sum())}")
        lines.append(f"- **Combined early-review flags:** {int(combined['combined_early_reject'].sum())}")
        lines.append("\n**Highest combined-risk components:**")
        for _, r in top.iterrows():
            lines.append(
                f"- **{r['component_id']}** — {r['combined_risk_score']:.1f}/100, dominant signal: **{r['dominant_risk_parameter']}**"
            )
        return "\n".join(lines)

    if intent == "prediction_range":
        q90 = float(meta.get("selected_regressor_error_abs_q90", 0.0))
        return (
            "### What the 90% prediction range means\n"
            f"For this demo model, the held-out validation errors were analysed and the **90th percentile absolute error is about {q90:.3f} {cfg.unit}**. "
            "The dashboard therefore displays the predicted 168h value ± that empirical error amount.\n\n"
            "It is an **estimated prototype prediction range**, not a formally calibrated statistical confidence interval and not an official engineering tolerance."
        )

    if intent == "risk_meaning":
        return (
            "### How the prototype risk score works\n"
            "The score combines three signals:\n"
            "- **45% anomaly behaviour** — how unusual the component looks compared with the analysed batch.\n"
            "- **35% predicted drift** — whether the forecasted 24h→168h slope is large.\n"
            "- **20% proximity to the prototype absolute limit.**\n\n"
            "Risk bands are **Normal (≤35), Warning (>35 to 65), Critical (>65)**. "
            "An early-review flag can also be triggered when predicted drift exceeds the prototype safety slope, the prediction exceeds the prototype limit, or a highly anomalous component also has elevated risk.\n\n"
            "These thresholds are demonstration assumptions and must be replaced with validated device-specific criteria in a real deployment."
        )

    if intent == "next_action":
        critical = scored[scored["risk_level"] == "Critical"].sort_values("risk_score", ascending=False)
        early = scored[scored["early_reject"]].sort_values("risk_score", ascending=False)
        targets = early if len(early) else critical
        if targets.empty:
            return (
                "No component currently triggers an early-review rule. I would still inspect the highest anomaly-percentile components, verify data quality, and continue normal burn-in testing."
            )
        ids = ", ".join(targets.head(5)["component_id"].astype(str).tolist())
        return (
            f"### Suggested engineer action\n"
            f"1. Review the early-review components first: **{ids}**.\n"
            "2. Check their raw 0h/24h readings and lot context for sensor/unit/data-entry errors.\n"
            "3. Compare the predicted 168h drift with the device-specific engineering limit when that limit is available.\n"
            "4. Record the engineer conclusion in **Component Inspector → Engineer feedback**.\n"
            "5. Do not use this prototype alone for production acceptance/rejection decisions."
        )

    if intent == "history":
        hist = load_history(db_path)
        if hist.empty:
            return "No historical batch snapshots have been saved yet. Go to **Overview → Save batch snapshot** first."
        latest = hist.iloc[0]
        return (
            f"### Local batch history\n"
            f"- **Saved snapshots:** {len(hist)}\n"
            f"- **Latest batch:** {latest['batch_name']} ({latest['parameter']})\n"
            f"- **Latest mean risk:** {_fmt(latest['mean_risk'], 1)}/100\n"
            f"- **Latest max risk:** {_fmt(latest['max_risk'], 1)}/100\n"
            f"- **Latest early-review count:** {int(latest['early_reject_count'])}\n\n"
            "Open **History & Feedback** for the complete table and trend chart."
        )

    if intent == "feedback":
        fb = load_feedback(db_path)
        if fb.empty:
            return "No engineer feedback has been saved yet. Open a component in **Component Inspector**, choose the engineer conclusion, and save it."
        counts = fb["engineer_label"].value_counts().to_dict()
        parts = [f"- **{k}:** {v}" for k, v in counts.items()]
        return "### Engineer feedback so far\n" + "\n".join(parts) + f"\n\nTotal reviewed decisions: **{len(fb)}**."

    if intent == "parameter_info":
        return (
            f"### Current primary parameter: {cfg.label}\n"
            f"- **Unit:** {cfg.unit}\n"
            f"- **Prototype absolute limit:** {cfg.absolute_max} {cfg.unit}\n"
            f"- **Prototype safety slope:** {cfg.safety_slope_per_hour} {cfg.unit}/hour\n"
            f"- **Model uses:** 0h and 24h readings plus derived early-drift/lot-relative features to predict 168h behaviour.\n\n"
            "These are demo thresholds in `src/config.py`, not official ISRO/device specifications."
        )

    if intent == "help" or confidence < 0.12:
        return (
            "### I can answer questions such as\n"
            "- **Summarize this batch**\n"
            "- **Which components should I inspect first?**\n"
            "- **Why is CMP-00042 critical?**\n"
            "- **How accurate is the model?**\n"
            "- **Is the uploaded data good?**\n"
            "- **Explain the 90% prediction range**\n"
            "- **What does the risk score mean?**\n"
            "- **Show combined risk across all parameters**\n"
            "- **What should the engineer do next?**\n"
            "- **Show batch history / engineer feedback**\n\n"
            "I answer from the **currently loaded batch, trained-model metadata, and local history**, so I stay grounded in the prototype rather than inventing acceptance rules."
        )

    return (
        "I’m not fully sure what you mean. Try asking about the batch summary, highest-risk components, a specific component ID, data quality, model performance, combined risk, prediction range, history, or next engineering action."
    )
