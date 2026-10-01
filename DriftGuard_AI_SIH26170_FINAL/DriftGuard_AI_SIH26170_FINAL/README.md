# DriftGuard - SIH26170 Product UI v6

A runnable prototype for **SIH26170: AI-Driven Anomaly Detection in Component Burn-In & Screening**.

The project analyses early burn-in measurements, detects dynamically unusual components, forecasts the 168h value, estimates risk, explains the decision, and supports an engineer-in-the-loop workflow.

> **Important:** Safety slopes, absolute limits and risk thresholds in `src/config.py` are prototype assumptions only. They are **NOT official ISRO/device acceptance limits**.

## Decision-first prototype + enhanced features

1. **Dynamic anomaly detection** using batch/lot-aware early features.
2. **Anomaly-model comparison**: Isolation Forest, Local Outlier Factor and One-Class SVM.
3. **Recall-aware model selection** using F2 so missed defects are penalized more strongly.
4. **168h prediction** with Linear Regression, Random Forest and Gradient Boosting comparison.
5. **Automatic regressor selection** using held-out Mean Absolute Error (MAE).
6. **Estimated 90% prediction range** based on held-out validation residuals.
7. **Multi-parameter combined risk** across Leakage Current, IDDQ and Propagation Delay when the uploaded file contains them.
8. **Normal / Warning / Critical risk scoring** and an engineering action recommendation.
9. **Root-cause style explanations** showing abnormal lot deviation, anomaly percentile, early drift and predicted safety-limit crossing.
10. **Batch comparison graph** with selected component, batch median and 10-90% population band.
11. **Data-quality gate** for duplicate IDs, missing/non-numeric values, negative values and extreme early jumps.
12. **Real-time burn-in simulator** that demonstrates readings arriving at 0h, 24h, 96h and 168h.
13. **What-if analysis** to change 0h/24h values and instantly see the changed 168h forecast and risk.
14. **Human-in-the-loop feedback**: engineers can confirm defects, mark false alarms, request further testing or confirm normal behaviour.
15. **Historical batch memory** stored locally in SQLite for comparing saved batch risk over time.
16. **Automatic PDF reliability report** for an individual component.
17. **Works with early-only uploads**: a new batch can contain only component ID, lot ID, 0h and 24h values for the chosen parameter.
18. **DriftGuard AI Copilot**: an offline, batch-aware chatbot that answers questions about risky components, explains individual decisions, summarizes batch health, reports model/data quality, combines parameter risk, and recommends what an engineer should inspect next. It uses TF-IDF intent matching plus live analytics, so it does not require a paid LLM API.


### Product-style UI/UX — v6

The interface is intentionally minimal, full-width and decision-first:

- fixed edge-to-edge navigation at the literal top of the viewport, with Streamlit chrome and heading-anchor artifacts removed
- stronger typography, controlled heading widths, and body-text contrast for large-screen presentation
- restrained white / soft-gray surfaces with one blue product accent
- a representative batch decision appears immediately after the hero, before feature marketing
- the Home page follows a clear product story: result preview → capabilities → workflow → human decision support → final CTA
- capabilities use flatter top-rule sections with small signal / forecast / evidence visuals instead of oversized rounded cards
- a connected Upload → Validate → Analyse → Assess process line explains the flow in seconds
- a balanced human-in-the-loop section shows engineering judgment and a grounded Copilot example
- subtle signal, dot-grid and circuit motifs continue across internal pages and the final CTA for visual consistency
- gentle load/reveal motion is used without neon, glassy AI aesthetics or distracting animation
- representative output metrics are flatter and use dividers instead of nested dashboard cards
- all CTA/button labels are protected from clipping or ellipsis; primary blue buttons explicitly force white text in every interaction state
- a real footer carries the SIH26170 / prototype-limit disclaimer consistently
- risk colors remain reserved for actual engineering state
- upload and model execution happen deliberately inside Analyse rather than silently on load
- Results remain decision-first: verdict, ranked Review Queue, component explanation and recommended action before plots
- technical graphs and raw ML signals stay available in Analytics / expandable detail views
- downloadable Engineering Batch Summary PDF and component PDF report remain available

## Official problem-statement mapping

The prototype is designed around the stated SIH requirements:

- Dynamic outlier detection instead of only static pass/fail thresholds.
- Predict `Value_168h` from early readings such as `Value_0h` and `Value_24h`.
- Reduce false negatives because missing a defective part is heavily penalized.
- Evaluate drift prediction using MAE.
- Give an understandable explanation for the classification.

## Quick start - Windows

The easiest method is to double-click:

```text
RUN_DASHBOARD_WINDOWS.bat
```

The first run creates a local virtual environment and installs packages from `requirements.txt`, so internet access is needed once.

### Manual setup

Open PowerShell / CMD inside this folder.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`).

## Retrain the demo models

The ZIP already contains trained demonstration models. To regenerate everything:

```bash
python src/data_generator.py --rows 1200 --out data/demo_burn_in.csv
python src/train_models.py --data data/demo_burn_in.csv --models models
python tests/smoke_test.py
```

## CSV format

### Full demonstration / evaluation file

```text
component_id,lot_id,is_defective,defect_type,
leakage_0h,leakage_24h,leakage_96h,leakage_168h,
iddq_0h,iddq_24h,iddq_96h,iddq_168h,
delay_0h,delay_24h,delay_96h,delay_168h
```

### Minimum early-stage file for one parameter

Example for leakage current:

```text
component_id,lot_id,leakage_0h,leakage_24h
```

The system will still produce:

- anomaly score
- 168h forecast
- estimated prediction range
- risk score
- action recommendation
- explainable reasons

## Dashboard navigation

### Home
Product-style introduction explaining what DriftGuard solves, its three core capabilities, the analysis flow, and example outputs generated from the bundled synthetic demo. No full report is shown here.

### Analyse
Upload a CSV, download a sample input file, inspect the detected parameters and data quality, then explicitly run the batch analysis. The demo dataset can also be selected here.

### Results
Contains five decision-focused views: Summary, Review Queue, Component Inspector, Combined Risk and Data Quality. Summary opens with the batch verdict and recommended review priorities rather than plots.

### Analytics
Contains the technical detail: Batch Analytics, Model Performance, Live Simulator and What-if Analysis. Graph-heavy material is kept here so it does not dominate the main workflow.

### Copilot
Grounded assistant for the active batch. Example questions: `Summarize this batch`, `Which components should I inspect first?`, `Why is CMP-00042 critical?`, and `What should the engineer do next?`.

### History
Saved batch snapshots and engineer review outcomes.

## Folder structure

```text
DriftGuard_AI_SIH26170_Product_UI/
├── app.py
├── cli_demo.py
├── requirements.txt
├── README.md
├── RUN_DASHBOARD_WINDOWS.bat
├── data/
│   ├── demo_burn_in.csv
│   ├── sample_early_only.csv
│   └── driftguard_history.db       # created/updated locally
├── models/
│   ├── *_anomaly.joblib
│   ├── *_regressor.joblib
│   ├── *_metadata.json
│   └── model_comparison_full.csv
├── src/
│   ├── config.py
│   ├── data_generator.py
│   ├── data_quality.py
│   ├── chatbot.py
│   ├── features.py
│   ├── inference.py
│   ├── multi_inference.py
│   ├── reports.py
│   ├── risk_engine.py
│   ├── storage.py
│   └── train_models.py
└── tests/
    └── smoke_test.py
```

## Suggested SIH live demo (3-4 minutes)

1. **Home:** explain the problem, the three capabilities and the example output snapshots in under 30 seconds.
2. **Analyse:** upload `data/sample_early_all_parameters.csv` and run the model using only early readings.
3. **Results → Summary:** show the batch verdict, plain-language findings and priority queue without opening a graph.
4. **Results → Component Inspector:** open the highest-priority component and explain the decision in plain language.
5. **Copilot:** ask it to summarize the batch and explain the highest-risk component.
6. **Analytics → What-if Analysis:** change a 24h value and show the recommendation updating.
7. **Analytics → Batch Analytics:** only open graphs if judges ask for deeper evidence.
8. **Analytics → Model Performance:** emphasize false negatives, F2/recall and MAE.
9. **PDF + feedback:** download the engineering summary or component report and show that a human engineer remains in control.

## Important limitations

This is a hackathon/academic prototype. The bundled dataset is synthetic and the current thresholds are demonstration assumptions. A production system would require real validated burn-in datasets, device-specific limits, calibrated uncertainty, domain review, model monitoring, audit controls and formal validation before any automated acceptance/rejection decision.

## UI quality check for v6

Before packaging, the v6 interface was visually reviewed using local rendered previews at a 1600x1000 desktop viewport, plus a compact-width check. The pass specifically checked:

- top navigation at the true top edge and visible while scrolling
- full-width Home composition rather than a narrow centered page
- CTA labels and primary-button text contrast
- hero headline wrapping and visual balance
- representative output hierarchy and reduced nested-card styling
- capability visual consistency
- continuous workflow line
- balanced human-in-the-loop / Copilot composition
- final CTA and footer consistency
- shared styling for internal Analyse / Results / Analytics workspaces

Backend smoke tests and Copilot smoke tests also pass in this build.

## Final merged build fixes

This final build combines the latest v6 interface with the model compatibility fix.

- Hero reliability graphic is constrained to the viewport so its right side is not clipped on wide screens.
- Hero columns are shrink-safe and use a balanced desktop ratio.
- Horizontal layout overflow is contained without masking oversized hero content.
- scikit-learn is pinned to 1.8.0 so bundled Gradient Boosting regressors deserialize consistently.
- `src/model_preflight.py` verifies leakage, IDDQ and propagation-delay models before the dashboard starts.
- If a saved sklearn model is incompatible, DriftGuard automatically rebuilds only the affected model from the bundled synthetic demo data.
- `REBUILD_MODELS_WINDOWS.bat` can be used to manually regenerate all models with the local environment.

On Windows, use `RUN_DASHBOARD_WINDOWS.bat`. It upgrades dependencies, checks model compatibility, then launches the dashboard.
