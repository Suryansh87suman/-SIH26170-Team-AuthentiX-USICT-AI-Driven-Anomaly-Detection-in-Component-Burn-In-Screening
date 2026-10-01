from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import PARAMETERS
from inference import score_dataframe
from multi_inference import score_all_parameters, supported_parameters
from risk_engine import explain_row, recommendation_for_row
from data_quality import assess_data_quality
from storage import init_db, save_batch_snapshot, save_feedback, load_history, load_feedback
from reports import component_report_pdf, batch_summary_pdf
from chatbot import answer_question

DB_PATH = ROOT / "data" / "driftguard_history.db"
init_db(DB_PATH)

st.set_page_config(
    page_title="DriftGuard | Reliability Analysis",
    page_icon="DG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pio.templates.default = "plotly_white"

st.markdown(
    r'''<style>
:root {
  --nav-h: 64px;
  --page-pad: clamp(24px, 4vw, 64px);
  --content-max: 1480px;
  --text: #1d1d1f;
  --body: #545a63;
  --muted: #757b84;
  --line: #e4e7eb;
  --line-strong: #d6dbe1;
  --soft: #f6f8fb;
  --soft-blue: #f3f7fc;
  --blue: #0a66c2;
  --blue-hover: #0758aa;
  --green: #248a3d;
  --amber: #a45f00;
  --red: #c9342f;
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  text-rendering: optimizeLegibility;
}
html { scroll-behavior: smooth; }
body, .stApp { background: #fff !important; color: var(--text) !important; }

/* Remove Streamlit chrome and heading anchors. */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu,
footer { display: none !important; height: 0 !important; min-height: 0 !important; }
h1 a, h2 a, h3 a, h4 a,
[data-testid="stHeaderActionElements"] { display: none !important; visibility: hidden !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stMain {
  padding-top: 0 !important;
  margin-top: 0 !important;
  overflow-x: hidden !important;
}
.block-container {
  width: 100% !important;
  max-width: none !important;
  box-sizing: border-box !important;
  padding-top: var(--nav-h) !important;
  padding-right: var(--page-pad) !important;
  padding-bottom: 4.5rem !important;
  padding-left: var(--page-pad) !important;
}

/* Global text hierarchy. */
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.035em; }
p, .stMarkdown p, .section-sub, .verdict-copy, .finding span { color: var(--body) !important; }
.stCaption, [data-testid="stCaptionContainer"], .micro-note { color: var(--muted) !important; }
hr { border-color: var(--line) !important; }

/* Fixed, true top navigation. */
.st-key-topbar_nav {
  position: fixed !important;
  inset: 0 0 auto 0 !important;
  width: 100vw !important;
  height: var(--nav-h) !important;
  min-height: var(--nav-h) !important;
  margin: 0 !important;
  padding: 0 var(--page-pad) !important;
  z-index: 999999 !important;
  background: rgba(255,255,255,.94) !important;
  border-bottom: 1px solid rgba(210,214,219,.88) !important;
  backdrop-filter: blur(18px) saturate(150%);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  box-shadow: 0 2px 16px rgba(30,42,58,.035) !important;
  overflow: visible !important;
}
.st-key-topbar_nav > div,
.st-key-topbar_nav [data-testid="stVerticalBlock"],
.st-key-topbar_nav [data-testid="stHorizontalBlock"],
.st-key-topbar_nav [data-testid="stColumn"] {
  margin-top: 0 !important;
  margin-bottom: 0 !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  transform: none !important;
  top: auto !important;
}
.st-key-topbar_nav > div,
.st-key-topbar_nav [data-testid="stHorizontalBlock"] {
  width: 100% !important;
  max-width: var(--content-max) !important;
  height: var(--nav-h) !important;
  min-height: var(--nav-h) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  align-items: center !important;
  gap: .3rem !important;
}
.st-key-topbar_nav .top-brand {
  display: flex !important;
  align-items: center !important;
  min-height: 2.25rem !important;
  padding: 0 !important;
  margin: 0 !important;
}
.st-key-topbar_nav .top-brand-name {
  font-size: 1.08rem !important;
  line-height: 1 !important;
  font-weight: 735 !important;
  letter-spacing: -.035em !important;
  color: var(--text) !important;
}
.st-key-topbar_nav .stButton,
.st-key-topbar_nav .stButton > button { margin: 0 !important; }
.st-key-topbar_nav .stButton > button {
  position: relative !important;
  width: auto !important;
  min-width: max-content !important;
  min-height: 2.35rem !important;
  height: 2.35rem !important;
  padding: .15rem .62rem !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #656a72 !important;
  font-size: .82rem !important;
  font-weight: 520 !important;
  overflow: visible !important;
}
.st-key-topbar_nav .stButton > button p,
.st-key-topbar_nav .stButton > button span,
.st-key-topbar_nav .stButton > button div {
  color: inherit !important;
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
.st-key-topbar_nav .stButton > button:hover { color: var(--text) !important; background: transparent !important; }
.st-key-topbar_nav .stButton > button[kind="primary"] { color: var(--text) !important; font-weight: 660 !important; }
.st-key-topbar_nav .stButton > button[kind="primary"]::after {
  content: "";
  position: absolute;
  left: .62rem;
  right: .62rem;
  bottom: -.72rem;
  height: 2px;
  border-radius: 2px;
  background: var(--text);
}

/* Global button system: never clip text; primary always white on blue. */
.stButton > button, .stDownloadButton > button {
  width: auto !important;
  min-width: max-content !important;
  min-height: 2.65rem !important;
  padding: .34rem 1.1rem !important;
  border-radius: 999px !important;
  border: 1px solid var(--line-strong) !important;
  background: #fff !important;
  color: var(--text) !important;
  font-weight: 610 !important;
  box-shadow: none !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
.stButton > button p, .stButton > button span, .stButton > button div,
.stDownloadButton > button p, .stDownloadButton > button span, .stDownloadButton > button div {
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #f7f8fa !important;
  border-color: #bfc5cc !important;
}
.stButton > button[kind="primary"] {
  background: var(--blue) !important;
  border-color: var(--blue) !important;
  color: #fff !important;
  box-shadow: 0 7px 18px rgba(10,102,194,.14) !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div { color: #fff !important; }
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active {
  background: var(--blue-hover) !important;
  border-color: var(--blue-hover) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span,
.stButton > button[kind="primary"]:focus p,
.stButton > button[kind="primary"]:active p { color: #fff !important; }
/* Nav active button is intentionally not blue. */
.st-key-topbar_nav .stButton > button[kind="primary"],
.st-key-topbar_nav .stButton > button[kind="primary"] p,
.st-key-topbar_nav .stButton > button[kind="primary"] span,
.st-key-topbar_nav .stButton > button[kind="primary"] div {
  background: transparent !important;
  border-color: transparent !important;
  color: var(--text) !important;
  box-shadow: none !important;
}

/* Home hero: wide, balanced, visible without excess scroll. */
.st-key-home_hero {
  position: relative;
  isolation: isolate;

  /* Full-bleed section: break out of Streamlit's padded .block-container. */
  width: 100vw !important;
  max-width: 100vw !important;
  margin-left: calc(50% - 50vw) !important;
  margin-right: calc(50% - 50vw) !important;

  padding: clamp(3.1rem, 4.8vw, 5.2rem) 0 clamp(3rem, 4.2vw, 4.4rem) !important;
  box-sizing: border-box !important;
  overflow: hidden !important;

  min-height: clamp(570px, 72vh, 720px);
  display: flex;
  align-items: center;

  background:
    radial-gradient(circle at 84% 18%, rgba(160,198,244,.33), transparent 29rem),
    radial-gradient(circle at 7% 92%, rgba(220,230,243,.48), transparent 31rem),
    linear-gradient(135deg, #fbfcfe 0%, #f4f7fb 52%, #fbfcfd 100%);
  border-bottom: 1px solid #ebeff4;
}
.st-key-home_hero::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  opacity: .24;
  background-image:
    linear-gradient(rgba(77,101,134,.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(77,101,134,.08) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: linear-gradient(90deg, transparent 0%, transparent 26%, #000 61%, #000 100%);
}
.st-key-home_hero > div {
  /* Keep content comfortably inset while the section background reaches both edges. */
  width: min(92vw, var(--content-max)) !important;
  max-width: 92vw !important;
  margin-left: auto !important;
  margin-right: auto !important;
  box-sizing: border-box !important;
}
.st-key-home_hero [data-testid="stHorizontalBlock"] {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  gap: clamp(2rem, 3vw, 3.75rem) !important;
  align-items: center !important;
}
.st-key-home_hero [data-testid="stColumn"] {
  min-width: 0 !important;
  width: 0 !important;
  max-width: none !important;
  box-sizing: border-box !important;
}
.st-key-home_hero [data-testid="stColumn"]:first-child {
  flex: 1.03 1 0% !important;
}
.st-key-home_hero [data-testid="stColumn"]:last-child {
  flex: .97 1 0% !important;
}
.hero-copy-v5 { max-width: 690px; }
.hero-copy-v5 .landing-kicker {
  color: #707780;
  font-size: .72rem;
  line-height: 1.25;
  font-weight: 700;
  letter-spacing: .055em;
  margin-bottom: 1rem;
}
.hero-copy-v5 h1 {
  margin: 0;
  max-width: 690px;
  color: var(--text);
  font-size: clamp(3.1rem, 5vw, 5.15rem);
  line-height: .95;
  letter-spacing: -.058em;
  font-weight: 745;
}
.hero-copy-v5 p {
  max-width: 650px;
  margin: 1.25rem 0 0;
  color: #5b6169;
  font-size: clamp(1rem, 1.15vw, 1.16rem);
  line-height: 1.58;
}
.hero-proof {
  display: flex;
  flex-wrap: wrap;
  gap: .55rem 1.15rem;
  margin-top: 1.25rem;
  color: #737b85;
  font-size: .75rem;
}
.hero-proof span { position: relative; padding-left: .8rem; }
.hero-proof span::before {
  content: "";
  position: absolute;
  width: 4px;
  height: 4px;
  left: 0;
  top: .48rem;
  border-radius: 50%;
  background: #8190a4;
}
.st-key-home_hero .stButton > button { min-width: 158px !important; }
.hero-visual {
  width: 100% !important;
  max-width: 680px !important;
  min-width: 0 !important;
  margin-left: auto !important;
  margin-right: 0 !important;
  box-sizing: border-box !important;
}
.signal-panel {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  box-sizing: border-box !important;
  background: rgba(255,255,255,.79);
  border: 1px solid rgba(202,211,224,.9);
  border-radius: 24px;
  padding: 1.05rem;
  box-shadow: 0 22px 60px rgba(44,61,86,.11);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  animation: panelFloat 8s ease-in-out infinite;
}
.signal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: .1rem .18rem .75rem;
  color: #7b8592;
  font-size: .62rem;
  letter-spacing: .055em;
  font-weight: 700;
}
.signal-dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  margin-right: .35rem;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 5px rgba(36,138,61,.08);
}
.board {
  position: relative;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  height: clamp(310px, 31vw, 420px);
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid #e2e7ee;
  background: linear-gradient(180deg, #fafcff 0%, #f2f6fb 100%);
}
.board::before {
  content: "";
  position: absolute;
  inset: 0;
  opacity: .32;
  background-image: radial-gradient(#bdc8d7 1px, transparent 1px);
  background-size: 18px 18px;
}
.board-glow {
  position: absolute;
  width: 250px;
  height: 250px;
  left: 50%;
  top: 48%;
  transform: translate(-50%,-50%);
  border-radius: 50%;
  background: radial-gradient(circle, rgba(37,111,203,.11), rgba(37,111,203,0) 67%);
  animation: pulseGlow 4.4s ease-in-out infinite;
}
.circuit { position: absolute; inset: 0; width: 100% !important; max-width: 100% !important; height: 100% !important; display: block; box-sizing: border-box !important; }
.signal-line { stroke-dasharray: 850; stroke-dashoffset: 850; animation: drawSignal 3.4s ease-in-out infinite alternate; }
.signal-pulse { transform-origin: 594px 112px; animation: signalPulse 1.9s ease-out infinite; }
.scan-line {
  position: absolute;
  left: 5%;
  right: 5%;
  top: 15%;
  height: 1px;
  opacity: .5;
  background: linear-gradient(90deg, transparent, rgba(25,118,210,.5), transparent);
  animation: scan 5.2s ease-in-out infinite;
}
.time-tag {
  position: absolute;
  padding: .46rem .58rem;
  border: 1px solid #dce3ec;
  border-radius: 10px;
  background: rgba(255,255,255,.93);
  box-shadow: 0 7px 18px rgba(43,56,78,.055);
}
.time-tag b { display: block; color: var(--text); font-size: .7rem; }
.time-tag span { color: #87909b; font-size: .56rem; }
.t0 { left: 5%; bottom: 8%; }
.t24 { left: 39%; bottom: 20%; }
.t168 { right: 5%; top: 14%; }
.signal-foot {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .55rem;
  padding: .8rem .1rem .05rem;
}
.signal-foot div { padding: .05rem .6rem; border-right: 1px solid #e7ebf0; }
.signal-foot div:last-child { border-right: 0; }
.signal-foot span { display: block; color: #939ba6; font-size: .58rem; margin-bottom: .12rem; }
.signal-foot b { color: #3b4149; font-size: .7rem; font-weight: 650; }

/* Home product sections: wide, controlled, consistent. */
.product-section {
  position: relative;
  width: min(90vw, var(--content-max));
  max-width: var(--content-max);
  margin: 0 auto !important;
  padding: clamp(4.4rem, 5.8vw, 6.1rem) 0 !important;
  border-top: 1px solid var(--line);
}
.product-section:first-of-type { border-top: 0; }
.section-kicker {
  color: #6e7680;
  font-size: .68rem;
  font-weight: 720;
  letter-spacing: .09em;
  margin-bottom: .85rem;
}
.product-section h2 {
  max-width: 880px;
  margin: 0;
  color: var(--text);
  font-size: clamp(2.05rem, 3.15vw, 3.25rem);
  line-height: 1.02;
  letter-spacing: -.047em;
  font-weight: 725;
}
.product-section > p {
  max-width: 800px;
  margin: .8rem 0 1.5rem;
  color: #5e646c;
  font-size: .99rem;
  line-height: 1.62;
}
/* Faint signal motif between sections, not generic circles. */
.product-section::after {
  content: "";
  position: absolute;
  right: 0;
  top: 2.4rem;
  width: 220px;
  height: 54px;
  opacity: .12;
  pointer-events: none;
  background:
    linear-gradient(160deg, transparent 0 18%, #7c97b8 18% 20%, transparent 20% 42%, #4f7fb9 42% 44%, transparent 44% 65%, #1976d2 65% 67%, transparent 67%);
  mask-image: linear-gradient(90deg, transparent, #000 35%, #000);
}

/* Representative output: flatter information, minimal nested cards. */
.decision-stage {
  display: grid;
  grid-template-columns: minmax(0, 1.12fr) minmax(330px, .88fr);
  margin-top: 1.85rem;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.decision-main, .review-snapshot {
  padding: 2.1rem 2rem;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}
.review-snapshot { border-left: 1px solid var(--line); background: #fafbfc; }
.decision-label, .preview-eyebrow {
  color: #7b828c;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.decision-title {
  margin: .55rem 0 .75rem;
  color: var(--text);
  font-size: clamp(2.2rem, 3.15vw, 3.55rem);
  line-height: .97;
  letter-spacing: -.053em;
  font-weight: 735;
}
.decision-lede { max-width: 700px; color: #5e646c; line-height: 1.58; font-size: .95rem; }
.decision-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0;
  margin-top: 1.8rem;
  border-top: 1px solid var(--line);
}
.decision-metrics div { padding: 1rem 1rem .8rem; border-right: 1px solid var(--line); }
.decision-metrics div:first-child { padding-left: 0; }
.decision-metrics div:last-child { border-right: 0; }
.decision-metrics span { display: block; color: #818892; font-size: .65rem; margin-bottom: .25rem; }
.decision-metrics b { color: var(--text); font-size: 1.3rem; letter-spacing: -.035em; }
.snapshot-head { display: flex; justify-content: space-between; color: #767e88; font-size: .67rem; font-weight: 700; letter-spacing: .055em; text-transform: uppercase; }
.snapshot-id { margin: 1.15rem 0 .16rem; color: var(--text); font-size: 1.9rem; font-weight: 735; letter-spacing: -.04em; }
.snapshot-risk { color: #676d75; font-size: .8rem; margin-bottom: 1rem; }
.snapshot-line {
  display: grid;
  grid-template-columns: minmax(105px,.65fr) minmax(0,1.35fr);
  gap: .75rem;
  padding: .72rem 0;
  border-top: 1px solid var(--line);
  font-size: .76rem;
  line-height: 1.4;
}
.snapshot-line span { color: #7b828c; }
.snapshot-line b { color: #2f343a; font-weight: 610; }
.snapshot-action { margin-top: .85rem; padding: .72rem .8rem; background: #f2f5f8; border-radius: 7px; color: #343a41; font-size: .77rem; line-height: 1.42; }

/* Compact capabilities with one visual family. */
.feature-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1.5rem; margin: 2.35rem 0 0; }
.feature-card {
  position: relative;
  min-height: 185px;
  padding: 1.25rem .2rem .8rem;
  border-top: 1px solid #dfe3e8;
  background: transparent;
}
.feature-top { display: flex; align-items: center; justify-content: space-between; min-height: 45px; margin-bottom: 1.75rem; }
.feature-top > span { color: #808792; font-size: .7rem; font-weight: 700; letter-spacing: .06em; }
.feature-card h3 { margin: 0 0 .42rem; font-size: 1.18rem; line-height: 1.12; letter-spacing: -.032em; }
.feature-card p { max-width: 330px; margin: 0; color: #646a72; font-size: .87rem; line-height: 1.5; }
.mini-signal, .mini-forecast, .mini-evidence { width: 92px; height: 36px; display: flex; align-items: flex-end; justify-content: flex-end; gap: 5px; }
.mini-signal i { width: 6px; border-radius: 6px; background: #a9b8cb; animation: miniBars 3.2s ease-in-out infinite; }
.mini-signal i:nth-child(1){height:11px}.mini-signal i:nth-child(2){height:18px}.mini-signal i:nth-child(3){height:13px}.mini-signal i:nth-child(4){height:27px;background:#6f92bf}.mini-signal i:nth-child(5){height:34px;background:#337cc4}
.mini-forecast { position: relative; border-bottom: 1px solid #ccd5e0; }
.mini-forecast::before { content:""; position:absolute; left:5px; right:3px; bottom:9px; height:2px; background:linear-gradient(155deg,transparent 0 20%,#aab9cc 20% 22%,transparent 22% 43%,#7d9dc4 43% 45%,transparent 45% 65%,#337cc4 65% 67%,transparent 67%); }
.mini-forecast i { width: 5px; height: 5px; border-radius: 50%; background: #6389b7; }
.mini-forecast i:nth-child(1){align-self:flex-end;margin-bottom:7px}.mini-forecast i:nth-child(2){align-self:center}.mini-forecast i:nth-child(3){align-self:flex-start;margin-top:3px;background:#1976d2}
.mini-evidence { flex-direction: column; align-items: stretch; justify-content: center; gap: 6px; }
.mini-evidence i { height: 2px; background: linear-gradient(90deg,#b3bfce,#7f9fc3); border-radius: 2px; }
.mini-evidence i:nth-child(2){width:78%}.mini-evidence i:nth-child(3){width:58%;background:#337cc4}

/* Continuous process line. */
.process-line {
  position: relative;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0;
  max-width: 1120px;
  margin: 2.3rem 0 0;
}
.process-line::before {
  content: "";
  position: absolute;
  left: 5%;
  right: 5%;
  top: 16px;
  height: 1.5px;
  background: linear-gradient(90deg,#cfd6df,#8da6c2);
}
.process-connector { display: none !important; }
.process-step { position: relative; z-index: 1; padding-right: 1rem; }
.process-step span {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  margin-bottom: .85rem;
  border: 1px solid #aebbc9;
  border-radius: 50%;
  background: #fff;
  color: #596b80;
  font-size: .67rem;
  font-weight: 700;
}
.process-step b { display: block; color: var(--text); font-size: .91rem; margin-bottom: .2rem; }
.process-step small { display: block; max-width: 180px; color: #777f89; font-size: .74rem; line-height: 1.4; }

/* Human-in-the-loop: balanced, flatter, less chat-template-like. */
.review-grid { display: grid; grid-template-columns: minmax(0,1fr) minmax(380px,.9fr); gap: 2.5rem; align-items: start; }
.review-grid > div:first-child > p { max-width: 720px; color: #5e646c; line-height: 1.6; }
.review-points { margin-top: 1.65rem; }
.review-points > div {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: .75rem;
  padding: .86rem 0;
  border-top: 1px solid var(--line);
}
.review-points > div > span { color: #7a8390; font-size: .68rem; font-weight: 700; }
.review-points p { margin: 0; font-size: .83rem; line-height: 1.45; }
.review-points b { color: #2f343a; }
.copilot-preview {
  padding-left: 2rem;
  border-left: 1px solid var(--line);
  background: transparent;
}
.copilot-top { display: flex; justify-content: space-between; gap: 1rem; padding-bottom: .9rem; color: #747c86; font-size: .66rem; font-weight: 700; }
.chat-bubble {
  margin: 0;
  padding: .82rem 0;
  border-top: 1px solid var(--line);
  border-radius: 0;
  background: transparent;
  color: #5b626b;
  font-size: .78rem;
  line-height: 1.5;
}
.chat-bubble.user-q { color: #2f343a; font-weight: 620; }
.chat-bubble b { color: #2e3339; font-weight: 650; }

/* Final CTA uses same signal language as hero; no giant promo card. */
.st-key-final_cta_v5 {
  position: relative;
  overflow: hidden;
  margin: .5rem calc(-1 * var(--page-pad)) 0 !important;
  padding: clamp(4rem, 5.8vw, 6rem) var(--page-pad) !important;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  background:
    linear-gradient(rgba(77,101,134,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(77,101,134,.035) 1px,transparent 1px),
    linear-gradient(135deg,#f5f8fc,#fbfcfe);
  background-size: 42px 42px, 42px 42px, auto;
}
.st-key-final_cta_v5 > div { width: min(90vw,var(--content-max)) !important; max-width: var(--content-max) !important; margin: 0 auto !important; }
.final-copy-v5 h2 { margin: .25rem 0 .9rem; max-width: 700px; font-size: clamp(2.15rem,3.45vw,3.65rem); line-height: 1; letter-spacing: -.052em; }
.final-copy-v5 p { max-width: 640px; color: #5e646d; line-height: 1.62; margin-bottom: 1.75rem; }
.st-key-final_cta_v5 .stButton > button { min-width: 160px !important; }
.cta-trace svg { display: block; width: 100%; max-width: 500px; margin-left: auto; opacity: .9; }

/* Proper footer. */
.product-footer {
  display: flex !important;
  width: min(90vw,var(--content-max));
  max-width: var(--content-max);
  margin: 0 auto;
  padding: 1.5rem 0 2rem;
  gap: 1.25rem;
  justify-content: space-between;
  align-items: flex-start;
  color: #737a83;
  font-size: .72rem;
  line-height: 1.5;
}
.product-footer b { color: #30343a; font-weight: 650; }
.product-footer .footer-note { max-width: 760px; text-align: right; }

/* Internal pages share the same product system. */
.page-banner {
  position: relative;
  overflow: hidden;
  width: min(90vw,var(--content-max));
  max-width: var(--content-max);
  min-height: 200px;
  margin: 2rem auto 2.1rem;
  padding: 2.15rem 2.35rem;
  display: grid;
  grid-template-columns: minmax(0,1.15fr) minmax(320px,.85fr);
  gap: 2rem;
  align-items: center;
  border: 1px solid #e9edf2;
  border-radius: 16px;
  background: radial-gradient(circle at 84% 24%, rgba(180,208,244,.28), transparent 30%), linear-gradient(135deg,#fbfcfe,#f4f7fb 60%,#fafbfd);
}
.page-banner::before {
  content:"";
  position:absolute;
  inset:0;
  opacity:.18;
  background-image: linear-gradient(rgba(82,105,137,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(82,105,137,.07) 1px,transparent 1px);
  background-size:34px 34px;
  mask-image:linear-gradient(90deg,transparent 0%,transparent 42%,#000 72%);
  pointer-events:none;
}
.page-banner-copy, .page-banner-visual { position:relative; z-index:1; }
.page-banner-kicker { color:#747c86; font-size:.67rem; font-weight:700; letter-spacing:.08em; margin-bottom:.7rem; }
.page-banner h1 { margin:0; font-size:clamp(2rem,3.05vw,3.05rem); line-height:1; letter-spacing:-.05em; }
.page-banner p { margin:.78rem 0 0; max-width:750px; color:#5f656d; font-size:.98rem; line-height:1.55; }
.page-banner-visual { width:100%; max-width:470px; justify-self:end; }
.page-banner-visual svg { width:100%; display:block; }
.page-signal-line { stroke-dasharray:620; stroke-dashoffset:620; animation:pageDraw 3.8s ease-in-out infinite alternate; }
.page-signal-dot { transform-origin:401px 31px; animation:signalPulse 2s ease-out infinite; }
.page-scan { position:absolute; left:4%; right:3%; top:20%; height:1px; background:linear-gradient(90deg,transparent,rgba(25,118,210,.34),transparent); animation:pageScan 5.5s ease-in-out infinite; }

.analysis-shell, .section, .verdict {
  position: relative;
  overflow: hidden;
  margin: 1rem 0 1.45rem;
  padding: 1.45rem 1.55rem;
  border: 1px solid #e9edf1;
  border-radius: 12px;
  background: #f7f9fb;
  box-shadow: none;
}
.analysis-shell h2, .section-title { margin:0; color:var(--text); font-size:1.35rem; letter-spacing:-.035em; }
.analysis-shell p, .section-sub { color:#5e646d; line-height:1.5; }
.verdict { padding:1.8rem 1.9rem; }
.verdict-label { color:#747b84; font-size:.75rem; font-weight:650; margin-bottom:.45rem; }
.verdict-title { margin:0; font-size:clamp(1.8rem,3vw,2.8rem); line-height:1.05; font-weight:720; letter-spacing:-.045em; }
.verdict-copy { max-width:820px; margin-top:.65rem; font-size:.96rem; line-height:1.5; }
.tone-red { color:var(--red)!important; }.tone-amber{color:var(--amber)!important}.tone-green{color:var(--green)!important}
.health-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.7rem; margin:.8rem 0 1rem; }
.health-item { padding:.92rem; border:1px solid #e5e9ed; border-radius:9px; background:#fff; min-height:86px; }
.health-label { color:#7b828b; font-size:.7rem; margin-bottom:.3rem; }.health-value{color:var(--text);font-size:1rem;font-weight:670;line-height:1.25}
.finding { padding:.8rem 0; border-bottom:1px solid #dfe3e8; }.finding:last-child{border-bottom:0}.finding b{display:block;color:var(--text);font-size:.94rem;margin-bottom:.12rem}
.explain { margin:.8rem 0; padding:1rem 1.1rem; border-left:3px solid #8294aa; background:#fafbfc; border-radius:0 8px 8px 0; }.explain-title{font-weight:670;color:var(--text);margin-bottom:.3rem}.explain-copy{color:#60666e;font-size:.86rem;line-height:1.48}
.active-strip { display:flex; gap:.65rem; align-items:center; flex-wrap:wrap; padding:.65rem .78rem; margin:.6rem 0 1.1rem; border:1px solid #e8ebef; border-radius:8px; background:#f7f9fb; color:#656b73; font-size:.74rem; }.active-strip b{color:var(--text)}
.steps { display:grid; grid-template-columns:repeat(5,1fr); gap:.65rem; margin:0 0 1.5rem; }.step{border-top:2px solid #d8dde3;padding-top:.55rem;color:#747c86;font-size:.74rem;line-height:1.25}.step b{display:block;color:var(--text);font-weight:650;margin-bottom:.1rem}.step.done{border-top-color:#8798ac}

/* Streamlit controls/data surfaces. */
[data-testid="stMetric"] { background:transparent !important; border:0 !important; padding:.1rem 0 !important; min-height:82px; }
[data-testid="stMetricLabel"] { color:#737b84 !important; font-weight:550; }
[data-testid="stMetricValue"] { color:var(--text) !important; font-weight:690; letter-spacing:-.035em; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #e5e9ed; border-radius:9px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid #e5e9ed !important; border-radius:9px !important; background:#fff !important; box-shadow:none !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stFileUploaderDropzone"] { border-radius:9px !important; border-color:#d7dce2 !important; background:#fff !important; }
[data-testid="stChatMessage"] { border:0 !important; border-bottom:1px solid #e8ebef !important; border-radius:0 !important; background:#fff !important; padding:.45rem 0 !important; }

.no-break { white-space: nowrap; }
.workflow-section h2 { max-width: 1080px !important; font-size: clamp(2rem, 2.9vw, 3rem) !important; }
.final-copy-v5 h2 { max-width: 820px !important; font-size: clamp(2.05rem, 3.2vw, 3.35rem) !important; }
/* Subtle motion only. */
.reveal { opacity:0; transform:translateY(10px); animation:softReveal .62s cubic-bezier(.2,.75,.25,1) forwards; }
.reveal-1{animation-delay:.05s}.reveal-2{animation-delay:.12s}.reveal-3{animation-delay:.19s}
@keyframes softReveal { to{opacity:1;transform:translateY(0)} }
@keyframes drawSignal { to{stroke-dashoffset:0} }
@keyframes scan { 0%,100%{top:14%;opacity:.18}50%{top:84%;opacity:.58} }
@keyframes panelFloat { 0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)} }
@keyframes pulseGlow { 0%,100%{transform:translate(-50%,-50%) scale(.94);opacity:.62}50%{transform:translate(-50%,-50%) scale(1.05);opacity:1} }
@keyframes signalPulse { 0%{transform:scale(.8);opacity:1}75%,100%{transform:scale(2.1);opacity:0} }
@keyframes miniBars { 0%,100%{transform:scaleY(.78);opacity:.7}50%{transform:scaleY(1);opacity:1} }
@keyframes pageDraw { to{stroke-dashoffset:0} }
@keyframes pageScan { 0%,100%{top:18%;opacity:.15}50%{top:82%;opacity:.5} }

@media (prefers-reduced-motion: reduce) {
  .reveal,.signal-panel,.signal-line,.signal-pulse,.scan-line,.board-glow,.mini-signal i,.page-signal-line,.page-signal-dot,.page-scan { animation:none !important; opacity:1 !important; transform:none !important; }
}
@media (max-width: 1100px) {
  .decision-stage, .review-grid { grid-template-columns:1fr; }
  .review-snapshot { border-left:0; border-top:1px solid var(--line); }
  .copilot-preview { border-left:0; border-top:1px solid var(--line); padding:1.5rem 0 0; }
  .page-banner { grid-template-columns:1fr; }
  .page-banner-visual { max-width:380px; justify-self:start; }
}
@media (max-width: 900px) {
  :root { --page-pad:18px; --nav-h:60px; }
  .st-key-topbar_nav { overflow-x:auto !important; overflow-y:hidden !important; }
  .st-key-topbar_nav [data-testid="stHorizontalBlock"] { min-width:720px !important; }
  .st-key-home_hero { min-height:auto; padding-top:2.6rem !important; }
  .st-key-home_hero [data-testid="stHorizontalBlock"] { display:block !important; }
  .st-key-home_hero [data-testid="stColumn"] { width: 100% !important; max-width: 100% !important; }
  .hero-visual { max-width:680px; margin:2rem 0 0; }
  .feature-grid { grid-template-columns:1fr; }
  .process-line { grid-template-columns:1fr; gap:.65rem; }
  .process-line::before { left:15px; right:auto; top:16px; bottom:16px; width:1.5px; height:auto; }
  .process-step { display:grid; grid-template-columns:42px 1fr; column-gap:.6rem; padding:0 0 .7rem; }
  .process-step span { grid-row:1 / span 2; margin:0; }
  .process-step b { align-self:end; }
  .decision-metrics { grid-template-columns:1fr 1fr; }
  .decision-metrics div:nth-child(2) { border-right:0; }
  .product-footer { flex-direction:column; }
  .product-footer .footer-note { text-align:left; }
}
@media (max-width: 640px) {
  .no-break { white-space: normal; }
  .hero-copy-v5 h1 { font-size:clamp(2.75rem,14vw,4.1rem); }
  .hero-copy-v5 p { font-size:1rem; }
  .signal-foot { grid-template-columns:1fr; }
  .signal-foot div { border-right:0; border-top:1px solid #e7ebf0; padding:.45rem 0; }
  .decision-metrics { grid-template-columns:1fr; }
  .decision-metrics div { border-right:0; border-bottom:1px solid var(--line); padding-left:0; }
  .snapshot-line { grid-template-columns:1fr; gap:.18rem; }
  .health-grid { grid-template-columns:1fr 1fr; }
  .page-banner-visual { display:none; }
}
</style>
'''
    , unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Minimal light visual system. Inspired by modern product sites: generous space,
# strong hierarchy, neutral palette, and color only when it carries meaning.
# -----------------------------------------------------------------------------

RISK_COLORS = {"Normal": "#248a3d", "Warning": "#b26a00", "Critical": "#d70015"}


def polish_chart(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#6e6e73", family="Arial"),
        title_font=dict(size=17, color="#1d1d1f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6e6e73")),
        margin=dict(l=20, r=18, t=52, b=20),
        hoverlabel=dict(bgcolor="#1d1d1f", font_color="#ffffff"),
    )
    fig.update_xaxes(gridcolor="#eeeeef", zerolinecolor="#e5e5e7", linecolor="#d2d2d7")
    fig.update_yaxes(gridcolor="#eeeeef", zerolinecolor="#e5e5e7", linecolor="#d2d2d7")
    if height:
        fig.update_layout(height=height)
    return fig


def page_heading(title, subtitle):
    # Shared visual header so every workspace feels like the same product.
    st.markdown(
        f'''
        <section class="page-banner">
          <div class="page-banner-copy">
            <div class="page-banner-kicker">DRIFTGUARD · {title.upper()}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div class="page-banner-visual" aria-hidden="true">
            <svg viewBox="0 0 420 170" preserveAspectRatio="xMidYMid meet">
              <defs>
                <linearGradient id="pageLine" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="#b9c7da"/>
                  <stop offset="100%" stop-color="#1976d2"/>
                </linearGradient>
              </defs>
              <g fill="none" stroke="#dce3ec" stroke-width="1.6">
                <path d="M10 36 H95 V68 H155"/><path d="M10 132 H84 V104 H155"/>
                <path d="M410 36 H326 V68 H265"/><path d="M410 132 H338 V104 H265"/>
              </g>
              <rect x="155" y="38" width="110" height="94" rx="20" fill="#ffffff" stroke="#cfd8e5"/>
              <rect x="176" y="58" width="68" height="54" rx="13" fill="#f5f8fc" stroke="#e0e6ef"/>
              <path class="page-signal-line" d="M18 145 C82 139 108 146 150 127 S221 121 258 102 S326 78 401 31" fill="none" stroke="url(#pageLine)" stroke-width="4" stroke-linecap="round"/>
              <circle class="page-signal-dot" cx="401" cy="31" r="6" fill="#1976d2"/>
            </svg>
            <div class="page-scan"></div>
          </div>
        </section>
        ''',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_demo():
    return pd.read_csv(ROOT / "data" / "demo_burn_in.csv")


@st.cache_data
def load_meta(parameter):
    with open(ROOT / "models" / f"{parameter}_metadata.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def score_cached(df_csv: str, parameter: str):
    from io import StringIO
    df = pd.read_csv(StringIO(df_csv))
    return score_dataframe(df, parameter, ROOT / "models")


@st.cache_data(show_spinner=False)
def score_all_cached(df_csv: str):
    from io import StringIO
    df = pd.read_csv(StringIO(df_csv))
    combined, frames = score_all_parameters(df, ROOT / "models")
    return combined, frames


def risk_counts(frame, level_col="risk_level"):
    return {level: int((frame[level_col] == level).sum()) for level in ["Normal", "Warning", "Critical"]}


def batch_verdict(scored):
    counts = risk_counts(scored)
    early = int(scored["early_reject"].sum())
    if counts["Critical"] > 0 or early > 0:
        return "Action required", "red", (
            f"{counts['Critical']} components are classified as critical and {early} are flagged for early engineering review. "
            "Start with the priority review queue below."
        )
    if counts["Warning"] > 0:
        return "Monitor this batch", "amber", (
            f"No critical components were identified, but {counts['Warning']} components show behaviour that deserves closer monitoring."
        )
    return "Within expected range", "green", "No component currently requires elevated review under the prototype decision rules."


def simple_reason(row, parameter):
    cfg = PARAMETERS[parameter]
    pct = float(row.get("anomaly_percentile", 0))
    early = float(row.get(f"{parameter}_slope_0_24", 0))
    future = float(row.get("future_slope_per_hour", 0))
    pred = float(row.get("predicted_168h", 0))
    hi = float(row.get("prediction_upper_90", pred))
    if pred > cfg.absolute_max:
        return "Forecast exceeds the prototype safety limit"
    if hi > cfg.absolute_max:
        return "Prediction range reaches beyond the prototype limit"
    if early > cfg.safety_slope_per_hour:
        return "Early 0–24h drift is faster than the prototype safety slope"
    if future > cfg.safety_slope_per_hour:
        return "Predicted future drift is faster than the prototype safety slope"
    if pct >= 0.93:
        return f"Early behaviour is more unusual than about {pct*100:.0f}% of this batch"
    if float(row.get("risk_score", 0)) >= 35:
        return "Several moderate risk signals combine to raise concern"
    return "Readings remain consistent with the reference population"


def primary_issue(scored, parameter):
    cfg = PARAMETERS[parameter]
    tests = {
        "unusually fast early drift": int((scored[f"{parameter}_slope_0_24"] > cfg.safety_slope_per_hour).sum()),
        "unusually fast predicted drift": int((scored["future_slope_per_hour"] > cfg.safety_slope_per_hour).sum()),
        "forecast values above the prototype limit": int((scored["predicted_168h"] > cfg.absolute_max).sum()),
        "strong outlier behaviour compared with the batch": int((scored["anomaly_percentile"] >= 0.93).sum()),
    }
    label, count = max(tests.items(), key=lambda x: x[1])
    if count == 0:
        return "No dominant risk pattern was found in this batch."
    return f"The most common elevated signal is {label}, affecting {count} components."


def engineering_summary_text(scored, quality, parameter, source_name):
    counts = risk_counts(scored)
    status, _, _ = batch_verdict(scored)
    top = scored.sort_values("risk_score", ascending=False).iloc[0]
    early = int(scored["early_reject"].sum())
    lines = [
        f"Batch status: {status}.",
        f"Analysed {len(scored):,} components for {PARAMETERS[parameter].label}.",
        f"Risk distribution: {counts['Normal']} normal, {counts['Warning']} warning, {counts['Critical']} critical.",
        f"Early engineering-review flags: {early}.",
        primary_issue(scored, parameter),
        f"Highest-priority component: {top['component_id']} with a risk score of {float(top['risk_score']):.1f}/100.",
        f"Data quality score: {quality['score']}/100.",
        "Recommended next step: review the priority queue from highest risk downward before using the model output for any acceptance decision.",
        "Prototype thresholds are demonstration assumptions and are not official ISRO or device acceptance limits.",
    ]
    return "\n".join(lines)


def anomaly_label(row):
    p = float(row.get("anomaly_percentile", 0))
    if p >= .93:
        return "Very unusual"
    if p >= .75:
        return "Elevated"
    return "Typical"


def drift_label(row, parameter):
    slope = float(row.get(f"{parameter}_slope_0_24", 0))
    limit = PARAMETERS[parameter].safety_slope_per_hour
    if slope > limit:
        return "High"
    if slope > .65 * limit:
        return "Elevated"
    return "Normal"


def forecast_label(row, parameter):
    cfg = PARAMETERS[parameter]
    pred = float(row.get("predicted_168h", 0))
    hi = float(row.get("prediction_upper_90", pred))
    if pred > cfg.absolute_max:
        return "Above limit"
    if hi > cfg.absolute_max:
        return "Range crosses limit"
    if pred > .8 * cfg.absolute_max:
        return "Near limit"
    return "Within range"


def plot_batch_band(df, row, parameter):
    cfg = PARAMETERS[parameter]
    times = [0, 24, 96, 168]
    med, q10, q90, selected, valid_times = [], [], [], [], []
    for h in times:
        col = f"{parameter}_{h}h"
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            valid_times.append(h)
            med.append(vals.median())
            q10.append(vals.quantile(0.10))
            q90.append(vals.quantile(0.90))
            selected.append(row.get(col, np.nan))
    fig = go.Figure()
    if valid_times:
        fig.add_trace(go.Scatter(x=valid_times, y=q90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=valid_times, y=q10, mode="lines", fill="tonexty", line=dict(width=0), name="Batch 10–90% band"))
        fig.add_trace(go.Scatter(x=valid_times, y=med, mode="lines+markers", name="Batch median"))
        fig.add_trace(go.Scatter(x=valid_times, y=selected, mode="lines+markers", name="Selected component"))
    fig.add_trace(go.Scatter(
        x=[24, 168], y=[row[f"{parameter}_24h"], row["predicted_168h"]], mode="lines+markers",
        name="Model forecast", line=dict(dash="dash")
    ))
    fig.add_hline(y=cfg.absolute_max, line_dash="dot", annotation_text="Prototype limit")
    fig.update_layout(title="Component vs batch behaviour", xaxis_title="Burn-in time (hours)", yaxis_title=f"{cfg.label} ({cfg.unit})")
    return fig


# -----------------------------------------------------------------------------
# Product shell: top navigation and deliberate analyse flow.
# -----------------------------------------------------------------------------

# Final visual overrides for the product shell. These intentionally keep motion subtle.

# Full-screen product shell and shared reliability motif.


# ----------------------------------------------------------------------------
# v4 visual polish: stronger hierarchy, quieter premium surfaces, sticky nav,
# decision previews before analytics, and consistent reliability motifs.
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# v5 product-system polish: fixed full-width nav, deliberate spacing, flatter
# information hierarchy, and consistent reliability visuals across the product.
# ----------------------------------------------------------------------------


def render_footer():
    st.markdown(
        '<div class="product-footer"><div><b>DriftGuard</b><br>SIH26170 &middot; Predictive component reliability</div><div class="footer-note">Prototype thresholds and safety slopes are demonstration assumptions unless validated device-specific engineering limits are supplied. This research/hackathon prototype is decision support, not production acceptance testing.</div></div>',
        unsafe_allow_html=True,
    )

def go_to(page_name):
    st.session_state["top_nav"] = page_name


if "top_nav" not in st.session_state:
    st.session_state["top_nav"] = "Home"
if "_next_nav" in st.session_state:
    st.session_state["top_nav"] = st.session_state.pop("_next_nav")

# A real product-style top bar. We use keyed Streamlit buttons so navigation
# keeps session state (uploaded batch, current results, etc.) without radio UI.
with st.container(key="topbar_nav"):
    nav_cols = st.columns([4.5, 0.72, 0.82, 0.78, 0.92, 0.78, 0.78], vertical_alignment="center")
    with nav_cols[0]:
        st.markdown(
            '<div class="top-brand"><div class="top-brand-name">DriftGuard</div></div>',
            unsafe_allow_html=True,
        )
    for col, page_name in zip(nav_cols[1:], ["Home", "Analyse", "Results", "Analytics", "Copilot", "History"]):
        with col:
            st.button(
                page_name,
                key=f"nav_{page_name.lower()}",
                type="primary" if st.session_state["top_nav"] == page_name else "secondary",
                use_container_width=False,
                on_click=go_to,
                args=(page_name,),
            )

top_page = st.session_state["top_nav"]


# Home is a product introduction, not an analysis report.
if top_page == "Home":
    demo_df = load_demo()
    demo_scored = score_cached(demo_df.to_csv(index=False), "leakage")
    demo_counts = risk_counts(demo_scored)
    demo_top = demo_scored.sort_values("risk_score", ascending=False).iloc[0]
    demo_cfg = PARAMETERS["leakage"]
    demo_status, _, _ = batch_verdict(demo_scored)
    demo_early = int(demo_scored["early_reject"].sum())

    with st.container(key="home_hero"):
        hero_left, hero_right = st.columns([1.03, .97], gap="large", vertical_alignment="center")
        with hero_left:
            st.markdown(
                '''<div class="hero-copy-v5 reveal reveal-1"><div class="landing-kicker">PREDICTIVE RELIABILITY SCREENING · SIH26170</div><h1>See risk earlier.<br><span class="no-break">Decide with evidence.</span></h1><p>DriftGuard turns early burn-in measurements into an explainable engineering decision: what looks unusual, what may happen by 168 hours, and what deserves engineering attention first.</p><div class="hero-proof"><span>Early anomaly detection</span><span>168h forecast</span><span>Traceable reasoning</span></div></div>''',
                unsafe_allow_html=True,
            )
            h1, h2, h3 = st.columns([1.42, 1.18, 2.4])
            h1.button("Upload & analyse", type="primary", use_container_width=False, on_click=go_to, args=("Analyse",), key="hero_upload_v5")
            if h2.button("Try demo data", use_container_width=False, key="hero_demo_v5"):
                st.session_state["analysis_df"] = demo_df
                st.session_state["analysis_source"] = "Bundled synthetic demo dataset"
                st.session_state["analysis_parameter"] = "leakage"
                st.session_state["analysis_ready"] = True
                st.session_state["_next_nav"] = "Results"
                st.rerun()
        with hero_right:
            st.markdown(
                '''<div class="hero-visual reveal reveal-2" aria-label="Reliability screening illustration"><div class="signal-panel"><div class="signal-head"><div><span class="signal-dot"></span> LIVE SCREENING</div><span>EARLY BURN-IN</span></div><div class="board"><div class="board-glow"></div><svg class="circuit" viewBox="0 0 640 340" role="img" aria-label="Component reliability signal"><defs><linearGradient id="lineFadeV5" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#9fbbe7"/><stop offset="100%" stop-color="#1976d2"/></linearGradient></defs><g fill="none" stroke="#d7dee9" stroke-width="2"><path d="M30 82 H145 V132 H214"/><path d="M30 258 H160 V215 H214"/><path d="M610 82 H494 V132 H426"/><path d="M610 258 H480 V215 H426"/><path d="M96 32 V62 H178 V105"/><path d="M542 308 V284 H462 V236"/></g><g fill="#c5cfdd"><circle cx="30" cy="82" r="4"/><circle cx="30" cy="258" r="4"/><circle cx="610" cy="82" r="4"/><circle cx="610" cy="258" r="4"/></g><rect x="215" y="102" width="210" height="135" rx="20" fill="#fff" stroke="#cfd8e5" stroke-width="2"/><rect x="245" y="129" width="150" height="80" rx="14" fill="#f6f9fd" stroke="#d9e1ec"/><path d="M265 185 C295 178, 315 184, 338 171 S374 142, 392 153" fill="none" stroke="#7698c7" stroke-width="4" stroke-linecap="round"/><circle cx="392" cy="153" r="7" fill="#1976d2"/><path class="signal-line" d="M46 287 C112 281, 142 287, 190 271 S270 262, 312 244 S392 224, 438 188 S518 140, 594 112" fill="none" stroke="url(#lineFadeV5)" stroke-width="5" stroke-linecap="round"/><circle class="signal-pulse" cx="594" cy="112" r="8" fill="#1976d2"/></svg><div class="scan-line"></div><div class="time-tag t0"><b>0h</b><span>baseline</span></div><div class="time-tag t24"><b>24h</b><span>early drift</span></div><div class="time-tag t168"><b>168h</b><span>forecast</span></div></div><div class="signal-foot"><div><span>Anomaly</span><b>Detected early</b></div><div><span>Decision</span><b>Engineer attention</b></div><div><span>Evidence</span><b>Traceable</b></div></div></div></div>''',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'''<section class="product-section decision-showcase reveal reveal-2"><div class="section-kicker">REPRESENTATIVE OUTPUT</div><h2>See the decision before opening a graph.</h2><p>This preview uses the bundled synthetic dataset. Uploaded batches follow the same decision-first workflow.</p><div class="decision-stage"><div class="decision-main"><div class="decision-label">Batch status</div><div class="decision-title">{demo_status}</div><div class="decision-lede">{demo_counts['Critical']} critical components and {demo_early} early-attention flags were identified. The highest-priority components are surfaced first, with a reason and recommended engineering action.</div><div class="decision-metrics"><div><span>Components</span><b>{len(demo_scored):,}</b></div><div><span>Critical</span><b>{demo_counts['Critical']}</b></div><div><span>Warning</span><b>{demo_counts['Warning']}</b></div><div><span>Early attention</span><b>{demo_early}</b></div></div></div><div class="review-snapshot"><div class="snapshot-head"><span>Priority component</span><span>01</span></div><div class="snapshot-id">{demo_top['component_id']}</div><div class="snapshot-risk">{demo_top['risk_level']} risk · {float(demo_top['risk_score']):.1f}/100</div><div class="snapshot-line"><span>Why flagged</span><b>{simple_reason(demo_top, 'leakage')}</b></div><div class="snapshot-line"><span>Predicted 168h</span><b>{float(demo_top['predicted_168h']):.2f} {demo_cfg.unit}</b></div><div class="snapshot-line"><span>90% range</span><b>{float(demo_top['prediction_lower_90']):.2f}–{float(demo_top['prediction_upper_90']):.2f} {demo_cfg.unit}</b></div><div class="snapshot-action">{recommendation_for_row(demo_top)}</div></div></div></section>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''<section class="product-section reveal reveal-2"><div class="section-kicker">CAPABILITIES</div><h2>Three jobs. One engineering workflow.</h2><p>The model stays in the background. The engineer gets the evidence, the forecast and the next action.</p><div class="feature-grid premium-features"><div class="feature-card"><div class="feature-top"><span>01</span><div class="mini-signal"><i></i><i></i><i></i><i></i><i></i></div></div><h3>Detect abnormal drift</h3><p>Compare each component with its batch and surface behaviour that fixed pass/fail limits can miss.</p></div><div class="feature-card"><div class="feature-top"><span>02</span><div class="mini-forecast"><i></i><i></i><i></i></div></div><h3>Forecast 168 hours</h3><p>Use early readings such as 0h and 24h to estimate later behaviour with an empirical prediction range.</p></div><div class="feature-card"><div class="feature-top"><span>03</span><div class="mini-evidence"><i></i><i></i><i></i></div></div><h3>Explain the decision</h3><p>Show why a part was flagged, how unusual it is, and the recommended engineering next step.</p></div></div></section>
        <section class="product-section workflow-section reveal reveal-3"><div class="section-kicker">WORKFLOW</div><h2>From measurements to a decision in <span class="no-break">four steps.</span></h2><p>A clear sequence keeps the system easy to understand during a live engineering assessment.</p><div class="process-line"><div class="process-step"><span>01</span><b>Upload</b><small>Structured burn-in readings</small></div><div class="process-connector"></div><div class="process-step"><span>02</span><b>Validate</b><small>Quality and consistency checks</small></div><div class="process-connector"></div><div class="process-step"><span>03</span><b>Analyse</b><small>Anomaly + 168h forecast</small></div><div class="process-connector"></div><div class="process-step"><span>04</span><b>Assess</b><small>Priority queue + evidence</small></div></div></section>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'''<section class="product-section review-section reveal reveal-3"><div class="review-grid"><div><div class="section-kicker">HUMAN-IN-THE-LOOP</div><h2>AI supports the decision. Engineers own it.</h2><p>DriftGuard makes its reasoning visible so the engineer can inspect the evidence, compare the component with its batch, record feedback and override the recommendation when context demands it.</p><div class="review-points"><div><span>01</span><p><b>Decision first.</b><br>Status, reason, prediction and next step appear before technical detail.</p></div><div><span>02</span><p><b>Evidence on demand.</b><br>Batch bands, model metrics and prediction details stay available when deeper validation is needed.</p></div><div><span>03</span><p><b>Engineer feedback.</b><br>Confirmed defect, false alarm and needs-attention outcomes can be stored for later validation.</p></div></div></div><div class="copilot-preview"><div class="copilot-top"><span>DriftGuard Copilot</span><span>Grounded in active batch</span></div><div class="chat-bubble user-q">Why is {demo_top['component_id']} critical?</div><div class="chat-bubble bot-a"><b>Reason</b><br>{simple_reason(demo_top, 'leakage')}.</div><div class="chat-bubble bot-a"><b>Forecast</b><br>{float(demo_top['predicted_168h']):.2f} {demo_cfg.unit} at 168h, with a 90% range of {float(demo_top['prediction_lower_90']):.2f}–{float(demo_top['prediction_upper_90']):.2f} {demo_cfg.unit}.</div><div class="chat-bubble bot-a"><b>Suggested next step</b><br>{recommendation_for_row(demo_top)}.</div></div></div></section>''',
        unsafe_allow_html=True,
    )

    with st.container(key="final_cta_v5"):
        fc1, fc2 = st.columns([1.05, .95], gap="large", vertical_alignment="center")
        with fc1:
            st.markdown('''<div class="final-copy-v5"><div class="section-kicker">READY TO ANALYSE A BATCH?</div><h2>Start with your early <span class="no-break">burn-in readings.</span></h2><p>Upload a CSV, validate the structure and turn early measurements into an explainable engineering priority list.</p></div>''', unsafe_allow_html=True)
            ca, cb, cc = st.columns([1.42, 1.18, 2.4])
            ca.button("Analyse a batch", type="primary", use_container_width=False, on_click=go_to, args=("Analyse",), key="home_final_analyse_v5")
            if cb.button("Try demo data", use_container_width=False, key="home_final_demo_v5"):
                st.session_state["analysis_df"] = demo_df
                st.session_state["analysis_source"] = "Bundled synthetic demo dataset"
                st.session_state["analysis_parameter"] = "leakage"
                st.session_state["analysis_ready"] = True
                st.session_state["_next_nav"] = "Results"
                st.rerun()
        with fc2:
            st.markdown('''<div class="cta-trace" aria-hidden="true"><svg viewBox="0 0 520 210"><defs><linearGradient id="ctaV5" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#bdc9d8"/><stop offset="100%" stop-color="#1976d2"/></linearGradient></defs><g fill="none" stroke="#dce3ec" stroke-width="1.5"><path d="M24 48 H122 V82 H190"/><path d="M24 174 H112 V142 H190"/><path d="M496 48 H400 V82 H330"/><path d="M496 174 H410 V142 H330"/></g><rect x="190" y="62" width="140" height="102" rx="18" fill="#fff" stroke="#d2dbe7"/><path class="page-signal-line" d="M25 184 C90 177 116 185 160 165 S235 159 270 136 S351 109 401 75 S450 48 497 33" fill="none" stroke="url(#ctaV5)" stroke-width="4" stroke-linecap="round"/><circle class="page-signal-dot" cx="497" cy="33" r="6" fill="#1976d2"/></svg></div>''', unsafe_allow_html=True)

    render_footer()
    st.stop()


# Analyse is the only place where data is uploaded or a new run is started.
if top_page == "Analyse":
    page_heading("Analyse a burn-in batch", "Upload early measurements, confirm the structure, choose a parameter, then run the analysis.")
    st.markdown('<div class="analysis-shell"><h2>1. Choose your data</h2><p>Use a CSV containing component IDs and at least 0h + 24h readings for one supported parameter.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Burn-in CSV", type=["csv"], help="Supported demo parameters: leakage current, IDDQ / standby current, and propagation delay.")
    sample_bytes = (ROOT / "data" / "sample_early_all_parameters.csv").read_bytes()
    d1, d2 = st.columns([1, 3])
    d1.download_button("Download sample CSV", data=sample_bytes, file_name="DriftGuard_sample_input.csv", mime="text/csv", use_container_width=True)
    d2.caption("The sample contains early readings only, so it can demonstrate a 168h forecast before the true 168h value is known.")
    st.markdown('</div>', unsafe_allow_html=True)

    candidate_df = None
    candidate_source = None
    if uploaded is not None:
        try:
            candidate_df = pd.read_csv(uploaded)
            candidate_source = uploaded.name
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
    elif st.session_state.get("analysis_ready", False):
        candidate_df = st.session_state["analysis_df"]
        candidate_source = st.session_state["analysis_source"]

    if candidate_df is None:
        st.markdown('<div class="section"><div class="section-title">No file selected yet</div><div class="section-sub">Upload your own CSV above, or use the bundled synthetic dataset for a quick demonstration.</div></div>', unsafe_allow_html=True)
        if st.button("Analyse bundled demo data", type="primary"):
            st.session_state["analysis_df"] = load_demo()
            st.session_state["analysis_source"] = "Bundled synthetic demo dataset"
            st.session_state["analysis_parameter"] = "leakage"
            st.session_state["analysis_ready"] = True
            st.session_state["_next_nav"] = "Results"
            st.rerun()
        st.stop()

    available = supported_parameters(candidate_df)
    st.markdown('<div class="analysis-shell"><h2>2. Check the file</h2><p>Confirm that usable early readings were found before running the models.</p>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Rows", f"{len(candidate_df):,}")
    a2.metric("Columns", len(candidate_df.columns))
    a3.metric("Supported parameters", len(available))
    if available:
        st.success("Usable 0h + 24h readings detected for: " + ", ".join(PARAMETERS[p].label for p in available))
        with st.expander("Preview first 20 rows"):
            st.dataframe(candidate_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.error("No supported parameter has both 0h and 24h readings. Use the sample CSV to see the expected structure.")
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="analysis-shell"><h2>3. Run analysis</h2><p>Select the measurement to review first. You can switch parameters later without re-uploading the file.</p>', unsafe_allow_html=True)
    previous = st.session_state.get("analysis_parameter")
    default_index = available.index(previous) if previous in available else 0
    selected = st.selectbox("Parameter", available, index=default_index, format_func=lambda p: f"{PARAMETERS[p].label} ({PARAMETERS[p].unit})")
    candidate_quality = assess_data_quality(candidate_df, available)
    q1, q2 = st.columns(2)
    q1.metric("Input quality score", f"{candidate_quality['score']}/100")
    q2.metric("Detected input issues", candidate_quality["issue_count"])
    if candidate_quality["issue_count"]:
        st.caption("The analysis can still run, but review Data Quality before treating the output as reliable.")
    if st.button("Analyse batch", type="primary", use_container_width=True):
        with st.spinner("Analysing early burn-in behaviour…"):
            score_cached(candidate_df.to_csv(index=False), selected)
        st.session_state["analysis_df"] = candidate_df
        st.session_state["analysis_source"] = candidate_source
        st.session_state["analysis_parameter"] = selected
        st.session_state["analysis_ready"] = True
        st.session_state["_next_nav"] = "Results"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer()
    st.stop()


# Results, Analytics and Copilot require a completed analysis.
if top_page in {"Results", "Analytics", "Copilot"} and not st.session_state.get("analysis_ready", False):
    page_heading(top_page, "Analyse a batch first to unlock this workspace.")
    st.info("No batch has been analysed in this session yet.")
    st.button("Go to Analyse", type="primary", on_click=go_to, args=("Analyse",))
    st.stop()


# Load the active analysed batch for the original result pages below.
if st.session_state.get("analysis_ready", False):
    df = st.session_state["analysis_df"]
    source_name = st.session_state["analysis_source"]
    available = supported_parameters(df)
    current_parameter = st.session_state.get("analysis_parameter", available[0])
    if current_parameter not in available:
        current_parameter = available[0]
        st.session_state["analysis_parameter"] = current_parameter
else:
    # History does not require an active batch. These values are only placeholders.
    df = load_demo()
    source_name = "No active batch"
    available = supported_parameters(df)
    current_parameter = available[0]


if top_page == "Results":
    c1, c2 = st.columns([2, 5], vertical_alignment="bottom")
    with c1:
        parameter = st.selectbox("Measurement", available, index=available.index(current_parameter), format_func=lambda p: f"{PARAMETERS[p].label} ({PARAMETERS[p].unit})", key="top_results_parameter")
        if parameter != current_parameter:
            st.session_state["analysis_parameter"] = parameter
            st.rerun()
    with c2:
        page = st.radio("Result view", ["Overview", "Review Queue", "Component Inspector", "Combined Risk", "Data Quality"], horizontal=True, label_visibility="collapsed")
    st.markdown(f'<div class="active-strip"><b>Current batch</b> {source_name}<span>·</span><b>{len(df):,} components</b><span>·</span><b>{PARAMETERS[parameter].label}</b></div>', unsafe_allow_html=True)

elif top_page == "Analytics":
    parameter = current_parameter
    page = st.radio("Analytics view", ["Batch Analytics", "Model Performance", "Live Simulator", "What-if Analysis"], horizontal=True, label_visibility="collapsed")
    st.markdown(f'<div class="active-strip"><b>Current batch</b> {source_name}<span>·</span><b>{PARAMETERS[parameter].label}</b></div>', unsafe_allow_html=True)

elif top_page == "Copilot":
    parameter = current_parameter
    page = "Assistant"
    st.markdown(f'<div class="active-strip"><b>Current batch</b> {source_name}<span>·</span><b>{PARAMETERS[parameter].label}</b></div>', unsafe_allow_html=True)

elif top_page == "History":
    parameter = current_parameter
    page = "History & Feedback"



quality = assess_data_quality(df, available)
try:
    scored = score_cached(df.to_csv(index=False), parameter)
except Exception as exc:
    st.error(f"Could not analyse this file: {exc}")
    st.stop()

cfg = PARAMETERS[parameter]
meta = load_meta(parameter)


# -----------------------------------------------------------------------------
# Overview: decision first, no graphs.
# -----------------------------------------------------------------------------
if page == "Overview":
    page_heading("Batch result", "The decision summary first. Technical plots are available under Analytics.")

    counts = risk_counts(scored)
    status, tone, verdict_copy = batch_verdict(scored)
    st.markdown(
        f'<div class="verdict"><div class="verdict-label">Batch status</div><div class="verdict-title tone-{tone}">{status}</div><div class="verdict-copy">{verdict_copy}</div></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Components analysed", f"{len(scored):,}")
    m2.metric("Critical", counts["Critical"])
    m3.metric("Need monitoring", counts["Warning"])
    m4.metric("Early review flags", int(scored["early_reject"].sum()))

    st.markdown('<div class="section"><div class="section-title">What we found</div><div class="section-sub">The main conclusions, without requiring you to interpret a chart.</div>', unsafe_allow_html=True)
    top = scored.sort_values("risk_score", ascending=False).iloc[0]
    st.markdown(
        f"""
        <div class="finding"><b>{counts['Critical']} components require immediate review</b><span>They have the strongest combination of unusual early behaviour, predicted drift, and proximity to the prototype limit.</span></div>
        <div class="finding"><b>{primary_issue(scored, parameter)}</b><span>The system surfaces the dominant pattern so an engineer knows what to investigate first.</span></div>
        <div class="finding"><b>{top['component_id']} is currently the highest-priority component</b><span>Risk score {float(top['risk_score']):.1f}/100 · {simple_reason(top, parameter)}.</span></div>
        <div class="finding"><b>Input quality: {quality['score']}/100</b><span>{'No major input-quality issue was detected.' if quality['issue_count'] == 0 else str(quality['issue_count']) + ' data-quality issue(s) should be reviewed before relying on the result.'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-white"><div class="section-title">Priority review queue</div><div class="section-sub">Start here. The highest-risk components are listed first with a plain-language reason and recommended action.</div>', unsafe_allow_html=True)
    queue = scored.sort_values("risk_score", ascending=False).head(12).copy()
    queue["Why flagged"] = queue.apply(lambda r: simple_reason(r, parameter), axis=1)
    queue["Predicted 168h"] = queue["predicted_168h"].map(lambda x: f"{x:.2f} {cfg.unit}")
    queue["Risk"] = queue.apply(lambda r: f"{r['risk_level']} · {float(r['risk_score']):.1f}/100", axis=1)
    queue["Recommended action"] = queue.apply(recommendation_for_row, axis=1)
    st.dataframe(
        queue[["component_id", "Risk", "Predicted 168h", "Why flagged", "Recommended action"]].rename(columns={"component_id": "Component"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    summary_text = engineering_summary_text(scored, quality, parameter, source_name)
    st.markdown('<div class="section"><div class="section-title">Engineering summary</div><div class="section-sub">A concise handoff for a reviewer, mentor, or QA engineer.</div>', unsafe_allow_html=True)
    st.text(summary_text)
    report_bytes = batch_summary_pdf(scored, parameter, source_name, quality["score"], summary_text)
    c1, c2 = st.columns([1, 3])
    c1.download_button(
        "Download summary PDF",
        data=report_bytes,
        file_name=f"DriftGuard_{parameter}_batch_summary.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    with c2.expander("Save this batch to local history"):
        batch_name = st.text_input("Batch name", value=f"{parameter.upper()} Demo Batch")
        if st.button("Save batch snapshot"):
            save_batch_snapshot(DB_PATH, batch_name, source_name, parameter, scored)
            st.success("Batch snapshot saved locally.")
    st.markdown('</div>', unsafe_allow_html=True)


elif page == "Review Queue":
    page_heading("Review Queue", "A ranked list of the components that deserve attention first. No graph reading required.")
    counts = risk_counts(scored)
    r1, r2, r3 = st.columns(3)
    r1.metric("Immediate review", counts["Critical"])
    r2.metric("Closer monitoring", counts["Warning"])
    r3.metric("Normal", counts["Normal"])

    filter_level = st.segmented_control("Show", ["Critical + Warning", "Critical only", "All"], default="Critical + Warning")
    if filter_level == "Critical only":
        review = scored[scored["risk_level"] == "Critical"].copy()
    elif filter_level == "All":
        review = scored.copy()
    else:
        review = scored[scored["risk_level"].isin(["Critical", "Warning"])].copy()
    review = review.sort_values("risk_score", ascending=False)
    review["Why flagged"] = review.apply(lambda r: simple_reason(r, parameter), axis=1)
    review["Predicted 168h"] = review["predicted_168h"].map(lambda x: f"{x:.2f} {cfg.unit}")
    review["Risk"] = review.apply(lambda r: f"{r['risk_level']} · {float(r['risk_score']):.1f}/100", axis=1)
    review["Action"] = review.apply(recommendation_for_row, axis=1)
    st.dataframe(review[["component_id", "lot_id", "Risk", "Predicted 168h", "Why flagged", "Action"]].rename(columns={"component_id":"Component", "lot_id":"Lot"}), use_container_width=True, hide_index=True)
    st.caption("Open Component Inspector for the full explanation, prediction range, and technical trend view.")


elif page == "Component Inspector":
    page_heading("Component Inspector", "A plain-language explanation first. Technical plots and raw model signals are available only when you need them.")
    ids = scored.sort_values("risk_score", ascending=False)["component_id"].astype(str).tolist()
    component = st.selectbox("Component", ids)
    row = scored.loc[scored.component_id.astype(str) == component].iloc[0]
    risk_tone = {"Normal":"green", "Warning":"amber", "Critical":"red"}.get(row.risk_level, "amber")

    st.markdown(
        f'<div class="verdict"><div class="verdict-label">Component {component}</div><div class="verdict-title tone-{risk_tone}">{row.risk_level} risk</div><div class="verdict-copy">{simple_reason(row, parameter)}.</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="health-grid">
          <div class="health-item"><div class="health-label">Overall risk</div><div class="health-value">{float(row.risk_score):.1f} / 100</div></div>
          <div class="health-item"><div class="health-label">Anomaly</div><div class="health-value">{anomaly_label(row)}</div></div>
          <div class="health-item"><div class="health-label">Early drift</div><div class="health-value">{drift_label(row, parameter)}</div></div>
          <div class="health-item"><div class="health-label">168h forecast</div><div class="health-value">{forecast_label(row, parameter)}</div></div>
          <div class="health-item"><div class="health-label">Recommended action</div><div class="health-value">{'Review now' if row.early_reject else ('Monitor' if row.risk_level == 'Warning' else 'Continue test')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section"><div class="section-title">What does this mean?</div>', unsafe_allow_html=True)
    reasons = explain_row(row, parameter)
    st.markdown(
        f'<div class="finding"><b>What happened?</b><span>{simple_reason(row, parameter)}.</span></div>'
        f'<div class="finding"><b>What do we predict?</b><span>At 168 hours, {cfg.label.lower()} is predicted to be {float(row.predicted_168h):.2f} {cfg.unit}, with an estimated 90% range of {float(row.prediction_lower_90):.2f}–{float(row.prediction_upper_90):.2f} {cfg.unit}.</span></div>'
        f'<div class="finding"><b>How unusual is it?</b><span>Its early behaviour is more unusual than about {float(row.anomaly_percentile)*100:.0f}% of the analysed batch.</span></div>'
        f'<div class="finding"><b>What should the engineer do?</b><span>{recommendation_for_row(row)}.</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-white"><div class="section-title">Why was it flagged?</div>', unsafe_allow_html=True)
    for reason in reasons:
        st.write("•", reason)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Technical trend view"):
        st.plotly_chart(polish_chart(plot_batch_band(df, row, parameter)), use_container_width=True)
        st.caption("The shaded band represents the central 80% of the current batch. This graph supports the decision; it is not the decision itself.")

    with st.expander("Raw model signals"):
        st.json({
            "lot": str(row.lot_id),
            "anomaly_percentile": round(float(row.anomaly_percentile), 3),
            "early_drift_per_hour": round(float(row[f"{parameter}_slope_0_24"]), 5),
            "predicted_future_slope_per_hour": round(float(row.future_slope_per_hour), 5),
            "prototype_safety_slope": cfg.safety_slope_per_hour,
            "prototype_absolute_limit": cfg.absolute_max,
            "prediction_range_half_width": round(float(row.prediction_range_half_width), 4),
        })

    p1, p2 = st.columns([1, 2])
    pdf_bytes = component_report_pdf(row, parameter, source_name)
    p1.download_button("Download component report", data=pdf_bytes, file_name=f"DriftGuard_{component}_{parameter}_report.pdf", mime="application/pdf", use_container_width=True)
    with p2.expander("Engineer feedback"):
        feedback_label = st.selectbox("Conclusion", ["Confirmed defect", "False alarm", "Needs further test", "Confirmed normal"])
        note = st.text_area("Note (optional)")
        if st.button("Save feedback"):
            save_feedback(DB_PATH, row, parameter, feedback_label, note)
            st.success("Feedback saved.")


elif page == "Batch Analytics":
    page_heading("Batch Analytics", "Technical plots live here so the main workflow stays simple and decision-focused.")
    fig = px.scatter(
        scored, x=f"{parameter}_24h", y="predicted_168h", color="risk_level",
        hover_data=["component_id", "lot_id", "risk_score", "anomaly_percentile", "prediction_lower_90", "prediction_upper_90"],
        title=f"24h reading vs predicted 168h · {cfg.label}",
        category_orders={"risk_level": ["Normal", "Warning", "Critical"]}, color_discrete_map=RISK_COLORS,
    )
    st.plotly_chart(polish_chart(fig), use_container_width=True)

    drift_fig = px.histogram(scored, x=f"{parameter}_slope_0_24", nbins=35, title="Early drift-rate distribution · 0h to 24h", color_discrete_sequence=["#0071e3"])
    st.plotly_chart(polish_chart(drift_fig), use_container_width=True)

    st.markdown("### Lot-level risk")
    lot_summary = scored.groupby("lot_id", as_index=False).agg(
        components=("component_id", "count"), mean_risk=("risk_score", "mean"), max_risk=("risk_score", "max"), early_rejects=("early_reject", "sum")
    ).sort_values("mean_risk", ascending=False)
    st.dataframe(lot_summary, use_container_width=True, hide_index=True)

    if f"{parameter}_168h" in scored.columns:
        valid = scored.dropna(subset=[f"{parameter}_168h"]).copy()
        if len(valid):
            comp = go.Figure()
            comp.add_trace(go.Scatter(x=valid[f"{parameter}_168h"], y=valid["predicted_168h"], mode="markers", name="Components"))
            lo = min(valid[f"{parameter}_168h"].min(), valid["predicted_168h"].min())
            hi = max(valid[f"{parameter}_168h"].max(), valid["predicted_168h"].max())
            comp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction"))
            comp.update_layout(title="Predicted vs actual 168h values", xaxis_title="Actual 168h", yaxis_title="Predicted 168h")
            st.plotly_chart(polish_chart(comp), use_container_width=True)


elif page == "Combined Risk":
    page_heading("Combined Risk", "A single cross-parameter view when leakage, IDDQ and propagation delay are all available.")
    combined, frames = score_all_cached(df.to_csv(index=False))
    cc = risk_counts(combined, "combined_risk_level")
    status = "Action required" if cc["Critical"] else ("Monitor" if cc["Warning"] else "Within expected range")
    st.markdown(f'<div class="section"><div class="section-title">{status}</div><div class="section-sub">{cc["Critical"]} critical · {cc["Warning"]} warning · {cc["Normal"]} normal across combined early measurements.</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Normal", cc["Normal"]); c2.metric("Warning", cc["Warning"]); c3.metric("Critical", cc["Critical"]); c4.metric("Early review", int(combined["combined_early_reject"].sum()))

    risk_cols = [c for c in combined.columns if c.endswith("_risk") and c != "combined_risk_score"]
    top = combined.sort_values("combined_risk_score", ascending=False).head(30)
    st.dataframe(top[["component_id", "lot_id"] + risk_cols + ["combined_risk_score", "combined_risk_level", "dominant_risk_parameter", "combined_early_reject"]], use_container_width=True, hide_index=True)
    with st.expander("Technical risk distribution"):
        plot_df = combined.melt(id_vars=["component_id"], value_vars=risk_cols, var_name="parameter", value_name="risk")
        plot_df["parameter"] = plot_df["parameter"].str.replace("_risk", "", regex=False)
        st.plotly_chart(polish_chart(px.box(plot_df, x="parameter", y="risk", title="Risk distribution by parameter")), use_container_width=True)


elif page == "Data Quality":
    page_heading("Data Quality", "Check the input before trusting the model. This page separates measurement problems from component problems.")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Quality score", f"{quality['score']}/100")
    q2.metric("Rows", quality["rows"])
    q3.metric("Issues", quality["issue_count"])
    q4.metric("Missing values", quality["summary"]["missing_numeric_values"])
    if not quality["issues"]:
        st.success("No data-quality issue was detected by the prototype checks.")
    else:
        st.warning("Resolve the issues below before treating model output as reliable.")
        st.dataframe(pd.DataFrame(quality["issues"]), use_container_width=True, hide_index=True)
    with st.expander("What is checked"):
        st.write("Required IDs, duplicate component IDs, missing readings, non-numeric values, negative measurements, and extreme early jumps that may indicate unit or sensor problems.")


elif page == "Live Simulator":
    page_heading("Live Simulator", "See how the decision becomes available as burn-in readings arrive over time.")
    ids = scored.sort_values("risk_score", ascending=False)["component_id"].astype(str).tolist()
    component = st.selectbox("Component", ids, key="live_component")
    row = scored.loc[scored.component_id.astype(str) == component].iloc[0]
    placeholder = st.empty()
    if st.button("Run simulation", type="primary"):
        for h in [0, 24, 96, 168]:
            col = f"{parameter}_{h}h"
            observed = row.get(col, np.nan)
            with placeholder.container():
                st.markdown(f"### Burn-in time: {h}h")
                if pd.notna(observed):
                    st.metric("Latest observed reading", f"{float(observed):.3f} {cfg.unit}")
                else:
                    st.metric("Latest observed reading", "Not available")
                if h < 24:
                    st.info("The system waits for the 24h reading before producing the early-drift forecast.")
                else:
                    x1, x2, x3 = st.columns(3)
                    x1.metric("Predicted 168h", f"{row.predicted_168h:.2f} {cfg.unit}")
                    x2.metric("Risk", f"{row.risk_score:.1f}/100")
                    x3.metric("Action", "Review now" if row.early_reject else "Continue")
                    st.markdown(f"**Why:** {simple_reason(row, parameter)}.")
            time.sleep(0.8)
        st.success("Simulation complete.")


elif page == "What-if Analysis":
    page_heading("What-if Analysis", "Change an early reading and immediately see how the forecast and recommended action respond.")
    ids = scored.sort_values("risk_score", ascending=False)["component_id"].astype(str).tolist()
    component = st.selectbox("Component", ids, key="whatif_component")
    base_row = scored.loc[scored.component_id.astype(str) == component].iloc[0]
    current0 = float(base_row[f"{parameter}_0h"]); current24 = float(base_row[f"{parameter}_24h"])
    w1, w2 = st.columns(2)
    new0 = w1.number_input("Hypothetical 0h value", min_value=0.0, value=current0, step=max(current0 * .02, .01), format="%.4f")
    new24 = w2.number_input("Hypothetical 24h value", min_value=0.0, value=current24, step=max(current24 * .02, .01), format="%.4f")
    scenario = df.copy(); mask = scenario["component_id"].astype(str) == component
    scenario.loc[mask, f"{parameter}_0h"] = new0; scenario.loc[mask, f"{parameter}_24h"] = new24
    scenario_scored = score_dataframe(scenario, parameter, ROOT / "models")
    sim_row = scenario_scored.loc[scenario_scored.component_id.astype(str) == component].iloc[0]
    b1, b2, b3 = st.columns(3)
    b1.metric("Risk", f"{sim_row.risk_score:.1f}/100", delta=f"{sim_row.risk_score-base_row.risk_score:+.1f}")
    b2.metric("Predicted 168h", f"{sim_row.predicted_168h:.2f} {cfg.unit}", delta=f"{sim_row.predicted_168h-base_row.predicted_168h:+.2f}")
    b3.metric("Action", "Review now" if sim_row.early_reject else ("Monitor" if sim_row.risk_level == "Warning" else "Continue"))
    st.markdown(f'<div class="explain"><div class="explain-title">What changed?</div><div class="explain-copy">{simple_reason(sim_row, parameter)}.</div></div>', unsafe_allow_html=True)


elif page == "Assistant":
    page_heading("DriftGuard Assistant", "Ask about the current batch in normal language. Answers are grounded in the analysis already produced by the prototype.")
    st.caption("Decision support only. Prototype thresholds are not official ISRO/device acceptance limits.")
    chat_context_key = f"{source_name}|{parameter}|{len(df)}"
    if st.session_state.get("copilot_context_key") != chat_context_key:
        st.session_state["copilot_context_key"] = chat_context_key
        st.session_state["copilot_messages"] = [{"role":"assistant", "content":f"I’m looking at **{len(scored):,} components** for **{cfg.label}**. Ask me for the batch summary, priority components, a component explanation, model performance, or the next engineering action."}]
    if "copilot_messages" not in st.session_state:
        st.session_state["copilot_messages"] = []
    q1, q2, q3, q4 = st.columns(4); quick_prompt = None
    if q1.button("Summarize batch", use_container_width=True): quick_prompt = "Summarize this batch"
    if q2.button("Priority components", use_container_width=True): quick_prompt = "Which components should I inspect first?"
    if q3.button("Model performance", use_container_width=True): quick_prompt = "How accurate is the model?"
    if q4.button("Next action", use_container_width=True): quick_prompt = "What should the engineer do next?"
    top_example = str(scored.sort_values("risk_score", ascending=False).iloc[0]["component_id"])
    st.caption(f"Example: Why is {top_example} critical?")
    for message in st.session_state["copilot_messages"]:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    typed_prompt = st.chat_input("Ask about this batch…"); prompt = quick_prompt or typed_prompt
    if prompt:
        st.session_state["copilot_messages"].append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = answer_question(prompt, df, scored, parameter, quality, meta, ROOT / "models", DB_PATH)
            st.markdown(response)
        st.session_state["copilot_messages"].append({"role":"assistant", "content":response})


elif page == "Model Performance":
    page_heading("Model Performance", "The technical evidence behind the decisions. Kept separate from the main review workflow.")
    st.markdown("### 168h regression comparison")
    reg_rows = [{"model": k, "MAE": v["mae"], "90% abs-error": v.get("abs_error_q90", np.nan)} for k, v in meta["regression_models"].items()]
    reg_df = pd.DataFrame(reg_rows).sort_values("MAE")
    st.dataframe(reg_df, use_container_width=True, hide_index=True)
    st.success(f"Selected regressor: {meta['selected_regressor'].replace('_',' ').title()} · lowest validation MAE")
    with st.expander("Show MAE chart"):
        fig = px.bar(reg_df, x="model", y="MAE", title="Validation MAE by regressor", color_discrete_sequence=["#0071e3"])
        st.plotly_chart(polish_chart(fig), use_container_width=True)

    st.markdown("### Anomaly detector comparison")
    anomaly_rows = []
    for name, m in meta.get("anomaly_models", {"selected": meta["anomaly_metrics"]}).items():
        anomaly_rows.append({"model":name, "Recall":m["recall"], "Precision":m["precision"], "F1":m["f1"], "F2":m.get("f2",np.nan), "False Negatives":m["false_negatives"]})
    anomaly_df = pd.DataFrame(anomaly_rows).sort_values(["F2","Recall"], ascending=False)
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    selected_anomaly = meta.get("selected_anomaly_model", "isolation_forest")
    st.success(f"Selected anomaly model: {selected_anomaly.replace('_',' ').title()} · F2 emphasizes recall to reduce missed defects")


elif page == "History & Feedback":
    page_heading("History & Feedback", "Saved batch snapshots and engineer conclusions form the feedback loop for future validation and retraining.")
    history = load_history(DB_PATH)
    if history.empty:
        st.info("No batch snapshots saved yet. Save one from Overview.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        with st.expander("Show history trend"):
            hist_plot = history.sort_values("id")
            st.plotly_chart(polish_chart(px.line(hist_plot, x="saved_at", y="mean_risk", color="parameter", markers=True, title="Mean risk across saved batches")), use_container_width=True)
    st.markdown("### Engineer feedback")
    feedback = load_feedback(DB_PATH)
    if feedback.empty:
        st.info("No engineer feedback saved yet. Add feedback from Component Inspector.")
    else:
        st.dataframe(feedback, use_container_width=True, hide_index=True)
        with st.expander("Show feedback breakdown"):
            summary = feedback.groupby(["parameter", "engineer_label"]).size().reset_index(name="count")
            st.plotly_chart(polish_chart(px.bar(summary, x="parameter", y="count", color="engineer_label", barmode="group", title="Human review outcomes")), use_container_width=True)


render_footer()

