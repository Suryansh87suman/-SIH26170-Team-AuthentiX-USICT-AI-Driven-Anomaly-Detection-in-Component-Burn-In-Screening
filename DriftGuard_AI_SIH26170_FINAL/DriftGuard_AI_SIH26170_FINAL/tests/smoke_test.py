from pathlib import Path
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference import score_dataframe
from multi_inference import score_all_parameters
from data_quality import assess_data_quality
from reports import component_report_pdf, batch_summary_pdf
from storage import save_batch_snapshot, save_feedback, load_history, load_feedback

full = pd.read_csv(ROOT / 'data' / 'demo_burn_in.csv')

for parameter in ['leakage', 'iddq', 'delay']:
    early = full[['component_id','lot_id',f'{parameter}_0h',f'{parameter}_24h']].head(50)
    scored = score_dataframe(early, parameter, ROOT / 'models')
    assert len(scored) == 50
    assert scored['predicted_168h'].notna().all()
    assert scored['risk_score'].between(0,100).all()
    assert scored['prediction_lower_90'].notna().all()
    assert scored['prediction_upper_90'].notna().all()
    assert (scored['prediction_upper_90'] >= scored['prediction_lower_90']).all()
    assert scored['recommendation'].notna().all()

combined, frames = score_all_parameters(full.head(80), ROOT / 'models')
assert len(combined) == 80
assert combined['combined_risk_score'].between(0,100).all()
assert set(frames) == {'leakage','iddq','delay'}

quality = assess_data_quality(full.head(100))
assert 0 <= quality['score'] <= 100
assert quality['rows'] == 100

scored = score_dataframe(full.head(40), 'leakage', ROOT / 'models')
row = scored.iloc[0]
pdf = component_report_pdf(row, 'leakage', 'smoke-test')
assert pdf[:4] == b'%PDF'
assert len(pdf) > 1000

summary_pdf = batch_summary_pdf(scored, 'leakage', 'smoke-test', 98, 'Batch status: Action required.\nReview priority components first.')
assert summary_pdf[:4] == b'%PDF'
assert len(summary_pdf) > 1000

with tempfile.TemporaryDirectory() as td:
    db = Path(td) / 'test.db'
    save_batch_snapshot(db, 'Smoke batch', 'synthetic', 'leakage', scored)
    save_feedback(db, row, 'leakage', 'Needs further test', 'Smoke test note')
    assert len(load_history(db)) == 1
    assert len(load_feedback(db)) == 1

print('All enhanced smoke tests passed.')

# AI Copilot smoke tests
from chatbot import answer_question
import json

quality = assess_data_quality(full.head(120), ['leakage','iddq','delay'])
scored_chat = score_dataframe(full.head(120), 'leakage', ROOT / 'models')
with open(ROOT / 'models' / 'leakage_metadata.json', encoding='utf-8') as f:
    meta_chat = json.load(f)

with tempfile.TemporaryDirectory() as td:
    db = Path(td) / 'chat.db'
    top_id = str(scored_chat.sort_values('risk_score', ascending=False).iloc[0]['component_id'])
    for question in [
        'Summarize this batch',
        'Which components should I inspect first?',
        f'Why is {top_id} critical?',
        'How accurate is the model?',
        'Is the data good?',
        'Explain the 90 percent prediction range',
        'What does the risk score mean?',
        'Show combined risk',
        'What should the engineer do next?',
    ]:
        ans = answer_question(question, full.head(120), scored_chat, 'leakage', quality, meta_chat, ROOT / 'models', db)
        assert isinstance(ans, str) and len(ans) > 30

print('AI Copilot smoke tests passed.')
