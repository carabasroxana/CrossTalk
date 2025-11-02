# Everyday-Knowledge Multilingual Benchmark (EN/ES/RO)

A beginner-friendly benchmark to assess LLMs on everyday, culturally grounded knowledge in three languages (English, Spanish, Romanian) across four domains (food, holidays, transportation, shopping). Multiple-choice format for easy scoring.

## Quickstart
1. `pip install -r requirements.txt`
2. Inspect `data/pilot.jsonl`
3. Run zero-shot baseline: `python eval/run_inference.py --model gpt-4o-mini --mode zero-shot --input data/pilot.jsonl --out eval/preds/gpt4o_zero.jsonl`
4. Score: `python eval/score.py --gold data/pilot.jsonl --pred eval/preds/gpt4o_zero.jsonl`

## Files
- `data/pilot.jsonl` — small pilot dataset
- `prompts/` — zero-shot and few-shot templates per language
- `eval/run_inference.py` — calls a model (API or HF) and saves predictions
- `eval/score.py` — computes accuracy per language/domain and overall
- `reports/task_spec.md` — task rules, schema, guidelines

## Notes
- Avoid stereotypes; favors stable conventions over volatile facts.
- For API usage, set environment variables in `.env` (see code comments).
