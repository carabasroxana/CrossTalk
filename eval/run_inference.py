import argparse, json, os, time
from tqdm import tqdm
from pathlib import Path

USE_HF = True

if USE_HF:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
else:
    from dotenv import load_dotenv
    load_dotenv()
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAI client. "
        ) from e

LANG_PROMPT_ZERO = {
    "en": "prompts/zero_shot_en.txt",
    "es": "prompts/zero_shot_es.txt",
    "ro": "prompts/zero_shot_ro.txt",
}

LANG_PROMPT_FEW = {
    "en": "prompts/few_shot_en.txt",
    "es": "prompts/few_shot_es.txt",
    "ro": "prompts/few_shot_ro.txt",
}

def ensure_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return p

def load_prompt(path):
    path = ensure_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(item, tmpl):
    opts = "\n".join(item["options"])
    return tmpl.format(question=item["question"], options=opts)

def call_model_api(prompt, model_name, temperature=0.2, max_tokens=4, retries=3):
    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(f"API call failed after retries: {last_err}")

def call_hf(prompt, generator):
    out = generator(prompt, max_new_tokens=8, do_sample=False)
    text = out[0]["generated_text"][len(prompt):].strip()
    return text

def normalize_letter(s):
    s = (s or "").strip().upper()
    for ch in ["A", "B", "C", "D"]:
        if s.startswith(ch):
            return ch
    for ch in ["A", "B", "C", "D"]:
        if ch in s:
            return ch
    return "?"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL dataset")
    ap.add_argument("--out", required=True, help="Path to write predictions JSONL")
    ap.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    ap.add_argument("--model", required=True, help="e.g., gpt-4o-mini or local HF name")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=4)
    args = ap.parse_args()

    if not USE_HF:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
            )

    if args.mode == "zero-shot":
        prompt_map = {k: load_prompt(v) for k, v in LANG_PROMPT_ZERO.items()}
    else:
        prompt_map = {k: load_prompt(v) for k, v in LANG_PROMPT_FEW.items()}

    if USE_HF:
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
        generator = pipeline("text-generation", model=mdl, tokenizer=tok)
    else:
        generator = None

    out_path = Path(args.out)
    if out_path.parent.as_posix() != "":
        out_path.parent.mkdir(parents=True, exist_ok=True)

    input_path = ensure_file(args.input)
    total = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Infer"):
            if not line.strip():
                continue
            item = json.loads(line)
            lang = item.get("language")
            if lang not in prompt_map:
                rec = {"id": item.get("id"), "pred": "?", "raw": "UNKNOWN_LANGUAGE"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            tmpl = prompt_map[lang]
            prompt = build_prompt(item, tmpl)

            if USE_HF:
                raw = call_hf(prompt, generator)
            else:
                raw = call_model_api(
                    prompt,
                    args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

            pred = normalize_letter(raw)
            rec = {"id": item["id"], "pred": pred, "raw": raw}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    print(f"Done. Wrote {total} predictions to {out_path}")

if __name__ == "__main__":
    main()
