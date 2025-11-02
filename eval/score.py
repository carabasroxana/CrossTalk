import argparse, json, sys, os, csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # safe for non-notebook scripts
import matplotlib.pyplot as plt

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue  # skip blank lines
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping non-JSON line {i} in {path}: {e}", file=sys.stderr)

def pct(c, t): 
    return 0.0 if t == 0 else 100.0 * c / t

def barplot(savepath, labels, values, title, xlabel):
    plt.figure()
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold JSONL file")
    ap.add_argument("--pred", required=True, help="Path to predictions JSONL file")
    ap.add_argument("--plots", default="reports", help="Directory to save plots/summary")
    args = ap.parse_args()

    os.makedirs(args.plots, exist_ok=True)

    # --- Load gold ---
    gold, meta = {}, {}
    for x in iter_jsonl(args.gold):
        gold[x["id"]] = x["answer"]
        meta[x["id"]] = (x["language"], x["domain"])

    # --- Score ---
    correct, total = 0, 0
    buckets = defaultdict(lambda: {"c": 0, "t": 0})
    for x in iter_jsonl(args.pred):
        qid = x.get("id")
        if qid not in gold:
            continue
        total += 1
        is_ok = (str(x.get("pred","")).strip().upper() == str(gold[qid]).strip().upper())
        correct += int(is_ok)
        lang, dom = meta[qid]
        buckets[("lang", lang)]["t"] += 1
        buckets[("lang", lang)]["c"] += int(is_ok)
        buckets[("dom", dom)]["t"] += 1
        buckets[("dom", dom)]["c"] += int(is_ok)

    # --- Print text summary ---
    overall = pct(correct, total)
    print(f"Overall accuracy: {correct}/{total} = {overall:.1f}%\n")

    lang_labels, lang_vals = [], []
    dom_labels, dom_vals = [], []

    print("By language:")
    for (typ, key), v in sorted(buckets.items()):
        if typ == "lang":
            acc = pct(v["c"], v["t"])
            print(f"  {key}: {v['c']}/{v['t']} = {acc:.1f}%")
            lang_labels.append(key); lang_vals.append(acc)

    print("\nBy domain:")
    for (typ, key), v in sorted(buckets.items()):
        if typ == "dom":
            acc = pct(v["c"], v["t"])
            print(f"  {key}: {v['c']}/{v['t']} = {acc:.1f}%")
            dom_labels.append(key); dom_vals.append(acc)

    # --- Save plots ---
    lang_png = os.path.join(args.plots, "acc_by_language.png")
    dom_png  = os.path.join(args.plots, "acc_by_domain.png")
    if lang_labels:
        barplot(lang_png, lang_labels, lang_vals, "Accuracy by Language", "Language")
    if dom_labels:
        barplot(dom_png, dom_labels, dom_vals, "Accuracy by Domain", "Domain")

    # --- Save CSV summary ---
    csv_path = os.path.join(args.plots, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "key", "correct", "total", "accuracy_percent"])
        w.writerow(["overall", "", correct, total, f"{overall:.2f}"])
        for (typ, key), v in sorted(buckets.items()):
            acc = pct(v["c"], v["t"])
            w.writerow([typ, key, v["c"], v["t"], f"{acc:.2f}"])

    print(f"\nSaved: {lang_png if lang_labels else '(no language plot)'}")
    print(f"Saved: {dom_png if dom_labels else '(no domain plot)'}")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
