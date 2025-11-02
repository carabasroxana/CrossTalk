import argparse, json
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold JSONL file (with answers)")
    ap.add_argument("--pred", required=True, help="Path to predictions JSONL file")
    args = ap.parse_args()

    gold = {}
    meta = {}
    with open(args.gold, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            gold[x["id"]] = x["answer"]
            meta[x["id"]] = (x["language"], x["domain"])

    correct, total = 0, 0
    buckets = defaultdict(lambda: {"c": 0, "t": 0})

    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            qid = x["id"]
            if qid not in gold:
                continue
            total += 1
            is_correct = (x["pred"].strip().upper() == gold[qid].strip().upper())
            if is_correct:
                correct += 1
            lang, dom = meta[qid]
            buckets[("lang", lang)]["t"] += 1
            buckets[("lang", lang)]["c"] += int(is_correct)
            buckets[("dom", dom)]["t"] += 1
            buckets[("dom", dom)]["c"] += int(is_correct)

    def pct(c, t): return 0.0 if t == 0 else 100.0 * c / t

    print(f"Overall accuracy: {correct}/{total} = {pct(correct, total):.1f}%\n")

    print("By language:")
    for (typ, key), stats in sorted(buckets.items()):
        if typ == "lang":
            print(f"  {key}: {stats['c']}/{stats['t']} = {pct(stats['c'], stats['t']):.1f}%")

    print("\nBy domain:")
    for (typ, key), stats in sorted(buckets.items()):
        if typ == "dom":
            print(f"  {key}: {stats['c']}/{stats['t']} = {pct(stats['c'], stats['t']):.1f}%")

if __name__ == "__main__":
    main()
