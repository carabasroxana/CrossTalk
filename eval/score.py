import argparse, json, sys, os, csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s: continue
            try: yield json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[warn] Skipping non-JSON line {i} in {path}: {e}", file=sys.stderr)

def pct(c,t): return 0.0 if t==0 else 100.0*c/t
def barplot(save, labels, vals, title, xlabel):
    plt.figure(); plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Accuracy (%)"); plt.xlabel(xlabel); plt.title(title)
    plt.tight_layout(); plt.savefig(save, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--plots", default="reports")
    args = ap.parse_args()
    os.makedirs(args.plots, exist_ok=True)

    gold, meta = {}, {}
    for x in iter_jsonl(args.gold):
        gold[x["id"]] = x["answer"]; meta[x["id"]] = (x["language"], x["domain"])

    correct=total=0; buckets=defaultdict(lambda: {"c":0,"t":0})
    for x in iter_jsonl(args.pred):
        qid=x.get("id"); 
        if qid not in gold: continue
        total+=1; ok=(str(x.get("pred","")).strip().upper()==str(gold[qid]).strip().upper())
        correct+=int(ok); lang,dom=meta[qid]
        for k in [("lang",lang),("dom",dom)]:
            buckets[k]["t"]+=1; buckets[k]["c"]+=int(ok)

    overall=pct(correct,total)
    print(f"Overall accuracy: {correct}/{total} = {overall:.1f}%\n")

    langs=[]; lvals=[]; doms=[]; dvals=[]
    print("By language:")
    for (t,k),v in sorted(buckets.items()):
        if t=="lang":
            acc=pct(v["c"],v["t"]); print(f"  {k}: {v['c']}/{v['t']} = {acc:.1f}%")
            langs.append(k); lvals.append(acc)
    print("\nBy domain:")
    for (t,k),v in sorted(buckets.items()):
        if t=="dom":
            acc=pct(v["c"],v["t"]); print(f"  {k}: {v['c']}/{v['t']} = {acc:.1f}%")
            doms.append(k); dvals.append(acc)

    if langs: barplot(os.path.join(args.plots,"acc_by_language.png"), langs, lvals, "Accuracy by Language", "Language")
    if doms:  barplot(os.path.join(args.plots,"acc_by_domain.png"), doms, dvals, "Accuracy by Domain", "Domain")

    with open(os.path.join(args.plots,"summary.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["metric","key","correct","total","accuracy_percent"])
        w.writerow(["overall","",correct,total,f"{overall:.2f}"])
        for (t,k),v in sorted(buckets.items()):
            w.writerow([t,k,v["c"],v["t"],f"{pct(v['c'],v['t']):.2f}"])

if __name__=="__main__": main()
