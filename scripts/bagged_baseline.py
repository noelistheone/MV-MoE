"""Fairly-bagged GUME baseline (5-seed z-score mean, IDENTICAL bagging to MV-MoE's GUME component) for
all four datasets, all metrics R/N/MAP@10/@20. This is the apples-to-apples baseline a strict reviewer
asks for: MV-MoE bags 5 GUME seeds, so the fair comparison is MV-MoE vs a 5-seed-BAGGED GUME, not vs a
single run. We also emit single-GUME (seed_0) for reference, and the MV-MoE-vs-bagged-GUME % improvement,
which isolates the genuinely non-bagging (user-kNN) contribution. Reuses ensemble_eval (in-memory, Amazon)
and finalize_microlens.chunk_eval (chunked, MicroLens 98k x 17k). Writes results/bai/bagged_baseline.json.
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/MechInterp/scripts")
import ensemble_eval as E

MET = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MAP@10", "MAP@20"]
SDIR = Path("/workspace/MechInterp/results/bai/scores")
finA = json.load(open("/workspace/MechInterp/results/bai/finalize.json"))
finM = json.load(open("/workspace/MechInterp/results/bai/finalize_microlens.json"))
mlB = json.load(open("/workspace/MechInterp/results/bai/microlens_baselines.json"))
seeds_of = lambda ds: sorted(int(re.search(r"_s(\d+)\.npy", p).group(1))
                             for p in glob.glob(str(SDIR / f"{ds}_bprdump_s*.npy")))

# Fair baseline = UNIFORM 5-seed bag (textbook bagging = unweighted average of z-scored seeds).
# Chosen over a validation-tuned bag because the tuned bag is non-deterministic at the +-2e-4 level
# (validation ties among 5 near-identical seeds), whereas the uniform bag is exactly reproducible.
out = {}

# ---- Amazon (in-memory) ----
for ds in ["baby", "sports", "clothing"]:
    E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter"); E._CACHE.clear(); torch.cuda.empty_cache()
    train, valid, test, nu, ni = E.load_split()
    sd = seeds_of(ds); sf = [f"{ds}_bprdump_s{s}.npy" for s in sd]
    mats = E.load_scores(sf)
    bagS = E.combine(mats, [1.0] * len(mats))           # uniform: z-score each seed, sum
    bagged = {m: round(E.evaluate(bagS.clone(), train, test, nu, ni)[m], 5) for m in MET}
    g0 = E.evaluate(E.combine(E.load_scores([sf[0]]), [1.0]).clone(), train, test, nu, ni)
    single = {m: round(g0[m], 5) for m in MET}
    mv = finA[ds]["bag"]
    out[ds] = {"n_seeds": len(sd), "gume_single": single, "gume_bagged": bagged, "mvmoe": mv,
               "mvmoe_vs_bagged_pct": {m: round(100 * (mv[m] - bagged[m]) / bagged[m], 1) for m in MET},
               "mvmoe_vs_single_pct": {m: round(100 * (mv[m] - single[m]) / single[m], 1) for m in MET},
               "bagging_share_of_R20_gain": round(100 * (bagged["Recall@20"] - single["Recall@20"]) /
                                                  (mv["Recall@20"] - single["Recall@20"]), 1)}
    del mats, bagS; torch.cuda.empty_cache()
    print(f"=== {ds} === single R@20={single['Recall@20']:.4f}  bagged R@20={bagged['Recall@20']:.4f}  "
          f"MV-MoE R@20={mv['Recall@20']:.4f} | MV vs bagged R@20 {out[ds]['mvmoe_vs_bagged_pct']['Recall@20']:+.1f}% "
          f"N@20 {out[ds]['mvmoe_vs_bagged_pct']['NDCG@20']:+.1f}% | bagging share of single-gain={out[ds]['bagging_share_of_R20_gain']}%", flush=True)

# ---- MicroLens (chunked) ----
import finalize_microlens as FM
ds = "microlens"
E.DATA = Path(f"/workspace/Recsys/data/{ds}/{ds}.inter"); E._CACHE.clear(); torch.cuda.empty_cache()
train, valid, test, nu, ni = E.load_split()
tru, tri = FM.pairs(train); teu, tei = FM.pairs(test)
sd = seeds_of(ds)
bagged_view = torch.zeros((nu, ni), dtype=torch.float16)
CH = FM.CHUNK
for a in range(0, nu, CH):
    b = min(a + CH, nu); acc = torch.zeros((b - a, ni), device=E.dev)
    for s in sd:
        M = torch.from_numpy(np.load(SDIR / f"{ds}_bprdump_s{s}.npy", mmap_mode="r")[a:b]).to(E.dev).float()
        m = M.mean(1, keepdim=True); sdv = M.std(1, keepdim=True).clamp_min(1e-6)
        acc += (M - m) / sdv; del M
    bagged_view[a:b] = (acc / len(sd)).half().cpu(); del acc; torch.cuda.empty_cache()
bagged = {m: round(FM.chunk_eval([bagged_view], [1.0], tru, tri, teu, tei, nu, ni)[m], 5) for m in MET}
del bagged_view; torch.cuda.empty_cache()
single = finM["gume_single_ensemble"]   # seed_0 GUME, already computed identically in finalize_microlens
mv = finM["bag"]
out[ds] = {"n_seeds": len(sd), "gume_single": single, "gume_bagged": bagged, "mvmoe": mv,
           "mvmoe_vs_bagged_pct": {m: round(100 * (mv[m] - bagged[m]) / bagged[m], 1) for m in MET},
           "mvmoe_vs_single_pct": {m: round(100 * (mv[m] - single[m]) / single[m], 1) for m in MET},
           "bagging_share_of_R20_gain": round(100 * (bagged["Recall@20"] - single["Recall@20"]) /
                                              (mv["Recall@20"] - single["Recall@20"]), 1)}
print(f"=== microlens === single R@20={single['Recall@20']:.4f}  bagged R@20={bagged['Recall@20']:.4f}  "
      f"MV-MoE R@20={mv['Recall@20']:.4f} | MV vs bagged R@20 {out[ds]['mvmoe_vs_bagged_pct']['Recall@20']:+.1f}% "
      f"N@20 {out[ds]['mvmoe_vs_bagged_pct']['NDCG@20']:+.1f}% | bagging share={out[ds]['bagging_share_of_R20_gain']}%", flush=True)

Path("/workspace/MechInterp/results/bai/bagged_baseline.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/bai/bagged_baseline.json")
# compact summary for reporting
print("\n--- MV-MoE vs BAGGED GUME (the fair delta), R@20 / N@20 ---")
for ds in ["baby", "sports", "clothing", "microlens"]:
    p = out[ds]["mvmoe_vs_bagged_pct"]
    print(f"  {ds:10s} R@20 {p['Recall@20']:+5.1f}%  N@20 {p['NDCG@20']:+5.1f}%  R@10 {p['Recall@10']:+5.1f}%  N@10 {p['NDCG@10']:+5.1f}%")
