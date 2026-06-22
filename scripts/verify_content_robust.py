"""Robustness: tune userKNN weight on VALID first (best base), THEN add content.
Confirms content's marginal value isn't just compensating a mistuned uniform base.
Also reports a 'no-tune' honest read: TEST R@20 at content weight selected on VALID."""
import sys, os, json
sys.path.insert(0, "/workspace/MechInterp/scripts")
import torch
from verify_content_null import (load_split, gt_matrix, load_mat, zscore_rows,
                                  evaluate, DEV)

def run(ds):
    tr, val, tst, nu, ni = load_split(ds)
    tu = torch.tensor(tr[0], device=DEV); ti = torch.tensor(tr[1], device=DEV)
    tidx = (tu, ti)
    REL_v, nrel_v = gt_matrix(val, nu, ni)
    REL_t, nrel_t = gt_matrix(tst, nu, ni)
    gume = None
    for s in (2024, 2025, 2026):
        z = zscore_rows(load_mat(f"{ds}_bprdump_s{s}.npy"))
        gume = z if gume is None else gume + z
    gume /= 3.0
    z_uk = zscore_rows(load_mat(f"{ds}_userknn.npy"))
    # tune w_uk on valid
    best_uk, best_r = 0.0, -1
    for w in [0.0,0.25,0.5,0.75,1.0,1.25,1.5]:
        r,_ = evaluate((gume + w*z_uk), tidx, REL_v, nrel_v)
        if r > best_r: best_r, best_uk = r, w
    base = gume + best_uk*z_uk
    br_v,_ = evaluate(base.clone(), tidx, REL_v, nrel_v)
    br_t, bn_t = evaluate(base.clone(), tidx, REL_t, nrel_t)
    res = {"ds":ds,"best_uk_w":best_uk,"tuned_base_valid_R20":br_v,
           "tuned_base_test_R20":br_t}
    print(f"[{ds}] VALID-tuned base: w_uk={best_uk} valid R@20={br_v:.5f} TEST R@20={br_t:.5f}", flush=True)
    for tag in ["img","text","joint"]:
        torch.cuda.empty_cache()
        zc = zscore_rows(load_mat(f"{ds}_bai2i_sig_{tag}_learned_k20.npy"))
        bw, brr = 0.0, -1
        for wc in [0.0,0.05,0.1,0.2,0.3,0.5]:
            r,_ = evaluate(base + wc*zc, tidx, REL_v, nrel_v)
            if r > brr: brr, bw = r, wc
        rt,nt = evaluate(base + bw*zc, tidx, REL_t, nrel_t)
        dr = rt - br_t
        print(f"  [{tag}] best_w(valid)={bw} => TEST R@20={rt:.5f} (Δ {dr:+.5f}, {100*dr/br_t:+.2f}%)", flush=True)
        res[tag] = {"best_w":bw,"test_R20":rt,"delta":dr,"pct":100*dr/br_t}
        del zc
    out=f"/workspace/Recsys/results/bai/content_robust_{ds}.json"
    json.dump(res, open(out,"w"), indent=2)

if __name__ == "__main__":
    run(sys.argv[1])
