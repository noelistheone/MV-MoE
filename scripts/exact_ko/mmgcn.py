"""Exact structural knockout for MMGCN (Wei et al., MM 2019).

SOURCE: /workspace/Recsys/src/models/mmgcn.py  (read-only; verified line by line)

==============================================================================
1. What the architecture actually does
==============================================================================
`_propagate()`:
    outs = [towers["v"](norm_adj, v_feat, id_embedding),      # if v_feat is not None
            towers["t"](norm_adj, t_feat, id_embedding)]      # if t_feat is not None
    rep  = torch.stack(outs, 0).mean(0)
`full_sort_predict()`:  rep[:nU][users] @ rep[nU:].T

`_ModalityTower.forward(norm_adj, feat, id_embedding)`:
    item_feat = feat_mlp(feat)                       # W_m f_i + b_m
    x = normalize(cat([preference, item_feat]), -1)  # ROW-wise: every node -> unit norm
    for l in range(n_layers):                        # n_layers = 3 in these runs
        h     = leaky_relu(sparse.mm(norm_adj, x @ conv_w[l]))
        x_hat = leaky_relu(linear_layers[l](x)) + id_embedding      # has_id=True
        x     = leaky_relu(g_layers[l](h) + x_hat)

==============================================================================
2. Five structural facts that decide what a knockout is here
==============================================================================
F1  The two towers are PARAMETRICALLY DISJOINT (`preference`, `feat_mlp`, `conv_w`,
    `linear_layers`, `g_layers` are per-tower). The only tensors they share are read-only
    inputs: `norm_adj` and `id_embedding`. Deleting one tower therefore leaves the other
    BIT-IDENTICAL -- the survey's "EXACT_ADDITIVE" premise holds.
    Certificate: attribution()["tower_disjointness_max_delta"].

F2  The `mean` over towers is the only cross-modal coupling and it is a GLOBAL POSITIVE
    SCALAR (1/M). Scoring is u @ i.T with BOTH sides read off the SAME `rep`, so scaling
    rep by c > 0 scales every score by c^2 and preserves each user's ranking exactly.
    "mean over survivors" and "sum of survivors / M_original" therefore give identical
    top-K. The tower mean is NOT an ambiguity-creating normaliser.
    Certificate: `*_knockout_hi` vs `*_knockout_hi_massremoved` agree to 0.000000.

F3  There is NO ID-only survivor. `id_embedding` reaches the score ONLY from inside a
    tower (`x_hat = ... + id_embedding`). Deleting BOTH towers leaves `torch.stack([])`
    -- literally no representation. `both_knockout` is therefore necessarily the
    content-level arm; a tower-level `both_knockout_hi` DOES NOT EXIST and is reported as
    UNDEFINED rather than faked. (`idonly_diag` in attribution() scores with the raw
    `id_embedding`; that is a diagnostic, NOT a knockout of this model.)

F4  Tower deletion also deletes that tower's `preference` table -- a FREE per-user
    parameter that is not content -- plus the tower's graph weights, which are what
    carries `id_embedding` and the CF signal. So tower deletion OVER-states content.
    We emit both ends and report them as a bracket:
        lo (headline) `<m>_knockout`      content term deleted INSIDE the tower
        hi            `<m>_knockout_hi`   the whole tower deleted
    The gap between them is the tower's non-content capacity, not modality content.

F5  `feat_mlp(f) = W f + b` is additive, so "delete the content term" = keep b exactly.
    BUT the row-wise `F.normalize` immediately after is a SHARED NORMALISER: the model's
    own counterfactual renormalises the surviving b to unit norm, while the mass actually
    removed corresponds to b/||W f_i + b||. Per CONTRACT rule 3 that makes the content
    arm DELETE_PLUS_RENORM, and both ends are emitted:
        `<m>_knockout`             = b/||b||           (renormalised; model's own counterfactual)
        `<m>_knockout_massremoved` = b/||W f_i + b||   (only the deleted mass removed)
    The mass-removed end drives item rows to ~1e-3 norm while user rows stay at 1, a state
    the network never sees in training, so it is the pessimistic end of the bracket.
    Hence EXACTNESS = "DELETE_PLUS_RENORM" for the module as a whole, even though the
    tower-level arms (F2) are strictly EXACT.

==============================================================================
3. What the input-mean screen actually measured on MMGCN
==============================================================================
Per-dim input-mean substitution gives every item the SAME feature vector, so after
`feat_mlp` and the row-wise `normalize` ALL items share ONE layer-0 direction in that
tower. That is a content deletion -- with an arbitrary residual direction
normalize(W f_mean + b) instead of normalize(b). So the screen is NOT structurally broken
on MMGCN the way it is on DAMRS (where it yields image_knockout == both_knockout); it is a
deletion whose residual constant is arbitrary. `*_screen_meanfeat` reproduces it inside
this module so the screen entry and the exact arms are measured under ONE protocol.

==============================================================================
4. Content-interface taxonomy
==============================================================================
The C1..C5 legend is not defined anywhere in the repo yet (grep: only CONTRACT.md names
it), so the legend assumed here is stated explicitly:
    C1  frozen content-derived item-item graph; content off-path at inference (FREEDOM)
    C2  live content feature projected and ADDED to the scored embedding (VBPR, LGMRec)
    C3  content as the layer-0 NODE FEATURE of a dedicated per-modality GNN tower whose
        outputs are pooled                                                  <-- MMGCN
    C4  content only in training-time objectives; off the inference path
    C5  ID-only: content never reaches the scoring function
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASS = "C3"
EXACTNESS = "DELETE_PLUS_RENORM"   # content arms: layer-0 F.normalize (F5).
                                   # tower arms (*_knockout_hi) are strictly EXACT (F2).

MODALITY_KEY = {"image": "v", "text": "t"}

ARMS = {
    "baseline": "model as trained, re-derived from parts (asserted == full_sort_predict)",
    "image_knockout": "LO/headline: W_v f deleted inside the image tower, layer-0 row "
                      "renormalised (model's own counterfactual). preference/id/graph kept.",
    "image_knockout_massremoved": "same deletion, original layer-0 denominator kept "
                                  "(only the removed mass is gone) -- pessimistic bracket end",
    "image_knockout_hi": "HI: the whole image tower is deleted (EXACT). Also removes the "
                         "tower's free `preference` table and graph weights -> upper bound.",
    "image_knockout_hi_massremoved": "tower deleted, denominator kept at M=2 -- rank-identical "
                                     "certificate for F2",
    "image_screen_meanfeat": "reproduction of the input-mean screen (per-dim input mean)",
    "text_knockout": "as image_knockout, text tower",
    "text_knockout_massremoved": "as above, text tower",
    "text_knockout_hi": "the whole text tower is deleted (EXACT)",
    "text_knockout_hi_massremoved": "text tower deleted, denominator kept at M=2",
    "text_screen_meanfeat": "input-mean screen, text",
    "both_knockout": "content deleted inside BOTH towers (renormalised). The ONLY definable "
                     "both-arm: deleting both towers leaves no representation (F3).",
    "both_knockout_massremoved": "content deleted in both towers, denominators kept",
    "both_screen_meanfeat": "input-mean screen, both",
}
BRACKET = {"image": ("image_knockout", "image_knockout_hi"),
           "text": ("text_knockout", "text_knockout_hi"),
           "image_renorm": ("image_knockout_massremoved", "image_knockout"),
           "text_renorm": ("text_knockout_massremoved", "text_knockout"),
           "both_renorm": ("both_knockout_massremoved", "both_knockout")}
UNDEFINED_ARMS = {"both_knockout_hi": "deleting both towers leaves no representation (F3)"}


# ------------------------------------------------------------------ tower replay
def _item_layer0(tower, feat: torch.Tensor, content: str) -> torch.Tensor:
    """Item block of the tower's layer-0 input, BEFORE the row-wise normalize.

    'full'      -> W f_i + b                    (the model)
    'delete'    -> b            (broadcast)     (delete the content term; renormalised later)
    'meanfeat'  -> W mean(f) + b                (the input-mean screen)
    """
    if content == "full":
        return tower.feat_mlp(feat)
    if content == "meanfeat":
        return tower.feat_mlp(feat.mean(0, keepdim=True).expand_as(feat))
    if content == "delete":
        return tower.feat_mlp.bias.unsqueeze(0).expand(tower.n_items, -1)
    raise ValueError(f"unknown content mode {content!r}")


def _layer0(tower, feat: torch.Tensor, content: str, massremoved: bool) -> torch.Tensor:
    """Full layer-0 input (users + items) after the model's row-wise normalize.

    massremoved=True keeps the ORIGINAL per-item denominator ||W f_i + b|| instead of
    renormalising the survivor, i.e. removes exactly the mass of the deleted term (F5)."""
    item = _item_layer0(tower, feat, content)
    if massremoved and content != "full":
        denom = tower.feat_mlp(feat).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        item = item / denom
        user = F.normalize(tower.preference, dim=-1)
        return torch.cat([user, item], dim=0)
    return F.normalize(torch.cat([tower.preference, item], dim=0), dim=-1)


def _tower_forward(tower, norm_adj, feat, id_embedding,
                   content: str = "full", massremoved: bool = False) -> torch.Tensor:
    """Byte-faithful replay of _ModalityTower.forward with a switchable content term."""
    x = _layer0(tower, feat, content, massremoved)
    for l in range(tower.n_layers):
        h = torch.sparse.mm(norm_adj, x @ tower.conv_w[l])
        h = F.leaky_relu(h)
        x_hat = F.leaky_relu(tower.linear_layers[l](x))
        if tower.has_id:
            x_hat = x_hat + id_embedding
        x = F.leaky_relu(tower.g_layers[l](h) + x_hat)
    return x


def _feat_of(model, key: str) -> torch.Tensor:
    return model.v_feat if key == "v" else model.t_feat


@torch.no_grad()
def _rep(model, content: Dict[str, str] | None = None, drop: Tuple[str, ...] = (),
         massremoved: bool = False, keep_denominator: bool = False) -> torch.Tensor:
    """Combined representation.

    content          per-tower layer-0 content mode ('full' | 'delete' | 'meanfeat')
    drop             towers deleted outright
    massremoved      F5 bracket end for the content deletion
    keep_denominator divide by the ORIGINAL tower count instead of the surviving count
                     (F2 certificate; rank-identical)"""
    content = content or {}
    m_total = len(model.towers)
    outs = []
    for key in ("v", "t"):                      # same order as _propagate()
        if key not in model.towers or key in drop:
            continue
        outs.append(_tower_forward(model.towers[key], model.norm_adj, _feat_of(model, key),
                                   model.id_embedding, content.get(key, "full"), massremoved))
    if not outs:
        raise ValueError("MMGCN has no representation once every tower is deleted (F3)")
    if len(outs) == m_total and not keep_denominator:
        return torch.stack(outs, dim=0).mean(dim=0)      # exactly the model's own op
    return torch.stack(outs, dim=0).sum(dim=0) / (m_total if keep_denominator else len(outs))


def _split(model, rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return rep[:model.n_users], rep[model.n_users:]


# ------------------------------------------------------------------ contract API
@torch.no_grad()
def variants(model, dataset=None, device=None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    both_present = len(model.towers) == 2
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {"baseline": _split(model, _rep(model))}
    for mod, key in MODALITY_KEY.items():
        if key not in model.towers:
            continue
        out[f"{mod}_knockout"] = _split(model, _rep(model, {key: "delete"}))
        out[f"{mod}_knockout_massremoved"] = _split(
            model, _rep(model, {key: "delete"}, massremoved=True))
        out[f"{mod}_screen_meanfeat"] = _split(model, _rep(model, {key: "meanfeat"}))
        if both_present:
            out[f"{mod}_knockout_hi"] = _split(model, _rep(model, drop=(key,)))
            out[f"{mod}_knockout_hi_massremoved"] = _split(
                model, _rep(model, drop=(key,), keep_denominator=True))
    all_keys = list(model.towers)
    out["both_knockout"] = _split(model, _rep(model, {k: "delete" for k in all_keys}))
    out["both_knockout_massremoved"] = _split(
        model, _rep(model, {k: "delete" for k in all_keys}, massremoved=True))
    out["both_screen_meanfeat"] = _split(model, _rep(model, {k: "meanfeat" for k in all_keys}))
    return out


@torch.no_grad()
def attribution(model, dataset=None, device=None) -> dict:
    nU = model.n_users
    rep = _rep(model)

    def item_norm(x):
        return float(x[nU:].norm(dim=-1).mean())

    a: dict = {"n_towers": len(model.towers), "n_layers": int(model.n_layers),
               "combiner": "mean over towers (global positive scalar -> rank-neutral, F2)",
               "||rep_i||": item_norm(rep), "||rep_u||": float(rep[:nU].norm(dim=-1).mean())}

    per_tower = {}
    for key in model.towers:
        t = model.towers[key]
        feat = _feat_of(model, key)
        out_full = _tower_forward(t, model.norm_adj, feat, model.id_embedding, "full")
        out_del = _tower_forward(t, model.norm_adj, feat, model.id_embedding, "delete")
        raw = t.feat_mlp(feat)
        b = t.feat_mlp.bias.unsqueeze(0)
        wf = raw - b
        n0_full = F.normalize(raw, dim=-1)
        n0_del = F.normalize(b.expand_as(raw), dim=-1)
        mean_raw = t.feat_mlp(feat.mean(0, keepdim=True).expand_as(feat))
        n0_mean = F.normalize(mean_raw, dim=-1)
        per_tower[key] = {
            "||tower_out_i||": item_norm(out_full),
            "||tower_out_i (content deleted)||": item_norm(out_del),
            "||tower_out_i - tower_out_i(content deleted)||":
                float((out_full - out_del)[nU:].norm(dim=-1).mean()),
            "||W f||_mean": float(wf.norm(dim=-1).mean()),
            "||b||": float(b.norm()),
            "||preference_u||_mean": float(t.preference.norm(dim=-1).mean()),
            "content_share_of_layer0_premorm": float(wf.norm(dim=-1).mean()
                                                     / raw.norm(dim=-1).mean()),
            # after the row-wise normalize the layer-0 vector is a pure DIRECTION, so what
            # matters is how far each counterfactual's constant direction sits from the real one
            "cos(layer0_i, deleted_const)": float((n0_full * n0_del).sum(-1).mean()),
            "cos(layer0_i, screen_const)": float((n0_full * n0_mean).sum(-1).mean()),
            "cos(deleted_const, screen_const)": float((n0_del[0] * n0_mean[0]).sum()),
            # the screen collapses all items onto one row: certificate that it is a
            # constant-direction DELETION here, not a leak (unlike DAMRS)
            "screen_item_rows_identical_max_delta": float((n0_mean - n0_mean[0:1]).abs().max()),
            "feat_is_frozen_buffer_not_parameter": not any(feat is p for p in model.parameters()),
        }
    a["per_tower"] = per_tower

    if len(model.towers) == 2:
        keys = list(model.towers)
        solo = _tower_forward(model.towers[keys[0]], model.norm_adj, _feat_of(model, keys[0]),
                              model.id_embedding, "full")
        a["tower_disjointness_max_delta"] = float(
            (solo - 2.0 * _rep(model, drop=(keys[1],), keep_denominator=True)).abs().max())
        u1, _ = _split(model, _rep(model, drop=(keys[1],)))
        u2, _ = _split(model, _rep(model, drop=(keys[1],), keep_denominator=True))
        a["renorm_is_global_scalar_max_delta"] = float((u1 - 2.0 * u2).abs().max())

    a["id_only_survivor_exists"] = False
    a["both_towers_deleted"] = "UNDEFINED (no representation survives; F3)"
    a["idonly_diag_note"] = ("id_embedding reaches the score only through a tower; scoring "
                             "with it directly is a diagnostic, not a knockout of this model")
    return a


@torch.no_grad()
def _recon_pair(model, u, i, chunk: int = 2048) -> float:
    dev = u.device
    worst = 0.0
    for s in range(0, model.n_users, chunk):
        users = torch.arange(s, min(s + chunk, model.n_users), device=dev)
        ref = model.full_sort_predict({"user": users})
        worst = max(worst, float((u[users] @ i.t() - ref).abs().max()))
        del ref
    return worst


@torch.no_grad()
def recon_error(model, dataset=None, device=None, chunk: int = 2048) -> float:
    """max |baseline_scores - model.full_sort_predict| over ALL users, in chunks."""
    u, i = variants(model)["baseline"]
    return _recon_pair(model, u, i, chunk)


@torch.no_grad()
def recon_error_detail(model, dataset=None, device=None, chunk: int = 2048,
                       repeats: int = 5) -> dict:
    """recon_error plus the model's OWN self-consistency floor.

    `torch.sparse.mm` on CUDA is not bit-reproducible, so calling `_propagate()` twice
    already disagrees. `self_floor` = the same comparison using the MODEL's own rep; any
    decomposition error below it is unmeasurable."""
    u, i = variants(model)["baseline"]
    rep = model._propagate()
    su, si = _split(model, rep)
    r2 = model._propagate()
    err = _recon_pair(model, u, i, chunk)
    floors = [_recon_pair(model, su, si, chunk) for _ in range(repeats)]
    scale = float((su[:chunk] @ si.t()).abs().max())
    # Scale-free certificate: does the re-derived baseline pick the SAME top-20 as the
    # model's own scoring path? (unmasked; history masking is applied identically to both)
    users = torch.arange(0, min(4096, model.n_users), device=u.device)
    mine_k = torch.topk(u[users] @ i.t(), 20, dim=-1).indices
    ref_k = torch.topk(model.full_sort_predict({"user": users}), 20, dim=-1).indices
    same = [set(a.tolist()) == set(b.tolist()) for a, b in zip(mine_k.cpu(), ref_k.cpu())]
    return {"recon_error": err,
            "self_floor_max": max(floors), "self_floor_runs": floors,
            "rel_recon_error": err / scale,
            "propagate_self_delta": float((rep - r2).abs().max()),
            "max_abs_score": scale,
            "top20_set_agreement_frac": float(sum(same)) / len(same),
            "top20_users_checked": len(same)}


# ------------------------------------------------------------------ standalone run
def _main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    from recsys_bridge import load_frozen                             # noqa: E402
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: E402

    pins = json.loads((ROOT / "results" / "phase_micro" / "ckpt_pins.json").read_text())
    pin = pins.get(f"mmgcn/{args.dataset}", {}).get("path") if args.ckpt is None else args.ckpt
    cfg, ds, model, test_loader = load_frozen("mmgcn", args.dataset, device, ckpt_path=pin)
    print(f"ckpt = {model._ckpt_name}  (pinned={model._ckpt_pinned})", flush=True)
    print(f"missing keys = {model._missing_keys}", flush=True)

    det = recon_error_detail(model, ds, device)
    print("recon: " + json.dumps(det), flush=True)

    attr = attribution(model, ds, device)
    print(json.dumps(attr, indent=2), flush=True)

    vs = variants(model, ds, device)
    res, base_topk = {}, None
    for name, (u, i) in vs.items():
        m, topk, _ = evaluate_item_matrix(u, i, test_loader, device)
        m = {k: float(v) for k, v in m.items()}
        if name == "baseline":
            base_topk = topk
            res[name] = {"metrics": m}
        else:
            res[name] = {"metrics": m,
                         "delta": {k: m[k] - res["baseline"]["metrics"][k] for k in m},
                         "ranking_change": ranking_change(base_topk, topk)}
        d = "" if name == "baseline" else f"  dR@20={m['Recall@20'] - res['baseline']['metrics']['Recall@20']:+.6f}"
        print(f"  {name:32s} R@20={m['Recall@20']:.6f}  N@20={m['NDCG@20']:.6f}{d}", flush=True)

    logged = json.loads((RECSYS / "logs" / model._ckpt_name.replace(".pt", "") /
                         "result.json").read_text())["test_result"]
    base_m = res["baseline"]["metrics"]
    out = {"model": "mmgcn", "dataset": args.dataset, "CLASS": CLASS, "EXACTNESS": EXACTNESS,
           "ckpt": model._ckpt_path, "recon": det,
           "logged_test": {k: float(v) for k, v in logged.items()},
           "baseline_minus_logged": {k: base_m[k] - float(logged[k]) for k in base_m
                                     if k in logged},
           "attribution": attr, "arms": res, "arm_docs": ARMS, "bracket": BRACKET,
           "undefined_arms": UNDEFINED_ARMS}
    path = Path(args.out) if args.out else (ROOT / "results" / "_scratch" /
                                            f"exact_ko_mmgcn_{args.dataset}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
