"""Exact structural knockout for VBPR (He & McAuley, AAAI'16).

Interface class C2 -- ADDITIVE PROJECTED FEATURE on the inference path.

    item_i = [ e_id_i  ||  W_t t_i + W_v v_i + b ]        (item_linear on [t||v])
    u      = [ u_cf    ||  u_ct ]                          (one [n_users, 2d] table)
    s_ui   = u_cf . e_id_i  +  u_ct . (W_t t_i)  +  u_ct . (W_v v_i)  +  u_ct . b

Every content term is additive and separable, so structural knockout is EXACT:
delete the modality's own additive term, touch nothing else.

THREE THINGS THAT MUST NOT BE GOT WRONG
---------------------------------------
1. MODALITY ORDER. `VBPR._item_embeddings` builds `torch.cat([self.t_feat, self.v_feat])`
   -- TEXT FIRST -- while `item_linear` is declared `nn.Linear(v_feat_dim + t_feat_dim, d)`.
   The declared name order is the reverse of the actual column order. On baby
   (v=4096, t=384) BOTH slicings are shape-valid, so slicing `W[:, :v_feat_dim]` as
   "image" silently swaps the two modalities instead of raising. Correct slicing is
       W_txt = W[:, :t_feat_dim],  W_img = W[:, t_feat_dim:].
   `attribution()` emits `recon_err_wrong_slice` as a live guard on this.
2. THE USER TABLE IS SHARED. `u_ct` (the second half of the [n_users, 2d] user
   embedding) is used by text, image AND the bias. Zeroing any part of the user
   vector would knock out all three at once. We never touch the user side.
3. THE BIAS IS RANK-INERT, SO KEEP IT. `u_ct . b` is the same scalar for every item,
   i.e. a per-user constant, so it cannot change any ranking. It is not content and
   it survives in every arm, `both_knockout` included.

EXTRA ARM -- why the coarse screen is valid HERE and invalid elsewhere.
For this class the published input-mean screen is *provably rank-equivalent* to exact
deletion: replacing v with its per-dimension mean makes W_v v_i the same vector for
every item, so the score changes by u_ct . (W_v v_bar) -- a per-user constant. Hence
`image_inputmean` must produce exactly the same ranking as `image_knockout`. We emit
it as an arm and certify the equivalence in `attribution` (per-user spread of the score
difference == 0), so the paper can show the screen is sound on C2 and unsound on the
classes where content passes through a kNN graph or a normaliser.

Outputs stay under MechInterp; /workspace/Recsys is read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/MechInterp")
RECSYS = Path("/workspace/Recsys")
# Both repos use the top-level package name `src`; `src` MUST resolve to Recsys.
for _p in (str(RECSYS), str(ROOT / "src" / "models"), str(ROOT / "src" / "interp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASS = "C2"          # additive projected-feature term, live on the inference path
EXACTNESS = "EXACT"

# Users per chunk when materialising [users x items] score blocks (GPU is shared).
_USER_CHUNK = 2048


# ------------------------------------------------------------------ decomposition
@torch.no_grad()
def _streams(model) -> dict:
    """Split VBPR's item embedding into its additive terms.

    Returns u_all [n_users, 2d] and the item-side blocks:
        id_e  [n_items, d]  collaborative (ID) block
        h_img [n_items, d]  W_v v   (zeros if the model has no image modality)
        h_txt [n_items, d]  W_t t   (zeros if the model has no text modality)
        bias  [n_items, d]  broadcast item_linear.bias (per-user constant in score)
    """
    W = model.item_linear.weight.detach()          # [d, v_dim + t_dim]
    b = model.item_linear.bias.detach()            # [d]
    v_dim, t_dim = int(model.v_feat_dim), int(model.t_feat_dim)
    assert W.shape[1] == v_dim + t_dim, (
        f"item_linear in_features {W.shape[1]} != v_feat_dim+t_feat_dim {v_dim + t_dim}")

    n_items, d = model.n_items, model.embedding_size
    dev, dt = W.device, W.dtype
    zeros = torch.zeros(n_items, d, device=dev, dtype=dt)

    if model.v_feat is not None and model.t_feat is not None:
        # _item_embeddings does cat([t_feat, v_feat]) -> TEXT occupies columns [0, t_dim).
        W_txt, W_img = W[:, :t_dim], W[:, t_dim:]
        h_txt = model.t_feat.detach() @ W_txt.t()
        h_img = model.v_feat.detach() @ W_img.t()
    elif model.v_feat is not None:                 # image only: feat = v_feat
        h_img = model.v_feat.detach() @ W[:, :v_dim].t()
        h_txt = zeros
    else:                                          # text only: feat = t_feat
        h_txt = model.t_feat.detach() @ W[:, :t_dim].t()
        h_img = zeros

    return {"u_all": model.user_embedding.detach(),
            "id_e": model.item_id_embedding.detach(),
            "h_img": h_img, "h_txt": h_txt,
            "bias": b.unsqueeze(0).expand(n_items, d),
            "d": d, "W": W, "v_dim": v_dim, "t_dim": t_dim}


def _item_matrix(s: dict, keep_img: bool, keep_txt: bool,
                 img_const: torch.Tensor | None = None,
                 txt_const: torch.Tensor | None = None) -> torch.Tensor:
    """[id_e || (h_img?) + (h_txt?) + bias].

    `*_const` injects a constant (input-mean) projected vector in place of the
    per-item term; that is the coarse screen, kept only to certify its equivalence.
    """
    proj = s["bias"].clone()
    proj = proj + (s["h_img"] if keep_img else (img_const if img_const is not None else 0.0))
    proj = proj + (s["h_txt"] if keep_txt else (txt_const if txt_const is not None else 0.0))
    return torch.cat([s["id_e"], proj], dim=-1)


@torch.no_grad()
def _inputmean_consts(model, s: dict) -> dict:
    """W_m * mean_i(feat_m) broadcast over items -- what the input-mean screen produces."""
    W, t_dim, v_dim = s["W"], s["t_dim"], s["v_dim"]
    n_items = model.n_items
    out = {}
    if model.v_feat is not None:
        Wv = W[:, t_dim:] if model.t_feat is not None else W[:, :v_dim]
        vbar = model.v_feat.detach().mean(0, keepdim=True)      # per-dim mean, as in the screen
        out["img"] = (vbar @ Wv.t()).expand(n_items, s["d"])
    if model.t_feat is not None:
        Wt = W[:, :t_dim]
        tbar = model.t_feat.detach().mean(0, keepdim=True)
        out["txt"] = (tbar @ Wt.t()).expand(n_items, s["d"])
    return out


# ------------------------------------------------------------------ contract API
@torch.no_grad()
def variants(model, dataset, device) -> dict:
    """name -> (user_emb [n_users, 2d], item_emb [n_items, 2d]) scored by u @ i.T."""
    s = _streams(model)
    u = s["u_all"]
    c = _inputmean_consts(model, s)
    out = {
        "baseline":       (u, _item_matrix(s, True,  True)),
        "image_knockout": (u, _item_matrix(s, False, True)),    # delete W_v v
        "text_knockout":  (u, _item_matrix(s, True,  False)),   # delete W_t t
        "both_knockout":  (u, _item_matrix(s, False, False)),   # ID + rank-inert bias only
    }
    # Screen-equivalence arms: exact deletion vs. per-dim input mean. Rank-identical by
    # construction on C2; emitted so the claim is measured, not asserted.
    if "img" in c:
        out["image_inputmean"] = (u, _item_matrix(s, False, True, img_const=c["img"]))
    if "txt" in c:
        out["text_inputmean"] = (u, _item_matrix(s, True, False, txt_const=c["txt"]))
    return out


@torch.no_grad()
def attribution(model, dataset, device) -> dict:
    s = _streams(model)
    def mn(x): return float(x.norm(dim=-1).mean())

    W, t_dim, v_dim = s["W"], s["t_dim"], s["v_dim"]
    both = model.v_feat is not None and model.t_feat is not None
    proj = s["h_img"] + s["h_txt"] + s["bias"]

    out = {
        "interface": "additive projected feature (item_linear) concatenated to the ID block",
        "fusion": "concatenation + a single linear map; no learned modality weight",
        "||id_e||": mn(s["id_e"]),
        "||h_img||": mn(s["h_img"]),
        "||h_txt||": mn(s["h_txt"]),
        "||bias||": float(s["bias"][0].norm()),
        "||proj_block||": mn(proj),
        "||W_img||_F": float(W[:, t_dim:].norm()) if both else None,
        "||W_txt||_F": float(W[:, :t_dim].norm()) if both else None,
        "v_feat_dim": v_dim, "t_feat_dim": t_dim,
        "column_order": "[t_feat || v_feat] -- text first, opposite to the declared "
                        "nn.Linear(v_feat_dim + t_feat_dim) name order",
        "bias_is_rank_inert": True,
        "note_bias": "item_linear.bias is identical for every item, so u.bias is a per-user "
                     "constant: it cannot change a ranking and is kept in every arm.",
    }

    if both:
        # ---- guard 1: the modality-order trap, measured live.
        # Slice as if the declared order were true (image = first v_dim columns).
        h_img_wrong = model.v_feat.detach() @ W[:, :v_dim].t()
        h_txt_wrong = model.t_feat.detach() @ W[:, v_dim:v_dim + t_dim].t()
        wrong = torch.cat([s["id_e"], h_img_wrong + h_txt_wrong + s["bias"]], dim=-1)
        out["recon_err_wrong_slice"] = _max_score_err(model, s["u_all"], wrong)
        out["recon_err_wrong_slice_note"] = (
            "max |score - model score| when slicing by v_feat_dim FIRST (the declared but "
            "wrong order). Large => the modality-order trap is real and we are not in it.")

    # ---- guard 2: input-mean screen == exact deletion, on the score surface.
    # s_inputmean - s_exact = u_ct . (W_m featbar), the SAME scalar for every item,
    # so the per-user spread over items must be 0.
    c = _inputmean_consts(model, s)
    if "img" in c:
        exact = _item_matrix(s, keep_img=False, keep_txt=True)
        mean_ = _item_matrix(s, keep_img=False, keep_txt=True, img_const=c["img"])
        out["image_inputmean_rank_equiv"] = _per_user_spread(s["u_all"], exact, mean_)
    if "txt" in c:
        exact = _item_matrix(s, keep_img=True, keep_txt=False)
        mean_ = _item_matrix(s, keep_img=True, keep_txt=False, txt_const=c["txt"])
        out["text_inputmean_rank_equiv"] = _per_user_spread(s["u_all"], exact, mean_)
    out["inputmean_equiv_note"] = (
        "max over users of (max_i - min_i) of the per-item score difference between the "
        "input-mean screen and exact deletion; 0 (to float precision) proves the screen is "
        "a per-user constant shift, hence rank-equivalent, for this interface class.")
    return out


@torch.no_grad()
def recon_error(model, dataset, device) -> float:
    """max |baseline_scores - VBPR.full_sort_predict| over ALL users x items."""
    s = _streams(model)
    base = _item_matrix(s, True, True)
    return _max_score_err(model, s["u_all"], base)


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def _max_score_err(model, u_all: torch.Tensor, item_matrix: torch.Tensor) -> float:
    """max |u @ item_matrix.T - model.full_sort_predict|, chunked over users."""
    worst = 0.0
    n = u_all.shape[0]
    for lo in range(0, n, _USER_CHUNK):
        idx = torch.arange(lo, min(lo + _USER_CHUNK, n), device=u_all.device)
        ours = u_all[idx] @ item_matrix.t()
        theirs = model.full_sort_predict({"user": idx})
        worst = max(worst, float((ours - theirs).abs().max()))
        del ours, theirs
    return worst


@torch.no_grad()
def _per_user_spread(u_all: torch.Tensor, mat_a: torch.Tensor, mat_b: torch.Tensor) -> float:
    """max over users of the range, across items, of (score_b - score_a).

    Zero <=> the two item matrices differ by a per-user constant <=> identical rankings.
    """
    worst = 0.0
    n = u_all.shape[0]
    for lo in range(0, n, _USER_CHUNK):
        u = u_all[lo:lo + _USER_CHUNK]
        d = (u @ mat_b.t()) - (u @ mat_a.t())
        worst = max(worst, float((d.max(dim=1).values - d.min(dim=1).values).max()))
        del d
    return worst


# ------------------------------------------------------------------ standalone check
def _selfcheck(dataset_name: str = "baby", gpu: int = 0, out_dir: str | None = None) -> dict:
    """Run the contract's three acceptance checks and every arm. Read-only w.r.t. Recsys."""
    from recsys_bridge import load_frozen, latest_ckpt           # noqa: PLC0415
    from ranking_effects import evaluate_item_matrix, ranking_change  # noqa: PLC0415

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    pins = json.loads((ROOT / "results" / "phase_micro" / "ckpt_pins.json").read_text())
    pin = pins.get(f"vbpr/{dataset_name}", {}).get("path")
    cfg, ds, model, loader = load_frozen("vbpr", dataset_name, device, ckpt_path=pin)

    rec = recon_error(model, ds, device)
    attr = attribution(model, ds, device)
    arms = variants(model, ds, device)

    res, base_topk, _ = evaluate_item_matrix(*arms["baseline"], loader, device)
    base_r20 = float(res["Recall@20"])

    logged = json.loads((RECSYS / "logs" / model._ckpt_name.replace(".pt", "") /
                         "result.json").read_text())["test_result"]

    rows, tops = {"baseline": {k: float(v) for k, v in res.items()}}, {}
    for name, (u, im) in arms.items():
        if name == "baseline":
            continue
        m, tk, _ = evaluate_item_matrix(u, im, loader, device)
        rows[name] = {k: float(m[k]) - float(res[k]) for k in m}
        tops[name] = tk
        del u, im
    if "image_knockout" in tops and "image_inputmean" in tops:
        rows["_screen_vs_exact_image"] = ranking_change(tops["image_knockout"],
                                                        tops["image_inputmean"])
    if "text_knockout" in tops and "text_inputmean" in tops:
        rows["_screen_vs_exact_text"] = ranking_change(tops["text_knockout"],
                                                       tops["text_inputmean"])

    out = {"model": "vbpr", "dataset": dataset_name, "class": CLASS, "exactness": EXACTNESS,
           "ckpt": model._ckpt_name, "ckpt_pinned": bool(pin),
           "recon_error": rec, "baseline_Recall@20": base_r20,
           "logged_test_Recall@20": float(logged["Recall@20"]),
           "baseline_minus_logged": base_r20 - float(logged["Recall@20"]),
           "attribution": attr, "arms": rows}
    d = Path(out_dir or (ROOT / "results" / "_scratch" / "exact_ko"))
    d.mkdir(parents=True, exist_ok=True)
    (d / f"vbpr_{dataset_name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="baby")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    _selfcheck(a.dataset, a.gpu, a.out)
