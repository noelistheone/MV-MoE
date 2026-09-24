# Exact structural knockout modules — contract

One module per architecture: `scripts/exact_ko/<model>.py`. The driver
`scripts/exp_exact_crossarch.py` imports each and treats them uniformly.

## Required module API

```python
CLASS = "C1" | "C2" | "C3" | "C4" | "C5"   # content-interface taxonomy class
EXACTNESS = "EXACT" | "DELETE_PLUS_RENORM" | "BRACKET_ONLY" | "UNDEFINED_ID_ONLY"

def variants(model, dataset, device) -> dict[str, tuple[Tensor, Tensor]]:
    """name -> (user_emb [n_users, d], item_emb [n_items, d]) scored by u @ i.T.
    MUST contain 'baseline'. SHOULD contain 'image_knockout', 'text_knockout',
    'both_knockout'. MAY contain extra arms (e.g. '<x>_knockout_hi' / '_lo' brackets
    for DELETE_PLUS_RENORM and BRACKET_ONLY models)."""

def attribution(model, dataset, device) -> dict:
    """mean per-item norms of each additive stream, plus any fusion coefficients."""

def recon_error(model, dataset, device) -> float:
    """max |baseline_scores - model's own scoring path|. The driver asserts this is
    below TOL and records it. For UNDEFINED_ID_ONLY models return 0.0 and make
    variants() prove content is off-path instead (see below)."""
```

## Non-negotiable rules

1. **Reconstruction assert.** `recon_error` must compare against what the model itself
   computes (`full_sort_predict` / `forward`), not against your own re-derivation. FREEDOM's
   reference is `<= 1e-6`; use `1e-5` as the driver tolerance to allow for larger models.
2. **Delete terms, never corrupt inputs.** Input-mean substitution is the thing being
   replaced. It leaks through constant-direction vectors and it silently produces
   `image_knockout == both_knockout` on DAMRS/baby (verified in the input-mean screen).
3. **Never touch the surviving pathway.** Do not zero shared user vectors, shared biases, or
   the ID/CF term. If a shared normaliser makes deletion non-exact, say so via
   `EXACTNESS = "DELETE_PLUS_RENORM"` and emit BOTH arms:
   `<m>_knockout` (renormalised, the model's own counterfactual) and
   `<m>_knockout_nonorm` (mass actually removed) — the true effect is bracketed by them.
4. **Training-only branches stay untouched.** Only ablate what `full_sort_predict` reads.
   FREEDOM's `image_trs` is training-auxiliary; zeroing it changes R@20 by exactly 0.
5. **Stochastic ops must be paired.** `model.eval()` disables `nn.Dropout` but NOT
   functionals like `F.gumbel_softmax`. Draw once, reuse across all arms.
6. **freeze=False content tables (taxonomy C6).** If the on-path "content" is
   `nn.Embedding.from_pretrained(feat, freeze=False)`, the raw feature is off-path after
   `__init__` and the trained table is what matters. Ablate the TABLE, and report the drift
   cosine `cos(trained_row, raw_row)` in `attribution` so the reader can see whether the
   model is still consuming encoder content at all.
7. **ID-only models.** If content never reaches the scoring function, do not fabricate a
   knockout. Set `EXACTNESS = "UNDEFINED_ID_ONLY"` and have `variants` return only
   `baseline`, plus prove it: `attribution` must include a certificate that a destructive
   random feature swap leaves scores bitwise identical.

## Loading

Use `load_frozen(model, dataset, device, ckpt_path=...)` from `src/models/recsys_bridge.py`
with an explicit pin from `results/phase_micro/ckpt_pins.json` where one exists. It now
asserts no learnable parameter is missing from the checkpoint.
