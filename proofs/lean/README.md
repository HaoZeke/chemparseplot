# Lean 4 proofs for landfold FES observation choice

Self-contained Lean 4 library. No Mathlib.

Same contract as the d-SEAMS 2.0 supplementary
(`TeX/dseams2_paper/lean`): Lea-tagged statements whose Lean names are
the paper identities, `lake build` sorry-free. The Lea marker contract
is that of https://github.com/VIDA-NYU/Lea.

```
lake build
python3 ../scripts/validate_lea.py
```

Declarations match the Lea-marked theorem blocks in
`../org/supplement.org` (`% lea: formalize label=...`).

| Lean name | Identity |
| --- | --- |
| `clip_of_low`, `clip_of_high` | Eq. (clip) |
| `clip_erases_tail_contrast` | Eq. (tail) |
| `clip_not_injective_on_tail` | Eq. (tail-inj) |
| `below_ceiling_of_tail` | Eq. (cut) |
| `clipped_tail_lerp_is_ceiling` | Eq. (lerp-ceil) |
| `drop_ceiling_keeps_contrast` | Eq. (cut-keep) |
| `kernel_symmetric`, `kernel_diag_nonneg` | Eq. (kernel) |
| `representer_interpolates`, `representer_interpolates_one` | Eq. (interp) |
| `representer_linear`, `representer_smul` | Eq. (linear) |
| `representer_unique_one` | Eq. (unique-one) |
| `gram_adjugate`, `adjugate_gram` | Eq. (gram-adj) |
| `obs_linear` | Eq. (obs-linear) |
| `gram_unique` | Eq. (unique-two) |
| `kernel2_det_pos` | Eq. (spd-two) |
