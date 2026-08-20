# Blueprint — landfold FES observation identities

Goal: every observation-choice equation in the howto has a Lea-tagged
statement and a Lean 4 declaration that `lake build` accepts.

## Assembly

- [x] `clip_of_low`, `clip_of_high` (Eq. clip)
- [x] `clip_erases_tail_contrast` (Eq. tail)
- [x] `clip_not_injective_on_tail` (Eq. tail-inj)
- [x] `below_ceiling_of_tail` (Eq. cut)
- [x] `lerp_const`
- [x] `clipped_tail_lerp_is_ceiling` (Eq. lerp-ceil)
- [x] `drop_ceiling_keeps_contrast` (Eq. cut-keep)
- [x] `kernel_symmetric`, `kernel_diag_nonneg` (Eq. kernel)
- [x] `representer_interpolates` (Eq. interp)
- [x] `representer_interpolates_one`
- [x] `representer_linear` (Eq. linear)
- [x] `representer_smul`
- [x] `representer_unique_one`
- [x] `gram_adjugate` (Eq. gram-adj)

Org statements live in `../org/supplement.org`. Validate with
`python3 ../scripts/validate_lea.py`.
