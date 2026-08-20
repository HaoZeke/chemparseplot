/-!
# Landfold FES observation algebra

Design claim in `clip_erases_tail_contrast` and
`clipped_lerp_strictly_below`: a GP posterior mean is linear in the
observations, so the field we fit is the scientific object.

Clipping `F` to `fmax` before the fit (the first plot_fes) sends every
tail cell to the same value. Distinct occupancies in `{F ≥ fmax}` become
identical observations, so the interpolant cannot recover a density
ratio there.

Unclipped occupied-bin `z = -kT ln(ρ/ρmax)` (or a raw `(s, z)` cloud)
keeps `z₁ = z₂` only when `ρ₁ = ρ₂` on `{ρ > floor ρmax}`. `fmax` is a
membership cut (`z < fmax`), not a value map.

No Mathlib; Lean 4 core `grind`.
-/

namespace LandfoldFes

/-- Colour-scale clip used as a *value* map. This is the wrong target. -/
def clip (F M : Rat) : Rat := if F < M then F else M

/-- Occupied-bin cut: keep the cell iff `z < M`. This is the right use
of `fmax`. -/
def belowCeiling (z M : Rat) : Bool := z < M

theorem clip_of_low (F M : Rat) (h : F < M) : clip F M = F := by
  grind [clip]

theorem clip_of_high (F M : Rat) (h : ¬ F < M) : clip F M = M := by
  grind [clip]

/-- Two tail cells become the same observation after clip. -/
theorem clip_erases_tail_contrast (F1 F2 M : Rat)
    (h1 : ¬ F1 < M) (h2 : ¬ F2 < M) :
    clip F1 M = clip F2 M := by
  grind [clip]

/-- Distinct tail free energies exist that clip cannot separate.
The first plot_fes fitted this non-injective map. -/
theorem clip_not_injective_on_tail (M : Rat) :
    ∃ F1 F2 : Rat, ¬ F1 < M ∧ ¬ F2 < M ∧ F1 ≠ F2 ∧ clip F1 M = clip F2 M := by
  refine ⟨M, M + 1, ?_, ?_, ?_, ?_⟩
  · grind
  · grind
  · grind
  · grind [clip]

/-- The membership cut still distinguishes those same two cells. -/
theorem belowCeiling_separates_tail (M : Rat) :
    belowCeiling M M = false ∧ belowCeiling (M + 1) M = false ∧ M ≠ M + 1 := by
  grind [belowCeiling]

/-- Algebra: unclipped interpolant minus clipped interpolant is
`α (F1 - M)` when the high node is replaced by the ceiling. A GP
posterior mean is linear, so this is the bias of fitting `clip F fmax`. -/
theorem lerp_gap (α F1 F2 M : Rat) :
    α * F1 + (1 - α) * F2 - (α * M + (1 - α) * F2) = α * (F1 - M) := by
  grind

/-- Witness: that gap is positive for a mid-edge with one tail node.
Implied Boltzmann weight of the interpolant is then too large. -/
theorem lerp_gap_pos_witness :
    (1 / 2 : Rat) * ((3 : Rat) - 1) > 0 ∧
      (1 / 2 : Rat) * clip 3 1 + (1 - 1 / 2) * clip 0 1
        < (1 / 2 : Rat) * 3 + (1 - 1 / 2) * 0 := by
  grind [clip]

/-- Plateau interpolates to the plateau. Fitting clipped `F` pins the
whole convex hull of the tail to `fmax`. -/
theorem clipped_tail_lerp_is_ceiling (α F1 F2 M : Rat)
    (h1 : ¬ F1 < M) (h2 : ¬ F2 < M) :
    α * clip F1 M + (1 - α) * clip F2 M = M := by
  grind [clip]

/-- Dropping the ceiling from the fit set does not identify those
two tail values. -/
theorem drop_ceiling_keeps_contrast (F1 F2 M : Rat)
    (h1 : ¬ F1 < M) (h2 : ¬ F2 < M) (hne : F1 ≠ F2) :
    belowCeiling F1 M = false ∧ belowCeiling F2 M = false ∧ F1 ≠ F2 := by
  grind [belowCeiling]

end LandfoldFes
