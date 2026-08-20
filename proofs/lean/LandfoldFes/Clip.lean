/-
Value-clip versus membership cut (Eqs. clip, cut, tail, lerp-ceil).

A GP posterior mean is linear in the observations. Clipping F to fmax
before the fit (Eq. clip) identifies every tail cell. The membership
cut (Eq. cut) used by plot.landfold keeps those values distinct.
-/

namespace LandfoldFes

/-- Eq. (clip): colour-scale clip as a *value* map. -/
def clipF (F M : Nat) : Nat :=
  min F M

/-- Eq. (cut): keep the cell iff F is strictly below the ceiling. -/
def BelowCeiling (F M : Nat) : Prop :=
  F < M

/-- Eq. (tail): the cell sits in the clipped plateau. -/
def InTail (F M : Nat) : Prop :=
  M ≤ F

/-- Discrete convex combination on Nat, weight n/d. -/
def lerp (n d a b : Nat) : Nat :=
  (n * a + (d - n) * b) / d

/-- Eq. (clip), low branch. -/
theorem clip_of_low (F M : Nat) (h : F < M) : clipF F M = F := by
  unfold clipF
  exact Nat.min_eq_left (Nat.le_of_lt h)

/-- Eq. (clip), high branch. -/
theorem clip_of_high (F M : Nat) (h : InTail F M) : clipF F M = M := by
  unfold clipF InTail at *
  exact Nat.min_eq_right h

/-- Eq. (tail): two tail cells become the same observation after clip. -/
theorem clip_erases_tail_contrast (F1 F2 M : Nat)
    (h1 : InTail F1 M) (h2 : InTail F2 M) :
    clipF F1 M = clipF F2 M := by
  rw [clip_of_high F1 M h1, clip_of_high F2 M h2]

/-- Eq. (tail-inj): clip is not injective on the tail. -/
theorem clip_not_injective_on_tail (M : Nat) :
    ∃ F1 F2 : Nat, InTail F1 M ∧ InTail F2 M ∧ F1 ≠ F2 ∧ clipF F1 M = clipF F2 M := by
  refine ⟨M, M + 1, Nat.le_refl M, Nat.le_succ M, ?_, ?_⟩
  · exact Nat.ne_of_lt (Nat.lt_succ_self M)
  · rw [clip_of_high M M (Nat.le_refl M)]
    rw [clip_of_high (M + 1) M (Nat.le_succ M)]

/-- Eq. (cut): the membership cut still reports both tail cells as dropped. -/
theorem below_ceiling_of_tail (F M : Nat) (h : InTail F M) :
    ¬ BelowCeiling F M :=
  Nat.not_lt.mpr h

/-- A constant field interpolates to itself when the denominator is positive. -/
theorem lerp_const (n d M : Nat) (hd : 0 < d) (hn : n ≤ d) :
    lerp n d M M = M := by
  unfold lerp
  have hsum : n * M + (d - n) * M = d * M := by
    have hsplit : n + (d - n) = d := Nat.add_sub_of_le hn
    calc
      n * M + (d - n) * M = (n + (d - n)) * M := (Nat.add_mul n (d - n) M).symm
      _ = d * M := by rw [hsplit]
  rw [hsum, Nat.mul_comm d M, Nat.mul_div_cancel M hd]

/-- Eq. (lerp-ceil): the convex hull of a clipped tail is the ceiling. -/
theorem clipped_tail_lerp_is_ceiling
    (n d F1 F2 M : Nat) (hd : 0 < d) (hn : n ≤ d)
    (h1 : InTail F1 M) (h2 : InTail F2 M) :
    lerp n d (clipF F1 M) (clipF F2 M) = M := by
  rw [clip_of_high F1 M h1, clip_of_high F2 M h2]
  exact lerp_const n d M hd hn

/-- Eq. (cut-keep): dropping the ceiling does not identify two tail values. -/
theorem drop_ceiling_keeps_contrast (F1 F2 M : Nat)
    (h1 : InTail F1 M) (h2 : InTail F2 M) (hne : F1 ≠ F2) :
    ¬ BelowCeiling F1 M ∧ ¬ BelowCeiling F2 M ∧ F1 ≠ F2 :=
  ⟨below_ceiling_of_tail F1 M h1, below_ceiling_of_tail F2 M h2, hne⟩

end LandfoldFes
