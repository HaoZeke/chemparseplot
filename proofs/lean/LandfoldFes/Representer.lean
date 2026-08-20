/-
Finite GP representer (Eqs. representer, interp, linear, unique).

The noise-free posterior mean on n sites is the kernel expansion
μ_i = Σ_j K_{ij} α_j with K α = y. That is the representation:
the field at the observation sites is a section of the Gram matrix.
A GP on landfold log-density is valid exactly when this interpolant
exists (and is unique if K is injective).
-/

import LandfoldFes.Kernel

namespace LandfoldFes

/-- Weighted sum of a Fin-indexed family. -/
def sumFin (n : Nat) (f : Fin n → Int) : Int :=
  (List.finRange n).foldl (fun acc j => acc + f j) 0

/-- Eq. (representer): μ_i = Σ_j K_{ij} α_j. -/
def representer {n : Nat} (K : Kernel n) (α : Fin n → Int) (i : Fin n) : Int :=
  sumFin n (fun j => K.k i j * α j)

/-- The expansion at the training sites is the Gram action Kα. -/
def gramAction {n : Nat} (K : Kernel n) (α : Fin n → Int) (i : Fin n) : Int :=
  representer K α i

/-- Eq. (interp): if Kα = y then the representer reproduces the table. -/
theorem representer_interpolates {n : Nat} (K : Kernel n)
    (α y : Fin n → Int) (h : ∀ i, representer K α i = y i) (i : Fin n) :
    representer K α i = y i :=
  h i

private theorem foldl_add_gen
    {n : Nat} (l : List (Fin n)) (f g : Fin n → Int) (zf zg : Int) :
    l.foldl (fun acc j => acc + (f j + g j)) (zf + zg) =
      l.foldl (fun acc j => acc + f j) zf +
        l.foldl (fun acc j => acc + g j) zg := by
  induction l generalizing zf zg with
  | nil => simp
  | cons j js ih =>
    simp [List.foldl]
    have hacc : zf + zg + (f j + g j) = zf + f j + (zg + g j) := by
      omega
    rw [hacc, ih]

/-- Eq. (linear): the representer is linear in the coefficients. -/
theorem representer_linear {n : Nat} (K : Kernel n)
    (α β : Fin n → Int) (i : Fin n) :
    representer K (fun j => α j + β j) i =
      representer K α i + representer K β i := by
  unfold representer sumFin
  have hmul : ∀ j, K.k i j * (α j + β j) = K.k i j * α j + K.k i j * β j :=
    fun j => Int.mul_add (K.k i j) (α j) (β j)
  simp [hmul]
  simpa using
    foldl_add_gen (List.finRange n)
      (fun j => K.k i j * α j) (fun j => K.k i j * β j) 0 0

private theorem foldl_smul_gen
    {n : Nat} (l : List (Fin n)) (c : Int) (f : Fin n → Int) (z : Int) :
    l.foldl (fun acc j => acc + c * f j) (c * z) =
      c * l.foldl (fun acc j => acc + f j) z := by
  induction l generalizing z with
  | nil => simp
  | cons j js ih =>
    simp [List.foldl]
    have hacc : c * z + c * f j = c * (z + f j) := (Int.mul_add c z (f j)).symm
    rw [hacc, ih]

/-- Eq. (smul): the representer scales with the coefficients. -/
theorem representer_smul {n : Nat} (K : Kernel n)
    (c : Int) (α : Fin n → Int) (i : Fin n) :
    representer K (fun j => c * α j) i = c * representer K α i := by
  unfold representer sumFin
  have hmul : ∀ j, K.k i j * (c * α j) = c * (K.k i j * α j) := by
    intro j
    rw [← Int.mul_assoc, Int.mul_comm (K.k i j) c, Int.mul_assoc]
  simp [hmul]
  simpa using foldl_smul_gen (List.finRange n) c (fun j => K.k i j * α j) 0

/-- One-site interpolant: μ = k α, and k α = y is reproduction. -/
theorem representer_interpolates_one (k y α : Int) (hk : 0 ≤ k)
    (h : k * α = y) :
    representer (Kernel1 k hk) (fun _ => α) ⟨0, Nat.zero_lt_one⟩ = y := by
  unfold representer sumFin Kernel1
  simp [List.finRange]
  exact h

/-- Eq. (unique-one): a nonzero one-site kernel has at most one coefficient. -/
theorem representer_unique_one (k : Int) (hk : 0 ≤ k) (hk0 : k ≠ 0)
    (α β : Int)
    (h : representer (Kernel1 k hk) (fun _ => α) ⟨0, Nat.zero_lt_one⟩ =
         representer (Kernel1 k hk) (fun _ => β) ⟨0, Nat.zero_lt_one⟩) :
    α = β := by
  have ha : representer (Kernel1 k hk) (fun _ => α) ⟨0, Nat.zero_lt_one⟩ = k * α := by
    unfold representer sumFin Kernel1
    simp [List.finRange]
  have hb : representer (Kernel1 k hk) (fun _ => β) ⟨0, Nat.zero_lt_one⟩ = k * β := by
    unfold representer sumFin Kernel1
    simp [List.finRange]
  have hmul : k * α = k * β := by
    rw [← ha, h, hb]
  have hsub : k * (α - β) = 0 := by
    rw [Int.mul_sub, hmul, Int.sub_self]
  have hzero : k = 0 ∨ α - β = 0 := Int.mul_eq_zero.mp hsub
  cases hzero with
  | inl hk' => exact absurd hk' hk0
  | inr hz => omega

/-- Two-site Gram action. -/
def applyK2 (a b c α1 α2 : Int) : Int × Int :=
  (a * α1 + b * α2, b * α1 + c * α2)

/-- Cramer adjugate action: adj(K) y. -/
def adjK2 (a b c y1 y2 : Int) : Int × Int :=
  (c * y1 - b * y2, -b * y1 + a * y2)

def detK2 (a b c : Int) : Int :=
  a * c - b * b

/-- Eq. (gram-adj): K adj(K) y = (det K) y. This is the 2-site GP
interpolant without division: the representer with Cramer coefficients
reproduces the observation table up to det K. -/
theorem gram_adjugate (a b c y1 y2 : Int) :
    applyK2 a b c (adjK2 a b c y1 y2).1 (adjK2 a b c y1 y2).2 =
      (detK2 a b c * y1, detK2 a b c * y2) := by
  unfold applyK2 adjK2 detK2
  grind

end LandfoldFes
