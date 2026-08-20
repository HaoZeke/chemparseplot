/-
Finite kernel (Eq. kernel). The GP mean is a section of this Gram
matrix. Symmetry and a nonnegative diagonal are the only axioms the
representation uses.
-/

namespace LandfoldFes

/-- Eq. (kernel): a symmetric kernel on `n` sites with nonnegative diagonal. -/
structure Kernel (n : Nat) where
  k : Fin n → Fin n → Int
  sym : ∀ i j, k i j = k j i
  diag_nonneg : ∀ i, 0 ≤ k i i

/-- Eq. (kernel-sym). -/
theorem kernel_symmetric {n : Nat} (K : Kernel n) (i j : Fin n) :
    K.k i j = K.k j i :=
  K.sym i j

/-- Eq. (kernel-diag). -/
theorem kernel_diag_nonneg {n : Nat} (K : Kernel n) (i : Fin n) :
    0 ≤ K.k i i :=
  K.diag_nonneg i

/-- One-site kernel. The training-point section is the scalar `k`. -/
def Kernel1 (k : Int) (h : 0 ≤ k) : Kernel 1 where
  k := fun _ _ => k
  sym := fun _ _ => rfl
  diag_nonneg := fun _ => h

end LandfoldFes
