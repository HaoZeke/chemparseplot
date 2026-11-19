import numpy as np
import pytest

from chemparseplot.parse.spectral_instanton import read_oracle_batch

SOURCE = """spectral-instanton test
generated coordinates
12 12 12
90 90 90


1
2
1.0
H
Coordinates of Component 1
1 2 3 0 9
4 5 6 1 10
"""


def test_reads_sidecars_with_numeric_indices_and_fixed_mask(tmp_path):
    for i in range(11):
        (tmp_path / f"geom-{i}.con").write_text(SOURCE)
        (tmp_path / f"result-{i}.txt").write_text(f"{i}\n1 2 3\n4 5 6\n")
    batch = read_oracle_batch(tmp_path)
    np.testing.assert_array_equal(batch.energy.magnitude, np.arange(11))
    np.testing.assert_array_equal(batch.mobile, [True, False])
    np.testing.assert_array_equal(batch.positions.shape, (11, 2, 3))
    np.testing.assert_array_equal(batch.forces.magnitude[0, 1], [4, 5, 6])
    assert batch.atom_ids == (9, 10)
    (tmp_path / "result-2.txt").unlink()
    with pytest.raises(ValueError, match="2"):
        read_oracle_batch(tmp_path)
