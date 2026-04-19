# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for shared eOn trajectory parser helpers."""

from pathlib import Path

import pytest

from chemparseplot.parse.eon._trajectory_common import (
    load_optional_payload,
    require_dat_file,
    resolve_movie_file,
)


class TestResolveMovieFile:
    def test_prefers_suffixless_movie(self, tmp_path):
        movie = tmp_path / "climb"
        movie.write_text("dummy")
        (tmp_path / "climb.con").write_text("fallback")
        assert resolve_movie_file(tmp_path, "climb") == movie

    def test_falls_back_to_con_suffix(self, tmp_path):
        movie = tmp_path / "minimization.con"
        movie.write_text("dummy")
        assert resolve_movie_file(tmp_path, "minimization") == movie

    def test_missing_movie_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="movie file"):
            resolve_movie_file(tmp_path, "climb")


class TestRequireDatFile:
    def test_returns_existing_path(self, tmp_path):
        dat = tmp_path / "climb.dat"
        dat.write_text("iteration\tstep_size\n")
        assert require_dat_file(tmp_path, "climb.dat") == dat

    def test_missing_dat_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"climb\.dat"):
            require_dat_file(tmp_path, "climb.dat")


class TestLoadOptionalPayload:
    def test_skips_missing_path(self, tmp_path):
        assert load_optional_payload(tmp_path / "missing", Path.read_text) is None

    def test_loads_existing_path(self, tmp_path):
        payload = tmp_path / "mode.dat"
        payload.write_text("1 0 0\n")
        assert load_optional_payload(payload, Path.read_text) == "1 0 0\n"
