# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for shared eOn trajectory parser helpers."""

from pathlib import Path

import polars as pl
import pytest
from ase.build import molecule

from chemparseplot.parse.eon._trajectory_common import (
    frame_rows_to_table,
    load_optional_payload,
    metadata_value,
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


class DummyFrame:
    def __init__(self, *, frame_index=None, energy=None, metadata=None):
        self.frame_index = frame_index
        self.energy = energy
        self.metadata = metadata or {}

    def to_ase(self):
        return molecule("H2O")


class TestMetadataHelpers:
    def test_metadata_value_reads_builtin_fields(self):
        frame = DummyFrame(frame_index=3, energy=-1.25, metadata={"step_size": 0.1})
        assert metadata_value(frame, "frame_index") == 3
        assert metadata_value(frame, "energy") == -1.25
        assert metadata_value(frame, "step_size") == 0.1

    def test_frame_rows_to_table_skips_incomplete_frames(self):
        frames = [
            DummyFrame(frame_index=0, metadata={"step_size": 0.0}),
            DummyFrame(frame_index=1, metadata={"step_size": 0.2}),
        ]
        df = frame_rows_to_table(frames, ("frame_index", "step_size"))
        assert isinstance(df, pl.DataFrame)
        assert df.height == 2

    def test_frame_rows_to_table_coerces_numeric_strings(self):
        frames = [
            DummyFrame(frame_index="0", energy="-1.25", metadata={"step_size": "0.0", "convergence": "0.5"}),
            DummyFrame(frame_index="1", energy="-1.50", metadata={"step_size": "0.2", "convergence": "0.1"}),
        ]
        df = frame_rows_to_table(frames, ("frame_index", "step_size", "convergence", "energy"))
        assert df.dtypes == [pl.Int64, pl.Float64, pl.Float64, pl.Float64]
        assert df["convergence"].to_list() == [0.5, 0.1]

    def test_frame_rows_to_table_allows_incomplete_prefix_when_requested(self):
        frames = [
            DummyFrame(frame_index=0, metadata={}),
            DummyFrame(frame_index=1, metadata={"step_size": 0.2}),
        ]
        df = frame_rows_to_table(
            frames,
            ("frame_index", "step_size"),
            allow_leading_incomplete=True,
        )
        assert df.height == 1
        assert df["frame_index"].to_list() == [1]

    def test_frame_rows_to_table_rejects_partial_metadata_after_start(self):
        frames = [
            DummyFrame(frame_index=0, metadata={"step_size": 0.1}),
            DummyFrame(frame_index=1, metadata={}),
        ]
        df = frame_rows_to_table(
            frames,
            ("frame_index", "step_size"),
            allow_leading_incomplete=True,
        )
        assert df.is_empty()
