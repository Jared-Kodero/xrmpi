"""Single-process tests: no MPI launcher needed."""

from __future__ import annotations

import io
import itertools
import math

import numpy as np
import pytest
from xrmpi.mpi.mpi_utils import SerialProgressBar
from xrmpi.mpp import mpp_data
from xrmpi.mpp.ext_collectives import materialize
from xrmpi.mpp.ext_domains import _define_layout_nd
from xrmpi.mpp.ext_efp import prod_decompose, prod_recombine
from xrmpi.mpp.mpp import extreme_identity, mpp_chksum
from xrmpi.mpp.mpp_domains_define import mpp_compute_extent, mpp_define_layout
from xrmpi.mpp.mpp_efp import _EFP_BLOCK, _NUMBIT, _from_digits, _to_digits


# --- domain decomposition --------------------------------------------------
@pytest.mark.parametrize("length", [0, 1, 7, 100, 1001])
@pytest.mark.parametrize("size", [1, 2, 3, 8, 13])
def test_compute_extent_tiles_the_axis(length: int, size: int) -> None:
    bounds = [mpp_compute_extent(length, rank, size) for rank in range(size)]
    assert bounds[0][0] == 0 and bounds[-1][1] == length
    assert all(stop == nxt for (_, stop), (nxt, _) in itertools.pairwise(bounds))
    widths = [stop - start for start, stop in bounds]
    assert max(widths) - min(widths) <= 1


def test_compute_extent_min_chunk() -> None:
    bounds = [mpp_compute_extent(10, rank, 8, 4) for rank in range(8)]
    widths = [stop - start for start, stop in bounds]
    assert widths == [5, 5, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("ndivs", [1, 2, 4, 6, 12, 16])
def test_define_layout_uses_every_rank(ndivs: int) -> None:
    rows, cols = mpp_define_layout(64, 64, ndivs)
    assert rows * cols == ndivs


def test_define_layout_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        mpp_define_layout(4, 4, 0)


def test_layout_nd_factorises() -> None:
    shape = _define_layout_nd([20, 30, 40], 12)
    assert math.prod(shape) == 12 and len(shape) == 3


# --- extended fixed point --------------------------------------------------
def _reference_digits(array: np.ndarray, axis: int) -> np.ndarray:
    """The original block-wise implementation, kept as an executable spec."""
    from xrmpi.mpp.mpp_efp import _NUMINT, _SCALES, _carry_overflow

    values = np.moveaxis(np.asarray(array, dtype=np.float64), axis, 0)
    digits = np.zeros((_NUMINT, *values.shape[1:]), dtype=np.int64)
    for start in range(0, values.shape[0], _EFP_BLOCK):
        block = values[start : start + _EFP_BLOCK]
        sign = np.where(block < 0.0, -1.0, 1.0)
        residual = np.abs(block)
        for n, scale in enumerate(_SCALES):
            digit = np.floor(residual / scale)
            digits[n] += (sign * digit).astype(np.int64).sum(axis=0)
            residual -= digit * scale
        _carry_overflow(digits)
    return digits


@pytest.mark.parametrize(
    "make",
    [
        lambda r: r.normal(size=(300, 7)),
        lambda r: r.normal(size=(500, 5)) * 10.0 ** r.integers(-30, 30, size=(500, 5)),
        lambda r: r.normal(size=(200, 4)) * 1e-30,
        lambda r: r.normal(size=(200, 4)) * 1e25,
        lambda r: r.integers(-1000, 1000, size=(100, 6)).astype(float),
        lambda r: np.zeros((50, 3)),
        lambda r: np.full((10, 2), -0.0),
        lambda r: np.log2(r.uniform(0.5, 1.0, size=(300, 9))),
        lambda r: r.normal(size=(70000, 3)),
    ],
    ids=[
        "normal",
        "wide",
        "tiny",
        "huge",
        "ints",
        "zeros",
        "negzero",
        "logmant",
        "long",
    ],
)
def test_to_digits_matches_reference(make) -> None:
    values = make(np.random.default_rng(0))
    assert np.array_equal(_to_digits(values, 0), _reference_digits(values, 0))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_to_digits_any_axis(axis: int) -> None:
    values = np.random.default_rng(1).normal(size=(5, 60, 11))
    assert np.array_equal(_to_digits(values, axis), _reference_digits(values, axis))


def test_digits_round_trip_is_exact_for_one_value() -> None:
    values = np.array([[1.0 + 2.0**-52], [-0.1]])
    total = _from_digits(_to_digits(values, 0))
    assert total[0] == pytest.approx(math.fsum(values.ravel()), rel=0, abs=2.0**-52)


def test_digits_sum_is_order_independent() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(512, 4)) * 10.0 ** rng.integers(-6, 6, size=(512, 4))
    shuffled = values[rng.permutation(512)]
    assert np.array_equal(
        _from_digits(_to_digits(values, 0)), _from_digits(_to_digits(shuffled, 0))
    )
    for column in range(4):
        assert _from_digits(_to_digits(values, 0))[column] == pytest.approx(
            math.fsum(values[:, column]), rel=1e-13
        )


def test_numbit_constant() -> None:
    assert _NUMBIT == 46


# --- reproducible product --------------------------------------------------
def test_prod_round_trip() -> None:
    values = np.random.default_rng(3).uniform(0.5, 1.5, size=(40, 3))
    got = prod_recombine(prod_decompose(values, (0,)).copy())
    assert got == pytest.approx(values.prod(axis=0), rel=1e-12)


def test_prod_special_values() -> None:
    values = np.array([[2.0, 0.0, np.inf, np.nan, -3.0], [3.0, 5.0, 2.0, 1.0, 2.0]])
    got = prod_recombine(prod_decompose(values, (0,)))
    assert (
        got[0] == pytest.approx(6.0, rel=1e-14) and got[1] == 0.0 and got[2] == np.inf
    )
    assert np.isnan(got[3]) and got[4] == pytest.approx(-6.0, rel=1e-14)


def test_integer_prod_rounds_to_nearest() -> None:
    rng = np.random.default_rng(7)
    values = rng.integers(1, 9, size=(12, 400)).astype(np.int64)
    exact = values.prod(axis=0)
    got = prod_recombine(prod_decompose(values, (0,)), np.dtype("int64"))
    assert got.dtype == np.int64 and np.array_equal(got, exact)
    pair = np.array([[2], [3]])
    assert prod_recombine(prod_decompose(pair, (0,)), np.dtype("int64"))[0] == 6


def test_prod_skipna_ignores_nan() -> None:
    values = np.array([[2.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    skipping = prod_recombine(prod_decompose(values, (0,), skipna=True))
    assert skipping[0] == pytest.approx(6.0) and skipping[1] == pytest.approx(1.0)
    assert np.isnan(prod_recombine(prod_decompose(values, (0,)))[0])


# --- collectives helpers ---------------------------------------------------
def test_extreme_identity() -> None:
    assert extreme_identity(np.dtype("f8"), minimum=True) == np.inf
    assert extreme_identity(np.dtype("f8"), minimum=False) == -np.inf
    assert extreme_identity(np.dtype("i4"), minimum=True) == np.iinfo("i4").max
    assert extreme_identity(np.dtype("?"), minimum=True) is True
    with pytest.raises(TypeError):
        extreme_identity(np.dtype("c16"), minimum=True)


def test_chksum_is_order_independent() -> None:
    values = np.random.default_rng(4).normal(size=100)
    assert mpp_chksum(values) == mpp_chksum(values[::-1].copy())
    assert mpp_chksum(values, mask_val=values[3]) != mpp_chksum(values)


def test_materialize_loads_lazy_xarray() -> None:
    dask = pytest.importorskip("dask.array")
    import xarray as xr

    lazy = xr.DataArray(dask.ones((4, 3), chunks=2), dims=("a", "b"))
    assert materialize(lazy).chunks is None
    nested = materialize((lazy, [lazy], {"k": lazy}, 5))
    assert nested[0].chunks is None and nested[1][0].chunks is None
    assert nested[2]["k"].chunks is None and nested[3] == 5
    plain = np.arange(3)
    assert materialize(plain) is plain


# --- buffer pool -----------------------------------------------------------
def test_stack_reuses_and_frees() -> None:
    mpp_data.free_stack()
    first = mpp_data.get_stack(100, np.dtype("f8"))
    mpp_data.put_stack(first)
    again = mpp_data.get_stack(60, np.dtype("f8"))
    assert again.size == 60
    mpp_data.put_stack(again)
    assert mpp_data.stack_size()["float64"] >= 100
    mpp_data.free_stack()


def test_stack_limit_bypasses_pool() -> None:
    mpp_data.free_stack()
    old = mpp_data.MPP_STACK_LIMIT
    try:
        mpp_data.mpp_domains_set_stack_size(10)
        big = mpp_data.get_stack(50, np.dtype("f8"))
        mpp_data.put_stack(big)
        assert not mpp_data._stack.get(np.dtype("f8"))
    finally:
        mpp_data.mpp_domains_set_stack_size(old)


# --- progress bar ----------------------------------------------------------
def test_progress_bar_yields_everything(tmp_path) -> None:
    sink = io.StringIO()
    bar = SerialProgressBar(
        range(5),
        description="Writing",
        file=sink,
        lockfile=tmp_path / ".lock",
        min_interval=0.0,
    )
    assert len(bar) == 5 and list(bar) == [0, 1, 2, 3, 4]
    assert "Writing: 5/5" in sink.getvalue() and sink.getvalue().endswith("\n")


def test_progress_bar_without_length() -> None:
    sink = io.StringIO()
    bar = SerialProgressBar((i for i in range(3)), file=sink)
    assert list(bar) == [0, 1, 2]
    with pytest.raises(TypeError):
        len(bar)
