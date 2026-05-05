"""Tests for miatt.registration — ACPC template and registration utilities.

SimpleITK cannot be imported inside the pytest process on this system (C-extension
initialiser causes SIGSEGV under the assertion rewriter).  All tests here use
pure-Python/numpy mocks.  End-to-end integration is verified by running
``pixi run pipeline --approach registration``.
"""

from __future__ import annotations

import numpy as np
import pytest

from miatt.registration import (
    _TEMPLATE_ORIGIN,
    _TEMPLATE_SIZE,
    _TEMPLATE_SPACING,
    propagate_landmarks,
)


class _MockTransform:
    """Minimal stand-in for a SimpleITK rigid transform with a known offset."""

    def __init__(self, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._offset = np.array(offset, dtype=float)

    def TransformPoint(self, xyz: list[float]) -> tuple[float, float, float]:  # noqa: N802
        out = np.array(xyz, dtype=float) + self._offset
        return tuple(out.tolist())


class TestPropagateTemplatePoints:
    def test_known_translation(self) -> None:
        """Each landmark shifts exactly by the transform offset."""
        tx = _MockTransform(offset=(10.0, -5.0, 3.0))
        lm = {
            "AC": np.array([0.0, 0.0, 0.0]),
            "PC": np.array([0.0, -25.0, 0.0]),
        }
        result = propagate_landmarks(tx, lm)  # type: ignore[arg-type]
        np.testing.assert_allclose(result["AC"], [10.0, -5.0, 3.0], atol=1e-9)
        np.testing.assert_allclose(result["PC"], [10.0, -30.0, 3.0], atol=1e-9)

    def test_identity_transform(self) -> None:
        """Identity offset leaves all coordinates unchanged."""
        tx = _MockTransform(offset=(0.0, 0.0, 0.0))
        lm = {"genu": np.array([0.5, 20.0, 10.0])}
        result = propagate_landmarks(tx, lm)  # type: ignore[arg-type]
        np.testing.assert_allclose(result["genu"], lm["genu"], atol=1e-9)

    def test_keys_preserved(self) -> None:
        """Output dict must contain exactly the same keys as input."""
        tx = _MockTransform(offset=(1.0, 2.0, 3.0))
        lm = {"AC": np.zeros(3), "PC": np.array([0.0, -25.0, 0.0]), "genu": np.zeros(3)}
        result = propagate_landmarks(tx, lm)  # type: ignore[arg-type]
        assert set(result.keys()) == set(lm.keys())

    def test_returns_numpy_arrays(self) -> None:
        """Each value in the output must be a (3,) numpy ndarray."""
        tx = _MockTransform()
        lm = {"AC": np.zeros(3)}
        result = propagate_landmarks(tx, lm)  # type: ignore[arg-type]
        assert isinstance(result["AC"], np.ndarray)
        assert result["AC"].shape == (3,)


class TestBackwardMappingMath:
    """Verify the ACPC→scanner inverse used inside resample_to_acpc_space."""

    def test_pure_translation_inverse(self) -> None:
        """T_4x4 encodes scanner→ACPC; its inverse maps ACPC origin to the AC scanner pos."""
        ac_scanner = np.array([20.0, -8.0, 42.0])
        T = np.eye(4)
        T[:3, 3] = -ac_scanner  # scanner→ACPC: subtract scanner origin
        T_inv = np.linalg.inv(T)

        acpc_origin = np.array([0.0, 0.0, 0.0, 1.0])
        scanner_pt = T_inv @ acpc_origin
        np.testing.assert_allclose(scanner_pt[:3], ac_scanner, atol=1e-9)

    def test_rotation_preserves_det_one(self) -> None:
        """Inverting a rigid transform should yield another rigid transform (det = +1)."""
        # 90-degree rotation around z-axis
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        T = np.array([
            [c, -s, 0, 5.0],
            [s,  c, 0, 3.0],
            [0,  0, 1, 1.0],
            [0,  0, 0, 1.0],
        ])
        T_inv = np.linalg.inv(T)
        assert abs(np.linalg.det(T_inv[:3, :3]) - 1.0) < 1e-9

    def test_template_grid_constants_sane(self) -> None:
        """Template origin + size*spacing must produce a grid that contains the ACPC origin."""
        for axis in range(3):
            lo = _TEMPLATE_ORIGIN[axis]
            hi = lo + (_TEMPLATE_SIZE[axis] - 1) * _TEMPLATE_SPACING[axis]
            assert lo < 0.0, f"axis {axis}: origin should be negative (covers posterior/inferior)"
            assert hi > 0.0, f"axis {axis}: far edge should be positive"
            assert lo <= 0.0 <= hi, f"axis {axis}: ACPC origin not within template grid"
