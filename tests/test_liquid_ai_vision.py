"""
tests/test_liquid_ai_vision.py — Comprehensive tests for liquid_ai_vision_kit.

Covers:
  - LiquidVisionCell output shape
  - LiquidVisionCell hidden state persistence
  - PatchEmbedder patch count
  - PatchEmbedder output shape
  - LiquidVisionClassifier forward pass shape
  - LiquidVisionClassifier with different image sizes
  - Batch processing (sequential)
  - ODE stability (no NaN / infinity)
  - Demo runs without error
  - Latency benchmark returns positive time
  - Edge cases: single patch, large hidden dim, tau sensitivity
"""

import sys
import time
import importlib
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo root is on the path so we can import liquid_ai_vision_kit
sys.path.insert(0, str(Path(__file__).parent.parent))

from liquid_ai_vision_kit import LiquidVisionCell, PatchEmbedder, LiquidVisionClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cell_small():
    return LiquidVisionCell(input_dim=8, hidden_dim=16, seed=0)


@pytest.fixture
def embedder():
    return PatchEmbedder(patch_size=4)


@pytest.fixture
def clf():
    return LiquidVisionClassifier(
        image_size=16,
        patch_size=4,
        hidden_dim=32,
        n_classes=10,
        n_cells=1,
        seed=42,
    )


@pytest.fixture
def img16():
    rng = np.random.default_rng(7)
    return rng.random((16, 16))


# ---------------------------------------------------------------------------
# 1. LiquidVisionCell — output shape
# ---------------------------------------------------------------------------

def test_cell_output_shape(cell_small):
    x = np.random.randn(8)
    out, h = cell_small.forward(x)
    assert out.shape == (16,), f"Expected (16,), got {out.shape}"
    assert h.shape == (16,), f"Expected (16,), got {h.shape}"


# ---------------------------------------------------------------------------
# 2. LiquidVisionCell — hidden state persistence across calls
# ---------------------------------------------------------------------------

def test_cell_hidden_state_persistence(cell_small):
    x = np.random.randn(8)
    out1, h1 = cell_small.forward(x)
    out2, h2 = cell_small.forward(x, h=h1)
    # With a non-zero initial hidden state the output should differ
    assert not np.allclose(out1, out2), (
        "Output should change when the hidden state carries over"
    )


# ---------------------------------------------------------------------------
# 3. PatchEmbedder — patch count
# ---------------------------------------------------------------------------

def test_embedder_patch_count(embedder, img16):
    patches = embedder.forward(img16)
    expected_n = (16 // 4) ** 2  # 16
    assert patches.shape[0] == expected_n, (
        f"Expected {expected_n} patches, got {patches.shape[0]}"
    )


# ---------------------------------------------------------------------------
# 4. PatchEmbedder — output shape
# ---------------------------------------------------------------------------

def test_embedder_output_shape(embedder, img16):
    patches = embedder.forward(img16)
    assert patches.ndim == 2
    assert patches.shape[1] == 4 * 4, (
        f"Expected patch_dim=16, got {patches.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 5. LiquidVisionClassifier — forward pass shape
# ---------------------------------------------------------------------------

def test_classifier_forward_shape(clf, img16):
    logits = clf.forward(img16)
    assert logits.shape == (10,), f"Expected (10,), got {logits.shape}"


# ---------------------------------------------------------------------------
# 6. LiquidVisionClassifier — different image sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [8, 16, 32])
def test_classifier_different_image_sizes(size):
    clf = LiquidVisionClassifier(
        image_size=size,
        patch_size=4,
        hidden_dim=16,
        n_classes=4,
        seed=1,
    )
    img = np.random.randn(size, size)
    logits = clf.forward(img)
    assert logits.shape == (4,)


# ---------------------------------------------------------------------------
# 7. Batch processing — sequential inference over multiple images
# ---------------------------------------------------------------------------

def test_batch_sequential_processing(clf):
    rng = np.random.default_rng(99)
    images = [rng.random((16, 16)) for _ in range(8)]
    results = [clf.forward(img) for img in images]
    assert len(results) == 8
    for logits in results:
        assert logits.shape == (10,)
    # Results for different images should not all be identical
    assert not all(np.allclose(results[0], r) for r in results[1:])


# ---------------------------------------------------------------------------
# 8. ODE stability — output is finite, no NaN
# ---------------------------------------------------------------------------

def test_ode_stability_no_nan():
    """Aggressive inputs should not cause NaN or infinity."""
    cell = LiquidVisionCell(input_dim=32, hidden_dim=64, seed=5)
    rng = np.random.default_rng(0)
    h = None
    for _ in range(50):
        x = rng.uniform(-10, 10, 32)
        out, h = cell.forward(x, h)
        assert np.all(np.isfinite(out)), "NaN or Inf detected in cell output"
        assert np.all(np.isfinite(h)), "NaN or Inf detected in hidden state"


def test_classifier_output_is_finite(clf, img16):
    logits = clf.forward(img16)
    assert np.all(np.isfinite(logits)), "Classifier output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 9. Demo runs without error
# ---------------------------------------------------------------------------

def test_demo_runs_without_error():
    import demo  # noqa: F401 — importing also executes module-level code if any
    # Call main() explicitly
    result = demo.main()
    # main() returns elapsed time
    assert result > 0


# ---------------------------------------------------------------------------
# 10. Latency benchmark returns positive time
# ---------------------------------------------------------------------------

def test_latency_benchmark_positive(clf, img16):
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        clf.forward(img16)
    elapsed = time.perf_counter() - t0
    assert elapsed > 0, "Elapsed time must be positive"
    avg_ms = (elapsed / n) * 1000
    assert avg_ms < 5000, f"Average latency {avg_ms:.1f} ms seems too high"


# ---------------------------------------------------------------------------
# 11. Edge case — single patch image
# ---------------------------------------------------------------------------

def test_single_patch_image():
    """Image equal to patch size → exactly 1 patch."""
    clf = LiquidVisionClassifier(
        image_size=4,
        patch_size=4,
        hidden_dim=8,
        n_classes=3,
        seed=2,
    )
    img = np.random.randn(4, 4)
    logits = clf.forward(img)
    assert logits.shape == (3,)


# ---------------------------------------------------------------------------
# 12. Edge case — tau sensitivity (different tau → different output)
# ---------------------------------------------------------------------------

def test_tau_sensitivity():
    """Changing tau should produce different hidden trajectories."""
    x = np.random.randn(16)
    cell_fast = LiquidVisionCell(input_dim=16, hidden_dim=16, tau=0.1, seed=10)
    cell_slow = LiquidVisionCell(input_dim=16, hidden_dim=16, tau=10.0, seed=10)
    out_fast, _ = cell_fast.forward(x)
    out_slow, _ = cell_slow.forward(x)
    assert not np.allclose(out_fast, out_slow), (
        "Different tau values should yield different outputs"
    )


# ---------------------------------------------------------------------------
# 13. Edge case — PatchEmbedder raises on non-divisible image size
# ---------------------------------------------------------------------------

def test_embedder_raises_on_bad_size():
    emb = PatchEmbedder(patch_size=4)
    img = np.zeros((15, 16))  # 15 not divisible by 4
    with pytest.raises(ValueError, match="divisible"):
        emb.forward(img)


# ---------------------------------------------------------------------------
# 14. Edge case — deterministic weights (same seed = same output)
# ---------------------------------------------------------------------------

def test_deterministic_with_same_seed(img16):
    clf_a = LiquidVisionClassifier(image_size=16, patch_size=4, hidden_dim=16, n_classes=5, seed=77)
    clf_b = LiquidVisionClassifier(image_size=16, patch_size=4, hidden_dim=16, n_classes=5, seed=77)
    assert np.allclose(clf_a.forward(img16), clf_b.forward(img16)), (
        "Same seed should give identical outputs"
    )
