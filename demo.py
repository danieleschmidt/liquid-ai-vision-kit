"""
demo.py — Demonstrate liquid_ai_vision_kit on synthetic 16x16 images.

Generates random synthetic images (with random Gaussian blobs), runs
inference through the LiquidVisionClassifier, and benchmarks latency
over 100 forward passes.
"""

import time
import numpy as np
from liquid_ai_vision_kit import LiquidVisionClassifier, PatchEmbedder, LiquidVisionCell


def make_synthetic_image(size: int = 16, n_blobs: int = 3, seed: int = None) -> np.ndarray:
    """Generate a synthetic grayscale image with random Gaussian blobs."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size))
    ys, xs = np.mgrid[0:size, 0:size]
    for _ in range(n_blobs):
        cx, cy = rng.uniform(2, size - 2, 2)
        sigma = rng.uniform(1, 4)
        amp = rng.uniform(0.5, 1.0)
        img += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    img = np.clip(img, 0, 1)
    return img


def main():
    print("=" * 60)
    print("  Liquid AI Vision Kit — Demo")
    print("=" * 60)

    # --- Setup ---
    IMAGE_SIZE = 16
    PATCH_SIZE = 4
    HIDDEN_DIM = 32
    N_CLASSES = 5
    N_IMAGES = 5

    clf = LiquidVisionClassifier(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_classes=N_CLASSES,
        n_cells=2,
        seed=7,
    )

    print(f"\nModel config:")
    print(f"  Image size : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Patch size : {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Patches    : {(IMAGE_SIZE // PATCH_SIZE) ** 2}")
    print(f"  Hidden dim : {HIDDEN_DIM}")
    print(f"  Classes    : {N_CLASSES}")

    # --- Inference on synthetic images ---
    print(f"\nRunning inference on {N_IMAGES} synthetic images…")
    for i in range(N_IMAGES):
        img = make_synthetic_image(IMAGE_SIZE, seed=i)
        logits = clf.forward(img)
        pred = int(np.argmax(logits))
        print(f"  Image {i+1}: logits=[{', '.join(f'{v:.3f}' for v in logits)}]  pred={pred}")

    # --- Latency benchmark ---
    N_BENCH = 100
    img_bench = make_synthetic_image(IMAGE_SIZE, seed=999)

    # Warm-up
    for _ in range(5):
        clf.forward(img_bench)

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        clf.forward(img_bench)
    elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / N_BENCH) * 1000
    print(f"\nLatency benchmark ({N_BENCH} forward passes):")
    print(f"  Total time : {elapsed*1000:.2f} ms")
    print(f"  Avg/pass   : {avg_ms:.4f} ms")
    print(f"  Throughput : {N_BENCH/elapsed:.1f} FPS")

    # --- PatchEmbedder standalone demo ---
    print("\nPatchEmbedder standalone:")
    emb = PatchEmbedder(patch_size=4)
    patches = emb.forward(img_bench)
    print(f"  Input  : {img_bench.shape}")
    print(f"  Output : {patches.shape}  (n_patches={patches.shape[0]}, patch_dim={patches.shape[1]})")

    # --- LiquidVisionCell standalone demo ---
    print("\nLiquidVisionCell standalone:")
    cell = LiquidVisionCell(input_dim=16, hidden_dim=8)
    x = np.random.randn(16)
    out, h = cell.forward(x)
    print(f"  Input shape  : {x.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Hidden shape : {h.shape}")

    print("\nDemo complete. ✓")
    return elapsed  # return for test verification


if __name__ == "__main__":
    main()
