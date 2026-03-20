"""
PatchEmbedder — splits an image into fixed-size patches and flattens each patch.
"""

import numpy as np


class PatchEmbedder:
    """
    Splits a 2-D grayscale image (H, W) into non-overlapping patches
    of shape (patch_size, patch_size) and flattens each to a 1-D vector.

    Output shape: (n_patches, patch_dim)
    where:
        n_patches = (H // patch_size) * (W // patch_size)
        patch_dim = patch_size * patch_size
    """

    def __init__(self, patch_size: int = 4):
        """
        Args:
            patch_size: Height/width of each square patch (pixels).
        """
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Embed an image into a sequence of flattened patches.

        Args:
            img: 2-D array of shape (H, W). H and W must be divisible by patch_size.

        Returns:
            patches: 2-D array of shape (n_patches, patch_dim).

        Raises:
            ValueError: If image dimensions are not divisible by patch_size.
        """
        img = np.asarray(img, dtype=float)
        if img.ndim != 2:
            raise ValueError(f"Expected 2-D image, got shape {img.shape}")

        H, W = img.shape
        ps = self.patch_size

        if H % ps != 0 or W % ps != 0:
            raise ValueError(
                f"Image size ({H}x{W}) must be divisible by patch_size ({ps})"
            )

        n_h = H // ps
        n_w = W // ps

        # Reshape to (n_h, ps, n_w, ps), then transpose and flatten patches
        patches = img.reshape(n_h, ps, n_w, ps)
        patches = patches.transpose(0, 2, 1, 3)          # (n_h, n_w, ps, ps)
        patches = patches.reshape(n_h * n_w, ps * ps)    # (n_patches, patch_dim)

        return patches
