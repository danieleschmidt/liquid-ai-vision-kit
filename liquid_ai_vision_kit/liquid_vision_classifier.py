"""
LiquidVisionClassifier — image classifier built on PatchEmbedder + LiquidVisionCells.
"""

import numpy as np
from .patch_embedder import PatchEmbedder
from .liquid_vision_cell import LiquidVisionCell


class LiquidVisionClassifier:
    """
    End-to-end image classifier using Liquid Neural Network cells.

    Pipeline:
        1. PatchEmbedder  — split image into patches
        2. LiquidVisionCells — process patch sequence recurrently
        3. Linear head    — map final hidden state to class logits

    All operations use NumPy; no deep-learning framework required.
    """

    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        hidden_dim: int = 64,
        n_classes: int = 10,
        n_cells: int = 1,
        tau: float = 1.0,
        dt: float = 0.1,
        n_steps: int = 5,
        seed: int = 0,
    ):
        """
        Args:
            image_size: Assumed square image side length (pixels).
            patch_size: Size of each square patch (pixels).
            hidden_dim: Hidden dimension of each LiquidVisionCell.
            n_classes:  Number of output classes.
            n_cells:    Number of stacked LiquidVisionCells.
            tau:        ODE time constant.
            dt:         Euler step size.
            n_steps:    Euler steps per forward call.
            seed:       Base random seed (incremented per cell).
        """
        self.embedder = PatchEmbedder(patch_size=patch_size)
        patch_dim = patch_size * patch_size

        self.cells = []
        for i in range(n_cells):
            in_dim = patch_dim if i == 0 else hidden_dim
            self.cells.append(
                LiquidVisionCell(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    tau=tau,
                    dt=dt,
                    n_steps=n_steps,
                    seed=seed + i,
                )
            )

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        rng = np.random.default_rng(seed + 999)
        self.W_out = rng.normal(0, 0.1, (n_classes, hidden_dim))
        self.b_out = np.zeros(n_classes)

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Classify an image.

        Args:
            img: 2-D array of shape (H, W).

        Returns:
            logits: 1-D array of shape (n_classes,).
        """
        patches = self.embedder.forward(img)   # (n_patches, patch_dim)

        # Process each patch through the cell stack sequentially
        hidden_states = [None] * len(self.cells)
        x = None

        for patch in patches:
            x = patch
            for j, cell in enumerate(self.cells):
                x, hidden_states[j] = cell.forward(x, hidden_states[j])

        # Final hidden state of the last cell → linear head
        final_h = hidden_states[-1]
        logits = self.W_out @ final_h + self.b_out
        return logits
