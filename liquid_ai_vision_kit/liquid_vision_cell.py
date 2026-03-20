"""
LiquidVisionCell — ODE-based liquid neural network cell for visual inputs.

Uses Euler integration:
    dh/dt = (-h + tanh(W_h @ h + W_in @ x + b)) / tau
"""

import numpy as np


class LiquidVisionCell:
    """
    A liquid neural network cell driven by visual feature input.

    The hidden state evolves via:
        dh/dt = (-h + tanh(W_h @ h + W_in @ x + b)) / tau

    Euler integration is used with a fixed time step dt.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tau: float = 1.0,
        dt: float = 0.1,
        n_steps: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            input_dim:  Dimensionality of the input feature vector.
            hidden_dim: Dimensionality of the hidden (state) vector.
            tau:        Time constant controlling ODE dynamics.
            dt:         Euler integration step size.
            n_steps:    Number of Euler steps per forward call.
            seed:       Random seed for weight initialisation.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.dt = dt
        self.n_steps = n_steps

        rng = np.random.default_rng(seed)
        scale = 0.1
        self.W_h = rng.normal(0, scale, (hidden_dim, hidden_dim))
        self.W_in = rng.normal(0, scale, (hidden_dim, input_dim))
        self.b = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray, h: np.ndarray = None):
        """
        Run one forward pass.

        Args:
            x: Input vector of shape (input_dim,).
            h: Previous hidden state of shape (hidden_dim,). If None, zeros.

        Returns:
            (output, hidden_state) — both np.ndarray of shape (hidden_dim,).
        """
        if h is None:
            h = np.zeros(self.hidden_dim)

        x = np.asarray(x, dtype=float)
        h = np.asarray(h, dtype=float).copy()

        for _ in range(self.n_steps):
            dh = (-h + np.tanh(self.W_h @ h + self.W_in @ x + self.b)) / self.tau
            h = h + self.dt * dh

        output = h  # output IS the hidden state (single-layer cell)
        return output, h
