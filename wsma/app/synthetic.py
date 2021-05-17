import numpy as np

rng = np.random.default_rng()


def sinusoid(
		start: float = 0,
		stop: float = 10,
		n: int = 1_000,
		freq: float = None,
		offset: float = None,
		noise: float = None):
	freq = rng.uniform(5, 10) if freq is None else freq
	offset = rng.uniform(0, 5) if offset is None else offset
	noise = rng.uniform(0, 1) if noise is None else noise
	x = np.linspace(start, stop, n)
	noise *= rng.normal(size=(n,))
	return x, np.sin(freq * (x + offset)) + noise
