from __future__ import annotations

import kittygrad.core as core


def inv_permutation(permutation: Size) -> Size:
    if not permutation:
        return tuple()

    permutation = core.np.array(permutation)
    inv = core.np.empty_like(permutation)
    inv[permutation] = core.np.arange(len(inv), dtype=inv.dtype)
    return inv.tolist()
