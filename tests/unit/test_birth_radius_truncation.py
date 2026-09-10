"""Confining births to a finite radius must renormalise, not clip.

``r_birth_max_re`` is the Section 8 birth-profile edge: the fiducial scatters
binaries along the full Hernquist stellar light, which reaches hundreds of kpc,
and the alternative confines them to the star-forming body.  The cut has to
reshape the distribution inside the cut rather than pile the rejected draws onto
it, and it has to reach the offsets the integrators return.
"""

from __future__ import annotations

import numpy as np


def test_truncation_keeps_the_analytic_mass_fraction():
    """M(<r)/M = (r/(r+a))^2 puts 81.1 percent of the light inside 5 R_e."""
    from grb_offsets import HERNQUIST_RE_OVER_A, hernquist_birth_radius

    a = 1.0
    r = hernquist_birth_radius(a, rng=np.random.default_rng(0), size=200000)
    r_max = 5.0 * HERNQUIST_RE_OVER_A * a
    kept = (r < r_max).mean()
    assert np.isclose(kept, (r_max / (r_max + a)) ** 2, atol=5e-3)
    assert np.isclose(kept, 0.811, atol=5e-3)


def test_truncated_draws_respect_the_cut_without_piling_up_on_it():
    from grb_offsets import HERNQUIST_RE_OVER_A, hernquist_birth_radius

    a = 1.0
    r_max = 5.0 * HERNQUIST_RE_OVER_A * a
    r = hernquist_birth_radius(a, rng=np.random.default_rng(0), size=200000, r_max_re=5.0)
    assert r.max() <= r_max * (1 + 1e-9)
    # A clip would leave ~19 percent of the sample in the top one percent of
    # radius; a renormalised draw leaves the profile's own share there.
    top = (r > 0.99 * r_max).mean()
    assert top < 0.02, top


def test_truncation_reaches_the_integrated_offsets():
    """The kwarg has to survive the wrapper chain down to the orbit integrator."""
    from grb_offsets import compute_offsets_mixed_hosts

    rng = np.random.default_rng(0)
    n = 800
    v = rng.uniform(20.0, 200.0, n)
    t = 10 ** rng.uniform(1.0, 3.0, n)

    def _run(cut):
        return compute_offsets_mixed_hosts(
            v,
            t,
            weights=np.ones(n),
            max_systems=n,
            rng=np.random.default_rng(1),
            r_birth_max_re=cut,
        )["mixed_offsets"]

    full, cut = _run(None), _run(5.0)
    assert np.median(cut) < np.median(full)
    # The wide tail is where the birth profile does its work, so it moves most.
    assert (cut > 30.0).mean() < 0.5 * (full > 30.0).mean()


def test_default_is_the_untruncated_fiducial():
    """Omitting the kwarg reproduces the full profile bit for bit."""
    from grb_offsets import hernquist_birth_radius

    a = 3.0
    kw = {"rng": np.random.default_rng(7), "size": 5000}
    assert np.array_equal(
        hernquist_birth_radius(a, **kw),
        hernquist_birth_radius(a, rng=np.random.default_rng(7), size=5000, r_max_re=None),
    )
