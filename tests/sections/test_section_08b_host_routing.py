"""Section 8b of grb_main.ipynb: host assignment that follows the delay time.

The static ``HOST_MODELS`` mixture gives every binary the same 75/25 chance of
a star-forming host regardless of when it merges.  Routing by delay time
instead moves weight between templates of different size, and because the
elliptical is the largest host in the mixture the direction of that move is
set by how prompt a population is.  These pin the sign of the effect, which is
what the section's conclusion rests on.
"""

from __future__ import annotations

import numpy as np

from grb_offsets import (
    HOST_MODELS,
    SF_HOST_TRANSITION_T_MYR,
    compute_offsets_mixed_hosts,
    weighted_offset_cdf,
)


def _median(res):
    x, cdf = weighted_offset_cdf(res["mixed_offsets"], res["mixed_weights"])
    return float(np.interp(0.5, cdf, x))


def _offsets(t_delay, t_host=None, seed=5):
    n = t_delay.size
    return compute_offsets_mixed_hosts(
        np.full(n, 120.0),
        t_delay,
        weights=np.ones(n),
        max_systems=n,
        rng=np.random.default_rng(seed),
        t_host_myr=t_host,
    )


def test_prompt_population_moves_inward_under_routing():
    """Stripping a short-delay population of its elliptical share compacts it.

    The elliptical template has R_e = 8 kpc against 3.6 kpc for the SF disk, so
    the 25 percent the static mixture puts there is carrying offsets outward
    that a prompt population has no business inheriting.
    """
    t = np.full(400, 200.0)
    assert _median(_offsets(t, t_host=t)) < _median(_offsets(t))


def test_slow_population_moves_outward_under_routing():
    """The same argument with the sign flipped, for a gigayear population."""
    t = np.full(400, 8000.0)
    assert _median(_offsets(t, t_host=t)) > _median(_offsets(t))


def test_routing_predicts_an_elliptical_fraction_instead_of_assuming_one():
    """The static mixture hands every class 25 percent; routing does not.

    Section 8b reads the routed fraction as a prediction to be checked against
    the observed early-type host fraction, so it has to be able to differ from
    the assumed one.
    """
    t = np.concatenate([np.full(320, 200.0), np.full(80, 8000.0)])
    res = _offsets(t, t_host=t)
    n_host = len(res["mixed_weights"]) // len(HOST_MODELS)
    ell = list(HOST_MODELS).index("Elliptical")
    w_ell = res["mixed_weights"][ell * n_host : (ell + 1) * n_host]
    frac = w_ell.sum() / res["mixed_weights"].sum()
    assert np.isclose(frac, 80 / 400)
    assert not np.isclose(frac, HOST_MODELS["Elliptical"]["weight"])


def test_routing_boundary_is_the_documented_constant():
    """A system either side of ``SF_HOST_TRANSITION_T_MYR`` routes differently."""
    t = np.array([SF_HOST_TRANSITION_T_MYR * 0.99, SF_HOST_TRANSITION_T_MYR * 1.01])
    res = _offsets(np.full(2, 500.0), t_host=t)
    n_host = len(res["mixed_weights"]) // len(HOST_MODELS)
    ell = list(HOST_MODELS).index("Elliptical")
    w_ell = res["mixed_weights"][ell * n_host : (ell + 1) * n_host]
    np.testing.assert_allclose(w_ell, [0.0, 1.0])
