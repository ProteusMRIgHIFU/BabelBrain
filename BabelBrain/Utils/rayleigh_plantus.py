"""
PlanTUS axial-profile sweep shared by the transducer creator (local GPU) and the
remote job server (standalone 'RayleighPlanTUS' function).

For every requested focal target the sweep computes the per-element steering
phases and then forward-propagates the transducer field along the axial line
`rf`, returning |u2| for each target. All ForwardSimple calls go through the
`forward` callable, so the same code runs locally and on the server, and a
remote run needs only ONE job for the whole sweep.

Pure NumPy (no Qt, no BabelBrain GUI imports) so the server can import it.
"""
import numpy as np

MODE_ANNULAR = 'annular'   # flat/focused annular arrays: per-ring phase via sub-panels
MODE_ARRAY = 'array'       # focused/flat 2D phased arrays: back-propagate from focus
MODE_NONE = 'none'         # single-element (simple focused): no steering


def plantus_axial_profiles(forward, mode, cwvnb_extlay, center, ds, n_subs, amp,
                           targets, rf, steer=None, elemcenter=None, focal_ds=None):
    """Return |u2| along `rf` for each focal target, shape (len(targets), len(rf)).

    forward     -- callable(cwvnb_extlay, center, ds, u0, rf) -> complex field
    mode        -- MODE_ANNULAR / MODE_ARRAY / MODE_NONE
    center, ds  -- all transducer sub-panel centres / areas (n_total x 3, n_total x 1)
    n_subs      -- number of sub-panels per element (sums to n_total)
    amp         -- source amplitude applied to every element
    targets     -- (n_targets x 3) focal points (grid-snapped)
    rf          -- (n_points x 3) axial evaluation points
    steer       -- MODE_ARRAY only: per-target bool, False keeps zero phase
    elemcenter  -- MODE_ARRAY only: element centres the focus back-propagates to
    focal_ds    -- MODE_ARRAY only: area assigned to the focal point source
    """
    mode = str(mode)
    center = np.asarray(center, np.float32)
    ds = np.asarray(ds, np.float32)
    rf = np.asarray(rf, np.float32)
    targets = np.asarray(targets, np.float32).reshape(-1, 3)
    n_subs = np.asarray(n_subs, np.int64).ravel()
    n_elems = len(n_subs)

    profiles = np.zeros((len(targets), rf.shape[0]), np.float32)
    for i, target in enumerate(targets):
        target = target.reshape(1, 3)
        phi = np.zeros(n_elems)

        if mode == MODE_ANNULAR:
            # Forward-propagate each element's sub-panels to the focal point and
            # take the conjugate phase needed to steer to that point.
            nBase = 0
            for n in range(n_elems):
                sl = slice(nBase, nBase + n_subs[n])
                u2back = forward(cwvnb_extlay, center[sl], ds[sl],
                                 np.ones(n_subs[n], np.complex64), target)
                phi[n] = -np.angle(u2back[0])
                nBase += n_subs[n]

        elif mode == MODE_ARRAY:
            if steer is not None and bool(np.asarray(steer).ravel()[i]):
                # Propagate from the focal point to each element centre (inverse
                # direction), then conjugate to obtain the steering phase.
                u2back = forward(cwvnb_extlay, target,
                                 np.ones(1, np.float32) * np.float32(focal_ds),
                                 np.ones(1, np.complex64),
                                 np.asarray(elemcenter, np.float32))
                phi = np.angle(np.conjugate(u2back[:n_elems]))

        elif mode != MODE_NONE:
            raise ValueError("unknown PlanTUS steering mode %r" % mode)

        u0 = np.repeat((amp * np.exp(1j * phi)).astype(np.complex64),
                       n_subs).reshape(-1, 1)
        profiles[i] = np.abs(np.asarray(forward(cwvnb_extlay, center, ds, u0, rf)).ravel())

    return profiles
