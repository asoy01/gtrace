'''
Consistency of the propagation convention of GaussianBeam.

propagate(), width() and R() all take a *geometric* distance and must
therefore advance the q-parameter by that same distance. R() used to
divide by the index of refraction, which made it disagree with the other
two for any beam inside a substrate (n != 1).

Run with:  python test_beam_propagation.py
'''

import numpy as np

import gtrace.beam as beam
import gtrace.optics.gaussian as gauss
from gtrace.unit import *

pi = np.pi

TOL = 1e-12

def rel(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale

def make_beam(n):
    # Not at a waist, so that the ROC is finite.
    return beam.GaussianBeam(q0=gauss.Rw2q(2.0, 1*mm), wl=1064*nm,
                             pos=[0.0, 0.0], dirAngle=0.0, n=n)

for n in [1.0, 1.45, 2.0]:
    for d in [0.0, 1*mm, 5*cm, 1.0, 30.0]:
        b = make_beam(n)
        p = b.copy()
        p.propagate(d)

        # propagate advances q by the geometric distance
        assert rel(complex(p.qx).real, complex(b.qx).real + d) < TOL, \
            'propagate: n=%g d=%g' % (n, d)
        assert rel(complex(p.qx).imag, complex(b.qx).imag) < TOL, \
            'propagate keeps the Rayleigh range: n=%g d=%g' % (n, d)

        # ... and so must width() and R()
        assert rel(b.width(d)[0], p.width(0.0)[0]) < TOL, \
            'width(d) != propagate(d).width(0): n=%g d=%g' % (n, d)
        assert rel(b.R(d)[0], p.R(0.0)[0]) < TOL, \
            'R(d) != propagate(d).R(0): n=%g d=%g' % (n, d)

        # ... which is the same as evaluating them at q + d
        assert rel(b.R(d)[0], gauss.q2R(b.qx + d)) < TOL, \
            'R(d) != q2R(q + d): n=%g d=%g' % (n, d)
        k = 2*pi*b.n/b.wl
        assert rel(b.width(d)[0],
                   np.sqrt(-2.0/(k*np.imag(1.0/(b.qx + d))))) < TOL, \
            'width(d) != w(q + d): n=%g d=%g' % (n, d)

        # optical distance is the only quantity scaled by the index
        assert rel(p.optDist, b.optDist + n*d) < TOL, \
            'optDist: n=%g d=%g' % (n, d)

# The reduced q-parameter carries the index of refraction.
for n in [1.0, 1.45]:
    b = make_beam(n)
    assert rel(complex(b.qrx).real, complex(b.qx).real/n) < TOL, 'qrx = qx/n'

# waist() must be consistent with width() and with Re(q).
for n in [1.0, 1.45]:
    b = make_beam(n)
    w = b.waist()
    assert rel(w['Waist Position'][0], -complex(b.qx).real) < TOL, 'waist position'
    assert rel(w['Waist Size'][0],
               b.width(w['Waist Position'][0])[0]) < TOL, 'waist size'
    # The ROC diverges at the waist, and the beam is widest away from it.
    assert b.width(w['Waist Position'][0])[0] < b.width(0.0)[0], 'waist is a minimum'

# The ROC is infinite at a waist. Returning inf there relies on numpy
# type promotion inside R(), so pin it down.
for n in [1.0, 1.45]:
    bw = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                           pos=[0.0, 0.0], dirAngle=0.0, n=n)
    assert np.isinf(bw.R(0.0)[0]) and np.isinf(bw.R(0.0)[1]), \
        'R at a waist must be inf, not an exception: n=%g' % n
    assert np.isfinite(bw.R(0.5)[0]), 'R away from the waist is finite: n=%g' % n

# A beam in air must be unaffected by all of the above.
b1 = make_beam(1.0)
b145 = make_beam(1.45)
assert rel(b1.R(0.5)[0], gauss.q2R(b1.qx + 0.5)) < TOL
assert rel(b1.R(0.5)[0], b145.R(0.5)[0]) < TOL, \
    'beams with the same q must have the same ROC regardless of n'

print('test_beam_propagation.py: all checks passed')
