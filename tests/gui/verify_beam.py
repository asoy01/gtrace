'''
A beam's derived quantities agree with the q it was given.

A ``GaussianBeam`` stores its q-parameter twice over. ``qx`` and ``qy``
are the q-parameters themselves; ``qrx`` and ``qry`` are the same
divided by the refractive index, and those are what an ABCD transform
is applied to. The width, and the best matching circular q, are derived
too. All of it is worked out by trait handlers when ``qx`` or ``qy`` is
assigned.

Which is a problem when nothing is assigned. Traits does not notify
when an assignment matches the value already there, and ``qx`` defaults
to ``1j``, so ``GaussianBeam(q0=1j)`` used to leave every derived value
at *its* default: a width of zero, a circular q of zero, and a reduced
q of zero. The first propagation then transformed that zero, so the
beam came out with a real q - infinite Rayleigh range, zero waist - and
the next thing to ask for its width divided by zero. Nothing in the
package went near it, because nothing in the package writes ``1j``,
which is why it survived from before 0.3.0 to 0.5.0.

So this suite asks the invariant directly, over q-parameters chosen to
include the awkward one: whatever a beam was given, and however it was
given, the four derived values agree with it. Then it propagates, since
the invariant surviving construction was never the hard part.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import numpy as np

import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.layout import OpticalLayout, TraceRules, q_from_waist
from gtrace.optics.gaussian import q2w, optimalMatching
from gtrace.unit import *

npass = 0
nfail = 0

def check(name, cond, detail=''):
    global npass, nfail
    if cond:
        npass += 1
        print('  PASS  %s %s' % (name, detail))
    else:
        nfail += 1
        print('  FAIL  %s %s' % (name, detail))

def close(a, b, tol=1e-15):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def derived_agree(label, b):
    '''
    The four values a beam derives from its q, against the same
    functions the model uses.
    '''
    check('%s: reduced qx is qx over n' % label,
          close(b.qrx, b.qx / b.n), '%s vs %s' % (b.qrx, b.qx / b.n))
    check('%s: reduced qy is qy over n' % label,
          close(b.qry, b.qy / b.n), '%s vs %s' % (b.qry, b.qy / b.n))
    check('%s: wx is the width of qx' % label,
          close(b.wx, q2w(b.qx, wl=b.wl / b.n)), str(b.wx))
    check('%s: wy is the width of qy' % label,
          close(b.wy, q2w(b.qy, wl=b.wl / b.n)), str(b.wy))
    check('%s: q is the circular match of the two' % label,
          close(b.q, optimalMatching(b.qx, b.qy)[0]), str(b.q))
    check('%s: the beam has a width at all' % label,
          b.wx > 0.0 and b.wy > 0.0, '%g, %g' % (b.wx, b.wy))
    check('%s: and a Rayleigh range' % label,
          np.imag(b.qx) > 0.0 and np.imag(b.qy) > 0.0, str(b.qx))


print('=== construction: whatever q it was given ===')

# 1j is the default of the trait, which is the case that was broken.
# The others are ordinary: a metre of Rayleigh range, a waist a way
# back, and what q_from_waist returns for a 0.3 mm waist.
for q0 in (1j, 1j * 1.0, 10j, -0.2 + 0.5j, q_from_waist(0.3 * mm, 0.0, 1064 * nm)):
    derived_agree('q0=%s' % q0, GaussianBeam(q0=q0, wl=1064 * nm))

# The same, in glass. The reduced q is where n shows up, so a beam
# inside a substrate is the case that tells the two representations
# apart.
for n in (1.45, 2.0):
    derived_agree('q0=1j n=%s' % n, GaussianBeam(q0=1j, wl=1064 * nm, n=n))

# Given separately, and given for one direction only.
b = GaussianBeam(q0x=1j, q0y=2j, wl=1064 * nm)
derived_agree('q0x/q0y', b)
check('q0x and q0y are kept apart',
      b.qx == 1j and b.qy == 2j, '%s, %s' % (b.qx, b.qy))
b = GaussianBeam(q0=1j, q0y=3j, wl=1064 * nm)
check('q0 fills the direction q0y did not give',
      b.qx == 1j and b.qy == 3j, '%s, %s' % (b.qx, b.qy))

# The default beam, constructed with no q at all.
derived_agree('no q given', GaussianBeam(wl=1064 * nm))


print()
print('=== propagation ===')

# A free-space propagation adds the distance to the reduced q, so the
# imaginary part - the Rayleigh range - is the thing that must survive.
b = GaussianBeam(q0=1j, wl=1064 * nm)
w0 = b.wx
b.propagate(0.15)
check('propagating adds the distance to q',
      close(b.qx, 0.15 + 1j, 1e-12), str(b.qx))
check('and keeps the Rayleigh range',
      close(np.imag(b.qx), 1.0, 1e-12), str(np.imag(b.qx)))
check('so the beam is wider than it was at its waist',
      b.wx > w0, '%g > %g' % (b.wx, w0))
derived_agree('after propagate', b)

# In glass the reduced q advances by d/n, so q advances by d.
b = GaussianBeam(q0=1j, wl=1064 * nm, n=1.45)
b.propagate(0.29)
check('in glass the q still advances by the distance',
      close(b.qx, 0.29 + 1j, 1e-12), str(b.qx))
derived_agree('after propagate in glass', b)

# Propagating in steps and in one go come to the same place.
one = GaussianBeam(q0=1j, wl=1064 * nm)
one.propagate(0.3)
many = GaussianBeam(q0=1j, wl=1064 * nm)
for _ in range(3):
    many.propagate(0.1)
check('three steps land where one step of the sum lands',
      close(one.qx, many.qx, 1e-12), '%s vs %s' % (one.qx, many.qx))

# copy() re-copies the reduced q after the deepcopy, and the copy has
# to be as consistent as the original.
b = GaussianBeam(q0=1j, wl=1064 * nm, n=1.45)
c = b.copy()
derived_agree('a copy', c)
check('a copy carries the same q',
      c.qx == b.qx and c.qrx == b.qrx, '%s, %s' % (c.qx, c.qrx))
c.propagate(0.2)
check('and propagates on its own',
      close(c.qx, 0.2 + 1j, 1e-12) and b.qx == 1j,
      '%s, original %s' % (c.qx, b.qx))

# The waist, read back. A beam at its waist reports the distance to it
# as zero, and one propagated a way past it reports how far back it is.
b = GaussianBeam(q0=1j, wl=1064 * nm)
wst = b.waist()
check('a beam given q=1j is at its waist',
      close(wst['Waist Position'][0], 0.0, 1e-12), str(wst))
b.propagate(0.4)
check('and 0.4 further on, the waist is 0.4 behind',
      close(b.waist()['Waist Position'][0], -0.4, 1e-9),
      str(b.waist()['Waist Position']))
check('the waist size is unchanged by propagating',
      close(b.waist()['Waist Size'][0], wst['Waist Size'][0], 1e-12),
      '%g vs %g' % (b.waist()['Waist Size'][0], wst['Waist Size'][0]))


print()
print('=== the whole trace, from such a beam ===')

# What the bug actually cost: a layout whose source was written this
# way could not be traced at all.
M1 = opt.Mirror(HRcenter=[0.0, 0.0], normAngleHR=deg2rad(135),
                diameter=25.4 * mm, thickness=6 * mm,
                wedgeAngle=deg2rad(0.25), n=1.45, name='M1')
M2 = opt.Mirror(HRcenter=[0.0, 0.2], normAngleHR=deg2rad(-45),
                diameter=25.4 * mm, thickness=6 * mm, n=1.45, name='M2')
src = GaussianBeam(q0=1j, pos=[-0.15, 0.0], dirAngle=0.0,
                   wl=1064 * nm, name='b0')
L = OpticalLayout(optics=[M1, M2], sources=[src],
                  rules=TraceRules(order=3, power_threshold=1e-4),
                  name='beamcheck')
beams = L.trace()
check('a layout whose source was given q=1j traces',
      len(beams) > 2, '%d beams' % len(beams))
check('and every beam that comes back has a width',
      all(b.wx > 0.0 and b.wy > 0.0 for b in beams),
      str([b.name for b in beams if not (b.wx > 0.0)]))
check('and a Rayleigh range',
      all(np.imag(b.qx) > 0.0 for b in beams),
      str([b.name for b in beams if not (np.imag(b.qx) > 0.0)]))
for b in beams[:4]:
    derived_agree('traced %s' % b.name, b)

# The same layout traced from an equivalent q written differently must
# come to the same numbers: the fix is about initialisation, not about
# the physics.
src2 = GaussianBeam(q0=1e-18 + 1j, pos=[-0.15, 0.0], dirAngle=0.0,
                    wl=1064 * nm, name='b0')
L2 = OpticalLayout(optics=[M1, M2], sources=[src2],
                   rules=TraceRules(order=3, power_threshold=1e-4),
                   name='beamcheck2')
beams2 = L2.trace()
check('a q of 1j and a q a hair off it trace the same beams',
      len(beams2) == len(beams), '%d vs %d' % (len(beams2), len(beams)))
same = all(close(a.qx, b.qx, 1e-9) and close(a.wx, b.wx, 1e-9)
           for a, b in zip(beams, beams2))
check('to the same q and the same width',
      same,
      str([(a.name, a.qx, b.qx) for a, b in zip(beams, beams2)
           if not close(a.qx, b.qx, 1e-9)][:2]))


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
