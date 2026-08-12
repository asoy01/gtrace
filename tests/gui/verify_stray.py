'''
What a stray order is, and what ``order`` caps.

Two numbers with similar names do different jobs, and confusing them
is what this suite exists to catch.

``beam.stray_order`` counts how many times a beam has done something
an element was not built to do: gone through a mirror, bounced off a
face meant to transmit, rattled around inside a substrate. It rides
with the beam.

``order``, the argument to ``hit`` and ``hitFromHR``, says how many
of those the call may *make*. It is a budget for new ghosts, not a
test the arriving beam has to pass.

So a reflection off a face that is meant to reflect is capped by
neither: it makes no ghost, and an already-stray beam bounces off a
mirror exactly as a fresh one does. A reflection off a face that is
not meant to reflect makes one, and that is what ``order`` limits.

Getting this wrong is not hypothetical, twice over. Between 0.3.1 and
0.4.0 the first external reflection was capped as though it were a
ghost, so a stray beam arriving at a steering mirror produced nothing
at the default order - fatal to code calling hitFromHR directly, and
invisible inside a trace, because of the second one: from 2025 until
2026-08 non_seq_trace zeroed the counter every time a beam left one
element for the next, so nothing arrived anywhere stray. Both are
fixed, and both are checked here.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import numpy as np

import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.layout import q_from_waist
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

def hit_it(m, stray=0, order=0, y=0.0):
    '''
    Send a beam of a given stray order at the HR face and return what
    comes back.
    '''
    b = GaussianBeam(pos=[0.0, y], dirAngle=0.0,
                     q0=q_from_waist(0.2*mm, 0.0, 1064*nm), wl=1064*nm,
                     name='b')
    b.stray_order = stray
    return m.hitFromHR(b, order=order)

def mirror(cls=opt.Mirror, **kw):
    kw.setdefault('name', 'M')
    if cls is opt.CyMirror:
        kw.setdefault('curve_direction', 'h')
    return cls(HRcenter=[0.1, 0.0], normAngleHR=np.pi, diameter=1*inch,
               thickness=10*mm, **kw)


print('--- a face meant to reflect reflects, whatever arrives ---')

for cls in (opt.Mirror, opt.CyMirror):
    label = cls.__name__
    m = mirror(cls)
    check('%s: a fresh beam reflects at the default order' % label,
          'r1' in hit_it(m))
    # The regression. A ghost that reaches a steering mirror is still
    # steered: the mirror is built to reflect, so the reflection is
    # not a new ghost and `order` has nothing to say about it.
    for stray in (1, 2, 7):
        d = hit_it(m, stray=stray)
        check('%s: a beam of stray order %d reflects too, at order 0'
              % (label, stray), 'r1' in d,
              '' if 'r1' in d else '(dropped)')
        if 'r1' in d:
            check('  and comes away at the order it arrived at',
                  d['r1'].stray_order == stray,
                  str(d['r1'].stray_order))
    check('%s: the reflected power is Refl_HR of what came in' % label,
          abs(hit_it(m, stray=3)['r1'].P - m.Refl_HR) < 1e-15)


print('--- a face not meant to reflect makes a ghost, and it is capped ---')

for cls in (opt.Mirror, opt.CyMirror):
    label = cls.__name__
    m = mirror(cls, HRreflective=False, Refl_HR=0.01, Trans_HR=0.99)
    d = hit_it(m, stray=0, order=0)
    check('%s: no budget, no ghost' % label, 'r1' not in d)
    d = hit_it(m, stray=0, order=1)
    check('%s: a budget of one buys it' % label, 'r1' in d)
    if 'r1' in d:
        check('  and it is counted as one', d['r1'].stray_order == 1,
              str(d['r1'].stray_order))
    check('%s: a beam already stray needs the budget for where it lands'
          % label,
          'r1' not in hit_it(m, stray=1, order=1)
          and 'r1' in hit_it(m, stray=1, order=2))

# A lens is the element the flag was added for.
lens = opt.Lens(f=0.3, center=[0.1, 0.0], normAngleHR=np.pi,
                diameter=1*inch, thickness=5*mm, Refl_HR=0.01,
                Trans_HR=0.99, name='L')
check('a lens does not reflect on the house',
      lens.HRreflective is False and 'r1' not in hit_it(lens))
check('  but says so when asked for one ghost',
      'r1' in hit_it(lens, order=1))


print('--- the transmitted side is unchanged by any of this ---')

m = mirror(Trans_HR=0.5, Refl_HR=0.5)
check('a mirror is a face meant to reflect and not to transmit',
      m.HRreflective is True and m.HRtransmissive is False)
check('so going through it costs an order, and is capped',
      's1' not in hit_it(m, order=0) and 's1' in hit_it(m, order=1),
      str(sorted(hit_it(m, order=1))))
bs = mirror(Trans_HR=0.5, Refl_HR=0.5, HRtransmissive=True)
check('a beam splitter transmits at no cost, as it always did',
      's1' in hit_it(bs, order=0))
check('  and reflects at no cost either',
      'r1' in hit_it(bs, stray=2, order=0))


print('--- the order rides with the beam, across elements ---')

# The counter used to be zeroed every time a beam left one element for
# the next, so a ghost arrived at the next mirror as though it were
# the main beam - drawn in the main beam's colour, and given a fresh
# allowance of `order` ghosts of its own. Nothing then bounded the
# recursion but the power threshold.
from gtrace.layout import OpticalLayout, TraceRules

def two_mirrors(order, threshold=1e-8):
    L = OpticalLayout(
        optics=[opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.deg2rad(135),
                           diameter=1*inch, thickness=10*mm, name='M1'),
                opt.Mirror(HRcenter=[0.5, 0.3], normAngleHR=np.deg2rad(-135),
                           diameter=1*inch, thickness=10*mm, name='M2')],
        sources=[GaussianBeam(pos=[0.0, 0.0], dirAngle=0.0,
                              q0=q_from_waist(0.2*mm, 0.0, 1064*nm),
                              wl=1064*nm, name='b0')],
        rules=TraceRules(order=order, power_threshold=threshold),
        name='stray')
    L.trace()
    return L

L = two_mirrors(4)
names = [b.name for b in L.beams]
check('the main path is traced', 'M1:r1' in names and 'M2:r1' in names,
      str(names))
check('the source and its steered beams are not stray',
      all(b.stray_order == 0
          for b in L.beams if b.name in ('b0', 'M1:r1'))
      and L.beams[0].stray_order == 0)
check('a ghost is marked as one',
      all(b.stray_order > 0 for b in L.beams if ':s' in b.name),
      str([(b.name, b.stray_order) for b in L.beams if ':s' in b.name]))
# The point of carrying the counter: a ghost that reaches the next
# mirror and is steered by it is still a ghost.
steered = [b for b in L.beams
           if b.name.endswith(':r1') and b.stray_order > 0]
check('and it is still one after another element steers it',
      bool(steered),
      str([(b.name, b.stray_order) for b in steered]))
check('no beam comes back at an order above the budget',
      all(b.stray_order <= 4 for b in L.beams),
      str(max(b.stray_order for b in L.beams)))

# And the budget is a budget: raising it buys more of the trace, which
# is what it could not do while the counter was being zeroed.
sizes = [len(two_mirrors(k).beams) for k in (0, 1, 2, 4)]
check('the order bounds the trace, and a bigger one buys more of it',
      sizes == sorted(sizes) and sizes[0] < sizes[-1], str(sizes))


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
