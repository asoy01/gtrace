'''
The CyLens class: a lens ordered by focal length that focuses in one
plane only.

The ordering machinery - the thick lens solve, the shapes, what the
constructor refuses - is Lens's own code and verify_lens.py holds it to
its contract. What is new here is the direction, so that is what this
suite measures: the focal length ordered has to land in the plane
curve_direction names, at the value ordered, and the other plane has to
be a plain window - not a weak lens, a window.

Two instruments, one per plane. In the plane of the drawing a ray is
traced through the lens and the focal length read off the angle it
leaves at, exactly as verify_lens.py does. Out of the plane nothing can
be traced - gtrace draws in two dimensions - but the beam's qy crosses
the lens through the same ray matrices a ray would, so the whole ABCD
matrix is extracted from how two different qy transform, and the focal
length is read off its C element. The extraction is calibrated first:
on a spherical Lens, where both planes are the same, the two
instruments have to agree with each other.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import numpy as np

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
import gtrace.draw as draw
from gtrace.layout import (OpticalLayout, TraceRules, EditError,
                           optic_to_dict, optic_from_dict)
from gtrace.optcomp import CyLens, Lens, LensGeometryError
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

def rel(got, want):
    return np.abs(got - want)/np.abs(want) if want else np.abs(got)

#{{{ The two instruments

#: As in verify_lens.py: small enough for aberration, large enough for
#: the arithmetic.
PROBE_HEIGHT = 10*um

#: How closely a traced focal length has to match the one ordered.
FOCAL_TOL = 1e-6

#: A plane that is supposed to be a window may carry no more power than
#: this, in 1/m. The solver itself only promises the ordered f to 1e-9
#: relative, so demanding less than this of the other plane would be
#: asking the flat plane to be flatter than the curved one is curved.
WINDOW_TOL = 1e-9

def ray_efl(lens, from_front=True, h=PROBE_HEIGHT):
    '''
    The in-plane focal length, measured by tracing: a ray parallel to
    the axis leaves towards the focus, so EFL = -h/slope.

    Returns None if the ray does not come out the far side.
    '''
    axis = lens.normVectHR if not from_front else -lens.normVectHR
    perp = np.array([-axis[1], axis[0]])

    start = lens.center - axis*1.0 + perp*h
    b = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                          pos=start, dirAngle=np.arctan2(axis[1], axis[0]),
                          name='probe')

    isHit, beams, face = lens.hit(b, order=0, threshold=1e-9)
    if not isHit or 't1' not in beams:
        return None
    out = beams['t1']

    h_in = np.dot(np.asarray(b.pos) - lens.center, perp)
    slope = np.dot(out.dirVect, perp)/np.dot(out.dirVect, axis)
    if slope == 0:
        return np.inf
    return -h_in/slope

#: The two probe beams the extraction below feeds through the lens.
#: Any two distinct q would do; a collimated one and a converging one
#: keep the linear system comfortably conditioned.
_PROBE_QS = [gauss.Rw2q(np.inf, 1*mm), gauss.Rw2q(2.0, 0.3*mm)]

def transit_matrix(lens, plane):
    '''
    The ABCD matrix of the whole transit, entry face to exit face, in
    one transverse plane, extracted from how q transforms.

    Two beams with different q are sent through on the axis. Each q is
    read at the entry face (the starting q advanced by the approach,
    which is pure propagation) and at the exit face (where the
    transmitted beam is born), and the bilinear map between them,

        q2 = (A q1 + B) / (C q1 + D),

    is solved for [A, B, C, D] as the null space of the linear system
    each pair contributes two complex rows to. The transit starts and
    ends in air, so det = 1 fixes the scale, and A > 0 the sign.

    Returns (A, B, C, D, residual): the residual is how far the fitted
    map is from reproducing the measured q2, and the caller checks it,
    so a poorly conditioned fit fails loudly rather than quietly.
    '''
    rows = []
    pairs = []
    for q0 in _PROBE_QS:
        axis = -np.asarray(lens.normVectHR)
        b = beam.GaussianBeam(q0=q0, wl=1064*nm,
                              pos=lens.center - axis*0.5,
                              dirAngle=np.arctan2(axis[1], axis[0]),
                              name='probe')
        q_start = b.qx if plane == 'x' else b.qy
        isHit, beams, face = lens.hit(b, order=0, threshold=1e-9)
        out = beams['t1']
        q1 = q_start + beams['input'].length
        q2 = out.qx if plane == 'x' else out.qy
        pairs.append((q1, q2))
        # A q1 + B - C q1 q2 - D q2 = 0, split into real and imaginary.
        rows.append([q1.real, 1.0, -(q1*q2).real, -q2.real])
        rows.append([q1.imag, 0.0, -(q1*q2).imag, -q2.imag])
    _, _, vt = np.linalg.svd(np.array(rows))
    v = vt[-1]
    det = v[0]*v[3] - v[1]*v[2]
    v = v/np.sqrt(np.abs(det))
    if v[0] < 0:
        v = -v
    A, B, C, D = v
    residual = max(np.abs((A*q1 + B)/(C*q1 + D) - q2)
                   for q1, q2 in pairs)
    return A, B, C, D, residual

def q_efl(lens, plane):
    '''
    The focal length one plane of the transit has: -1/C.
    '''
    A, B, C, D, res = transit_matrix(lens, plane)
    return (np.inf if C == 0 else -1.0/C), res

#}}}

print('--- calibrating the instruments on a sphere ---')

#A spherical Lens is the same lens in both planes, so the ray-traced
#focal length and the q-extracted one have to agree - with the order,
#and with each other. This is what lets the q extraction speak for the
#out-of-plane behaviour below.
S = Lens(f=500*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='S')
efl_ray = ray_efl(S)
for plane in ['x', 'y']:
    got, res = q_efl(S, plane)
    check('a sphere reads the same in %s' % plane,
          rel(got, 500*mm) <= FOCAL_TOL, '(%.9f m)' % got)
    check('with a residual that says the fit is real',
          res < 1e-12, '(%.1e)' % res)
check('and the ray agrees with the extraction',
      rel(efl_ray, 500*mm) <= FOCAL_TOL, '(%.9f m)' % efl_ray)

#And on a substrate with no power at all, both planes are windows: no
#power, unit magnification, and the length of glass it amounts to.
F = Lens(f=None, center=[0.0, 0.0], normAngleHR=np.pi, name='F')
for plane in ['x', 'y']:
    A, B, C, D, res = transit_matrix(F, plane)
    check('a flat substrate carries no power in %s' % plane,
          np.abs(C) < WINDOW_TOL, '(C = %.2e)' % C)
    check('and does not magnify in %s' % plane,
          np.abs(A - 1) < 1e-9 and np.abs(D - 1) < 1e-9,
          '(A = %.12f, D = %.12f)' % (A, D))
    check('and is the thickness of glass it is in %s' % plane,
          np.abs(B - F.center_thickness/F.n) < 1e-12,
          '(B = %.9f vs d/n = %.9f)' % (B, F.center_thickness/F.n))

print('--- the focal length ordered lands in one plane only ---')

#: (label, kwargs): both signs, every family, the shapes and blanks
#: verify_lens.py spreads its cases over. Each is built twice, once per
#: curve_direction.
CASES = [
    ('biconvex 500mm',      dict(f=500*mm)),
    ('biconvex 50mm',       dict(f=50*mm)),
    ('biconvex 2 inch',     dict(f=100*mm, diameter=2*inch)),
    ('biconvex dense',      dict(f=200*mm, n=1.76)),
    ('plano-convex',        dict(f=500*mm, shape='plano-convex')),
    ('convex-plano',        dict(f=500*mm, shape='convex-plano')),
    ('biconcave',           dict(f=-500*mm)),
    ('plano-concave',       dict(f=-100*mm, shape='plano-concave',
                                 thickness=3*mm)),
    ('meniscus',            dict(f=200*mm, shape='meniscus',
                                 ROC_HR=-50*mm)),
    ('meniscus negative',   dict(f=-200*mm, shape='meniscus',
                                 ROC_HR=50*mm, thickness=8*mm)),
]

for label, kw in CASES:
    want = kw['f']

    #Curved in the plane of the drawing: the ray sees the lens, the
    #out-of-plane q sees a window.
    L = CyLens(center=[0.0, 0.0], normAngleHR=np.pi, name='L',
               curve_direction='h', **kw)
    check("h %s: f property" % label, rel(L.f, want) <= 1e-12,
          '(%.9f m)' % L.f)
    for side in [True, False]:
        got = ray_efl(L, from_front=side)
        where = 'front' if side else 'back'
        if got is None:
            check("h %s: traced from the %s" % (label, where), False,
                  '(the ray did not come out)')
            continue
        check("h %s: traced from the %s" % (label, where),
              rel(got, want) <= FOCAL_TOL,
              '(%.9f vs %.9f m)' % (got, want))
    A, B, C, D, res = transit_matrix(L, 'y')
    check("h %s: no power out of the plane" % label,
          np.abs(C) < WINDOW_TOL and res < 1e-9, '(C = %.2e)' % C)

    #Curved out of the plane: the very same order lands in y, and the
    #drawing plane gets the window.
    V = CyLens(center=[0.0, 0.0], normAngleHR=np.pi, name='V',
               curve_direction='v', **kw)
    check("v %s: f property" % label, rel(V.f, want) <= 1e-12,
          '(%.9f m)' % V.f)
    got, res = q_efl(V, 'y')
    check("v %s: focuses out of the plane" % label,
          rel(got, want) <= FOCAL_TOL and res < 1e-9,
          '(%.9f vs %.9f m)' % (got, want))
    got = ray_efl(V)
    check("v %s: a ray in the plane leaves parallel" % label,
          got is not None and np.abs(1.0/got) < WINDOW_TOL,
          '(power %.2e)' % (0.0 if got is None else 1.0/got))
    A, B, C, D, res = transit_matrix(V, 'x')
    check("v %s: the plane of the drawing is a window" % label,
          np.abs(C) < WINDOW_TOL and np.abs(A - 1) < 1e-9
          and res < 1e-9,
          '(C = %.2e, A = %.12f)' % (C, A))

print('--- the two directions are the same lens, turned ---')

#The 'h' lens measured in x and the 'v' lens measured in y are the same
#matrix: same order, same blank, same solve. Not just the same C - the
#same transit.
L = CyLens(f=150*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='L',
           curve_direction='h')
V = CyLens(f=150*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='V',
           curve_direction='v')
Mh = transit_matrix(L, 'x')
Mv = transit_matrix(V, 'y')
check('the curved planes carry one matrix',
      all(np.abs(a - b) < 1e-9 for a, b in zip(Mh[:4], Mv[:4])),
      '(%s vs %s)' % (np.round(Mh[:4], 6), np.round(Mv[:4], 6)))
check('and the same curvatures',
      L.inv_ROC_HR == V.inv_ROC_HR and L.inv_ROC_AR == V.inv_ROC_AR)

print('--- what the section through the drawing is ---')

#A cylinder curved out of the page has no curvature in it, so the
#section is a rectangle running through the apexes - the thick part -
#while an 'h' section runs between the chord planes. CyMirror already
#owns this distinction; a CyLens has to inherit it, not lose it.
L = CyLens(f=200*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='L',
           curve_direction='h')
V = CyLens(f=200*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='V',
           curve_direction='v')
for lens, want, which in [(L, L.thickness, 'the chord planes'),
                           (V, V.center_thickness, 'the apexes')]:
    c = lens.get_corners()
    depth = np.linalg.norm(np.asarray(c[0]) - np.asarray(c[3]))
    check("%s: the section runs between %s"
          % (lens.curve_direction, which),
          np.abs(depth - want) < 1e-15,
          '(%.6f vs %.6f mm)' % (depth/mm, want/mm))

cv = draw.Canvas()
L.draw(cv)
polys = [s for s in cv.layers['Mirrors'].shapes
         if isinstance(s, draw.PolyLine)]
lines = [s for s in cv.layers['Mirrors'].shapes if isinstance(s, draw.Line)]
check('an h lens draws its two faces as arcs',
      len(polys) == 2 and len(lines) == 2,
      '(%d arcs, %d lines)' % (len(polys), len(lines)))

cv = draw.Canvas()
V.draw(cv)
polys = [s for s in cv.layers['Mirrors'].shapes
         if isinstance(s, draw.PolyLine)]
lines = [s for s in cv.layers['Mirrors'].shapes if isinstance(s, draw.Line)]
check('a v lens draws as the rectangle the plane cuts out of it',
      len(polys) == 0 and len(lines) == 4,
      '(%d arcs, %d lines)' % (len(polys), len(lines)))

b = opt._ProbeRay([-0.2, 0.0], [1.0, 0.0])
for lens in [L, V]:
    ans = lens.isHit(b)
    check("%s: an on-axis ray lands on the HR face"
          % lens.curve_direction,
          ans['isHit'] and ans['face'] == 'HR', '(%r)' % ans['face'])

print('--- the defaults, and what the constructor refuses ---')

L = CyLens(f=500*mm)
check('a CyLens curves in the plane by default',
      L.curve_direction == 'h', '(%r)' % L.curve_direction)
check('and is named for what it is', L.name == 'CyLens')
check('anchors on its centre like a Lens', L.ROC_anchor == 'center')
check('transmits on the front face', L.HRtransmissive is True)
check('reflects nothing by default',
      L.Refl_HR == 0.0 and L.Refl_AR == 0.0)
check('has no wedge', L.wedgeAngle == 0.0)
check('draws no HR marker', L.draw_HR_marker is False)
check('one inch across, six millimetres at the rim',
      np.abs(L.diameter - 1*inch) < 1e-15
      and np.abs(L.thickness - 6*mm) < 1e-15)
check('is a Lens to everything that asks',
      isinstance(L, Lens) and isinstance(L, opt.CyMirror))

#The refusals are Lens's own code; one from each family shows they
#reach a CyLens intact, plus the one refusal that is new.
REFUSALS = [
    ('a direction that is neither h nor v',
     dict(f=500*mm, curve_direction='x'), "must be 'h' or 'v'"),
    ('two concave faces through the middle',
     dict(f=-25*mm, thickness=6*mm), 'meet inside the substrate'),
    ('an unknown shape',
     dict(f=500*mm, shape='banana'), 'Unknown lens shape'),
    ('a meniscus with no radius given',
     dict(f=500*mm, shape='meniscus'), 'not determined by f alone'),
    ('biconvex with a negative f',
     dict(f=-500*mm, shape='biconvex'), 'wants a convex HR face'),
    ('center and HRcenter together',
     dict(f=500*mm, center=[0, 0], HRcenter=[0, 0]), 'not both'),
]
for label, kw, wanted in REFUSALS:
    try:
        CyLens(name='L', **kw)
    except LensGeometryError as exc:
        check('refuses %s' % label, wanted in str(exc),
              '(%s)' % (str(exc)[:70] if wanted in str(exc)
                        else 'wrong reason: ' + str(exc)[:110]))
    except Exception as exc:
        check('refuses %s' % label, False,
              '(raised %s: %s)' % (type(exc).__name__, exc))
    else:
        check('refuses %s' % label, False, '(built one instead)')

print('--- retuning, in the plane the power lives in ---')

for direction in ['h', 'v']:
    L = CyLens(f=150*mm, center=[0.12, -0.03], normAngleHR=np.pi,
               name='L', curve_direction=direction)
    where = np.array(L.center)
    for f in [300*mm, -80*mm, 150*mm]:
        L.f = f
        check('%s set f = %g mm: reads back' % (direction, f/mm),
              rel(L.f, f) <= 1e-12, '(%.9f m)' % L.f)
        if direction == 'h':
            got = ray_efl(L)
        else:
            got, _ = q_efl(L, 'y')
        check('%s set f = %g mm: measures to it' % (direction, f/mm),
              rel(got, f) <= FOCAL_TOL, '(%.9f m)' % got)
        check('%s set f = %g mm: the lens stays put' % (direction, f/mm),
              np.allclose(L.center, where, atol=1e-15),
              '(%s)' % (L.center,))
        check('%s set f = %g mm: still a cylinder' % (direction, f/mm),
              L.curve_direction == direction)

#A retune that cannot be had leaves everything alone, the direction
#included.
L = CyLens(f=500*mm, curve_direction='v', name='L')
before = (L.f, L.inv_ROC_HR, L.inv_ROC_AR, L.curve_direction)
try:
    L.f = 0.0
except LensGeometryError:
    check('a zero focal length is refused', True)
else:
    check('a zero focal length is refused', False)
check('and the lens is left exactly as it was',
      (L.f, L.inv_ROC_HR, L.inv_ROC_AR, L.curve_direction) == before)

L.set_focal_length(220*mm, shape='plano-convex')
check('set_focal_length can reshape a CyLens',
      L.shape == 'plano-convex' and rel(L.f, 220*mm) <= 1e-12,
      '(%r, %.6f m)' % (L.shape, L.f))
check('without touching the direction', L.curve_direction == 'v')

print('--- copying and saving ---')

L = CyLens(f=-120*mm, shape='plano-concave', thickness=8*mm,
           diameter=2*inch, n=1.52, center=[0.1, 0.2],
           normAngleHR=0.7, name='CL1', curve_direction='v',
           max_stray_order=3)
L.ROC_anchor = 'HRcenter'          # not the default, so it has to travel
c = L.copy()
check('a copy is a CyLens', type(c) is CyLens, '(%s)' % type(c).__name__)
check('a copy keeps the direction', c.curve_direction == 'v')
check('a copy keeps ROC_anchor', c.ROC_anchor == 'HRcenter')
for attr in ['inv_ROC_HR', 'inv_ROC_AR', 'diameter', 'thickness',
             'wedgeAngle', 'n', 'Refl_HR', 'Trans_HR', 'Refl_AR',
             'Trans_AR', 'normAngleHR', 'max_stray_order']:
    check('a copy keeps %s' % attr,
          np.allclose(getattr(c, attr), getattr(L, attr), atol=1e-15))
check('a copy keeps HRcenter',
      np.allclose(c.HRcenter, L.HRcenter, atol=1e-15))
check('a copy has the same focal length', rel(c.f, L.f) <= 1e-15)

d = optic_to_dict(L)
check('it saves as a CyLens', d['type'] == 'CyLens', '(%r)' % d['type'])
check('with its direction', d['curve_direction'] == 'v')
r = optic_from_dict(d)
check('and loads as one', type(r) is CyLens, '(%s)' % type(r).__name__)
check('curved the way it was saved', r.curve_direction == 'v')
check('with ROC_anchor as it was saved', r.ROC_anchor == 'HRcenter')
check('and the same focal length', rel(r.f, L.f) <= 1e-15)
check('a file without a direction curves in the plane',
      optic_from_dict({k: v for k, v in d.items()
                       if k != 'curve_direction'}).curve_direction == 'h')

#An impossible saved lens is refused on the way back in, like a Lens.
bad = optic_to_dict(CyLens(f=-25*mm, thickness=9*mm, name='L'))
bad['thickness'] = 1*mm
try:
    optic_from_dict(bad)
    check('an impossible saved CyLens is refused', False, '(it loaded)')
except LensGeometryError:
    check('an impossible saved CyLens is refused', True)

print('--- a CyLens in a layout ---')

b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 0.5*mm), wl=1064*nm,
                       pos=[-0.2, 0.0], dirAngle=0.0, name='b0')
lay = OpticalLayout(sources=[b0],
                    rules=TraceRules(order=0, power_threshold=1e-3),
                    name='cyl')
#A mirror first, so there is a template a lens must not inherit from.
lay.apply_edit({'op': 'add', 'name': 'M1', 'type': 'Mirror',
                'params': {'HRcenter': [0.0, 0.3], 'normAngleHR': 0.0}})
lay.apply_edit({'op': 'add', 'name': 'CL1', 'type': 'CyLens',
                'params': {'HRcenter': [0.0, 0.0],
                           'normAngleHR': np.pi,
                           'curve_direction': 'v'}})
made = lay.get_optics('CL1')
check('an add message makes a CyLens', type(made).__name__ == 'CyLens')
check('curved the way the message says', made.curve_direction == 'v')
check('at the catalogue focal length, inheriting nothing',
      rel(made.f, 0.5) <= 1e-9 and made.Refl_HR == 0.0
      and np.abs(made.diameter - 1*inch) < 1e-15,
      '(f = %.4f m)' % made.f)

lay.apply_edit({'op': 'set', 'target': 'CL1', 'attrs': {'f': 0.25}})
check('f can be set through the protocol', rel(made.f, 0.25) <= 1e-9,
      '(%.6f m)' % made.f)
lay.apply_edit({'op': 'set', 'target': 'CL1',
                'attrs': {'curve_direction': 'h'}})
check('so can the direction', made.curve_direction == 'h')
try:
    lay.apply_edit({'op': 'set', 'target': 'CL1',
                    'attrs': {'curve_direction': 'x'}})
    check('but not to something that is neither', False, '(it took it)')
except EditError:
    check('but not to something that is neither', True)
try:
    lay.apply_edit({'op': 'add', 'name': 'CL2', 'type': 'CyLens',
                    'params': {'f': -0.025}})
    check('an impossible lens comes back as an EditError', False,
          '(it was made)')
except EditError as e:
    check('an impossible lens comes back as an EditError',
          'meet inside the substrate' in str(e), '(%s)' % str(e)[:60])

lay.apply_edit({'op': 'remove', 'target': 'CL1'})
lay.apply_edit({'op': 'undo'})
check('undoing a removal brings the same object back',
      lay.get_optics('CL1') is made)

lay.trace()
ent = [o for o in lay.scene_dict()['optics'] if o['name'] == 'CL1'][0]
check('the scene names the type', ent['type'] == 'CyLens')
check('and carries the power for the panel',
      rel(1.0/ent['inv_f'], made.f) <= 1e-9, '(1/f = %.4f)' % ent['inv_f'])
check('and the direction', ent['curve_direction'] == 'h')

#The main beam goes through a CyLens of either direction at order 0,
#which is what HRtransmissive is for.
for direction in ['h', 'v']:
    lens = CyLens(f=100*mm, center=[0.0, 0.0], normAngleHR=np.pi,
                  name='CL', curve_direction=direction)
    tr = OpticalLayout(optics=[lens], sources=[b0.copy()],
                       rules=TraceRules(order=0, power_threshold=1e-3),
                       name='through').trace()
    check("the main beam passes a '%s' lens at order 0" % direction,
          any(bm.name == 'CL:t1' and bm.stray_order == 0 for bm in tr),
          '(%s)' % ', '.join(bm.name for bm in tr))

lay.render_html(os.path.join(WORK, 'cylens_view.html'))
check('a layout with a CyLens renders',
      os.path.exists(os.path.join(WORK, 'cylens_view.html')),
      '(%d bytes)'
      % os.path.getsize(os.path.join(WORK, 'cylens_view.html')))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
