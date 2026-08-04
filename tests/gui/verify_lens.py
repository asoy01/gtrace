'''
The Lens class: does a lens ordered by focal length have it?

The central check does not trust any formula in gtrace. A ray is sent
in parallel to the axis at a small height and the focal length is read
off the angle it leaves at,

    EFL = -h / slope_out

which is what an effective focal length is: measured from the principal
plane, wherever that turns out to be. Doing it this way makes the check
independent of the lensmaker's equation the constructor solved, of
where the principal planes are, and of the thickness of the substrate.
It is also the check that says whether solving as a thick lens was
worth it - against a thin lens solve the same measurement comes out a
couple of parts in a thousand off.

The rest is the constructor's contract: what it refuses, what it
defaults to, and that a lens survives being copied and saved.
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
from gtrace.layout import OpticalLayout, TraceRules, optic_to_dict, optic_from_dict
from gtrace.optcomp import Lens, LensGeometryError, lens_power, sagitta
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

#{{{ Measuring a focal length by tracing

#: Height of the probe ray. Small enough that spherical aberration
#: stays far below the tolerance, large enough to keep well clear of
#: the noise in an angle computed from a direction vector.
PROBE_HEIGHT = 10*um

#: How closely the measured focal length has to match the one ordered.
#: A thin lens solve misses by some 2e-3, so this separates the two by
#: three orders of magnitude.
FOCAL_TOL = 1e-6

def measure_focal_length(lens, from_front=True, h=PROBE_HEIGHT):
    '''
    Send a ray in parallel to the axis and read the focal length off
    the angle it leaves at.

    Returns None if the ray does not come out the far side.
    '''
    #The axis, and the transverse direction 90 degrees from it.
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

#}}}

#{{{ The lenses probed

#: (label, kwargs). Spread over both signs, all the shapes, both
#: apertures a catalogue sells and a range of focal lengths short
#: enough that the thickness of the substrate matters.
CASES = [
    ('biconvex default',      dict(f=500*mm)),
    ('biconvex named',        dict(f=500*mm, shape='biconvex')),
    ('biconvex short',        dict(f=50*mm)),
    ('biconvex 2 inch',       dict(f=100*mm, diameter=2*inch)),
    ('biconvex thick',        dict(f=200*mm, thickness=12*mm)),
    ('biconvex dense',        dict(f=200*mm, n=1.76)),
    ('plano-convex',          dict(f=500*mm, shape='plano-convex')),
    ('convex-plano',          dict(f=500*mm, shape='convex-plano')),
    ('convex-plano short',    dict(f=60*mm, shape='convex-plano')),
    ('biconcave',             dict(f=-500*mm)),
    ('biconcave named',       dict(f=-500*mm, shape='biconcave')),
    ('biconcave short',       dict(f=-75*mm, thickness=3*mm)),
    ('plano-concave',         dict(f=-100*mm, shape='plano-concave',
                                   thickness=3*mm)),
    ('concave-plano',         dict(f=-100*mm, shape='concave-plano',
                                   thickness=3*mm)),
    ('meniscus positive',     dict(f=200*mm, shape='meniscus', ROC_HR=-50*mm)),
    ('meniscus concave front', dict(f=200*mm, shape='meniscus', ROC_HR=300*mm)),
    ('meniscus negative',     dict(f=-200*mm, shape='meniscus', ROC_HR=50*mm,
                                   thickness=8*mm)),
]

#}}}

print('--- the focal length ordered is the focal length traced ---')

for label, kw in CASES:
    L = Lens(center=[0.0, 0.0], normAngleHR=np.pi, name='L', **kw)
    want = kw['f']

    check('%s: f property' % label, rel(L.f, want) <= 1e-12,
          '(%.9f vs %.9f m)' % (L.f, want))

    for side in [True, False]:
        got = measure_focal_length(L, from_front=side)
        where = 'front' if side else 'back'
        if got is None:
            check('%s: traced from the %s' % (label, where), False,
                  '(the ray did not come out)')
            continue
        check('%s: traced from the %s' % (label, where),
              rel(got, want) <= FOCAL_TOL,
              '(%.9f vs %.9f m, %.1e rel)' % (got, want, rel(got, want)))

print('--- the thickness term is what makes that work ---')

#The same lens solved thin, to show the check above can tell the
#difference. Not a property of gtrace: a demonstration that the
#tolerance means something.
for label, kw in [('f=500mm biconvex', dict(f=500*mm)),
                  ('f=50mm biconvex', dict(f=50*mm))]:
    f = kw['f']
    n = 1.45
    thin = -1./(2.*(n - 1.)*f)          # both faces, thin lens
    L = Lens(f=None, inv_ROC_HR=thin, inv_ROC_AR=thin,
             center=[0.0, 0.0], normAngleHR=np.pi, name='thin', **{
                 k: v for k, v in kw.items() if k != 'f'})
    got = measure_focal_length(L)
    check('%s: a thin lens solve misses' % label,
          rel(got, f) > 10*FOCAL_TOL,
          '(%.6f m instead of %.6f, %.1e rel)' % (got, f, rel(got, f)))

print('--- what the constructor refuses ---')

#: (what is wrong, arguments, words the message has to contain). The
#: third column is the point: several of these would raise something
#: for the wrong reason and still look like a pass. A concave lens too
#: thin for its own faces was for a while reported as a solver failure,
#: which is true but useless to whoever has to widen the blank.
REFUSALS = [
    ('two concave faces through the middle',
     dict(f=-25*mm, thickness=6*mm), 'meet inside the substrate'),
    ('a face steeper than its aperture',
     dict(f=-8*mm, thickness=60*mm), 'cannot span an aperture'),
    ('a focal length the substrate cannot reach',
     dict(f=2*mm), 'No symmetric lens'),
    ('biconcave with a positive f',
     dict(f=500*mm, shape='biconcave'), 'wants a concave HR face'),
    ('biconvex with a negative f',
     dict(f=-500*mm, shape='biconvex'), 'wants a convex HR face'),
    ('plano-concave with a positive f',
     dict(f=500*mm, shape='plano-concave'), 'wants a concave AR face'),
    ('an unknown shape',
     dict(f=500*mm, shape='banana'), 'Unknown lens shape'),
    ('a meniscus with no radius given',
     dict(f=500*mm, shape='meniscus'), 'not determined by f alone'),
    ('a meniscus that is not one',
     dict(f=200*mm, shape='meniscus', ROC_HR=-500*mm), 'not a meniscus'),
    ('a radius given for a shape that solves for both',
     dict(f=500*mm, ROC_HR=-100*mm), 'Only a meniscus'),
    ('both curvatures given as well as f',
     dict(f=500*mm, inv_ROC_AR=1.0), 'over-determine'),
    ('ROC_HR and inv_ROC_HR together',
     dict(f=200*mm, shape='meniscus', ROC_HR=-50*mm, inv_ROC_HR=-20.0),
     'not both'),
    ('a zero focal length',
     dict(f=0.0), 'finite, non-zero focal length'),
    ('an infinite focal length',
     dict(f=np.inf), 'finite, non-zero focal length'),
    ('a substrate no denser than the air',
     dict(f=500*mm, n=1.0), 'denser than its surroundings'),
    ('a shape with no focal length to shape',
     dict(shape='biconvex'), 'no meaning without f'),
    ('center and HRcenter together',
     dict(f=500*mm, center=[0, 0], HRcenter=[0, 0]), 'not both'),
]

for label, kw, wanted in REFUSALS:
    try:
        L = Lens(name='L', **kw)
    except LensGeometryError as exc:
        check('refuses %s' % label, wanted in str(exc),
              '(%s)' % (str(exc)[:80] if wanted in str(exc)
                        else 'wrong reason: ' + str(exc)[:110]))
    except Exception as exc:
        check('refuses %s' % label, False,
              '(raised %s, not LensGeometryError: %s)'
              % (type(exc).__name__, exc))
    else:
        check('refuses %s' % label, False,
              '(built one instead: f=%.6g, ROC=(%.4g, %.4g))'
              % (L.f, 1/L.inv_ROC_HR if L.inv_ROC_HR else np.inf,
                 1/L.inv_ROC_AR if L.inv_ROC_AR else np.inf))

check('LensGeometryError is a ValueError',
      issubclass(LensGeometryError, ValueError))

print('--- what it accepts that looks like it should not ---')

#A short concave lens is makeable as long as the blank is thick enough,
#and the message from the refusal above says how thick. Taking it at
#its word has to work.
try:
    L = Lens(f=-25*mm, thickness=9*mm, name='L')
    check('a concave lens in a blank thick enough', L.center_thickness > 0,
          '(centre %.3f mm of a %.3f mm rim)'
          % (L.center_thickness/mm, L.thickness/mm))
    check('and it still has its focal length',
          rel(measure_focal_length(L), -25*mm) <= FOCAL_TOL)
except LensGeometryError as exc:
    check('a concave lens in a blank thick enough', False, '(%s)' % exc)

print('--- the defaults a lens needs ---')

L = Lens(f=500*mm, name='L')
check('no wedge by default', L.wedgeAngle == 0.0, '(%r)' % L.wedgeAngle)
check('the front face transmits', L.HRtransmissive is True)
check('both faces are coated alike',
      L.Refl_HR == L.Refl_AR and L.Trans_HR == L.Trans_AR,
      '(R %.4g, T %.4g)' % (L.Refl_HR, L.Trans_HR))
check('power is conserved at each face',
      abs(L.Refl_HR + L.Trans_HR - 1) < 1e-12
      and abs(L.Refl_AR + L.Trans_AR - 1) < 1e-12)
# A real lens reflects, but a bench full of them makes so many faint
# ghosts that the picture is unreadable. Whoever wants those says so.
check('neither face reflects by default',
      L.Refl_HR == 0.0 and L.Refl_AR == 0.0,
      '(%.4g / %.4g)' % (L.Refl_HR, L.Refl_AR))
Lg = Lens(f=500*mm, Refl_HR=0.005, Trans_HR=0.995,
          Refl_AR=0.005, Trans_AR=0.995, name='Lg')
b_ghost = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 0.5*mm), wl=1064*nm,
                            pos=[-0.2, 0.0], dirAngle=0.0, name='bg')
n_plain = len(OpticalLayout(
    optics=[Lens(f=500*mm, center=[0, 0], normAngleHR=np.pi, name='Lp')],
    sources=[b_ghost.copy()],
    rules=TraceRules(order=3, power_threshold=1e-9)).trace())
Lg.center = [0, 0]
Lg.normAngleHR = np.pi
n_coated = len(OpticalLayout(optics=[Lg], sources=[b_ghost.copy()],
                             rules=TraceRules(order=3,
                                              power_threshold=1e-9)).trace())
check('so an uncoated lens makes fewer beams than a coated one',
      n_plain < n_coated, '(%d vs %d)' % (n_plain, n_coated))
check('one inch across by default', abs(L.diameter - 25.4*mm) < 1e-15,
      '(%.4f mm)' % (L.diameter/mm))
check('six millimetres at the rim', abs(L.thickness - 6*mm) < 1e-15,
      '(%.4f mm)' % (L.thickness/mm))
check('no HR marker is drawn', L.draw_HR_marker is False)
check('a Mirror still marks its HR', opt.Mirror().draw_HR_marker is True)

cv = draw.Canvas()
L.draw(cv)
lines = [s for s in cv.layers['Mirrors'].shapes if isinstance(s, draw.Line)]
polys = [s for s in cv.layers['Mirrors'].shapes
         if isinstance(s, draw.PolyLine)]
check('a biconvex lens draws two arcs and two sides',
      len(polys) == 2 and len(lines) == 2,
      '(%d arcs, %d lines)' % (len(polys), len(lines)))

print('--- placing it ---')

L = Lens(f=500*mm, center=[0.3, -0.2], normAngleHR=np.pi/3, name='L')
check('center places the middle of the substrate',
      np.allclose(L.center, [0.3, -0.2], atol=1e-15), '(%s)' % (L.center,))
check('the two chord planes straddle it',
      abs(np.linalg.norm(L.HRcenterC - L.ARcenterC) - L.thickness) < 1e-15)

L2 = Lens(f=500*mm, HRcenter=[0.3, -0.2], name='L')
check('HRcenter places the apex of the front face',
      np.allclose(L2.HRcenter, [0.3, -0.2], atol=1e-15),
      '(%s)' % (L2.HRcenter,))
check('and then center is not at the origin',
      not np.allclose(L2.center, [0.0, 0.0], atol=1e-9))

check('the default is the middle at the origin',
      np.allclose(Lens(f=500*mm).center, [0.0, 0.0], atol=1e-15))

print('--- derived quantities follow the lens, not the order ---')

L = Lens(f=500*mm, name='L')
check('shape names itself', L.shape == 'biconvex', '(%r)' % L.shape)
check('centre thickness is the rim plus two sagittae',
      abs(L.center_thickness - (L.thickness + L.sagHR + L.sagAR)) < 1e-15,
      '(%.6f mm)' % (L.center_thickness/mm))
check('a biconvex lens is thicker in the middle',
      L.center_thickness > L.thickness)

before = L.f
L.inv_ROC_AR = 0.0
#Flattening one face of a symmetric lens roughly doubles the focal
#length - roughly, because the thickness term goes with it.
check('f follows an edited curvature',
      1.9 < L.f/before < 2.1, '(%.6f -> %.6f m)' % (before, L.f))
check('and so does shape', L.shape == 'convex-plano', '(%r)' % L.shape)
check('f now agrees with a lens built that way',
      rel(L.f, Lens(f=None, inv_ROC_HR=L.inv_ROC_HR, inv_ROC_AR=0.0).f)
      <= 1e-12)
check('and the trace agrees with it too',
      rel(measure_focal_length(
          Lens(f=None, inv_ROC_HR=L.inv_ROC_HR, inv_ROC_AR=0.0,
               center=[0, 0], normAngleHR=np.pi)), L.f) <= FOCAL_TOL)

flat = Lens(f=None, name='flat')
check('a lens with no curvature has no power', flat.f == np.inf,
      '(%r)' % flat.f)
check('and names itself plano-plano', flat.shape == 'plano-plano',
      '(%r)' % flat.shape)

print('--- retuning a lens by its focal length ---')

def substrate_is_consistent(m, tol=1e-15):
    '''
    Every point of the substrate against the one description of it.
    Assigning a curvature updates the near chord plane and stops there,
    so this is what a retune has to put back in step.
    '''
    return (np.linalg.norm(m.HRcenter
                           - (m.HRcenterC + m.normVectHR*m.sagHR)) <= tol
            and np.linalg.norm(m.ARcenterC
                               - (m.HRcenterC
                                  - m.normVectHR*m.thickness)) <= tol
            and np.linalg.norm(m.ARcenter
                               - (m.ARcenterC
                                  + m.normVectAR*m.sagAR)) <= tol
            and np.linalg.norm(m.center
                               - (m.HRcenterC + m.ARcenterC)/2) <= tol)

#The loop this is all for: sweep the focal length and trace at each
#step. Every one of these has to be the lens it says it is.
L = Lens(f=150*mm, center=[0.12, -0.03], normAngleHR=np.pi, name='L')
where = np.array(L.center)
for f in [200*mm, 300*mm, 80*mm, 1000*mm, 150*mm]:
    L.f = f
    check('set f = %g mm: reads back' % (f/mm), rel(L.f, f) <= 1e-12,
          '(%.9f m)' % L.f)
    check('set f = %g mm: traces to it' % (f/mm),
          rel(measure_focal_length(L), f) <= FOCAL_TOL,
          '(%.9f m)' % measure_focal_length(L))
    check('set f = %g mm: the lens stays put' % (f/mm),
          np.allclose(L.center, where, atol=1e-15), '(%s)' % (L.center,))
    check('set f = %g mm: the substrate stays consistent' % (f/mm),
          substrate_is_consistent(L))
    check('set f = %g mm: still biconvex' % (f/mm), L.shape == 'biconvex',
          '(%r)' % L.shape)

#A lens anchors on its centre, unlike a mirror: the beam goes through
#it, so there is no reflection point for the faces to stay under, and
#the substrate is what sits at a position on the bench. That is
#anchor_point, not something set_focal_length arranges, so assigning a
#curvature by hand behaves the same way.
direct = Lens(f=150*mm, center=[0.4, 0.1], normAngleHR=np.pi, name='direct')
check('a lens anchors on its centre', direct.anchor_point == 'center',
      '(%r)' % direct.anchor_point)
check('a mirror anchors on its HR surface',
      opt.Mirror().anchor_point == 'HRcenter', '(%r)' % opt.Mirror().anchor_point)

apex = np.array(direct.HRcenter)
direct.inv_ROC_HR = -1.0/0.1
check('assigning a curvature leaves the lens where it is',
      np.allclose(direct.center, [0.4, 0.1], atol=1e-15),
      '(%s)' % (direct.center,))
check('and moves the face on it',
      np.linalg.norm(direct.HRcenter - apex) > 0.1*mm,
      '(apex moved %.6f mm)'
      % (np.linalg.norm(direct.HRcenter - apex)/mm))
check('leaving every point of it in step', substrate_is_consistent(direct))
direct.inv_ROC_AR = -1.0/0.1
check('the same for the back face',
      substrate_is_consistent(direct)
      and np.allclose(direct.center, [0.4, 0.1], atol=1e-15))

#Turning the anchor round makes a lens behave like a mirror.
asmirror = Lens(f=150*mm, center=[0.4, 0.1], normAngleHR=np.pi, name='m')
asmirror.anchor_point = 'HRcenter'
apex = np.array(asmirror.HRcenter)
asmirror.f = 300*mm
check('a lens anchored on HRcenter keeps its face still instead',
      np.allclose(asmirror.HRcenter, apex, atol=1e-15)
      and not np.allclose(asmirror.center, [0.4, 0.1], atol=1e-9),
      '(centre now %s)' % (asmirror.center,))
check('and is still a consistent substrate',
      substrate_is_consistent(asmirror))
check('and still has the focal length asked for',
      rel(measure_focal_length(asmirror), 300*mm) <= FOCAL_TOL)

print('--- the anchor is the turning point too ---')

#Assigning an orientation turns the optics about its anchor point: a
#mirror pivots the reflection point, which is what gtrace has always
#done, and a lens pivots the middle of its substrate.
M = opt.Mirror(HRcenter=[0.3, 0.2], normAngleHR=0.5)
h0, c0 = np.array(M.HRcenter), np.array(M.center)
M.normAngleHR = 1.3
check('a mirror assigned an angle keeps its HR apex still',
      np.allclose(M.HRcenter, h0, atol=1e-15)
      and not np.allclose(M.center, c0, atol=1e-6),
      '(centre moved %.6f mm)' % (np.linalg.norm(M.center - c0)/mm))

L = Lens(f=200*mm, center=[0.4, 0.1], normAngleHR=np.pi, name='L')
h0, c0 = np.array(L.HRcenter), np.array(L.center)
L.normAngleHR = np.pi + 0.7
check('a lens assigned an angle keeps its middle still',
      np.allclose(L.center, c0, atol=1e-15)
      and not np.allclose(L.HRcenter, h0, atol=1e-6),
      '(apex moved %.6f mm)' % (np.linalg.norm(L.HRcenter - h0)/mm))
check('and is still a consistent substrate', substrate_is_consistent(L))
check('and still focuses where it says',
      rel(measure_focal_length(L), 200*mm) <= FOCAL_TOL)

L2 = Lens(f=200*mm, center=[0.4, 0.1], normAngleHR=np.pi, name='L2')
c0 = np.array(L2.center)
L2.normVectHR = [0.6, 0.8]
check('assigning the vector turns about the same point',
      np.allclose(L2.center, c0, atol=1e-15)
      and substrate_is_consistent(L2), '(%s)' % (L2.center,))

#rotate() pivots the anchor point by default, so a mirror - anchored
#on its HR apex - turns exactly as it always has, and a lens turns
#about its middle. True asks for the middle whatever the anchor says.
for label, pivot_arg, pin in [
        ('the default pivots the anchor of a lens, its middle',
         None, 'center'),
        ('True pivots the middle', True, 'center'),
        ('False spells the default out', False, 'center')]:
    Lr = Lens(f=200*mm, center=[0.4, 0.1], normAngleHR=np.pi, name='Lr')
    p0 = np.array(getattr(Lr, pin))
    if pivot_arg is None:
        Lr.rotate(0.3)
    else:
        Lr.rotate(0.3, center=pivot_arg)
    check('rotate: %s' % label,
          np.allclose(np.array(getattr(Lr, pin)), p0, atol=1e-15)
          and substrate_is_consistent(Lr),
          '(%s at %s)' % (pin, np.array(getattr(Lr, pin))))

for label, kwargs in [('the default pivots the anchor of a mirror, '
                       'its HR apex - as rotate() always has', {}),
                      ('False spells that out', {'center': False})]:
    Mr = opt.Mirror(HRcenter=[0.3, 0.2], normAngleHR=0.5)
    h0 = np.array(Mr.HRcenter)
    Mr.rotate(0.3, **kwargs)
    check('rotate: %s' % label,
          np.allclose(Mr.HRcenter, h0, atol=1e-15))

#An explicit point still works, and is a rigid turn about it.
Mp = opt.Mirror(HRcenter=[0.3, 0.2], normAngleHR=0.5)
p = np.array([1.0, -0.5])
r0 = np.linalg.norm(Mp.HRcenter - p)
a0 = float(Mp.normAngleHR)
Mp.rotate(0.4, center=p)
check('rotate about a given point keeps the distance to it',
      abs(np.linalg.norm(Mp.HRcenter - p) - r0) < 1e-12)
check('and turns by the angle asked',
      abs(float(Mp.normAngleHR) - (a0 + 0.4)) < 1e-12)

print('--- and it keeps the shape it had ---')

for label, kw, expect in [
        ('biconvex', dict(f=150*mm), 'biconvex'),
        ('plano-convex', dict(f=150*mm, shape='plano-convex'),
         'plano-convex'),
        ('convex-plano', dict(f=150*mm, shape='convex-plano'),
         'convex-plano'),
        ('meniscus', dict(f=150*mm, shape='meniscus', ROC_HR=-40*mm),
         'convex-concave'),
        ('biconcave', dict(f=-150*mm), 'biconcave')]:
    L = Lens(center=[0.0, 0.0], normAngleHR=np.pi, name='L', **kw)
    ratio = ((L.inv_ROC_AR/L.inv_ROC_HR) if L.inv_ROC_HR
             else (L.inv_ROC_HR/L.inv_ROC_AR))
    target = 2*kw['f']
    L.f = target
    now = ((L.inv_ROC_AR/L.inv_ROC_HR) if L.inv_ROC_HR
           else (L.inv_ROC_HR/L.inv_ROC_AR))
    check('%s keeps its shape' % label, L.shape == expect,
          '(%r)' % L.shape)
    check('%s keeps the ratio between its faces' % label,
          rel(now, ratio) <= 1e-9 if ratio else abs(now) < 1e-12,
          '(%.9f vs %.9f)' % (now, ratio))
    check('%s reaches the new focal length' % label,
          rel(measure_focal_length(L), target) <= FOCAL_TOL,
          '(%.9f vs %.9f m)' % (measure_focal_length(L), target))

#Turning it inside out.
L = Lens(f=150*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='L')
L.f = -150*mm
check('a converging lens asked for a negative f turns inside out',
      L.shape == 'biconcave', '(%r)' % L.shape)
check('and traces to it',
      rel(measure_focal_length(L), -150*mm) <= FOCAL_TOL)

L = Lens(f=150*mm, shape='plano-convex', center=[0.0, 0.0],
         normAngleHR=np.pi, name='L')
L.f = -150*mm
check('so does a plano-convex one', L.shape == 'plano-concave',
      '(%r)' % L.shape)
check('and its flat face is still flat', L.inv_ROC_HR == 0.0)

print('--- changing the shape without rebuilding the lens ---')

L = Lens(f=150*mm, center=[0.05, 0.0], normAngleHR=np.pi, name='L')
L.set_focal_length(150*mm, shape='convex-plano')
check('set_focal_length can reshape', L.shape == 'convex-plano',
      '(%r)' % L.shape)
check('and keeps the focal length', rel(L.f, 150*mm) <= 1e-12)
check('and the place', np.allclose(L.center, [0.05, 0.0], atol=1e-15))
check('and the name', L.name == 'L')

L.set_focal_length(220*mm, shape='meniscus', ROC_HR=-40*mm)
check('including into a meniscus', L.shape == 'convex-concave',
      '(%r)' % L.shape)
check('with the radius asked for',
      rel(1/L.inv_ROC_HR, -40*mm) <= 1e-12, '(%.6f mm)' % (1/L.inv_ROC_HR/mm))
check('and the focal length asked for',
      rel(measure_focal_length(L), 220*mm) <= FOCAL_TOL)

print('--- a retune that cannot be had leaves the lens alone ---')

#Three different ways a retune can fail, each reached by a target that
#trips that one first: the faces meet in the middle while still
#spanning the aperture; the faces span nothing while the focal length
#is still reachable; the focal length is out of reach of this shape
#altogether.
RETUNE_REFUSALS = [
    ('a focal length that would eat through the middle',
     dict(f=-500*mm, thickness=3*mm), -50*mm, 'meet inside the substrate'),
    ('a face steeper than the aperture',
     dict(f=500*mm), 8*mm, 'cannot span an aperture'),
    ('a focal length this shape cannot reach at all',
     dict(f=500*mm), 1*mm, 'cannot reach that focal length'),
    ('a zero focal length', dict(f=500*mm), 0.0,
     'finite, non-zero focal length'),
]
for label, kw, target, wanted in RETUNE_REFUSALS:
    L = Lens(center=[0.0, 0.0], normAngleHR=np.pi, name='L', **kw)
    before = (L.f, L.inv_ROC_HR, L.inv_ROC_AR, np.array(L.center))
    try:
        L.f = target
    except LensGeometryError as exc:
        check('refuses %s' % label, wanted in str(exc),
              '(%s)' % (str(exc)[:70] if wanted in str(exc)
                        else 'wrong reason: ' + str(exc)[:110]))
    else:
        check('refuses %s' % label, False, '(retuned to %.6g m)' % L.f)
    check('and leaves %s untouched' % label,
          (L.f, L.inv_ROC_HR, L.inv_ROC_AR) == before[:3]
          and np.array_equal(L.center, before[3]),
          '(f = %.6g m)' % L.f)

flat = Lens(f=None, name='flat')
try:
    flat.f = 500*mm
except LensGeometryError as exc:
    check('a flat substrate has no shape to scale',
          'no shape to scale' in str(exc), '(%s)' % str(exc)[:70])
else:
    check('a flat substrate has no shape to scale', False,
          '(it became f = %.6g m)' % flat.f)
flat.set_focal_length(500*mm, shape='biconvex')
check('but it takes a shape and a focal length together',
      flat.shape == 'biconvex' and rel(flat.f, 500*mm) <= 1e-12,
      '(%r, %.6f m)' % (flat.shape, flat.f))

L = Lens(f=150*mm, name='L')
try:
    L.set_focal_length(200*mm, ROC_HR=-40*mm)
except LensGeometryError as exc:
    check('a radius without a shape is refused', 'nothing to pin' in str(exc),
          '(%s)' % str(exc)[:70])
else:
    check('a radius without a shape is refused', False)

print('--- copying and saving ---')

L = Lens(f=-120*mm, shape='concave-plano', thickness=8*mm,
         diameter=2*inch, n=1.52, center=[0.1, 0.2],
         normAngleHR=0.7, name='L1', max_stray_order=3)
L.anchor_point = 'HRcenter'          # not the default, so it has to travel
c = L.copy()
check('a copy is a Lens', type(c) is Lens, '(%s)' % type(c).__name__)
check('a copy keeps anchor_point', c.anchor_point == 'HRcenter',
      '(%r)' % c.anchor_point)
for attr in ['inv_ROC_HR', 'inv_ROC_AR', 'diameter', 'thickness',
             'wedgeAngle', 'n', 'Refl_HR', 'Trans_HR', 'Refl_AR',
             'Trans_AR', 'normAngleHR', 'max_stray_order']:
    check('a copy keeps %s' % attr,
          np.allclose(getattr(c, attr), getattr(L, attr), atol=1e-15))
check('a copy keeps HRcenter', np.allclose(c.HRcenter, L.HRcenter, atol=1e-15))
check('a copy keeps the name', c.name == L.name)
check('a copy has the same focal length', rel(c.f, L.f) <= 1e-15,
      '(%.9f vs %.9f)' % (c.f, L.f))

d = optic_to_dict(L)
check('it saves as a Lens', d['type'] == 'Lens', '(%r)' % d['type'])
r = optic_from_dict(d)
check('and loads as a Lens', type(r) is Lens, '(%s)' % type(r).__name__)
check('with anchor_point as it was saved', r.anchor_point == 'HRcenter',
      '(%r)' % r.anchor_point)
check('while a file without one takes the class default',
      optic_from_dict({k: v for k, v in d.items()
                       if k != 'anchor_point'}).anchor_point == 'center')
check('the loaded lens has the same focal length', rel(r.f, L.f) <= 1e-15,
      '(%.9f vs %.9f)' % (r.f, L.f))
for attr in ['inv_ROC_HR', 'inv_ROC_AR', 'diameter', 'thickness', 'n']:
    check('the loaded lens keeps %s' % attr,
          np.allclose(getattr(r, attr), getattr(L, attr), atol=1e-15))

#A lens whose radii were edited into something impossible should be
#refused on the way back in rather than loaded as a broken substrate.
bad = optic_to_dict(Lens(f=-25*mm, thickness=9*mm, name='L'))
bad['thickness'] = 1*mm
try:
    optic_from_dict(bad)
    check('an impossible saved lens is refused', False, '(it loaded)')
except LensGeometryError:
    check('an impossible saved lens is refused', True)

print('--- a lens in a layout ---')

#The point of HRtransmissive: the main beam through a lens must not be
#counted as a stray beam, or a trace of modest order stops at it.
b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 0.5*mm), wl=1064*nm,
                       pos=[-0.2, 0.0], dirAngle=0.0, name='b0')
L1 = Lens(f=100*mm, center=[0.0, 0.0], normAngleHR=np.pi, name='L1')
L2 = Lens(f=-50*mm, center=[0.25, 0.0], normAngleHR=np.pi, name='L2')
layout = OpticalLayout(optics=[L1, L2], sources=[b0],
                       rules=TraceRules(order=0, power_threshold=1e-3),
                       name='two lenses')
layout.trace()
names = [bm.name for bm in layout.beams]
check('the beam goes through both lenses at order 0',
      any(nm.startswith('L1') for nm in names)
      and any(nm.startswith('L2') for nm in names),
      '(%d beams: %s)' % (len(names), ', '.join(names[:6])))
main = [bm for bm in layout.beams if bm.layer == 'main_beam']
check('and stays a main beam throughout',
      all(bm.stray_order == 0 for bm in main),
      '(orders %s)' % sorted(set(bm.stray_order for bm in main)))

#The same layout with a Mirror standing in for the lens: the beam
#through it is a ghost, which is the behaviour Lens has to differ from.
M = opt.Mirror(HRcenter=[0.0, 0.0], normAngleHR=np.pi, diameter=1*inch,
               thickness=6*mm, wedgeAngle=0.0, inv_ROC_HR=0.0,
               inv_ROC_AR=0.0, Refl_HR=0.0, Trans_HR=1.0,
               Refl_AR=0.0, Trans_AR=1.0, n=1.45, name='M')
layout2 = OpticalLayout(optics=[M], sources=[b0.copy()],
                        rules=TraceRules(order=0, power_threshold=1e-3),
                        name='mirror instead')
layout2.trace()
check('a Mirror in its place would have stopped the beam',
      not any(bm.name.startswith('M:t') for bm in layout2.beams),
      '(%s)' % ', '.join(bm.name for bm in layout2.beams))

layout.render_html(os.path.join(WORK, 'lens_view.html'))
check('a layout with lenses renders',
      os.path.exists(os.path.join(WORK, 'lens_view.html')),
      '(%d bytes)'
      % os.path.getsize(os.path.join(WORK, 'lens_view.html')))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
