'''
The surface geometry of Mirror and CyMirror.

Not a GUI check, but it belongs to the same harness: it is run by
run_all.py and reports counted assertions, which the two smoke scripts
in tests/ do not.

Everything here is about where the two faces of a substrate actually
are. The HR face was always described by its chord centre, the AR face
by ARcenter - which is the apex, one sagitta further out - so an AR with
any curvature sat a sagitta behind where the sides of the substrate end.
That is invisible while the AR is flat, which it is for nearly every
mirror, and first order as soon as it is not. Nothing in the suites
built a curved AR, so nothing caught it.

The checks compare gtrace against an independently derived arc. A face
of inverse ROC c (positive for a concave face, as everywhere in gtrace)
stands off its own chord plane by

    offset(y) = sag + 1/c - sign(c)*sqrt(1/c^2 - y^2)

which is the sagitta at the axis and zero at the rim. Placing the
substrate at the origin facing +x, HRcenter - the apex of the HR arc,
which is what the constructor is given - lands the chord planes at

    x_HR_chord = -sag_H          x_AR_chord = -sag_H - t

and the two faces at x_HR_chord + offset_H(y) and x_AR_chord -
offset_A(y), the sign following each face's outward normal. Landing on
the chord plane at the rim is exactly the property that was broken.
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

def close(a, b, tol=1e-12):
    return np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol)

#{{{ The arc, derived here rather than taken from the class

def sagitta(c, diameter):
    '''
    Sagitta of a face of inverse ROC c spanning an aperture, positive
    when the face bulges out of the substrate.
    '''
    if c == 0.0:
        return 0.0
    R = 1.0/c
    r = diameter/2.0
    return -np.sign(R)*(np.abs(R) - np.sqrt(R**2 - r**2))

def offset(y, c, diameter):
    '''
    How far a face of inverse ROC c stands off its own chord plane at
    height y, along its outward normal. The sagitta at the axis, zero at
    the rim.
    '''
    if c == 0.0:
        return 0.0
    return sagitta(c, diameter) + 1.0/c - np.sign(c)*np.sqrt(1.0/c**2 - y**2)

def chord_planes(c_HR, diameter, thickness):
    '''
    x of the two chord planes of a substrate whose HRcenter - the apex
    of the HR arc - sits at the origin, facing +x.
    '''
    xc_HR = -sagitta(c_HR, diameter)
    return xc_HR, xc_HR - thickness

def x_HR(y, c_HR, diameter, thickness):
    '''
    Where the HR face is, at height y. Its outward normal is +x.
    '''
    xc_HR, _ = chord_planes(c_HR, diameter, thickness)
    return xc_HR + offset(y, c_HR, diameter)

def x_AR(y, c_HR, c_AR, diameter, thickness):
    '''
    Where the AR face is, at height y. Its outward normal is -x, and its
    chord plane is one HR sagitta further back than a reading of
    thickness alone would suggest.
    '''
    _, xc_AR = chord_planes(c_HR, diameter, thickness)
    return xc_AR - offset(y, c_AR, diameter)

#}}}

#{{{ Fixtures

DIAM = 25.4*mm
THICK = 6*mm
FAR = 1.0            # where the probe beams start

def make(cls, c_HR, c_AR, **kw):
    '''
    A substrate at the origin facing +x, with no wedge so that the two
    faces stay coaxial and the analytic arcs above apply.
    '''
    return cls(HRcenter=[0.0, 0.0], normAngleHR=0.0, diameter=DIAM,
               thickness=THICK, wedgeAngle=0.0, inv_ROC_HR=c_HR,
               inv_ROC_AR=c_AR, Refl_HR=0.5, Trans_HR=0.5,
               Refl_AR=0.5, Trans_AR=0.5, n=1.45, name='S', **kw)

def probe(y, from_front):
    '''
    A beam parallel to the axis at height y, aimed at the substrate.
    '''
    if from_front:
        return beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                                 pos=[FAR, y], dirAngle=np.pi, name='p')
    return beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                             pos=[-FAR, y], dirAngle=0.0, name='p')

#: Curvatures probed on each face: flat, convex and concave, in both a
#: gentle and a steep flavour. A lens-like face is the point of the
#: exercise, so the radii are the ones a 1 inch lens actually has.
CURVATURES = [('flat', 0.0),
              ('convex', -1.0/0.45),
              ('concave', +1.0/0.45),
              ('convex steep', -1.0/0.10),
              ('concave steep', +1.0/0.10)]

#: Pairs (HR, AR) probed together, so that the AR chord plane is once
#: measured from an HR that is itself curved. A lens has two curved
#: faces; nothing else in the suites ever builds one.
PAIRS = [('biconvex', -1.0/0.45, -1.0/0.45),
         ('biconcave', +1.0/0.45, +1.0/0.45),
         ('meniscus', -1.0/0.30, +1.0/0.50)]

#: Heights at which each face is probed, as a fraction of the radius.
HEIGHTS = [0.0, 0.3, 0.7, 0.95]

#}}}

print('--- where the faces are, against an independent arc ---')

def check_faces(cls, label, cname, c_HR, c_AR):
    kw = {'curve_direction': 'h'} if cls is opt.CyMirror else {}
    m = make(cls, c_HR, c_AR, **kw)
    for h in HEIGHTS:
        y = h*DIAM/2

        beams = m.hitFromHR(probe(y, from_front=True))
        got = beams['input'].length
        want = FAR - x_HR(y, c_HR, DIAM, THICK)
        check('%s %s HR at y=%.2fr' % (label, cname, h),
              close(got, want), '(%.12f vs %.12f)' % (got, want))

        beams = m.hitFromAR(probe(y, from_front=False))
        got = beams['input'].length
        want = FAR + x_AR(y, c_HR, c_AR, DIAM, THICK)
        check('%s %s AR at y=%.2fr' % (label, cname, h),
              close(got, want), '(%.12f vs %.12f)' % (got, want))

for cls in [opt.Mirror, opt.CyMirror]:
    label = cls.__name__
    # One face at a time: the HR is a control on the arcs derived above
    # as much as on the class, the AR is where the sagitta went missing.
    for cname, c in CURVATURES:
        check_faces(cls, label, 'HR %s' % cname, c, 0.0)
        check_faces(cls, label, 'AR %s' % cname, 0.0, c)
    # Both at once.
    for cname, c_HR, c_AR in PAIRS:
        check_faces(cls, label, cname, c_HR, c_AR)

print('--- the substrate is closed: each arc ends on its own chord ---')

for cls in [opt.Mirror, opt.CyMirror]:
    label = cls.__name__
    kw = {'curve_direction': 'h'} if cls is opt.CyMirror else {}
    for cname, c_HR, c_AR in PAIRS:
        m = make(cls, c_HR, c_AR, **kw)

        # With no wedge the sides are the two segments spanning the
        # chord planes at y = +-r, so their ends are the four corners of
        # the substrate.
        corners = []
        for centre, normVect, length in m.get_side_info():
            along = np.array([-normVect[1], normVect[0]])
            corners.append(centre + along*length/2)
            corners.append(centre - along*length/2)
        corners = np.array(corners)

        r = DIAM/2
        rims = [('HR rim +r', [x_HR(r, c_HR, DIAM, THICK), r]),
                ('HR rim -r', [x_HR(-r, c_HR, DIAM, THICK), -r]),
                ('AR rim +r', [x_AR(r, c_HR, c_AR, DIAM, THICK), r]),
                ('AR rim -r', [x_AR(-r, c_HR, c_AR, DIAM, THICK), -r])]
        for name, want in rims:
            d = np.min(np.linalg.norm(corners - np.array(want), axis=1))
            check('%s %s %s meets a corner of the substrate'
                  % (label, cname, name), d <= 1e-12, '(gap %.3e m)' % d)

print('--- isHit answers for the same surfaces hitFrom* traces ---')

for cls in [opt.Mirror, opt.CyMirror]:
    label = cls.__name__
    kw = {'curve_direction': 'h'} if cls is opt.CyMirror else {}

    for cname, c in CURVATURES:
        m = make(cls, 0.0, c, **kw)
        for h in HEIGHTS:
            y = h*DIAM/2
            p = probe(y, from_front=False)
            ans = m.isHit(p)
            traced = m.hitFromAR(p)
            check('%s isHit finds the AR (%s, y=%.2fr)' % (label, cname, h),
                  ans['isHit'] and ans['face'] == 'AR',
                  '(face %r)' % ans['face'])
            if ans['isHit'] and 'input' in traced:
                check('%s isHit distance matches the trace (%s, y=%.2fr)'
                      % (label, cname, h),
                      close(ans['distance'], traced['input'].length),
                      '(%.12f vs %.12f)'
                      % (ans['distance'], traced['input'].length))
            else:
                check('%s isHit distance matches the trace (%s, y=%.2fr)'
                      % (label, cname, h), False, '(no intersection)')

print('--- the drawn AR arc lies on the drawn substrate ---')

for cls in [opt.Mirror, opt.CyMirror]:
    label = cls.__name__
    kw = {'curve_direction': 'h'} if cls is opt.CyMirror else {}
    m = make(cls, 0.0, -1.0/0.45, **kw)

    cv = draw.Canvas()
    m.draw(cv)
    shapes = cv.layers['Mirrors'].shapes
    polys = [s for s in shapes if isinstance(s, draw.PolyLine)]
    lines = [s for s in shapes if isinstance(s, draw.Line)]
    check('%s draws one arc for the curved AR' % label, len(polys) == 1,
          '(%d polylines, %d lines)' % (len(polys), len(lines)))

    if polys:
        arc = polys[0]
        ends = np.array([[arc.x[0], arc.y[0]], [arc.x[-1], arc.y[-1]]])
        # Every endpoint of every straight segment of the outline.
        pts = []
        for l in lines:
            pts.append(np.asarray(l.start, dtype=float))
            pts.append(np.asarray(l.stop, dtype=float))
        pts = np.array(pts)
        worst = 0.0
        for e in ends:
            worst = max(worst, np.min(np.linalg.norm(pts - e, axis=1)))
        check('%s the arc ends on the outline' % label, worst <= 1e-12,
              '(largest gap %.3e m)' % worst)

print('--- changing a curvature carries the rest of the substrate ---')

#anchor_point says which end of the sagitta stays put. 'HRcenter', the
#default, keeps the arc under the beam and moves the substrate back
#behind it: regrinding a telescope mirror changes the magnification and
#must not move the spot. 'center' keeps the substrate still and moves
#the face on it, which is what an optics the beam goes through wants.
#
#Either way the rest of the substrate has to follow. The notification
#that carries it used to be suppressed, leaving ARcenterC, ARcenter and
#center where the old sagitta had put them.

def in_step(m, tol=1e-15):
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

for cls in [opt.Mirror, opt.CyMirror]:
    label = cls.__name__
    kw = {'curve_direction': 'h'} if cls is opt.CyMirror else {}

    check('%s anchors on HRcenter by default' % label,
          make(cls, 0.0, 0.0, **kw).anchor_point == 'HRcenter',
          '(%r)' % make(cls, 0.0, 0.0, **kw).anchor_point)

    for cname, c in CURVATURES:
        if c == 0.0:
            continue

        # Anchored on the HR surface, the default.
        m = make(cls, 0.0, 0.0, **kw)
        m.translate([0.4, -0.2])
        m.rotate(0.6)
        apex = np.array(m.HRcenter)
        centre = np.array(m.center)

        m.inv_ROC_HR = c
        check('%s HR %s: the HR surface stays put' % (label, cname),
              np.allclose(m.HRcenter, apex, atol=1e-15),
              '(moved %.3e m)' % np.linalg.norm(m.HRcenter - apex))
        check('%s HR %s: the substrate moves with it' % (label, cname),
              in_step(m), '(centre off by %.6f mm)'
              % (np.linalg.norm(m.center
                                - (m.HRcenterC + m.ARcenterC)/2)/mm))
        #A whole sagitta, not half of one: both chord planes move
        #together, since the thickness between them has not changed.
        #Half a sagitta was the signature of the bug, where HRcenterC
        #moved and ARcenterC stayed.
        check('%s HR %s: by a whole sagitta, both planes together'
              % (label, cname),
              np.allclose(m.center - centre, -m.sagHR*m.normVectHR,
                          atol=1e-15),
              '(%.6f mm, sag %.6f mm)'
              % (np.linalg.norm(m.center - centre)/mm, m.sagHR/mm))

        # Anchored on the substrate instead.
        m3 = make(cls, 0.0, 0.0, **kw)
        m3.translate([0.4, -0.2])
        m3.rotate(0.6)
        m3.anchor_point = 'center'
        apex3 = np.array(m3.HRcenter)
        centre3 = np.array(m3.center)
        planes3 = (np.array(m3.HRcenterC), np.array(m3.ARcenterC))

        m3.inv_ROC_HR = c
        check("%s HR %s: anchored on center, the substrate stays put"
              % (label, cname),
              np.allclose(m3.center, centre3, atol=1e-15)
              and np.allclose(m3.HRcenterC, planes3[0], atol=1e-15)
              and np.allclose(m3.ARcenterC, planes3[1], atol=1e-15),
              '(centre moved %.3e m)'
              % np.linalg.norm(m3.center - centre3))
        check('%s HR %s: and the apex moves by the sagitta' % (label, cname),
              np.allclose(m3.HRcenter - apex3, m3.sagHR*m3.normVectHR,
                          atol=1e-15),
              '(%.6f mm, sag %.6f mm)'
              % (np.linalg.norm(m3.HRcenter - apex3)/mm, m3.sagHR/mm))
        check('%s HR %s: with the substrate still in step' % (label, cname),
              in_step(m3))
        check('%s HR %s: the two anchors differ by one sagitta'
              % (label, cname),
              abs(np.linalg.norm(m3.center - m.center)
                  - abs(m.sagHR)) <= 1e-15,
              '(%.6f mm)' % (np.linalg.norm(m3.center - m.center)/mm))

        m2 = make(cls, 0.0, 0.0, **kw)
        m2.translate([0.4, -0.2])
        m2.rotate(0.6)
        before = (np.array(m2.center), np.array(m2.HRcenterC),
                  np.array(m2.ARcenterC), np.array(m2.HRcenter))
        m2.inv_ROC_AR = c
        check('%s AR %s: nothing moves but its own apex' % (label, cname),
              all(np.allclose(got, want, atol=1e-15) for got, want in
                  zip([m2.center, m2.HRcenterC, m2.ARcenterC, m2.HRcenter],
                      before)))
        check('%s AR %s: and the substrate stays in step' % (label, cname),
              in_step(m2), '(ARcenter off by %.6f mm)'
              % (np.linalg.norm(m2.ARcenter
                                - (m2.ARcenterC
                                   + m2.normVectAR*m2.sagAR))/mm))

print('--- a flat AR is untouched by any of this ---')

for cls in [opt.Mirror, opt.CyMirror]:
    label = cls.__name__
    kw = {'curve_direction': 'h'} if cls is opt.CyMirror else {}
    m = make(cls, 0.0, 0.0, **kw)
    for h in HEIGHTS:
        y = h*DIAM/2
        beams = m.hitFromAR(probe(y, from_front=False))
        got = beams['input'].length
        check('%s flat AR at y=%.2fr is the plane at -t' % (label, h),
              close(got, FAR - THICK), '(%.12f)' % got)

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
