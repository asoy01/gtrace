'''
CyMirror: that a cylinder focuses in one plane and leaves the other be.

Not a GUI check, but it belongs to the same harness as verify_surfaces:
run by run_all.py, reporting counted assertions.

CyMirror used to be cylindrical in shape only. Its two hit methods put
the cross-section it presents to the trace and the optical power of the
surface into a single variable, and with the curvature out of the plane
that variable had to be zero - the section really is a straight line -
so the power went with it. What came out was a mirror that focused in
both planes when the curvature was in the plane of the trace, and in
neither when it was not. The one function that knew the difference,
cyl_refl_defl_angle, was never called by anything.

The theory is Siegman, Lasers, Table 15.1, and the entries are written
out again here rather than imported, so that the check is against the
book and not against gtrace's reading of it. For a surface of radius R
at incidence theta, in the reduced-slope convention:

  (d) reflection      C_x = -2 n1 / (R cos t)      C_y = -2 n1 cos t / R
  (f) refraction, in the plane of incidence
        A = cos t2 / cos t1,  D = cos t1 / cos t2,
        C = (n2 cos t2 - n1 cos t1) / (R cos t1 cos t2)
  (g) refraction, perpendicular to it
        A = D = 1,  C = (n2 cos t2 - n1 cos t1) / R

A cylinder is then the surface that presents R to one plane and no
curvature to the other, which is 1/R -> 0 in the entry for that plane.
Note what that does *not* say: only for reflection does the flat plane
become the identity. A tilted flat interface still has A and D, because
the beam is wider on one side of it than the other, and every interface
carries the index change. Losing that was the second half of the same
bug - the transmission matrices were a copy of the spherical ones and
had never been told which plane was which.

The last section is the physical statement the rest is for: send a
Gaussian beam at a cylindrical mirror and the waist moves in one plane
and not the other, and the focal lengths a spherical mirror would have
in the two planes differ by cos^2(theta), which at 45 degrees is a
factor of two.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import numpy as np

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.optics.geometric import refl_defl_angle, cyl_refl_defl_angle
from gtrace.unit import *

pi = np.pi

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

#{{{ Table 15.1, written out here

def siegman(theta1, n1, n2, invROC_x, invROC_y):
    '''
    The four matrices of Table 15.1, given the curvature each plane
    sees. Derived here from the book, independently of gtrace.
    '''
    c1 = np.cos(theta1)
    theta2 = np.arcsin(n1*np.sin(theta1)/n2)
    c2 = np.cos(theta2)

    Mrx = np.array([[1., 0.], [-2*n1*invROC_x/c1, 1.]])          # (d)
    Mry = np.array([[1., 0.], [-2*n1*invROC_y*c1, 1.]])          # (d)
    Mtx = np.array([[c2/c1, 0.],                                  # (f)
                    [(n2*c2 - n1*c1)*invROC_x/(c1*c2), c1/c2]])
    Mty = np.array([[1., 0.], [(n2*c2 - n1*c1)*invROC_y, 1.]])    # (g)
    return Mrx, Mry, Mtx, Mty

#}}}

#{{{ The surface matrices

print('--- the surface matrices against Table 15.1 ---')

ANGLES = [0.0, np.deg2rad(10.), np.deg2rad(20.), np.deg2rad(45.),
          np.deg2rad(60.)]
CURVES = [1./2.0, -1./3.0, 1./0.5]
INDICES = [(1.0, 1.45), (1.45, 1.0), (1.0, 2.0)]

def call(theta, n1, n2, invROC, cyl=None):
    '''
    Drive one surface. The normal points back along the beam, tilted by
    theta; the beam runs along +x.
    '''
    if cyl is None:
        r = refl_defl_angle(0.0, pi - theta, n1, n2, invROC=invROC)
    else:
        r = cyl_refl_defl_angle(0.0, pi - theta, n1, n2, invROC=invROC,
                                curve_direction=cyl)
    return r[2:]

def tir(theta, n1, n2):
    '''
    Whether the surface is past its critical angle, where there is no
    transmitted ray and the refraction matrices are not numbers. Both
    the spherical and the cylindrical entry point say NaN there, which
    is a comparison no assertion can make; the transmission checks skip
    these and the reflection checks below do not, since reflection is
    perfectly well defined past the critical angle.
    '''
    return n1*np.sin(theta)/n2 > 1.0

for theta in ANGLES:
    deg = np.rad2deg(theta)
    for n1, n2 in INDICES:
        if tir(theta, n1, n2):
            continue
        for c in CURVES:
            # A sphere: the same curvature to both planes.
            want = siegman(theta, n1, n2, c, c)
            got = call(theta, n1, n2, c)
            check('sphere  %4.1fdeg n=%.2f->%.2f R=%+5.2f' % (deg, n1, n2, 1/c),
                  all(close(g, w) for g, w in zip(got, want)))

            # 'h': curvature in the plane of the trace, which is x.
            want = siegman(theta, n1, n2, c, 0.0)
            got = call(theta, n1, n2, c, cyl='h')
            check("cyl 'h' %4.1fdeg n=%.2f->%.2f R=%+5.2f" % (deg, n1, n2, 1/c),
                  all(close(g, w) for g, w in zip(got, want)))

            # 'v': curvature out of the plane, which is y.
            want = siegman(theta, n1, n2, 0.0, c)
            got = call(theta, n1, n2, c, cyl='v')
            check("cyl 'v' %4.1fdeg n=%.2f->%.2f R=%+5.2f" % (deg, n1, n2, 1/c),
                  all(close(g, w) for g, w in zip(got, want)))

print('--- and what the curvature does not reach ---')

for theta in ANGLES:
    deg = np.rad2deg(theta)
    for n1, n2 in INDICES:
        if tir(theta, n1, n2):
            continue
        c = 1./2.0
        for cyl, flat in [('h', 1), ('v', 0)]:
            Mr = call(theta, n1, n2, c, cyl=cyl)[flat]
            Mt = call(theta, n1, n2, c, cyl=cyl)[2 + flat]
            # Reflection off the flat plane is the identity, and only
            # because the angle out equals the angle in.
            check("cyl '%s' %4.1fdeg refl of the flat plane is identity"
                  % (cyl, deg), close(Mr, np.eye(2)))
            # Refraction is not: no power, but the tilt scaling stays.
            c1 = np.cos(theta)
            c2 = np.cos(np.arcsin(n1*np.sin(theta)/n2))
            want = (np.array([[c2/c1, 0.], [0., c1/c2]]) if flat == 0
                    else np.eye(2))
            check("cyl '%s' %4.1fdeg n=%.2f->%.2f trans of the flat plane "
                  'keeps its A and D' % (cyl, deg, n1, n2), close(Mt, want),
                  '(A=%.6f D=%.6f)' % (Mt[0, 0], Mt[1, 1]))

print('--- every matrix is symplectic ---')

for theta in ANGLES:
    for n1, n2 in INDICES:
        for cyl in [None, 'h', 'v']:
            Ms = call(theta, n1, n2, 1./2.0, cyl=cyl)
            if tir(theta, n1, n2):
                Ms = Ms[:2]
            for M in Ms:
                check('det=1  %4.1fdeg n=%.2f->%.2f %s'
                      % (np.rad2deg(theta), n1, n2, cyl or 'sphere'),
                      close(np.linalg.det(M), 1.0))

print('--- past the critical angle there is still a reflection ---')

for theta, n1, n2 in [(np.deg2rad(45.), 1.45, 1.0), (np.deg2rad(60.), 1.45, 1.0)]:
    c = 1./2.0
    want = siegman(theta, n1, n2, c, 0.0)[:2]
    got = call(theta, n1, n2, c, cyl='h')[:2]
    check('%4.1fdeg n=%.2f->%.2f: reflection is a number and is right'
          % (np.rad2deg(theta), n1, n2),
          all(np.all(np.isfinite(g)) and close(g, w) for g, w in zip(got, want)))
    check('%4.1fdeg n=%.2f->%.2f: and transmission is not'
          % (np.rad2deg(theta), n1, n2),
          all(np.any(np.isnan(M)) for M in call(theta, n1, n2, c, cyl='h')[2:]))

print('--- a flat cylinder is a flat surface, whichever way it faces ---')

for theta in ANGLES:
    for n1, n2 in INDICES:
        if tir(theta, n1, n2):
            continue
        a = call(theta, n1, n2, 0.0, cyl='h')
        b = call(theta, n1, n2, 0.0, cyl='v')
        s = call(theta, n1, n2, 0.0)
        check('no curvature, no difference  %4.1fdeg n=%.2f->%.2f'
              % (np.rad2deg(theta), n1, n2),
              all(close(x, y) and close(x, z) for x, y, z in zip(a, b, s)))

#}}}

#{{{ The mirror itself

print('--- CyMirror against Mirror, plane by plane ---')

WL = 1064*nm
Q0 = gauss.Rw2q(np.inf, 1*mm)

def mirror(cls, theta, invROC_HR=1./2.0, invROC_AR=0.0, **kw):
    '''
    A substrate 1 m along +x from the beam origin, its HR facing back at
    an incidence of theta.
    '''
    return cls(HRcenter=[1.0, 0], normAngleHR=pi - theta,
               diameter=10*cm, thickness=2*cm, wedgeAngle=0.0,
               inv_ROC_HR=invROC_HR, inv_ROC_AR=invROC_AR,
               Refl_HR=0.5, Trans_HR=0.5, Refl_AR=0.5, Trans_AR=0.5,
               n=1.45, name='M', **kw)

def probe():
    return beam.GaussianBeam(q0=Q0, wl=WL, pos=[0.0, 0.0], dirAngle=0.0)

# The probe runs down the axis and the apex of every one of these
# surfaces is at [1, 0], so the sphere and both cylinders are hit at the
# same point after the same distance. That is what makes these
# comparisons exact rather than approximate: only the matrices differ.
for deg in [0., 10., 20., 45.]:
    theta = np.deg2rad(deg)
    sph = mirror(opt.Mirror, theta).hitFromHR(probe())['r1']
    cyh = mirror(opt.CyMirror, theta, curve_direction='h').hitFromHR(probe())['r1']
    cyv = mirror(opt.CyMirror, theta, curve_direction='v').hitFromHR(probe())['r1']

    at_hr = probe()
    at_hr.propagate(1.0)

    check('%4.1fdeg  h focuses in x exactly as a sphere does' % deg,
          close(cyh.qx, sph.qx), '(%s)' % cyh.qx)
    check('%4.1fdeg  h leaves y as it arrived' % deg,
          close(cyh.qy, at_hr.qy), '(%s)' % cyh.qy)
    check('%4.1fdeg  v focuses in y exactly as a sphere does' % deg,
          close(cyv.qy, sph.qy), '(%s)' % cyv.qy)
    check('%4.1fdeg  v leaves x as it arrived' % deg,
          close(cyv.qx, at_hr.qx), '(%s)' % cyv.qx)
    # Not a swap: a sphere focuses harder in the plane of incidence
    # than out of it, so 'h' and 'v' of the same radius do different
    # things to their respective planes. Only at normal incidence do
    # the two coincide.
    check('%4.1fdeg  h and v differ unless the incidence is normal' % deg,
          close(cyh.qx, cyv.qy) == (deg == 0.),
          '(qx_h=%.6f qy_v=%.6f)' % (cyh.qx.real, cyv.qy.real))

print('--- at normal incidence the two planes cannot be told apart ---')

for cyl in ['h', 'v']:
    m = mirror(opt.CyMirror, 0.0, curve_direction=cyl)
    r = m.hitFromHR(probe())['r1']
    s = mirror(opt.Mirror, 0.0).hitFromHR(probe())['r1']
    at_hr = probe()
    at_hr.propagate(1.0)
    powered, flat = (r.qx, r.qy) if cyl == 'h' else (r.qy, r.qx)
    check("'%s' at 0deg: the powered plane is the sphere's" % cyl,
          close(powered, s.qx), '(%s)' % powered)
    # 'Unfocused' is not 'collimated': the beam arrives at the mirror
    # already diverging from a waist a metre back, and a plane surface
    # returns it exactly as it came.
    check("'%s' at 0deg: the flat plane is returned untouched" % cyl,
          close(flat, at_hr.qx),
          '(R=%.6f, was %.6f)' % (gauss.q2R(flat), gauss.q2R(at_hr.qx)))

print('--- the focal lengths differ by cos^2(theta) ---')

def focal(q_before, q_after, n=1.0):
    '''
    The focal length implied by the change in 1/q across a thin element.
    1/q_after - 1/q_before = -1/f.
    '''
    return -1.0/((1.0/q_after) - (1.0/q_before)).real

for deg in [20., 45., 60.]:
    theta = np.deg2rad(deg)
    R = 2.0
    at_hr = probe()
    at_hr.propagate(1.0)

    h = mirror(opt.CyMirror, theta, invROC_HR=1./R,
               curve_direction='h').hitFromHR(probe())['r1']
    v = mirror(opt.CyMirror, theta, invROC_HR=1./R,
               curve_direction='v').hitFromHR(probe())['r1']

    f_h = focal(at_hr.qx, h.qx)
    f_v = focal(at_hr.qy, v.qy)
    check('%4.1fdeg  h: f = R cos(t)/2 = %.6f' % (deg, R*np.cos(theta)/2),
          close(f_h, R*np.cos(theta)/2, tol=1e-9), '(%.9f)' % f_h)
    check('%4.1fdeg  v: f = R/(2 cos(t)) = %.6f' % (deg, R/(2*np.cos(theta))),
          close(f_v, R/(2*np.cos(theta)), tol=1e-9), '(%.9f)' % f_v)
    check('%4.1fdeg  and their ratio is cos^2(t) = %.6f'
          % (deg, np.cos(theta)**2), close(f_h/f_v, np.cos(theta)**2, tol=1e-9),
          '(%.9f)' % (f_h/f_v))

print('--- the untouched plane really is untouched ---')

for deg in [0., 20., 45.]:
    theta = np.deg2rad(deg)
    at_hr = probe()
    at_hr.propagate(1.0)
    for cyl, powered in [('h', 'x'), ('v', 'y')]:
        r = mirror(opt.CyMirror, theta,
                   curve_direction=cyl).hitFromHR(probe())['r1']
        flat_q = r.qy if cyl == 'h' else r.qx
        was = at_hr.qy if cyl == 'h' else at_hr.qx
        check("%4.1fdeg  '%s': q in the flat plane is the q that arrived"
              % (deg, cyl), close(flat_q, was), '(%s)' % flat_q)

print('--- transmission: the flat plane keeps the tilt, not the power ---')

# This is the half of the bug that a check on reflection alone would
# miss. A surface with no power still has A and D whenever it is tilted,
# because the beam is wider on one side of the interface than the other,
# and it still carries the index change. Across an interface whose
# matrix is diagonal, gtrace's q scales by exactly (n2/n1)*A^2, so the
# claim can be tested to the last bit rather than by eye.
N = 1.45

for deg in [0., 20., 45.]:
    theta = np.deg2rad(deg)
    theta2 = np.arcsin(np.sin(theta)/N)

    for cyl in ['h', 'v']:
        m = mirror(opt.CyMirror, theta, invROC_HR=1./2.0, invROC_AR=1./3.0,
                   curve_direction=cyl)
        bs = m.hitFromHR(probe(), order=2)
        at_hr = probe()
        at_hr.propagate(bs['input'].length)
        inside = bs['s1'].copy()
        inside.propagate(bs['s1'].length)

        flat_in = bs['s1'].qy if cyl == 'h' else bs['s1'].qx
        flat_at_ar = inside.qy if cyl == 'h' else inside.qx
        flat_out = bs['t1'].qy if cyl == 'h' else bs['t1'].qx
        was = at_hr.qy if cyl == 'h' else at_hr.qx

        # Sagittal: A = 1, so only the index shows. Tangential: the tilt
        # scaling as well, and the wedge is zero so the AR undoes it.
        A_in = 1.0 if cyl == 'h' else np.cos(theta2)/np.cos(theta)
        A_out = 1.0 if cyl == 'h' else np.cos(theta)/np.cos(theta2)

        check("%4.1fdeg  '%s': entering, the flat plane scales by n*A^2 "
              '= %.9f' % (deg, cyl, N*A_in**2),
              close(flat_in, N*A_in**2*was, tol=1e-12),
              '(%.9f)' % (flat_in/was).real)
        check("%4.1fdeg  '%s': leaving, by A^2/n = %.9f"
              % (deg, cyl, A_out**2/N),
              close(flat_out, A_out**2/N*flat_at_ar, tol=1e-12),
              '(%.9f)' % (flat_out/flat_at_ar).real)
        # End to end, from the q that arrived: in, across, out. Nothing
        # here is a curvature, which is the whole point - the only way a
        # 1/R could reach this plane is the bug.
        want = A_out**2/N*(N*A_in**2*was + bs['s1'].length)
        check("%4.1fdeg  '%s': in, across the glass, out - and no 1/R "
              'anywhere' % (deg, cyl), close(flat_out, want, tol=1e-12),
              '(%s)' % flat_out)

#}}}

#{{{ The spherical path is untouched

print('--- Mirror is exactly what it was ---')

#: Everything refl_defl_angle returned, for five surfaces, captured from
#: the implementation as it stood before the two entry points were given
#: a shared body. Pinned rather than recomputed: the spherical path is
#: the one every existing layout depends on, and 'we did not move it'
#: is a claim about the old numbers, not about the new ones agreeing
#: with the book a second time. Includes a surface past its critical
#: angle, where the transmission was NaN then and must be NaN now, and
#: a 7 km radius at 8 degrees, which is a KAGRA arm mirror.
REFERENCE = [
    (0.0, 1.0, 1.45, 0.5,
     [3.141592653589793, 0.0,
      1.0, 0.0, -1.0, 1.0,
      1.0, 0.0, -1.0, 1.0,
      1.0, 0.0, 0.22499999999999998, 1.0,
      1.0, 0.0, 0.22499999999999998, 1.0]),
    (0.3490658503988659, 1.0, 1.45, 0.5,
     [2.443460952792061, 6.172239321845366,
      1.0, 0.0, -1.0641777724759123, 1.0,
      1.0, 0.0, -0.9396926207859083, 1.0,
      1.034150050038463, 0.0, 0.2570108165904961, 0.9669776643755005,
      1.0, 0.0, 0.234696488441774, 1.0]),
    (0.7853981633974483, 1.0, 1.45, -0.3333333333333333,
     [1.5707963267948966, 6.007194402085411,
      1.0, 0.0, 0.9428090415820632, 1.0,
      1.0, 0.0, 0.4714045207910317, 1.0,
      1.2346561234460796, 0.0, -0.3017261774725364, 0.8099421215430213,
      1.0, 0.0, -0.18626403631022284, 1.0]),
    (1.0471975511965976, 1.45, 1.0, 2.0,
     [1.0471975511965983, float('nan'),
      1.0, 0.0, -11.599999999999993, 1.0,
      1.0, 0.0, -2.9000000000000017, 1.0,
      float('nan'), 0.0, float('nan'), float('nan'),
      1.0, 0.0, float('nan'), 1.0]),
    (0.13962634015954636, 1.0, 1.45, 0.00014285714285714287,
     [2.8623399732707, 6.23968840080142,
      1.0, 0.0, -0.00028852216357674805, 1.0,
      1.0, 0.0, -0.000282933733926163, 1.0,
      1.0051653227776798, 0.0, 6.565881266908986e-05, 0.994861220676211,
      1.0, 0.0, 6.471963707719318e-05, 1.0]),
]

for theta, n1, n2, c in [(r[0], r[1], r[2], r[3]) for r in REFERENCE]:
    want = [r[4] for r in REFERENCE
            if (r[0], r[1], r[2], r[3]) == (theta, n1, n2, c)][0]
    r = refl_defl_angle(0.0, pi - theta, n1, n2, invROC=c)
    got = [float(r[0]), float(r[1])] + [float(x) for M in r[2:] for x in M.ravel()]
    check('%5.1fdeg n=%.2f->%.2f R=%+8.2f: every bit as before'
          % (np.rad2deg(theta), n1, n2, 1/c),
          np.array_equal(np.array(got), np.array(want), equal_nan=True),
          '(%d values)' % len(got))

# And, separately, that those numbers are the book's.
rng = np.random.default_rng(20260804)
same = True
worst = 0.0
n = 0
while n < 2000:
    theta = rng.uniform(0., np.deg2rad(80.))
    n1, n2 = rng.uniform(1., 2.), rng.uniform(1., 2.)
    if tir(theta, n1, n2):
        continue
    n += 1
    c = rng.uniform(-1., 1.)
    a = refl_defl_angle(0.0, pi - theta, n1, n2, invROC=c)
    b = siegman(theta, n1, n2, c, c)
    for x, y in zip(a[2:], b):
        # Relative: an element of a steeply curved surface at grazing
        # incidence is large, and the last bit of it is not 1e-14.
        d = float(np.max(np.abs(x - y)/np.maximum(np.abs(y), 1.0)))
        worst = max(worst, d)
        same = same and d <= 1e-12
check('2000 random spherical surfaces match Table 15.1', same,
      '(worst %.2e relative)' % worst)

#}}}

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
