'''
The geometry kernel: the routines every hit test and every drawing goes
through.

Not a GUI check, but it belongs to the same harness for the reason
verify_surfaces.py gives: it wants counted assertions, and this is where
those live.

These four routines are written on plain floats rather than on numpy
arrays, because a trace calls them tens of thousands of times on
2-vectors, where a numpy call costs more than the arithmetic inside it.
Written that way, two things can break without anything saying so:

- ``vector_rotation_2D`` takes a single vector on its fast path, and
  arrays of shape (2, N) - a whole beam outline, or a whole arc - on the
  slow one. Sending an outline down the fast path would draw nonsense
  while every number in the model stayed right.
- ``_surface_matrices`` must produce nan transmission past the critical
  angle. That nan is how a caller tells total internal reflection from
  an ordinary refraction, and ``math.asin`` raises where the numpy call
  it replaces returned the nan by itself.

So both are checked here directly, along with the two intersection
routines against geometry worked out independently of them.
'''

import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import numpy as np

import gtrace.optics.geometric as geo

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


rng = np.random.default_rng(20260821)


print('--- vector_rotation_2D: one vector, and a whole outline ---')

def rotate_by_matrix(vect, angle):
    '''
    The rotation written as a matrix product, which is what the routine
    did for everything before it grew a fast path for single vectors.
    '''
    M = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return np.dot(M, np.array(vect))


#: (2, 60) is a drawn arc and (2, 100) a drawn beam outline. Those are
#: the shapes the drawing code passes, and the ones a scalar-only
#: rewrite would silently mangle.
for shape in [(2,), (2, 1), (2, 3), (2, 60), (2, 100)]:
    worst = 0.0
    shape_ok = True
    for _ in range(200):
        v = rng.normal(size=shape)
        a = float(rng.uniform(-10, 10))
        got = geo.vector_rotation_2D(v, a)
        want = rotate_by_matrix(v, a)
        if got.shape != want.shape:
            shape_ok = False
            break
        worst = max(worst, float(np.max(np.abs(got - want))))
    check('shape %-9s comes back the same shape' % str(shape), shape_ok,
          '' if shape_ok else 'got %s' % (got.shape,))
    check('shape %-9s turns as the matrix does' % str(shape),
          shape_ok and worst <= 1e-15, '(worst %.3e)' % worst)

#: What callers actually hand it. A list and a tuple are not arrays, and
#: an integer array is not a float one; all reach the fast path through
#: plain indexing.
for label, v in [('a list', [1.0, 2.0]),
                 ('a tuple', (1.0, 2.0)),
                 ('an array', np.array([1.0, 2.0])),
                 ('an integer array', np.array([1, 2])),
                 ('a list of ints', [1, 2])]:
    got = geo.vector_rotation_2D(v, 0.4)
    want = rotate_by_matrix(v, 0.4)
    check('%s is accepted' % label,
          got.shape == (2,) and np.allclose(got, want, atol=1e-15),
          str(got))

#: Two rows written as lists rather than as an array. This is the case
#: the fast path has to decline: multiplying a Python list by a float
#: repeats it or refuses, where multiplying a row of an array scales it.
#: An array of shape (2, N) would come out right either way, so it is
#: this one that holds the test apart.
for label, v in [('two rows as lists', [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                 ('two rows as tuples', ((1.0, 2.0), (3.0, 4.0)))]:
    try:
        got = geo.vector_rotation_2D(v, 0.4)
        want = rotate_by_matrix(v, 0.4)
        ok = got.shape == want.shape and np.allclose(got, want, atol=1e-15)
        detail = str(got.shape)
    except Exception as exc:
        ok = False
        detail = '%s: %s' % (type(exc).__name__, exc)
    check('%s goes through the matrix' % label, ok, detail)

#A turn of zero leaves a vector alone, and a turn of 2*pi brings it back.
v = np.array([0.3, -0.7])
check('turning by nothing changes nothing',
      np.array_equal(geo.vector_rotation_2D(v, 0.0), v))
check('turning by a full circle comes back',
      np.allclose(geo.vector_rotation_2D(v, 2*np.pi), v, atol=1e-15),
      str(geo.vector_rotation_2D(v, 2*np.pi)))
check('turning twice by a is turning once by 2a',
      np.allclose(geo.vector_rotation_2D(geo.vector_rotation_2D(v, 0.6), 0.6),
                  geo.vector_rotation_2D(v, 1.2), atol=1e-15))
#The length of a vector cannot change when it is turned.
check('and the length is untouched',
      all(abs(np.linalg.norm(geo.vector_rotation_2D(v, a))
              - np.linalg.norm(v)) <= 1e-15
          for a in np.linspace(0, 2*np.pi, 37)))


print()
print('--- _surface_matrices: past the critical angle there is no ray ---')

n1, n2 = 1.45, 1.0
critical = math.asin(n2/n1)
for factor in [0.1, 0.5, 0.9, 0.99, 0.999, 1.001, 1.05, 1.3]:
    theta = critical*factor
    beyond = theta > critical
    Mrx, Mry, Mtx, Mty = geo._surface_matrices(theta, n1, n2, 1/0.5, 1/0.5)
    t_nan = bool(np.isnan(Mtx).any() and np.isnan(Mty).any())
    check('theta = %.3f x critical: transmission is %s'
          % (factor, 'nan' if beyond else 'a number'),
          t_nan == beyond, '(nan=%s)' % t_nan)
    #Reflection is defined whatever the angle: past the critical angle
    #it is the only thing left.
    check('theta = %.3f x critical: reflection is still a number' % factor,
          bool(np.isfinite(Mrx).all() and np.isfinite(Mry).all()))

#Below the critical angle there is nothing special about the matrices,
#and every one of them has determinant 1 in the reduced-slope
#convention the module uses.
for theta in [0.0, 0.2, 0.5, critical*0.9]:
    for invROC in [0.0, 1/7000.0, 1/0.45, -1/0.30]:
        Ms = geo._surface_matrices(theta, n1, n2, invROC, invROC)
        dets = [float(np.linalg.det(M)) for M in Ms]
        check('theta=%.3f invROC=%.4g: every matrix has determinant 1'
              % (theta, invROC),
              all(abs(d - 1.0) <= 1e-12 for d in dets),
              str(['%.15f' % d for d in dets]))

#Going in and coming back out of the same interface undoes itself.
for theta in [0.0, 0.3, 0.6]:
    Mt_in = geo._surface_matrices(theta, 1.0, 1.45, 0.0, 0.0)[2]
    theta2 = math.asin(math.sin(theta)/1.45)
    Mt_out = geo._surface_matrices(theta2, 1.45, 1.0, 0.0, 0.0)[2]
    prod = np.dot(Mt_out, Mt_in)
    check('a flat interface crossed both ways is the identity (theta=%.1f)'
          % theta,
          np.allclose(prod, np.eye(2), atol=1e-12), str(prod.ravel()))


print()
print('--- _surface_angles: reflection is the mirror of incidence ---')

for beam_deg in range(0, 360, 17):
    for norm_deg in range(0, 360, 23):
        b = math.radians(beam_deg)
        n = math.radians(norm_deg)
        inc, refl = geo._surface_angles(b, n)
        #The angle of incidence is measured from the normal, and the
        #reflected ray leaves on the other side of it by the same
        #amount. Both are put back into a circle before comparing,
        #since the routine returns them in different ranges.
        got = (refl - n) % (2*math.pi)
        want = (-inc) % (2*math.pi)
        if abs(got - want) > math.pi:
            got = got - 2*math.pi*(1 if got > want else -1)
        check('beam %3d deg, normal %3d deg: reflection mirrors incidence'
              % (beam_deg, norm_deg),
              abs(got - want) <= 1e-12,
              '(incidence %.6f)' % inc)


print()
print('--- line_plane_intersection, against the line worked out here ---')

#A line and a plane, both written down here. The point where they meet
#is found by putting the parametric line into the plane equation, which
#shares no step with what the routine does.
for _ in range(300):
    c = rng.normal(scale=0.5, size=2)
    na = float(rng.uniform(0, 2*np.pi))
    nv = np.array([np.cos(na), np.sin(na)])
    diameter = float(rng.uniform(0.05, 1.0))
    #Aim at a chosen point of the plane from a chosen distance in front
    #of it, so that the case is a hit by construction.
    offset = float(rng.uniform(-0.45, 0.45))*diameter
    along = np.array([-nv[1], nv[0]])
    target = c + offset*along
    dist = float(rng.uniform(0.05, 3.0))
    #Somewhere on the front side, meaning the side the normal points to.
    spread = float(rng.uniform(-0.6, 0.6))
    start = target + dist*nv + spread*along
    d = target - start
    d = d/np.linalg.norm(d)

    ans = geo.line_plane_intersection(start, d, c, nv, diameter)
    hit_expected = abs(offset) <= diameter/2.0
    ok = ans['isHit'] == hit_expected
    ok = ok and np.allclose(ans['Intersection Point'], target, atol=1e-12)
    ok = ok and abs(ans['distance'] - np.linalg.norm(target - start)) <= 1e-12
    ok = ok and abs(abs(ans['distance from center']) - abs(offset)) <= 1e-12
    if not ok:
        check('a line aimed at a point of the plane lands on it', False,
              '(%s, wanted %s at %s)' % (ans, hit_expected, target))
        break
else:
    check('300 lines aimed at a point of a plane all land on it', True)

#A line running behind the plane, or away from it, is not a hit.
c = np.array([0.0, 0.0])
nv = np.array([1.0, 0.0])
check('a line travelling away from the plane misses',
      not geo.line_plane_intersection([1.0, 0.0], [1.0, 0.0], c, nv, 1.0)['isHit'])
check('a line coming at the back of the plane misses',
      not geo.line_plane_intersection([-1.0, 0.0], [1.0, 0.0], c, nv, 1.0)['isHit'])
check('a line parallel to the plane misses',
      not geo.line_plane_intersection([1.0, -1.0], [0.0, 1.0], c, nv, 1.0)['isHit'])
check('a line passing outside the aperture misses',
      not geo.line_plane_intersection([1.0, 0.9], [-1.0, 0.0], c, nv, 1.0)['isHit'])
check('and just inside it does not',
      geo.line_plane_intersection([1.0, 0.49], [-1.0, 0.0], c, nv, 1.0)['isHit'])


print()
print('--- line_arc_intersection, against the circle worked out here ---')

#The arc is part of a circle, so wherever the routine says the line met
#it, that point has to be on the circle, on the line, and inside the
#chord. The circle is rebuilt here from the chord and the curvature
#rather than taken from the routine.
for invROC in [1/7000.0, 1/0.45, -1/0.45, 1/0.10, -1/0.10]:
    ROC = 1.0/invROC
    diameter = 0.0254
    c = np.array([0.31, -0.22])
    na = 0.7
    nv = np.array([np.cos(na), np.sin(na)])
    arc_centre = c + nv*(ROC*math.cos(math.asin(diameter/(2*ROC))))
    along = np.array([-nv[1], nv[0]])

    worst_circle = 0.0
    worst_line = 0.0
    hits = 0
    for _ in range(200):
        offset = float(rng.uniform(-0.6, 0.6))*diameter
        target = c + offset*along
        start = target + float(rng.uniform(0.05, 2.0))*nv \
            + float(rng.uniform(-0.5, 0.5))*along
        d = target - start
        d = d/np.linalg.norm(d)
        ans = geo.line_arc_intersection(start, d, c, nv, invROC, diameter)
        if not ans.get('isHit'):
            continue
        hits += 1
        p = ans['Intersection Point']
        #On the circle.
        worst_circle = max(worst_circle,
                           abs(np.linalg.norm(p - arc_centre) - abs(ROC)))
        #On the line, and the reported distance is the distance to it.
        t = float(np.dot(p - start, d))
        worst_line = max(worst_line, float(np.linalg.norm(start + t*d - p)))
        worst_line = max(worst_line, abs(ans['distance'] - t))

    #A radius of 7000 m carries its own rounding: the point is built by
    #adding and subtracting two quantities of that size, so it can only
    #be as good as one part in 1e16 of the radius.
    tol = max(1e-12, abs(ROC)*1e-13)
    check('invROC=%9.5g: the point found is on the circle' % invROC,
          hits > 0 and worst_circle <= tol,
          '(%d hits, worst %.3e, tolerance %.3e)'
          % (hits, worst_circle, tol))
    check('invROC=%9.5g: and on the line, at the distance reported' % invROC,
          hits > 0 and worst_line <= tol,
          '(worst %.3e)' % worst_line)

    #The normal returned points from the surface towards the centre of
    #curvature for a concave face, and away from it for a convex one.
    ans = geo.line_arc_intersection(c + 1.0*nv, -nv, c, nv, invROC, diameter)
    if ans.get('isHit'):
        p = ans['Intersection Point']
        towards = arc_centre - p
        towards = towards/np.linalg.norm(towards)
        want = towards if ROC > 0 else -towards
        check('invROC=%9.5g: the local normal points the right way' % invROC,
              np.allclose(ans['localNormVect'], want, atol=1e-9),
              str(ans['localNormVect']))
        check('invROC=%9.5g: and the angle agrees with the vector' % invROC,
              abs((math.atan2(ans['localNormVect'][1],
                              ans['localNormVect'][0]) % (2*math.pi))
                  - ans['localNormAngle']) <= 1e-12)

#A face flat enough is handled as a plane, and has to agree with the
#plane routine exactly.
flat = geo.line_arc_intersection([1.0, 0.02], [-1.0, 0.0], [0.0, 0.0],
                                 [1.0, 0.0], 1e-7, 0.1)
plane = geo.line_plane_intersection([1.0, 0.02], [-1.0, 0.0], [0.0, 0.0],
                                    [1.0, 0.0], 0.1)
check('a nearly flat arc gives what the plane gives',
      flat['isHit'] == plane['isHit']
      and np.array_equal(flat['Intersection Point'],
                         plane['Intersection Point'])
      and flat['distance'] == plane['distance'],
      str(flat['Intersection Point']))
check('and carries a local normal along the chord normal',
      np.allclose(flat['localNormVect'], [1.0, 0.0], atol=1e-15),
      str(flat['localNormVect']))

#A line that goes nowhere reaches nothing, rather than dividing by zero.
check('a line with no direction hits nothing (plane)',
      not geo.line_plane_intersection([1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                      [1.0, 0.0], 1.0)['isHit'])
check('a line with no direction hits nothing (arc)',
      not geo.line_arc_intersection([1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                    [1.0, 0.0], 1/0.45, 0.1)['isHit'])

#The inputs may be any sequence of numbers, which is what lets the
#optics classes pass arrays and the suites pass lists.
forms = [([1.0, 0.02], [-1.0, 0.0]),
         ((1.0, 0.02), (-1.0, 0.0)),
         (np.array([1.0, 0.02]), np.array([-1.0, 0.0])),
         (np.array([1, 0]), np.array([-1, 0]))]
answers = [geo.line_plane_intersection(p, d, [0.0, 0.0], [1.0, 0.0], 1.0)
           for p, d in forms]
check('a list, a tuple, a float array and an integer array all work',
      all(a['isHit'] for a in answers)
      and all(np.allclose(a['Intersection Point'],
                          answers[0]['Intersection Point'], atol=1e-15)
              for a in answers[:3]))


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
