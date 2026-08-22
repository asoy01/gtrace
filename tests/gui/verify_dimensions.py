'''
Dimensions: the substrate geometry they rest on, and the model.

A dimension is two points and a name. What makes it worth checking is
the one question it asks of the optics - does this span run inside a
substrate - because that is what decides whether an optical distance is
written next to the physical one, and getting it wrong would put a
number on the drawing that is not a distance of anything.

The test behind that question is Optics.contains_segment, and the way it
is built is worth stating, since the checks below are shaped by it.
isHit() reports a surface only when it is approached from outside: it
refuses any face the ray is leaving through. So from inside a substrate
isHit() finds nothing at all in any direction, and that is what the
inside test rests on - not on a second description of where the faces
are, which is exactly the kind of duplicate that put the AR surface a
sagitta out of place until 2026-08-03.

The concave case is the one to watch. The hollow of a concave face is
enclosed by the substrate on three sides, so a test that only asked how
far the material reaches along a line would call it inside. It is not:
it is air, and a span across it is a span through air.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json

import numpy as np

import gtrace.beam as beam
import gtrace.draw as draw
import gtrace.draw.renderer as renderer
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.optcomp import _ProbeRay
from gtrace.layout import (OpticalLayout, TraceRules, Dimension, EditError,
                           UNDO_DEPTH, dimension_to_dict, dimension_from_dict,
                           optic_snap_points)
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

def refused(layout, msg, why):
    '''
    Check that an edit message is rejected without side effects.
    '''
    before = json.dumps(layout.to_dict())
    try:
        layout.apply_edit(msg)
    except EditError as e:
        check('refuses %s' % why, True, '(%s)' % str(e)[:60])
    except Exception as e:
        check('refuses %s' % why, False,
              '(raised %s instead)' % type(e).__name__)
        return
    else:
        check('refuses %s' % why, False, '(it went through)')
        return
    check('  and leaves the layout alone',
          json.dumps(layout.to_dict()) == before)

#: One of each shape of substrate the geometry has to cope with. The
#: sign of each curvature is what matters: a face that bulges out has
#: its chord inside the glass, a face that dishes in has it in air.
def cases():
    return [
        ('flat mirror',
         opt.Mirror(HRcenter=[0.5, 0.2], normAngleHR=np.deg2rad(35),
                    diameter=10*cm, thickness=5*cm, inv_ROC_HR=0.0,
                    wedgeAngle=np.deg2rad(0.25), n=1.45, name='flat')),
        ('concave HR, convex AR',
         opt.Mirror(HRcenter=[0.5, 0.2], normAngleHR=np.deg2rad(35),
                    diameter=10*cm, thickness=5*cm,
                    wedgeAngle=np.deg2rad(0.25),
                    inv_ROC_HR=1/2.0, inv_ROC_AR=-1/3.0, n=1.45,
                    name='curved')),
        ('biconvex lens',
         opt.Lens(f=500*mm, HRcenter=[0.2, 0.1], normAngleHR=np.pi,
                  name='lens')),
        ('biconcave lens',
         opt.Lens(f=-100*mm, thickness=3*mm, HRcenter=[0.2, 0.1],
                  normAngleHR=np.pi, name='lensneg')),
        ('cylinder, curved in plane',
         opt.CyMirror(HRcenter=[0.5, 0.2], normAngleHR=np.deg2rad(35),
                      diameter=10*cm, thickness=5*cm, inv_ROC_HR=1/2.0,
                      curve_direction='h', name='cyh')),
        ('cylinder, curved across',
         opt.CyMirror(HRcenter=[0.5, 0.2], normAngleHR=np.deg2rad(35),
                      diameter=10*cm, thickness=5*cm, inv_ROC_HR=1/2.0,
                      curve_direction='v', name='cyv')),
    ]

print('--- get_corners agrees with get_side_info ---')

# The two describe the same substrate: get_side_info for hit testing,
# get_corners for anything pointing at it. They are written out
# separately - the wedge and the sagitta appear in both - so they are
# held to each other here rather than left to drift.
for label, o in cases():
    c = o.get_corners()
    check('%s: four corners' % label, len(c) == 4, str(len(c)))
    sides = o.get_side_info()
    # Side 1 joins the corners at one end of the two faces, side 2 the
    # other: corners come round the substrate, so 0-3 and 1-2.
    for (i, j), (ctr, nv, ln) in zip(((0, 3), (1, 2)), sides):
        mid = (np.asarray(c[i]) + np.asarray(c[j])) / 2
        check('%s: side centre %d-%d' % (label, i, j),
              np.linalg.norm(mid - np.asarray(ctr)) < 1e-15,
              '(%.2e)' % np.linalg.norm(mid - np.asarray(ctr)))
        got = np.linalg.norm(np.asarray(c[i]) - np.asarray(c[j]))
        check('%s: side length %d-%d' % (label, i, j),
              abs(got - ln) < 1e-15, '(%.2e)' % abs(got - ln))

print('--- isHit reads nothing of a ray but where it is and where it goes ---')

# contains_segment asks isHit geometry questions with a bare ray rather
# than a GaussianBeam, since none of a beam's physics takes part in the
# answer. That is only true while isHit reads nothing else, so it is
# checked rather than assumed.
for label, o in cases():
    pos = np.asarray(o.center, dtype=float) - np.asarray(o.normVectHR,
                                                         dtype=float)*0.3
    dirv = np.asarray(o.normVectHR, dtype=float) * -1
    real = o.isHit(beam.GaussianBeam(pos=pos, dirVect=dirv, wl=1064*nm,
                                     P=1.0, q0=gauss.Rw2q(np.inf, 1*mm)))
    bare = o.isHit(_ProbeRay(pos, dirv))
    check('%s: same answer either way' % label,
          real['isHit'] == bare['isHit'] and real['face'] == bare['face']
          and abs(real['distance'] - bare['distance']) < 1e-15,
          '%s/%s' % (real['face'], bare['face']))

print('--- contains_segment: the cases a measurement actually meets ---')

for label, o in cases():
    HR = np.asarray(o.HRcenter, dtype=float)
    AR = np.asarray(o.ARcenter, dtype=float)
    nv = np.asarray(o.normVectHR, dtype=float)
    ctr = np.asarray(o.center, dtype=float)

    # The measurement this whole feature exists for: the optical
    # thickness of a substrate, taken between the apexes of its faces.
    # Both ends sit exactly on a surface, which is the awkward part.
    check('%s: HR apex to AR apex is inside' % label,
          o.contains_segment(HR, AR))
    check('%s: and the other way round' % label,
          o.contains_segment(AR, HR))
    check('%s: a short span about the middle is inside' % label,
          o.contains_segment(ctr - nv*1e-4, ctr + nv*1e-4))
    check('%s: a span in front of the face is not' % label,
          not o.contains_segment(HR + nv*0.1, HR + nv*0.2))
    check('%s: one that starts in front and ends inside is not' % label,
          not o.contains_segment(HR + nv*0.05, AR))
    check('%s: one that goes right through is not' % label,
          not o.contains_segment(HR + nv*0.1, AR - nv*0.1))
    check('%s: one nowhere near is not' % label,
          not o.contains_segment(HR + nv*1.0,
                                 HR + nv*1.0 + np.array([0.1, 0.1])))
    check('%s: a span of no length is not' % label,
          not o.contains_segment(ctr, ctr))

print('--- the hollow of a concave face is air, not glass ---')

# The chord of a face is the line joining its two corners. For a face
# that bulges out, the chord lies inside the glass; for one that dishes
# in, it lies in the hollow, which is air enclosed on three sides. The
# second is what a naive inside test gets wrong.
for label, o, face, concave in [
        ('concave HR, convex AR', None, 'HR', True),
        ('concave HR, convex AR', None, 'AR', False),
        ('biconvex lens', None, 'HR', False),
        ('biconvex lens', None, 'AR', False),
        ('biconcave lens', None, 'HR', True),
        ('biconcave lens', None, 'AR', True)]:
    o = dict(cases())[label]
    c = o.get_corners()
    a, b = (c[0], c[1]) if face == 'HR' else (c[2], c[3])
    got = o.contains_segment(a, b)
    check('%s: the %s chord is %s' % (label, face,
                                      'outside' if concave else 'inside'),
          got == (not concave), str(got))

print('--- Dimension.measure ---')

lens = dict(cases())['biconvex lens']
mirror = dict(cases())['flat mirror']
d = Dimension(lens.HRcenter, lens.ARcenter, name='D1')
m = d.measure([mirror, lens])
check('the length is the distance between the ends',
      abs(m['length'] - float(np.linalg.norm(np.asarray(lens.ARcenter)
                                             - np.asarray(lens.HRcenter))))
      < 1e-15, str(m['length']))
check('a span inside a substrate names it', m['inside'] == 'lens',
      str(m['inside']))
check('and reports its index', m['n'] == float(lens.n), str(m['n']))
check('the optical distance is n times the physical one',
      abs(m['optical'] - m['n']*m['length']) < 1e-15, str(m['optical']))

far = Dimension([2.0, 2.0], [3.0, 2.0], name='D2').measure([mirror, lens])
check('a span in the open has no optical distance',
      far['optical'] is None and far['inside'] is None and far['n'] is None,
      str(far))
check('but still has a length', abs(far['length'] - 1.0) < 1e-15,
      str(far['length']))
check('measuring against nothing is the same as measuring in the open',
      Dimension(lens.HRcenter, lens.ARcenter).measure()['optical'] is None)
check('a dimension of no length reports none of it',
      Dimension([1.0, 1.0], [1.0, 1.0]).measure([lens])['optical'] is None)

print('--- where the line is drawn ---')

# The offset carries the dimension line aside so that it can be read
# clear of whatever it measures. It is a choice about the drawing, so it
# must not touch what the measurement comes to.
span = Dimension([0.0, 0.0], [1.0, 0.0], name='D')
check('a fresh dimension has no offset', span.offset == 0.0)
a, b = span.line_ends()
check('so its line runs between its ends',
      np.allclose(a, [0.0, 0.0]) and np.allclose(b, [1.0, 0.0]))
check('the normal is to the left of the way it runs',
      np.allclose(span.normal, [0.0, 1.0]), str(list(span.normal)))
span.offset = 0.05
a, b = span.line_ends()
check('a positive offset carries the line to the left',
      np.allclose(a, [0.0, 0.05]) and np.allclose(b, [1.0, 0.05]),
      '%s %s' % (list(a), list(b)))
span.offset = -0.05
a, b = span.line_ends()
check('and a negative one to the right',
      np.allclose(a, [0.0, -0.05]) and np.allclose(b, [1.0, -0.05]),
      '%s %s' % (list(a), list(b)))
check('the length is unchanged by any of it', span.length == 1.0,
      str(span.length))
back = Dimension([1.0, 0.0], [0.0, 0.0], name='D')
check('reversing the ends reverses which side is which',
      np.allclose(back.normal, [0.0, -1.0]), str(list(back.normal)))
check('a dimension of no length has no normal to speak of',
      np.allclose(Dimension([0, 0], [0, 0]).normal, [0.0, 0.0]))

# The offset is a drawing choice, so it cannot decide whether a span
# runs inside a substrate: the span is still between p1 and p2.
lens0 = dict(cases())['biconvex lens']
inside = Dimension(lens0.HRcenter, lens0.ARcenter, name='D', offset=0.5)
check('a span inside a substrate stays inside however the line is drawn',
      inside.measure([lens0])['inside'] == 'lens',
      str(inside.measure([lens0])['inside']))

print('--- snap points ---')

pts = optic_snap_points(mirror)
kinds = [p['kind'] for p in pts]
check('four corners, three named points and two side middles',
      len(pts) == 9, str(len(pts)))
check('the corners come first', kinds[:4] == ['corner']*4, str(kinds))
check('then the two faces and the middle',
      kinds[4:7] == ['face', 'face', 'centre'], str(kinds[4:7]))
check('and the side middles last, so a named point at the same place wins',
      kinds[7:] == ['midpoint', 'midpoint'], str(kinds[7:]))
corners = [np.asarray(p['point']) for p in pts[:4]]
for i, c in enumerate(mirror.get_corners()):
    check('corner %d is the corner' % (i + 1),
          np.linalg.norm(corners[i] - np.asarray(c)) < 1e-15)
# The apex, not the chord centre: it is the point on the face, which is
# where a beam on the axis lands and what the panel calls the centre.
hr = [p for p in pts if p['label'].endswith(' HR')][0]
check('the HR point is the apex',
      np.linalg.norm(np.asarray(hr['point'])
                     - np.asarray(mirror.HRcenter)) < 1e-15,
      str(hr['point']))
# The two sides of the substrate are straight lines, so their middles
# are points on the drawing. Side 1 runs between corners 1 and 4, side 2
# between corners 2 and 3 - which is what makes them the sides rather
# than the faces.
mids = [np.asarray(p['point']) for p in pts if p['kind'] == 'midpoint']
for i, (a, b) in enumerate([(0, 3), (1, 2)]):
    want = (corners[a] + corners[b]) / 2.0
    check('side %d middle is halfway along that side' % (i + 1),
          np.linalg.norm(mids[i] - want) < 1e-15,
          '(%s vs %s)' % (mids[i], want))

# A curved face gets none. The middle of an arc is not a place anything
# is lined up on, and the middle of its chord is inside the glass with
# nothing drawn there. The apex is already offered, as 'HR'.
curved = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.deg2rad(135),
                    diameter=10*cm, thickness=5*cm,
                    wedgeAngle=np.deg2rad(0.25), inv_ROC_HR=1./(50*cm),
                    n=1.45, name='curved')
cpts = optic_snap_points(curved)
chord = np.asarray(curved.HRcenterC)
check('a curved face has a chord centre away from its apex',
      np.linalg.norm(chord - np.asarray(curved.HRcenter)) > 1e-4,
      '(%.6f mm apart)'
      % (np.linalg.norm(chord - np.asarray(curved.HRcenter))/mm))
check('and nothing is offered there',
      all(np.linalg.norm(np.asarray(q['point']) - chord) > 1e-6
          for q in cpts), str([q['label'] for q in cpts
                               if np.linalg.norm(np.asarray(q['point'])
                                                 - chord) <= 1e-6]))
check('the curved substrate still gets its two side middles',
      len([q for q in cpts if q['kind'] == 'midpoint']) == 2)

check('every point is named after its optics',
      all(p['optic'] == 'flat' and p['label'].startswith('flat ') for p in pts))
json.dumps(pts)
check('and the lot is JSON-clean', True)

print('--- registration ---')

def make_layout():
    b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                           pos=[0, 0], dirAngle=0, name='b0')
    M1 = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.deg2rad(135),
                    diameter=10*cm, thickness=5*cm,
                    wedgeAngle=np.deg2rad(0.25), Refl_HR=0.99, Trans_HR=0.01,
                    n=1.45, name='M1')
    L1 = opt.Lens(f=500*mm, HRcenter=[0.25, 0.0], normAngleHR=np.pi,
                  name='L1')
    lay = OpticalLayout(optics=[M1, L1], sources=[b0],
                        rules=TraceRules(order=4, power_threshold=1e-3),
                        name='dimensions')
    return lay, M1, L1

lay, M1, L1 = make_layout()
check('a fresh layout has no dimensions', lay.dimensions == [])
lay.add_dimension(Dimension([0, 0], [1, 0], name='D1'))
check('one can be registered', len(lay.dimensions) == 1)
check('and found by name', lay.get_dimension('D1').name == 'D1')
try:
    lay.get_dimension('nope')
    check('an unknown name raises', False)
except KeyError:
    check('an unknown name raises', True)

# Optics and dimensions share one namespace, because an edit message
# names its target and nothing else: a name that meant one thing in one
# message and another in the next would be a trap.
try:
    lay.add_dimension(Dimension([0, 0], [1, 1], name='M1'))
    check('a name an optics holds is refused', False)
except ValueError as e:
    check('a name an optics holds is refused', True, '(%s)' % str(e)[:50])
try:
    lay.add_optics(opt.Mirror(name='D1'))
    check('and a name a dimension holds is refused too', False)
except ValueError as e:
    check('and a name a dimension holds is refused too', True,
          '(%s)' % str(e)[:50])
check('a fresh name avoids both',
      lay.unique_dimension_name() == 'D2'
      and lay.unique_optics_name('M') == 'M2',
      '%s / %s' % (lay.unique_dimension_name(), lay.unique_optics_name('M')))

print('--- the edit protocol ---')

lay, M1, L1 = make_layout()
hr = [float(x) for x in np.asarray(L1.HRcenter)]
ar = [float(x) for x in np.asarray(L1.ARcenter)]

lay.trace()
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                'params': {'p1': hr, 'p2': ar}})
check('add registers it', [d.name for d in lay.dimensions] == ['D1'])
# A dimension is a note on the layout, not a part of it: no beam has
# moved, so the trace still stands.
check('and does not invalidate the trace', lay.beams is not None)
check('it can be undone', lay.can_undo)

lay.apply_edit({'op': 'add', 'type': 'Dimension',
                'params': {'p1': [0.0, 0.0], 'p2': [0.2, 0.0]}})
check('a nameless add is named for you',
      [d.name for d in lay.dimensions] == ['D1', 'D2'],
      str([d.name for d in lay.dimensions]))

refused(lay, {'op': 'add', 'type': 'Dimension', 'params': {'p1': [0, 0]}},
        'an add with one end')
refused(lay, {'op': 'add', 'type': 'Dimension',
              'params': {'p1': [0, 0], 'p2': [0, 0]}},
        'an add with both ends in one place')
refused(lay, {'op': 'add', 'type': 'Dimension',
              'params': {'p1': [0, 0], 'p2': [None, 1]}},
        'an end that is not a pair of numbers')
refused(lay, {'op': 'add', 'type': 'Dimension',
              'params': {'p1': [0, 0], 'p2': [float('inf'), 1]}},
        'an end at infinity')
refused(lay, {'op': 'add', 'type': 'Dimension',
              'params': {'p1': [0, 0], 'p2': [1, 1], 'colour': 'red'}},
        'a parameter a dimension does not have')
refused(lay, {'op': 'add', 'type': 'Dimension', 'name': 'M1',
              'params': {'p1': [0, 0], 'p2': [1, 1]}},
        'a name already taken by an optics')
refused(lay, {'op': 'set', 'target': 'D1', 'attrs': {'diameter': 0.1}},
        'an attribute a dimension does not have')
refused(lay, {'op': 'set', 'target': 'D1', 'attrs': {'offset': 'far'}},
        'an offset that is not a number')
refused(lay, {'op': 'set', 'target': 'D1',
              'attrs': {'offset': float('inf')}},
        'an offset at infinity')
refused(lay, {'op': 'add', 'type': 'Dimension',
              'params': {'p1': [0, 0], 'p2': [1, 1], 'offset': None}},
        'an add with an offset that is not a number')
refused(lay, {'op': 'set', 'target': 'D1', 'attrs': {'p2': ['x', 'y']}},
        'an end that is not numbers')
refused(lay, {'op': 'rename', 'target': 'D1', 'name': 'M1'},
        'renaming onto an optics')
refused(lay, {'op': 'move', 'target': 'D1', 'center': [0, 0]},
        'moving a dimension as a body')
refused(lay, {'op': 'rotate', 'target': 'D1', 'normAngleHR': 1.0},
        'turning one')

d1 = lay.get_dimension('D1')
lay.trace()
lay.apply_edit({'op': 'set', 'target': 'D1', 'attrs': {'p2': [0.9, 0.3]}})
check('set moves an end', list(d1.p2) == [0.9, 0.3], str(list(d1.p2)))
check('on the registered object itself', lay.get_dimension('D1') is d1)
check('and leaves the trace standing', lay.beams is not None)
lay.apply_edit({'op': 'set', 'target': 'D1',
                'attrs': {'p1': [0.1, 0.1], 'p2': [0.2, 0.2]}})
check('both ends at once', list(d1.p1) == [0.1, 0.1]
      and list(d1.p2) == [0.2, 0.2], str([list(d1.p1), list(d1.p2)]))
refused(lay, {'op': 'set', 'target': 'D1', 'attrs': {'p1': [0.2, 0.2]}},
        'a set that would put both ends in one place')

lay.trace()
lay.apply_edit({'op': 'set', 'target': 'D1', 'attrs': {'offset': 0.02}})
check('set moves the line aside', d1.offset == 0.02, str(d1.offset))
check('without touching the trace', lay.beams is not None)
check('or what the measurement comes to',
      abs(d1.length - float(np.linalg.norm(np.asarray(d1.p2)
                                           - np.asarray(d1.p1)))) < 1e-15)
lay.apply_edit({'op': 'undo'})
check('and undo puts it back', d1.offset == 0.0, str(d1.offset))

lay.apply_edit({'op': 'rename', 'target': 'D1', 'name': 'thickness'})
check('rename works on a dimension', d1.name == 'thickness'
      and lay.get_dimension('thickness') is d1, d1.name)
lay.apply_edit({'op': 'rename', 'target': 'thickness', 'name': 'D1'})

n0 = len(lay.dimensions)
lay.apply_edit({'op': 'remove', 'target': 'D1'})
check('remove takes it out', len(lay.dimensions) == n0 - 1
      and 'D1' not in [d.name for d in lay.dimensions],
      str([d.name for d in lay.dimensions]))
check('and the name is free again',
      lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                      'params': {'p1': hr, 'p2': ar}}) is lay)

print('--- undo and redo reach dimensions ---')

lay, M1, L1 = make_layout()
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                'params': {'p1': hr, 'p2': ar}})
d1 = lay.get_dimension('D1')
lay.apply_edit({'op': 'undo'})
check('undoing an add takes it away', lay.dimensions == [])
lay.apply_edit({'op': 'redo'})
check('redoing puts it back', [d.name for d in lay.dimensions] == ['D1'])
# The history holds the objects, not only their values, so what comes
# back is the dimension that was taken out.
check('as the very object', lay.get_dimension('D1') is d1)

lay.apply_edit({'op': 'set', 'target': 'D1', 'attrs': {'p2': [0.9, 0.3]}})
lay.apply_edit({'op': 'undo'})
check('undoing a set puts the end back',
      np.allclose(np.asarray(d1.p2), np.asarray(ar)), str(list(d1.p2)))
check('on the same object', lay.get_dimension('D1') is d1)

lay.apply_edit({'op': 'rename', 'target': 'D1', 'name': 'span'})
lay.apply_edit({'op': 'undo'})
check('undoing a rename gives the name back to the same object',
      d1.name == 'D1' and lay.get_dimension('D1') is d1, d1.name)

lay.apply_edit({'op': 'remove', 'target': 'D1'})
lay.apply_edit({'op': 'undo'})
check('undoing a remove brings back the object that was removed',
      lay.get_dimension('D1') is d1)

print('--- save and load ---')

lay, M1, L1 = make_layout()
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                'params': {'p1': hr, 'p2': ar}})
d1 = lay.get_dimension('D1')
path = os.path.join(WORK, 'dimensions.json')
lay.save(path)
with open(path) as f:
    raw = json.load(f)
check('a saved layout carries its dimensions',
      [d['name'] for d in raw['dimensions']] == ['D1'],
      str(raw.get('dimensions')))
check('written as their ends, where the line goes, and nothing else',
      set(raw['dimensions'][0]) == {'type', 'name', 'p1', 'p2', 'offset'},
      str(sorted(raw['dimensions'][0])))

back = OpticalLayout.load(path)
check('loading brings them back', len(back.dimensions) == 1)
check('with the ends they had',
      np.allclose(np.asarray(back.get_dimension('D1').p1), hr)
      and np.allclose(np.asarray(back.get_dimension('D1').p2), ar))
lay.update_from_file(path)
check('loading in place keeps the object', lay.get_dimension('D1') is d1)

# A file written before dimensions existed is a layout with none on it,
# not a file gtrace cannot read.
del raw['dimensions']
old = os.path.join(WORK, 'dimensions_old.json')
with open(old, 'w') as f:
    json.dump(raw, f)
check('a file from before dimensions loads as one without any',
      OpticalLayout.load(old).dimensions == [])
lay.update_from_file(old)
check('and loading it in place drops the ones that were there',
      lay.dimensions == [])

print('--- the scene ---')

lay, M1, L1 = make_layout()
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                'params': {'p1': hr, 'p2': ar}})
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D2',
                'params': {'p1': [0.0, 0.0], 'p2': [0.2, 0.0]}})
scene = lay.scene_dict()
check('the scene has a dimensions channel', 'dimensions' in scene)
check('and a snap channel', 'snap' in scene)
check('one entry per dimension', len(scene['dimensions']) == 2)
d = scene['dimensions'][0]
check('carrying the ends, the line and the measurement',
      set(d) == {'type', 'name', 'p1', 'p2', 'offset', 'line', 'length',
                 'optical', 'inside', 'n'}, str(sorted(d)))
check('the span inside the lens says so',
      d['inside'] == 'L1' and d['optical'] is not None, str(d['inside']))
check('the span in the open does not',
      scene['dimensions'][1]['optical'] is None
      and scene['dimensions'][1]['inside'] is None)
check('nine snap points per optics', len(scene['snap']) == 9*len(lay.optics),
      str(len(scene['snap'])))
# The line is worked out here rather than in a front end, so that only
# one place has an opinion about which side the offset goes.
check('with no offset the line runs between the ends',
      d['line'] == [d['p1'], d['p2']], str(d['line']))
lay.apply_edit({'op': 'set', 'target': 'D1', 'attrs': {'offset': 0.03}})
moved = lay.scene_dict()['dimensions'][0]
check('an offset carries the line and leaves the ends alone',
      moved['p1'] == d['p1'] and moved['line'] != d['line'],
      str(moved['line']))
check('by the offset, square to the span',
      abs(np.linalg.norm(np.asarray(moved['line'][0])
                         - np.asarray(moved['p1'])) - 0.03) < 1e-15
      and abs(np.dot(np.asarray(moved['line'][0]) - np.asarray(moved['p1']),
                     np.asarray(moved['p2'])
                     - np.asarray(moved['p1']))) < 1e-15,
      str(moved['line'][0]))
check('and the measurement is untouched',
      moved['length'] == d['length'] and moved['inside'] == d['inside'],
      str(moved['length']))
lay.apply_edit({'op': 'set', 'target': 'D1', 'attrs': {'offset': 0.0}})
json.dumps(scene)
check('the whole scene is JSON-clean', True)

# The measurement is worked out when the scene is built, never stored,
# so an optics that moves onto a span cannot leave a stale answer.
lay.apply_edit({'op': 'set', 'target': 'D1',
                'attrs': {'p1': [3.0, 3.0], 'p2': [3.1, 3.0]}})
check('moving a span out of the glass drops the optical distance',
      lay.scene_dict()['dimensions'][0]['optical'] is None)
lay.apply_edit({'op': 'set', 'target': 'D1',
                'attrs': {'p1': hr, 'p2': ar}})
check('and moving it back brings it again',
      lay.scene_dict()['dimensions'][0]['optical'] is not None)

print('--- drawing them into a DXF ---')

# A dimension is a note about the system rather than part of it, so it
# goes on a layer of its own: a layer is exactly the mechanism CAD
# offers for something you want to be able to switch off.
lay, M1, L1 = make_layout()
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                'params': {'p1': hr, 'p2': ar, 'offset': 0.05}})
lay.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D2',
                'params': {'p1': [0.0, 0.0], 'p2': [0.2, 0.0]}})

plain = lay.draw()
check('draw() leaves the dimensions out',
      'dimensions' not in plain.layers, str(sorted(plain.layers)))
# It has to: the viewer draws them itself from the scene, so a draw()
# that included them would draw them twice there.
canvas = lay.draw_dimensions(lay.draw())
check('draw_dimensions puts them on their own layer',
      'dimensions' in canvas.layers, str(sorted(canvas.layers)))
shapes = canvas.layers['dimensions'].shapes
texts = [s for s in shapes if isinstance(s, draw.Text)]
check('one label per dimension', len(texts) == 2, str(len(texts)))
check('the label carries the distance',
      any('6.359 mm' in t.text for t in texts),
      str([t.text for t in texts]))
# Only the span inside the lens has one.
check('and the optical distance where there is one',
      len([t for t in texts if 'optical' in t.text]) == 1,
      str([t.text for t in texts]))
lines = [s for s in shapes if isinstance(s, draw.Line)]
# Per dimension: the line, two ticks, and an extension line at each end
# where the line was carried aside. D1 has an offset, D2 does not.
check('the line and its ticks are drawn', len(lines) == 3 + 5,
      str(len(lines)))

check('an empty layout adds no layer',
      'dimensions' not in make_layout()[0].draw_dimensions(
          make_layout()[0].draw()).layers)

dxf_path = os.path.join(WORK, 'dimensions.dxf')
lay.export_dxf(dxf_path)
with open(dxf_path, encoding='utf-8', errors='replace') as f:
    text = f.read()
check('export_dxf writes a file', os.path.getsize(dxf_path) > 1000,
      '(%d bytes)' % os.path.getsize(dxf_path))
check('with the dimensions layer in it', 'dimensions' in text)
check('and the measurement written on it', '6.359 mm' in text)

nodim = os.path.join(WORK, 'dimensions_off.dxf')
lay.export_dxf(nodim, dimensions=False)
with open(nodim, encoding='utf-8', errors='replace') as f:
    text2 = f.read()
check('dimensions=False leaves them out',
      'dimensions' not in text2 and 'main_beam' in text2)

# The renderer's own error must be catchable. It derived from
# BaseException, which walks through every 'except Exception' between
# here and the top - including the one the widget uses to turn a
# failure into something the user can see.
check('an unsupported shape raises a catchable error',
      issubclass(renderer.UnknownShapeError, Exception)
      and not issubclass(renderer.UnknownShapeError, KeyboardInterrupt))
try:
    renderer.UnknownShapeError('why')
    check('and it takes a message', True)
except Exception as e:
    check('and it takes a message', False, str(e))

print('--- serialization helpers ---')

dim = Dimension([0.1, 0.2], [0.3, 0.4], name='D9')
d = dimension_to_dict(dim)
check('to_dict writes plain floats',
      all(isinstance(x, float) for x in d['p1'] + d['p2']), str(d))
again = dimension_from_dict(d)
check('and from_dict reads them back',
      again.name == 'D9' and np.allclose(again.p1, [0.1, 0.2])
      and np.allclose(again.p2, [0.3, 0.4]))
check('copy() is a new object with the same ends',
      dim.copy() is not dim and np.allclose(dim.copy().p2, dim.p2)
      and dim.copy().name == 'D9' and dim.copy().offset == dim.offset)
check('length is the distance between the ends',
      abs(dim.length - np.hypot(0.2, 0.2)) < 1e-15, str(dim.length))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
