'''
A rectangle that is turned: its corners, what it is turned about, and
everything downstream that has to agree about where it is.

A Rectangle used to be a corner, a width and a height with its sides
along the axes. It now carries an angle and the point that angle is
taken about, so a plate on a bench at 30 degrees is a rectangle
rather than a polyline that used to be one.

Three things carry it and are checked hardest.

The first is that the corners are the only statement of where it is.
The angle and the pivot are numbers; the corners follow from them,
and everything that draws, bounds, picks or writes the rectangle
takes them from the one place - so the checks here derive them a
second time, from the rotation written out by hand, and hold the
class to that.

The second is the pivot left unsaid. None is the middle of the
rectangle, worked out when it is asked for rather than written down,
which is what lets a rectangle that is carried keep turning about
itself. A pivot that was given is a point like any other and travels
with the shape.

The third is that nothing square to the axes moved. A rectangle with
no angle has to serialize, bound, draw and export exactly as it did
before there was an angle at all - the DXF of a breadboard is the
same file - since every part gtrace ships is drawn from rectangles.

The numbers the page has to agree with are written to
_work/rect_cases.json, which verify_rect.js reads.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json

import numpy as np

import gtrace.draw as draw
import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.draw.renderer import renderDXF
from gtrace.draw.serialize import shape_to_dict, shape_from_dict
from gtrace.draw.viewer.editor import ShapeEditor, EDITABLE_SHAPE_ATTRS
from gtrace.layout import OpticalLayout, EditError
from gtrace.mechanics import (Mechanics, breadboard, point_in_polygon,
                              rotate_shape, shape_centre, turned_shape)
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

def turn(p, pivot, angle):
    '''
    One point turned about another, written out here rather than taken
    from gtrace: a check that used the same rotation the class used
    would only be saying that the class agrees with itself.
    '''
    dx = p[0] - pivot[0]
    dy = p[1] - pivot[1]
    return np.array([pivot[0] + dx*np.cos(angle) - dy*np.sin(angle),
                     pivot[1] + dx*np.sin(angle) + dy*np.cos(angle)])

def corners_of(point, w, h):
    x, y = point
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

print()
print('--- what a rectangle is ---')

r = draw.Rectangle([0.0, 0.0], 2.0, 1.0)
check('a rectangle with no angle has one', r.angle == 0.0)
check('  and no pivot of its own', r.pivot is None)
check('  its corners are the four it is written from',
      np.allclose(r.corners(), corners_of([0, 0], 2, 1)))
check('  and the pivot it uses is its middle',
      np.allclose(r.pivot_point(), [1.0, 0.5]))

# --- the turn, against the rotation written out by hand ---
for angle in [0.1, np.pi/6, np.pi/2, -np.pi/3, 2.7, 4.5, -6.0]:
    r = draw.Rectangle([0.3, -0.2], 0.4, 0.15, angle=angle)
    want = np.array([turn(c, [0.3 + 0.2, -0.2 + 0.075], angle)
                     for c in corners_of([0.3, -0.2], 0.4, 0.15)])
    check('turned %7.3f rad about its middle' % angle,
          np.allclose(r.corners(), want))

for pivot in [[0.0, 0.0], [0.3, -0.2], [-1.0, 2.0], [0.5, 0.5]]:
    angle = 0.7
    r = draw.Rectangle([0.3, -0.2], 0.4, 0.15, angle=angle, pivot=pivot)
    want = np.array([turn(c, pivot, angle)
                     for c in corners_of([0.3, -0.2], 0.4, 0.15)])
    check('turned about %s' % (pivot,), np.allclose(r.corners(), want))
    check('  and says that is what it turns about',
          np.allclose(r.pivot_point(), pivot))

# --- it is still a rectangle ---
r = draw.Rectangle([0.3, -0.2], 0.4, 0.15, angle=0.7, pivot=[0.0, 0.0])
cs = r.corners()
sides = [cs[(i + 1) % 4] - cs[i] for i in range(4)]
check('a turned rectangle keeps its width',
      np.isclose(np.hypot(*sides[0]), 0.4)
      and np.isclose(np.hypot(*sides[2]), 0.4))
check('  and its height',
      np.isclose(np.hypot(*sides[1]), 0.15)
      and np.isclose(np.hypot(*sides[3]), 0.15))
check('  and its right angles',
      all(abs(np.dot(sides[i], sides[(i + 1) % 4])) < 1e-15 for i in range(4)))

check('a turn of nothing is the untouched rectangle',
      np.allclose(draw.Rectangle([1, 2], 3, 4, angle=0.0).corners(),
                  corners_of([1, 2], 3, 4)))
check('a turn of a whole revolution is too',
      np.allclose(draw.Rectangle([1, 2], 3, 4, angle=2*np.pi).corners(),
                  corners_of([1, 2], 3, 4)))
check('degrees say the same thing as radians',
      np.allclose(draw.Rectangle([1, 2], 3, 4, angle=30,
                                 angle_in_rad=False).corners(),
                  draw.Rectangle([1, 2], 3, 4, angle=np.pi/6).corners()))

# --- the pivot left unsaid travels; one that was given is a point ---
a = draw.Rectangle([0.0, 0.0], 2.0, 1.0, angle=0.5)
b = draw.Rectangle([10.0, 10.0], 2.0, 1.0, angle=0.5)
check('a rectangle with no pivot turns about itself wherever it is',
      np.allclose(b.corners() - [10.0, 10.0], a.corners()))
c = draw.Rectangle([0.0, 0.0], 2.0, 1.0, angle=0.5, pivot=[0.0, 0.0])
check('  and one given a pivot does not',
      not np.allclose(c.corners(), a.corners()))

check('a rectangle is not asked to store the middle it works out',
      draw.Rectangle([0, 0], 1, 1, angle=0.4).pivot is None)

print()
print('--- serialization ---')

for s in [draw.Rectangle([0.1, 0.2], 0.3, 0.4),
          draw.Rectangle([0.1, 0.2], 0.3, 0.4, angle=0.6),
          draw.Rectangle([0.1, 0.2], 0.3, 0.4, angle=-1.2, pivot=[9.0, -8.0]),
          draw.Rectangle([0.1, 0.2], 0.3, 0.4, thickness=0.002, angle=3.0)]:
    d = shape_to_dict(s)
    back = shape_from_dict(d)
    check('a rectangle survives a round trip (angle %.1f)' % s.angle,
          np.allclose(back.corners(), s.corners())
          and back.angle == s.angle
          and back.thickness == s.thickness)
    check('  and its pivot with it',
          (back.pivot is None and s.pivot is None)
          or np.allclose(back.pivot, s.pivot))
    check('  and the dict is strict JSON',
          json.loads(json.dumps(d)) == d)

d = shape_to_dict(draw.Rectangle([0, 0], 1, 1))
check('a rectangle square to the axes writes a null pivot',
      d['pivot'] is None and d['angle'] == 0.0)
check('  and nothing else about it changed',
      set(d) == {'type', 'point', 'width', 'height', 'angle', 'pivot',
                 'thickness'})

old = {'type': 'rectangle', 'point': [1.0, 2.0], 'width': 3.0,
       'height': 4.0, 'thickness': 0.0}
back = shape_from_dict(old)
check('a file written before there was an angle loads square to the axes',
      back.angle == 0.0 and back.pivot is None
      and np.allclose(back.corners(), corners_of([1, 2], 3, 4)))

print()
print('--- the bodies that carry one ---')

r = draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=np.pi/4)
m = Mechanics(shapes=[r], name='B1')
lo, hi = m.local_bbox()
want = r.corners()
check('a body is bounded by the corners the rectangle actually has',
      np.allclose(lo, want.min(axis=0)) and np.allclose(hi, want.max(axis=0)))
check('  which reaches past the box it was written from',
      hi[1] > 0.1 + 1e-9)
check('the middle of a turned rectangle is the middle of its corners',
      np.allclose(shape_centre(r), want.mean(axis=0)))

carried = turned_shape(r, 0.0, offset=[1.0, 2.0])
check('carrying a rectangle leaves it a rectangle',
      isinstance(carried, draw.Rectangle))
check('  with its own angle', carried.angle == r.angle)
check('  and its corners moved and nothing else',
      np.allclose(carried.corners(), r.corners() + [1.0, 2.0]))
check('  and no pivot written out for it',
      carried.pivot is None)

rp = draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=0.3, pivot=[0.5, 0.5])
carried = turned_shape(rp, 0.0, offset=[1.0, 2.0])
check('a pivot that was given is carried with the rectangle',
      np.allclose(carried.pivot, [1.5, 2.5])
      and np.allclose(carried.corners(), rp.corners() + [1.0, 2.0]))

t = turned_shape(rp, 0.9)
check('a rectangle turned by its body comes back as a polyline',
      isinstance(t, draw.PolyLine))
check('  closed, through the corners it really has',
      np.allclose(np.column_stack([t.x, t.y])[:4],
                  np.array([turn(c, [0, 0], 0.9) for c in rp.corners()]))
      and np.allclose([t.x[0], t.y[0]], [t.x[4], t.y[4]]))

t = rotate_shape(rp, 0.9, pivot=[3.0, 1.0])
check('turning about a point is turning about the origin and carrying back',
      np.allclose(np.column_stack([t.x, t.y])[:4],
                  np.array([turn(c, [3.0, 1.0], 0.9) for c in rp.corners()])))

m = Mechanics(shapes=[r], center=[1.0, 1.0], rotationAngle=0.6, name='B2')
ws = m.world_shapes()[0]
check('a body draws the corners of the rectangle it carries',
      np.allclose(np.column_stack([ws.x, ws.y])[:4],
                  np.array([turn(c, [0, 0], 0.6) for c in r.corners()])
                  + [1.0, 1.0]))

# A point inside the turned rectangle, outside the one it was written
# from: the outline is a box round the whole thing, so it holds both,
# but the drawing has to have moved.
r = draw.Rectangle([0.0, 0.0], 0.2, 0.02, angle=np.pi/2)
m = Mechanics(shapes=[r], name='B3')
inside_turned = np.array([0.1, 0.09])
check('a point the turned rectangle covers is on the body',
      m.contains(inside_turned))
check('  and the polygon test says so on its own',
      point_in_polygon(inside_turned, r.corners()))
check('  while the box it was written from does not reach it',
      not point_in_polygon(inside_turned, corners_of([0, 0], 0.2, 0.02)))

print()
print('--- DXF ---')

def dxf_polylines(path):
    '''
    Every LWPOLYLINE in a DXF file, as its list of points. Read out of
    the file itself rather than out of the writer: the point of the
    check is what a CAD program will find in there.
    '''
    lines = [ln.strip() for ln in open(path).read().splitlines()]
    out, pts, i = [], None, 0
    while i + 1 < len(lines):
        code, value = lines[i], lines[i + 1]
        if code == '0':
            if pts is not None:
                out.append(pts)
            pts = [] if value == 'LWPOLYLINE' else None
        elif pts is not None and code == '10':
            # x under group code 10, y under 20 right behind it.
            pts.append([float(value), float(lines[i + 3])])
        i += 2
    if pts is not None:
        out.append(pts)
    return out

cv = draw.Canvas(unit='m')
cv.add_layer('plain')
cv.add_shape(draw.Rectangle([0.0, 0.0], 0.2, 0.1), 'plain')
plain_path = os.path.join(WORK, 'rect_plain.dxf')
renderDXF(cv, plain_path)
got = dxf_polylines(plain_path)
check('a rectangle square to the axes is the DXF it always was',
      len(got) == 1 and np.allclose(got[0],
                                    [[0, 0], [200, 0], [200, 100],
                                     [0, 100], [0, 0]]))

turned = draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=np.pi/3, pivot=[0.05, 0.0])
cv = draw.Canvas(unit='m')
cv.add_layer('turned')
cv.add_shape(turned, 'turned')
turned_path = os.path.join(WORK, 'rect_turned.dxf')
renderDXF(cv, turned_path)
got = dxf_polylines(turned_path)
want = np.array([turn(c, [0.05, 0.0], np.pi/3)
                 for c in corners_of([0, 0], 0.2, 0.1)]) * 1000.0
check('a turned rectangle is written through the corners it has',
      len(got) == 1 and np.allclose(got[0][:4], want, atol=1e-9))
check('  closed, like any other rectangle',
      np.allclose(got[0][4], got[0][0]))

# Every part gtrace ships is drawn from rectangles: the file a
# breadboard writes must not have moved at all.
cv = draw.Canvas(unit='m')
breadboard(0.3, 0.2, name='BB').draw(cv)
bb_path = os.path.join(WORK, 'rect_breadboard.dxf')
renderDXF(cv, bb_path)
got = dxf_polylines(bb_path)
check('a breadboard writes its plate exactly where it did',
      len(got) == 1 and np.allclose(got[0],
                                    [[-150, -100], [150, -100], [150, 100],
                                     [-150, 100], [-150, -100]]))

print()
print('--- the editor protocol ---')

check('a rectangle may be given an angle and a pivot',
      {'angle', 'pivot'} <= EDITABLE_SHAPE_ATTRS['rectangle'])

body = Mechanics(shapes=[draw.Rectangle([0.0, 0.0], 0.2, 0.1)], name='E1')
ed = ShapeEditor(body)
ed.apply_edit({'op': 'set_shape', 'index': 0, 'attrs': {'angle': 0.5}})
check('an angle set through the protocol reaches the shape',
      body.shapes[0].angle == 0.5)
check('  and turns it about its own middle',
      np.allclose(body.shapes[0].corners(),
                  [turn(c, [0.1, 0.05], 0.5)
                   for c in corners_of([0, 0], 0.2, 0.1)]))
ed.apply_edit({'op': 'set_shape', 'index': 0, 'attrs': {'pivot': [0.0, 0.0]}})
check('a pivot set through the protocol reaches it too',
      np.allclose(body.shapes[0].pivot, [0.0, 0.0])
      and np.allclose(body.shapes[0].corners(),
                      [turn(c, [0, 0], 0.5)
                       for c in corners_of([0, 0], 0.2, 0.1)]))
ed.apply_edit({'op': 'undo'})
check('undo takes the pivot back off',
      body.shapes[0].pivot is None and body.shapes[0].angle == 0.5)

for bad, why in [(float('nan'), 'a nan'), (float('inf'), 'an infinity')]:
    try:
        ed.apply_edit({'op': 'set_shape', 'index': 0, 'attrs': {'angle': bad}})
        ok = False
    except EditError:
        ok = True
    check('an angle of %s is refused' % why, ok)
    check('  and the shape is untouched', body.shapes[0].angle == 0.5)

try:
    ed.apply_edit({'op': 'set_shape', 'index': 0,
                   'attrs': {'pivot': [float('inf'), 0.0]}})
    ok = False
except EditError:
    ok = True
check('a pivot with an infinity in it is refused', ok)
check('  and the shape is untouched', body.shapes[0].pivot is None)

# The editor's turn keeps a rectangle a rectangle - it has an angle to
# put the turn in - and lands the corners exactly where taking it apart
# into a polyline would have landed them.
for start in [draw.Rectangle([0.0, 0.0], 0.2, 0.1),
              draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=0.4),
              draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=0.4,
                             pivot=[0.5, 0.5])]:
    for pv in [None, [0.0, 0.0], [1.0, -1.0]]:
        body = Mechanics(shapes=[start], name='T')
        ed = ShapeEditor(body)
        msg = {'op': 'rotate_shape', 'index': 0, 'angle': 0.9}
        if pv is not None:
            msg['pivot'] = pv
        ed.apply_edit(msg)
        got = body.shapes[0]
        want = rotate_shape(start, 0.9,
                            shape_centre(start) if pv is None else pv)
        want_corners = (np.column_stack([want.x, want.y])[:4]
                        if isinstance(want, draw.PolyLine) else want.corners())
        check('the editor turns a rectangle and it is still one (%.1f, %s)'
              % (start.angle, pv), isinstance(got, draw.Rectangle))
        check('  and its corners are where the polyline would have been',
              np.allclose(got.corners(), want_corners))
        check('  and the turn is in its angle',
              np.isclose(got.angle, start.angle + 0.9))
        ed.apply_edit({'op': 'undo'})
        check('  and undo puts back exactly what was there',
              isinstance(body.shapes[0], draw.Rectangle)
              and np.allclose(body.shapes[0].corners(), start.corners())
              and body.shapes[0].angle == start.angle)

# A shape that is not a rectangle turns as it always did.
body = Mechanics(shapes=[draw.Line([0.0, 0.0], [0.1, 0.0])], name='T2')
ShapeEditor(body).apply_edit({'op': 'rotate_shape', 'index': 0,
                              'angle': np.pi/2, 'pivot': [0.0, 0.0]})
check('a line still turns the way it did',
      isinstance(body.shapes[0], draw.Line)
      and np.allclose(body.shapes[0].stop, [0.0, 0.1]))

snaps = ShapeEditor(Mechanics(
    shapes=[draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=np.pi/2)],
    name='E2')).scene_dict()['snap']
pts = np.array([p['point'] for p in snaps if p['kind'] == 'corner'])
want = draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=np.pi/2).corners()
check('the corners a drag settles on are the turned ones',
      all(any(np.allclose(p, c) for p in pts) for c in want))

print()
print('--- what the page is handed ---')

L = OpticalLayout(optics=[], sources=[], name='RectScene')
L.add_mechanics(Mechanics(
    shapes=[draw.Rectangle([0.0, 0.0], 0.2, 0.1, angle=0.4)], name='S1'))
scene = L.scene_dict()
shapes = [s for ly in scene['canvas']['layers'] for s in ly['shapes']]
rects = [s for s in shapes if s['type'] == 'rectangle']
check('a turned rectangle reaches the page as a rectangle',
      len(rects) == 1 and rects[0]['angle'] == 0.4)
check('  and the scene is still strict JSON',
      json.loads(json.dumps(scene)) is not None)

# The numbers verify_rect.js holds viewer.js to.
CASES = [
    {'type': 'rectangle', 'point': [0.0, 0.0], 'width': 0.2, 'height': 0.1,
     'angle': 0.0, 'pivot': None, 'thickness': 0.0},
    {'type': 'rectangle', 'point': [0.0, 0.0], 'width': 0.2, 'height': 0.1,
     'angle': 0.7, 'pivot': None, 'thickness': 0.0},
    {'type': 'rectangle', 'point': [-0.05, 0.02], 'width': 0.3, 'height': 0.04,
     'angle': np.pi/2, 'pivot': None, 'thickness': 0.0},
    {'type': 'rectangle', 'point': [0.0, 0.0], 'width': 0.2, 'height': 0.1,
     'angle': 0.7, 'pivot': [0.0, 0.0], 'thickness': 0.0},
    {'type': 'rectangle', 'point': [0.1, 0.1], 'width': 0.05, 'height': 0.25,
     'angle': -1.1, 'pivot': [0.4, -0.3], 'thickness': 0.0},
]
probes = [[0.05, 0.05], [0.15, 0.02], [0.0, 0.0], [-0.02, 0.06],
          [0.12, 0.12], [0.3, 0.3]]

def on_edge(point, corners, tol=1e-9):
    '''
    Whether a probe lands on the outline itself. Which side of its own
    edge a point falls on is not something two polygon tests have to
    agree about - it is the one place where a difference in the last
    bit changes the answer - so a probe that lands there is not asked
    of the page.
    '''
    p = np.asarray(point, dtype='float64')
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        d = b - a
        t = np.clip(np.dot(p - a, d) / np.dot(d, d), 0.0, 1.0)
        if np.hypot(*(p - (a + t*d))) <= tol:
            return True
    return False
cases = []
for d in CASES:
    s = shape_from_dict(d)
    # Every case is asked about a point that is certainly inside it and
    # one that is certainly outside, on top of the fixed probes: a hit
    # test that only ever answers 'no' is not a hit test.
    mine = ([[float(v) for v in shape_centre(s)]]
            + [[float(v) for v in s.corners().max(axis=0) + [1.0, 1.0]]])
    cases.append({
        'probes': probes + mine,
        'shape': d,
        'pivot': [float(v) for v in s.pivot_point()],
        'corners': [[float(v) for v in c] for c in s.corners()],
        'centre': [float(v) for v in shape_centre(s)],
        'bbox': [[float(v) for v in s.corners().min(axis=0)],
                 [float(v) for v in s.corners().max(axis=0)]],
        'encloses': [None if on_edge(p, s.corners())
                     else bool(point_in_polygon(p, s.corners()))
                     for p in probes + mine],
        'turned': [[float(v) for v in turn(c, [0.03, -0.04], 0.55)]
                   for c in s.corners()],
        'moved': [[float(v) for v in c] for c in (s.corners() + [1.0, -2.0])],
    })
out = {'turn': {'angle': 0.55, 'pivot': [0.03, -0.04]},
       'move': [1.0, -2.0], 'cases': cases}
path = os.path.join(WORK, 'rect_cases.json')
with open(path, 'w') as f:
    json.dump(out, f, indent=1)
check('the numbers for the page are written', os.path.exists(path),
      '(%d cases)' % len(cases))
check('  and every case is asked about a point inside it and one outside',
      all(True in c['encloses'] and False in c['encloses'] for c in cases))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
