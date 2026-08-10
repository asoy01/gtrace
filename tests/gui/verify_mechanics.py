'''
Mechanics: the hardware the trace never sees, and everything the layout
does with it.

A Mechanics is a named body - a breadboard, a mount, a housing - whose
geometry is a list of drawing primitives in local coordinates, placed on
the bench by a pose. Three choices carry the design and are checked
hardest here.

The first is that the pose is the only statement of where the body is.
The shapes never move; the world coordinates are derived on the way to
the canvas and the scene, so there is no second description to fall out
of step - the class of mistake that put the AR surface a sagitta out of
place in 2026-08-03.

The second is that the trace never sees it. Moving a mechanics redraws
the picture and must not invalidate the trace: the beams did not move,
and a layout that re-traced for every nudge of a breadboard would be
paying for physics that cannot have changed.

The third is the polygon. A mechanics is picked by area, because the
enclosing circle the optics use would let a breadboard cover the whole
bench; point_in_polygon is the primitive that decision rests on, so it
is checked on its own and against rotated bodies.
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
from gtrace.draw.serialize import (shape_to_dict, shape_from_dict,
                                   UnknownShapeError)
from gtrace.layout import (OpticalLayout, TraceRules, EditError,
                           q_from_waist, mechanics_to_dict,
                           mechanics_from_dict, mechanics_scene_dict,
                           mechanics_snap_points)
from gtrace.mechanics import (Mechanics, point_in_polygon, DEFAULT_LAYER,
                              LAYER_COLOR, breadboard, mirror_mount,
                              register_model, models, model_shapes,
                              from_model)
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

def close(a, b, tol=1e-12):
    return np.allclose(np.asarray(a, dtype='float64'),
                       np.asarray(b, dtype='float64'), atol=tol)


print('--- point_in_polygon ---')

square = [[0, 0], [1, 0], [1, 1], [0, 1]]
check('inside a square', point_in_polygon([0.5, 0.5], square))
check('outside a square', not point_in_polygon([1.5, 0.5], square))
check('outside, below', not point_in_polygon([0.5, -0.1], square))
check('well inside a corner region', point_in_polygon([0.01, 0.01], square))

# A concave polygon: an L, with the notch at the top right. The point in
# the notch is inside the bounding box and outside the polygon, which is
# exactly what a bounding-box test would get wrong.
ell = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
check('inside the foot of an L', point_in_polygon([1.5, 0.5], ell))
check('inside the leg of an L', point_in_polygon([0.5, 1.5], ell))
check('the notch of an L is outside', not point_in_polygon([1.5, 1.5], ell))

tri = [[0, 0], [1, 0], [0, 1]]
check('inside a triangle', point_in_polygon([0.2, 0.2], tri))
check('outside its hypotenuse', not point_in_polygon([0.7, 0.7], tri))


print('--- the local bounding box ---')

m = Mechanics(shapes=[draw.Line([-0.1, 0.0], [0.2, 0.05])])
lo, hi = m.local_bbox()
check('a line is bounded by its ends',
      close(lo, [-0.1, 0.0]) and close(hi, [0.2, 0.05]))

m = Mechanics(shapes=[draw.Rectangle([-0.15, -0.1], 0.3, 0.2)])
lo, hi = m.local_bbox()
check('a rectangle by its corners',
      close(lo, [-0.15, -0.1]) and close(hi, [0.15, 0.1]))

m = Mechanics(shapes=[draw.Circle([0.05, 0.0], 0.02)])
lo, hi = m.local_bbox()
check('a circle by centre +/- radius',
      close(lo, [0.03, -0.02]) and close(hi, [0.07, 0.02]))

m = Mechanics(shapes=[draw.Arc([0.0, 0.0], 0.1, 0.0, np.pi / 4)])
lo, hi = m.local_bbox()
check('an arc by the circle it lies on (a bound, not a fit)',
      close(lo, [-0.1, -0.1]) and close(hi, [0.1, 0.1]))

m = Mechanics(shapes=[draw.PolyLine([0.0, 0.1, -0.05], [0.0, 0.2, 0.1])])
lo, hi = m.local_bbox()
check('a polyline by its extremes',
      close(lo, [-0.05, 0.0]) and close(hi, [0.1, 0.2]))

m = Mechanics(shapes=[draw.Rectangle([-0.1, -0.1], 0.2, 0.2),
                      draw.Circle([0.25, 0.0], 0.05)])
lo, hi = m.local_bbox()
check('several shapes bound together',
      close(lo, [-0.1, -0.1]) and close(hi, [0.3, 0.1]))

lo, hi = Mechanics().local_bbox()
check('no shapes is a point at the origin',
      close(lo, [0, 0]) and close(hi, [0, 0]))


print('--- the outline, and containment through the pose ---')

board = Mechanics(shapes=[draw.Rectangle([-0.15, -0.1], 0.3, 0.2)],
                  center=[0.4, 0.2], name='BB')
ol = board.outline()
check('the outline is four corners', ol.shape == (4, 2))
check('unrotated, it is the bbox stood at the centre',
      close(ol, [[0.25, 0.1], [0.55, 0.1], [0.55, 0.3], [0.25, 0.3]]))
check('the centre is inside', board.contains([0.4, 0.2]))
check('a corner region is inside', board.contains([0.26, 0.11]))
check('past the edge is outside', not board.contains([0.56, 0.2]))

# Turned 90 degrees the box trades its width for its height: a point
# that was inside along x is now outside, and one that was outside
# along y is now inside. That cannot pass by accident of an unrotated
# test.
board.rotationAngle = np.pi / 2
check('turned 90deg: the long side now runs along y',
      board.contains([0.4, 0.34]) and not board.contains([0.54, 0.2]))
ol90 = board.outline()
check('the outline turned with it',
      close(sorted(ol90[:, 0]), [0.3, 0.3, 0.5, 0.5])
      and close(sorted(ol90[:, 1]), [0.05, 0.05, 0.35, 0.35]))
board.rotationAngle = 0.0

# The area of the outline is invariant under the pose: a turn is rigid.
def shoelace(p):
    p = np.asarray(p)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

a0 = shoelace(board.outline())
board.rotationAngle = 0.7
check('a turn preserves the outline area',
      abs(shoelace(board.outline()) - a0) < 1e-12)
board.rotationAngle = 0.0


print('--- world shapes ---')

m = Mechanics(shapes=[draw.Line([0.0, 0.0], [0.1, 0.0])],
              center=[1.0, 2.0], rotationAngle=np.pi / 2)
w = m.world_shapes()[0]
check('a line is carried and turned',
      close(w.start, [1.0, 2.0]) and close(w.stop, [1.0, 2.1]))

m = Mechanics(shapes=[draw.Rectangle([0.0, 0.0], 0.2, 0.1)],
              center=[1.0, 0.0])
w = m.world_shapes()[0]
check('an unturned rectangle stays a rectangle',
      isinstance(w, draw.Rectangle) and close(w.point, [1.0, 0.0])
      and w.width == 0.2 and w.height == 0.1)

m.rotationAngle = np.pi / 2
w = m.world_shapes()[0]
check('a turned rectangle becomes a polyline', isinstance(w, draw.PolyLine))
check('  closed', close([w.x[0], w.y[0]], [w.x[-1], w.y[-1]]))
check('  through the turned corners',
      close(sorted(zip(w.x[:4], w.y[:4])),
            sorted([(1.0, 0.0), (1.0, 0.2), (0.9, 0.0), (0.9, 0.2)])))

m = Mechanics(shapes=[draw.Circle([0.1, 0.0], 0.05)], center=[0.0, 0.0],
              rotationAngle=np.pi)
w = m.world_shapes()[0]
check('a circle moves and keeps its radius',
      close(w.center, [-0.1, 0.0], tol=1e-15) and w.radius == 0.05)

m = Mechanics(shapes=[draw.Arc([0.0, 0.0], 0.05, 0.0, np.pi / 2)],
              rotationAngle=np.pi / 4)
w = m.world_shapes()[0]
check('an arc turns by having its angles carried',
      abs(w.startangle - np.pi / 4) < 1e-15
      and abs(w.stopangle - 3 * np.pi / 4) < 1e-15)

m = Mechanics(shapes=[draw.Text('hole', [0.02, 0.0], height=0.005)],
              center=[0.5, 0.5], rotationAngle=np.pi / 2)
w = m.world_shapes()[0]
check('text is carried and its rotation follows',
      close(w.point, [0.5, 0.52]) and abs(w.rotation - np.pi / 2) < 1e-15)

m = Mechanics(shapes=[draw.PolyLine([0.0, 0.1], [0.0, 0.1])])
check('the local shapes are left untouched by the derivation',
      m.world_shapes()[0] is not m.shapes[0])


print('--- drawing into a canvas ---')

cv = draw.Canvas()
board = Mechanics(shapes=[draw.Rectangle([-0.15, -0.1], 0.3, 0.2),
                          draw.Circle([0.0, 0.0], 0.003)],
                  center=[0.4, 0.0], name='BB1')
board.draw(cv)
check('the hardware layer appears', DEFAULT_LAYER in cv.layers)
check('  with its own colour',
      cv.layers[DEFAULT_LAYER].color == LAYER_COLOR)
check('  carrying the shapes',
      len(cv.layers[DEFAULT_LAYER].shapes) == 2)
check('no name unless asked', 'text' not in cv.layers)

board.draw(cv, drawName=True)
check('the name goes on the text layer, like an optics name',
      'text' in cv.layers and any(
          isinstance(s, draw.Text) and s.text == 'BB1'
          for s in cv.layers['text'].shapes))

cv2 = draw.Canvas()
Mechanics(shapes=[draw.Line([0, 0], [0.1, 0])], layer='posts').draw(cv2)
check('a layer of its own choosing is honoured',
      'posts' in cv2.layers and len(cv2.layers['posts'].shapes) == 1)


print('--- translate, rotate, copy ---')

m = Mechanics(shapes=[draw.Circle([0, 0], 0.01)], center=[0.1, 0.2])
m.translate([0.05, -0.1])
check('translate moves the centre', close(m.center, [0.15, 0.1]))

m.rotate(np.pi / 2)
check('rotate about its own centre leaves the centre',
      close(m.center, [0.15, 0.1]) and abs(m.rotationAngle - np.pi/2) < 1e-15)

m = Mechanics(center=[1.0, 0.0])
m.rotate(np.pi / 2, center=[0.0, 0.0])
check('rotate about a pivot carries the centre around it',
      close(m.center, [0.0, 1.0], tol=1e-15)
      and abs(m.rotationAngle - np.pi / 2) < 1e-15)

m = Mechanics(shapes=[draw.Circle([0, 0], 0.01)], center=[0.1, 0.2],
              rotationAngle=0.3, name='A', layer='posts', model='P-1')
c = m.copy()
c.center[0] = 9.9
c.rotationAngle = 0.0
check('a copy carries the pose and the labels',
      c.name == 'A' and c.layer == 'posts' and c.model == 'P-1')
check('  and its pose is its own', m.center[0] == 0.1
      and m.rotationAngle == 0.3)
check('  while the primitives are shared', c.shapes[0] is m.shapes[0])


print('--- shape serialization, both ways ---')

SHAPES = [draw.Line([0.0, 0.1], [0.2, 0.3], thickness=0.001),
          draw.PolyLine([0.0, 0.1, 0.2], [0.0, 0.05, 0.0]),
          draw.Rectangle([-0.1, -0.2], 0.2, 0.4),
          draw.Circle([0.05, 0.05], 0.025),
          draw.Arc([0.0, 0.0], 0.1, 0.1, 2.0),
          draw.Text('label', [0.01, 0.02], height=0.005, rotation=0.3)]
for s in SHAPES:
    d = shape_to_dict(s)
    r = shape_from_dict(d)
    check('%s survives the round trip' % d['type'],
          type(r) is type(s) and shape_to_dict(r) == d)
    check('  and is strict JSON', json.dumps(d) is not None)

try:
    shape_from_dict({'type': 'blob'})
    check('an unknown shape type is refused', False)
except UnknownShapeError as e:
    check('an unknown shape type is refused', True, '(%s)' % e)

try:
    shape_from_dict('not a dict')
    check('a non-dict is refused', False)
except UnknownShapeError:
    check('a non-dict is refused', True)

# The class the refusal arrives on has to be catchable. The copy of it
# in renderer.py derived from BaseException until 2026-08-04, and sailed
# through every 'except Exception' between it and the user.
check('UnknownShapeError is an Exception',
      issubclass(UnknownShapeError, Exception)
      and not issubclass(UnknownShapeError, KeyboardInterrupt))


print('--- mechanics serialization ---')

m = Mechanics(shapes=SHAPES, center=[0.3, -0.2], rotationAngle=0.25,
              name='BB1', layer='hardware', model='MB3045/M')
d = mechanics_to_dict(m)
check('the dict is strict JSON', json.dumps(d) is not None)
check('it says what it is', d['type'] == 'Mechanics' and d['name'] == 'BB1')
check('the shapes are saved by value, the model as a label',
      len(d['shapes']) == len(SHAPES) and d['model'] == 'MB3045/M')

r = mechanics_from_dict(d)
check('the round trip returns the same body',
      mechanics_to_dict(r) == d)

r2 = mechanics_from_dict({'name': 'bare'})
check('a minimal dict fills the defaults',
      r2.layer == 'hardware' and r2.model is None and r2.shapes == []
      and close(r2.center, [0, 0]) and r2.rotationAngle == 0.0)


print('--- registration and the one namespace ---')

def fresh():
    L = OpticalLayout(name='mech', rules=TraceRules(order=2,
                                                    power_threshold=1e-6))
    L.add_optics(opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.pi,
                            diameter=1*inch, name='M1'))
    L.add_source(GaussianBeam(pos=[0.0, 0.0], dirAngle=0.0,
                              q0=q_from_waist(0.2*mm, 0.0, 1064*nm),
                              wl=1064*nm, name='b0'))
    L.add_mechanics(Mechanics(
        shapes=[draw.Rectangle([-0.3, -0.2], 0.6, 0.4)],
        center=[0.3, 0.0], name='BB1'))
    return L

L = fresh()
check('a mechanics registers', L.get_mechanics('BB1').name == 'BB1')
check('it is held by reference',
      L.get_mechanics('BB1') is L.mechanics[0])

for taken in ['M1', 'b0', 'BB1']:
    try:
        L.add_mechanics(Mechanics(name=taken))
        check('the name %r is refused' % taken, False)
    except ValueError:
        check('the name %r is refused' % taken, True)
try:
    L.add_optics(opt.Mirror(name='BB1'))
    check('and an optics cannot take a mechanics name', False)
except ValueError:
    check('and an optics cannot take a mechanics name', True)

check('unique_mechanics_name walks the whole namespace',
      L.unique_mechanics_name() == 'H1')
L.add_mechanics(Mechanics(name='H1'))
check('  and moves on when taken', L.unique_mechanics_name() == 'H2')
L.remove_mechanics('H1')


print('--- the scene ---')

L = fresh()
scene = L.scene_dict()
check('the scene is strict JSON', json.dumps(scene) is not None)
check('the mechanics channel is there', len(scene['mechanics']) == 1)
mch = scene['mechanics'][0]
check('it carries the pose and the labels',
      mch['name'] == 'BB1' and mch['type'] == 'Mechanics'
      and mch['layer'] == 'hardware' and mch['model'] is None
      and close(mch['center'], [0.3, 0.0]) and mch['rotationAngle'] == 0.0)
check('and the outline the model computes',
      close(mch['outline'], L.get_mechanics('BB1').outline()))

check('the canvas carries the hardware layer',
      any(ly['name'] == DEFAULT_LAYER and len(ly['shapes']) == 1
          for ly in scene['canvas']['layers']))

snaps = [p for p in scene['snap'] if p['optic'] == 'BB1']
check('the corners and the centre are snap points',
      len(snaps) == 5
      and sorted(p['kind'] for p in snaps) == ['centre'] + ['corner'] * 4)
corner_pts = [p['point'] for p in snaps if p['kind'] == 'corner']
check('  at the outline corners',
      close(sorted(map(tuple, corner_pts)),
            sorted(map(tuple, L.get_mechanics('BB1').outline()))))

L.apply_edit({'op': 'draw', 'params': {'drawOpticsNames': False}})
scene = L.scene_dict()
texts = [s for ly in scene['canvas']['layers'] if ly['name'] == 'text'
         for s in ly['shapes'] if s.get('text') == 'BB1']
check('the name follows the element-names option', texts == [])


print('--- the edit protocol ---')

L = fresh()
bb = L.get_mechanics('BB1')

L.trace()
check('trace stands before the test', L.beams is not None)
L.apply_edit({'op': 'move', 'target': 'BB1', 'center': [0.35, 0.05]})
check('move lands on the centre', close(bb.center, [0.35, 0.05]))
check('and does not invalidate the trace', L.beams is not None)

L.apply_edit({'op': 'rotate', 'target': 'BB1', 'rotationAngle': 0.2})
check('rotate lands on the angle', bb.rotationAngle == 0.2)
check('still no re-trace', L.beams is not None)

L.apply_edit({'op': 'set', 'target': 'BB1',
              'attrs': {'center': [0.3, 0.0], 'rotationAngle': 0.0}})
check('set takes both at once',
      close(bb.center, [0.3, 0.0]) and bb.rotationAngle == 0.0)

L.apply_edit({'op': 'rename', 'target': 'BB1', 'name': 'Board'})
check('rename is the same object under a new name',
      bb.name == 'Board' and L.get_mechanics('Board') is bb)

L.apply_edit({'op': 'add', 'type': 'Mechanics',
              'params': {'center': [0.1, 0.1],
                         'shapes': [{'type': 'circle', 'center': [0, 0],
                                     'radius': 0.02, 'thickness': 0}],
                         'layer': 'posts', 'model': 'P-2'}})
check('add without a name picks one', L._is_mechanics('H1'))
h1 = L.get_mechanics('H1')
check('  with the parameters given',
      h1.layer == 'posts' and h1.model == 'P-2'
      and isinstance(h1.shapes[0], draw.Circle))

L.apply_edit({'op': 'remove', 'target': 'H1'})
check('remove takes it out', not L._is_mechanics('H1'))

refused(L, {'op': 'set', 'target': 'Board', 'attrs': {'shapes': []}},
        'setting the shapes')
refused(L, {'op': 'set', 'target': 'Board', 'attrs': {'layer': 'x'}},
        'setting the layer')
refused(L, {'op': 'move', 'target': 'Board', 'center': [None, 0.0]},
        'a centre that is not a point')
refused(L, {'op': 'rotate', 'target': 'Board',
            'rotationAngle': float('nan')}, 'an angle that is not finite')
refused(L, {'op': 'move', 'target': 'Board'}, 'a move with nothing in it')
refused(L, {'op': 'add', 'type': 'Mechanics', 'name': 'Board'},
        'a name already taken')
refused(L, {'op': 'add', 'type': 'Mechanics', 'name': ' '},
        'a blank name')
refused(L, {'op': 'add', 'type': 'Mechanics',
            'params': {'shapes': [{'type': 'blob'}]}},
        'a shape gtrace cannot draw')
refused(L, {'op': 'add', 'type': 'Mechanics',
            'params': {'shapes': [{'type': 'circle'}]}},
        'a shape missing its geometry')
refused(L, {'op': 'add', 'type': 'Mechanics',
            'params': {'shapes': 'no'}}, 'shapes that are not a list')
refused(L, {'op': 'add', 'type': 'Mechanics', 'params': {'layer': ''}},
        'a blank layer')
refused(L, {'op': 'add', 'type': 'Mechanics', 'params': {'model': 3}},
        'a model that is not a name')
refused(L, {'op': 'add', 'type': 'Mechanics', 'params': {'f': 0.5}},
        'a parameter a mechanics does not take')
refused(L, {'op': 'align', 'target': 'Board', 'beam': 'b0',
            'beam_index': 0, 'point': [0.1, 0.0]},
        'aligning hardware to a beam')
refused(L, {'op': 'slide', 'target': 'Board', 'beam': 'b0',
            'beam_index': 0, 'distance': 0.05},
        'sliding hardware along a beam')


print('--- undo and redo keep identity ---')

L = fresh()
bb = L.get_mechanics('BB1')
c0 = bb.center.copy()

L.apply_edit({'op': 'move', 'target': 'BB1', 'center': [0.9, 0.9]})
L.apply_edit({'op': 'undo'})
check('undo of a move restores the centre', close(bb.center, c0))
L.apply_edit({'op': 'redo'})
check('redo brings it back', close(bb.center, [0.9, 0.9]))
L.apply_edit({'op': 'undo'})

L.apply_edit({'op': 'remove', 'target': 'BB1'})
check('removed', not L._is_mechanics('BB1'))
L.apply_edit({'op': 'undo'})
check('undo of a removal returns the very object',
      L._is_mechanics('BB1') and L.get_mechanics('BB1') is bb)

L.apply_edit({'op': 'rename', 'target': 'BB1', 'name': 'Board'})
L.apply_edit({'op': 'undo'})
check('undo of a rename keeps the object too',
      bb.name == 'BB1' and L.get_mechanics('BB1') is bb)


print('--- attachment: the pose is the host\'s ---')

L = fresh()
M1 = L.get_optics('M1')
mt = Mechanics(shapes=[draw.Circle([0, 0], 0.02)], name='MT1',
               attached_to=M1)
L.add_mechanics(mt)

check('offset [0,0] stands at the host centre',
      close(mt.center, M1.center))
check('and turns as the host faces',
      abs(mt.rotationAngle - float(M1.normAngleHR)) < 1e-15)

# The whole point: the host moves and the mount follows, with no
# notification anywhere to miss. The 2026-08-03 bugs were all a
# notification not arriving; a derived value has none to lose.
M1.HRcenter = [0.6, 0.1]
check('the mount follows a host moved directly in Python',
      close(mt.center, M1.center))
M1.normAngleHR = np.pi / 2
check('and follows a turn', abs(mt.rotationAngle - np.pi / 2) < 1e-15)

mt.offset = np.array([0.0, -0.05])
check('the offset lives in the host frame',
      close(mt.center, np.asarray(M1.center) + [0.05, 0.0]))

L.apply_edit({'op': 'rotate', 'target': 'M1', 'normAngleHR': np.pi})
# The same local offset [0, -0.05], now seen through a host facing pi:
# R(pi) flips both axes, so it lands at +0.05 across.
check('and through the edit protocol too',
      close(mt.center, np.asarray(M1.center) + [0.0, 0.05]))

for fn, what in [(lambda: setattr(mt, 'center', [0, 0]), 'the centre'),
                 (lambda: setattr(mt, 'rotationAngle', 0.1), 'the angle'),
                 (lambda: mt.translate([0.1, 0]), 'translate()'),
                 (lambda: mt.rotate(0.1), 'rotate()')]:
    try:
        fn()
        check('writing %s of an attached body is refused' % what, False)
    except ValueError as e:
        check('writing %s of an attached body is refused' % what, True,
              '(%s)' % str(e)[:40])

c0 = mt.center.copy()
a0 = mt.rotationAngle
mt.detach()
check('detach bakes the derived pose in',
      close(mt.center, c0) and mt.rotationAngle == a0
      and mt.attached_to is None)
mt.center = [0.2, 0.2]
check('and the freed body edits again', close(mt.center, [0.2, 0.2]))

mt.attach(M1)
# The baked angle was the host's pi, and keeping the pose keeps it.
check('attach with no offset keeps the body where it stands',
      close(mt.center, [0.2, 0.2], tol=1e-12)
      and abs(mt.rotationAngle - a0) < 1e-12)

try:
    mt.attach(Mechanics(name='other'))
    check('hardware cannot stand on hardware', False)
except ValueError as e:
    check('hardware cannot stand on hardware', True, '(%s)' % str(e)[:40])
try:
    Mechanics(name='x').attach(object())
    check('nor on something with no pose', False)
except ValueError:
    check('nor on something with no pose', True)
try:
    Mechanics(name='x', attached_to=M1, center=[0, 0])
    check('a pose alongside an attachment is refused', False)
except ValueError:
    check('a pose alongside an attachment is refused', True)

c = mt.copy()
check('a copy stands on the same host', c.attached_to is M1
      and close(c.offset, mt.offset))
mt.detach()


print('--- attachment by name, resolved at registration ---')

L = fresh()
M1 = L.get_optics('M1')
p = Mechanics(name='P1', attached_to='M1', offset=[0.0, 0.02])
try:
    p.center
    check('an unresolved link refuses to give a pose', False)
except ValueError as e:
    check('an unresolved link refuses to give a pose', True,
          '(%s)' % str(e)[:40])
L.add_mechanics(p)
check('registration joins it to the optics of that name',
      p.attached_to is M1 and close(p.offset, [0.0, 0.02]))
L.remove_mechanics('P1')

try:
    L.add_mechanics(Mechanics(name='P2', attached_to='NoSuch'))
    check('a host name nothing answers to is refused', False)
except ValueError as e:
    check('a host name nothing answers to is refused', True,
          '(%s)' % str(e)[:50])


print('--- attachment through the protocol ---')

L = fresh()
M1 = L.get_optics('M1')
bb = L.get_mechanics('BB1')

L.trace()
L.apply_edit({'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': 'M1'}})
check('set attached_to attaches where it stands',
      bb.attached_to is M1 and close(bb.center, [0.3, 0.0]))
check('and does not invalidate the trace', L.beams is not None)

refused(L, {'op': 'move', 'target': 'BB1', 'center': [0, 0]},
        'moving an attached body')
refused(L, {'op': 'rotate', 'target': 'BB1', 'rotationAngle': 0.1},
        'turning an attached body')
refused(L, {'op': 'set', 'target': 'BB1',
            'attrs': {'attached_to': 'M1', 'center': [0, 0]}},
        'a pose alongside an attachment')
refused(L, {'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': 'b0'}},
        'attaching to a source')
refused(L, {'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': 'BB1'}},
        'attaching to hardware')
refused(L, {'op': 'remove', 'target': 'M1'},
        'removing an optics with hardware attached')

L.apply_edit({'op': 'set', 'target': 'BB1',
              'attrs': {'offset': [0.05, 0.0], 'offset_angle': 0.1}})
check('the offset edits while attached',
      close(bb.offset, [0.05, 0.0]) and bb.offset_angle == 0.1)

pose_before = (bb.center.copy(), bb.rotationAngle)
L.apply_edit({'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': None}})
check('null detaches, and the body stays where it stood',
      bb.attached_to is None and close(bb.center, pose_before[0])
      and bb.rotationAngle == pose_before[1])
refused(L, {'op': 'set', 'target': 'BB1', 'attrs': {'offset': [0, 0]}},
        'an offset on a body standing on its own')

L.apply_edit({'op': 'undo'})
check('undo of a detach re-attaches', bb.attached_to is M1)
L.apply_edit({'op': 'redo'})
check('redo detaches again', bb.attached_to is None)
L.apply_edit({'op': 'undo'})

pos0 = bb.center.copy()
L.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.7, 0.2]})
moved = bb.center.copy()
check('the mount moves with its host under the protocol',
      not close(moved, pos0))
L.apply_edit({'op': 'undo'})
check('and back when the host move is undone', close(bb.center, pos0))

L.apply_edit({'op': 'add', 'type': 'Mechanics', 'name': 'MT2',
              'params': {'attached_to': 'M1', 'offset': [0.0, -0.08],
                         'shapes': [{'type': 'circle', 'center': [0, 0],
                                     'radius': 0.01, 'thickness': 0}]}})
mt2 = L.get_mechanics('MT2')
check('add with attached_to lands on the host',
      mt2.attached_to is M1)
refused(L, {'op': 'add', 'type': 'Mechanics',
            'params': {'attached_to': 'NoSuch'}},
        'adding onto an optics that is not there')
refused(L, {'op': 'add', 'type': 'Mechanics',
            'params': {'attached_to': 'M1', 'center': [0, 0]}},
        'adding with a pose and an attachment at once')

L.apply_edit({'op': 'remove', 'target': 'MT2'})
check('the mount itself removes freely', not L._is_mechanics('MT2'))
L.apply_edit({'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': None}})
L.apply_edit({'op': 'remove', 'target': 'M1'})
check('and the host removes once nothing stands on it',
      'M1' not in [o.name for o in L.optics])


print('--- attachment saved by name, and relinked ---')

L = fresh()
M1 = L.get_optics('M1')
bb = L.get_mechanics('BB1')
L.apply_edit({'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': 'M1'}})

d = mechanics_to_dict(bb)
check('an attached body saves its host and offset, and no pose',
      d['attached_to'] == 'M1' and 'offset' in d
      and 'center' not in d and 'rotationAngle' not in d)
d_free = mechanics_to_dict(Mechanics(name='f', center=[1, 2]))
check('a free body saves its pose, and no attachment',
      'attached_to' not in d_free and close(d_free['center'], [1, 2]))

L.apply_edit({'op': 'rename', 'target': 'M1', 'name': 'PRM'})
check('a renamed host saves under its current name',
      mechanics_to_dict(bb)['attached_to'] == 'PRM')
L.apply_edit({'op': 'undo'})

full = L.to_dict()
L2 = OpticalLayout.from_dict(full)
bb2 = L2.get_mechanics('BB1')
check('a fresh load joins the mount to its own host',
      bb2.attached_to is L2.get_optics('M1'))
L2.get_optics('M1').HRcenter = [0.9, 0.9]
# The offset was derived when BB1 attached where it stood: [0.225, 0]
# in the frame of a host at [0.525, 0] facing pi. The moved host stands
# at [0.925, 0.9], still facing pi, so the mount lands at [0.7, 0.9].
check('and it follows that host, not the original',
      close(bb2.center, [0.7, 0.9]) and close(bb.center, [0.3, 0.0]))

path = os.path.join(WORK, 'mech_attach_layout.json')
L.save(path)
L.apply_edit({'op': 'set', 'target': 'BB1', 'attrs': {'attached_to': None}})
L.apply_edit({'op': 'load', 'path': path})
check('loading into a layout re-attaches the same objects',
      L.get_mechanics('BB1') is bb and bb.attached_to is M1)

dangling = L.to_dict()
dangling['optics'] = []
try:
    OpticalLayout.from_dict(dangling)
    check('a file whose host is missing is refused', False)
except ValueError as e:
    check('a file whose host is missing is refused', True,
          '(%s)' % str(e)[:50])

scene = L.scene_dict()
mch = [m for m in scene['mechanics'] if m['name'] == 'BB1'][0]
check('the scene names the host', mch['attached_to'] == 'M1')
check('and carries the derived pose', close(mch['center'], bb.center))
check('a free body rides as null',
      json.dumps(scene) is not None
      and all('attached_to' in m for m in scene['mechanics']))


print('--- save, load, and the merge that keeps identity ---')

L = fresh()
bb = L.get_mechanics('BB1')
path = os.path.join(WORK, 'mech_layout.json')
L.apply_edit({'op': 'save', 'path': path})
with open(path, encoding='utf-8') as f:
    on_disk = json.load(f)
check('the file carries the mechanics',
      len(on_disk['mechanics']) == 1
      and on_disk['mechanics'][0]['name'] == 'BB1')

L.apply_edit({'op': 'move', 'target': 'BB1', 'center': [0.9, 0.9]})
L.apply_edit({'op': 'load', 'path': path})
check('loading puts the saved pose back', close(bb.center, [0.3, 0.0]))
check('  onto the same object', L.get_mechanics('BB1') is bb)

L2 = OpticalLayout.load(path)
check('a fresh load rebuilds the body',
      L2.get_mechanics('BB1').contains([0.3, 0.0])
      and isinstance(L2.get_mechanics('BB1').shapes[0], draw.Rectangle))

old = {'name': 'old', 'optics': [], 'sources': [], 'rules': {}}
check('a file from before mechanics existed still loads',
      OpticalLayout.from_dict(old).mechanics == [])


print('--- the builders ---')

bb = breadboard(0.30, 0.30, name='BB')
plate = [s for s in bb.shapes if isinstance(s, draw.Rectangle)]
holes = [s for s in bb.shapes if isinstance(s, draw.Circle)]
check('a 300 mm board on a 25 mm grid drills 12 x 12',
      len(plate) == 1 and len(holes) == 144, str(len(holes)))
xs = sorted(set(round(float(s.center[0]), 9) for s in holes))
check('the grid is symmetric about the centre',
      xs[0] == -xs[-1] and abs(xs[0] + 0.1375) < 1e-9,
      '(%s .. %s)' % (xs[0], xs[-1]))
check('rows land on the pitch',
      all(abs((xs[i + 1] - xs[i]) - 0.025) < 1e-9
          for i in range(len(xs) - 1)))
lo, hi = bb.local_bbox()
check('the plate bounds the body, holes inside it',
      close(lo, [-0.15, -0.15]) and close(hi, [0.15, 0.15]))
check('holes=False is just the outline',
      len(breadboard(0.3, 0.3, holes=False).shapes) == 1)
check('a board too small for its margin drills nothing',
      len(breadboard(0.02, 0.02).shapes) == 1)
check('a wider margin thins the grid',
      len([s for s in breadboard(0.30, 0.30, margin=0.05).shapes
           if isinstance(s, draw.Circle)]) == 81)
check('the pose kwargs pass through',
      close(breadboard(0.1, 0.1, center=[1, 2], name='X').center, [1, 2]))

mm_ = mirror_mount()
lo, hi = mm_.local_bbox()
check('the mount stands wholly behind the origin', hi[0] < 0)
check('  a clearance in front of its plate',
      abs(max(float(s.point[0]) + s.width
              for s in mm_.shapes if isinstance(s, draw.Rectangle))
          + 0.01) < 1e-12)
check('  as wide as asked', abs((hi[1] - lo[1]) - 0.05) < 1e-12)
check('knobs=False is just the plate',
      len(mirror_mount(knobs=False).shapes) == 1)
knobs = [s for s in mirror_mount().shapes if isinstance(s, draw.Circle)]
check('the knobs stick out of the back',
      len(knobs) == 2 and all(float(s.center[0]) < -0.022 for s in knobs))


print('--- the model library ---')

stock = models()
check('the generic stock is registered',
      all(k in stock for k in ['BB3030', 'BB4530', 'BB6045',
                               'MOUNT-25', 'MOUNT-50']),
      str(list(stock)))

b = from_model('BB3030', name='Bench', center=[0.4, 0.0])
check('from_model builds the shapes and carries the label',
      b.model == 'BB3030' and len(b.shapes) == 145
      and close(b.center, [0.4, 0.0]))
b2 = from_model('BB3030', name='Bench2')
b2.shapes.pop()
check('two bodies from one model do not share shapes',
      len(b.shapes) == 145 and len(b2.shapes) == 144)
try:
    from_model('NO-SUCH-MODEL')
    check('an unknown model is refused', False)
except KeyError as e:
    check('an unknown model is refused', True, '(%s)' % str(e)[:40])

src = Mechanics(shapes=[draw.Rectangle([-0.01, -0.01], 0.02, 0.02)],
                name='c', layer='clamps')
register_model('TEST-CLAMP', src, 'a test clamp')
src.shapes.append(draw.Circle([0, 0], 0.001))
check('a model is registered by value, not by reference',
      len(model_shapes('TEST-CLAMP')) == 1)
check('and carries its layer', from_model('TEST-CLAMP').layer == 'clamps')
check('a bare shape list registers on the default layer',
      (register_model('TEST-BLOCK', [draw.Circle([0, 0], 0.01)]) or True)
      and from_model('TEST-BLOCK').layer == DEFAULT_LAYER)
check('models() describes them',
      models()['TEST-CLAMP'] == 'a test clamp')


print('--- relink: the saved values are the truth until asked ---')

L = fresh()
M1 = L.get_optics('M1')
c1 = from_model('TEST-CLAMP', name='C1', attached_to=M1)
v1 = Mechanics(shapes=[draw.Circle([0, 0], 0.005)], name='V1',
               center=[0.2, 0.2], model='VENDOR-ONLY')
L.add_mechanics(c1)
L.add_mechanics(v1)

# The library moves on after the layout was built.
register_model('TEST-CLAMP', [draw.Rectangle([-0.02, -0.02], 0.04, 0.04)],
               'a bigger clamp')

d = L.to_dict()
L2 = OpticalLayout.from_dict(d)
check('a load keeps the shapes it was saved with',
      abs(L2.get_mechanics('C1').shapes[0].width - 0.02) < 1e-12)

out = L2.relink_mechanics()
check('relink redraws the labelled body from the library',
      out == ['C1']
      and abs(L2.get_mechanics('C1').shapes[0].width - 0.04) < 1e-12,
      str(out))
check('  a model the library does not know is left alone',
      isinstance(L2.get_mechanics('V1').shapes[0], draw.Circle))
check('  a body with no label is left alone',
      L2.get_mechanics('BB1').shapes[0].width == 0.6)
check('  and the pose and attachment stay',
      L2.get_mechanics('C1').attached_to is L2.get_optics('M1')
      and L2.get_mechanics('C1').model == 'TEST-CLAMP')

check('relink can be aimed at names',
      L.relink_mechanics(names=['V1']) == []
      and abs(L.get_mechanics('C1').shapes[0].width - 0.02) < 1e-12)

full = L2.to_dict()
c1d = [m for m in full['mechanics'] if m['name'] == 'C1'][0]
check('a relinked body saves the new shapes, by value',
      abs(c1d['shapes'][0]['width'] - 0.04) < 1e-12
      and c1d['model'] == 'TEST-CLAMP')


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
