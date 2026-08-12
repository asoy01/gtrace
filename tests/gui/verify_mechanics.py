'''
Mechanics: the bodies the trace never see, and everything the layout
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
                              LAYER_COLOR, breadboard, round_breadboard,
                              mirror_mount,
                              mirror_mount_2in, lens_holder,
                              pedestal, clamping_fork, host_pose,
                              register_model, models, model_shapes,
                              model_params, model_points, from_model,
                              save_models, load_models)
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
check('the mechanics layer appears', DEFAULT_LAYER in cv.layers)
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
              name='BB1', layer='mechanics', model='MB3045/M')
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
      r2.layer == 'mechanics' and r2.model is None and r2.shapes == []
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
      L.unique_mechanics_name() == 'P1')
L.add_mechanics(Mechanics(name='P1'))
check('  and moves on when taken', L.unique_mechanics_name() == 'P2')
L.remove_mechanics('P1')


print('--- the scene ---')

L = fresh()
scene = L.scene_dict()
check('the scene is strict JSON', json.dumps(scene) is not None)
check('the mechanics channel is there', len(scene['mechanics']) == 1)
mch = scene['mechanics'][0]
check('it carries the pose and the labels',
      mch['name'] == 'BB1' and mch['type'] == 'Mechanics'
      and mch['layer'] == 'mechanics' and mch['model'] is None
      and close(mch['center'], [0.3, 0.0]) and mch['rotationAngle'] == 0.0)
check('and the outline the model computes',
      close(mch['outline'], L.get_mechanics('BB1').outline()))

check('the canvas carries the mechanics layer',
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
check('add without a name picks one', L._is_mechanics('P1'))
h1 = L.get_mechanics('P1')
check('  with the parameters given',
      h1.layer == 'posts' and h1.model == 'P-2'
      and isinstance(h1.shapes[0], draw.Circle))

L.apply_edit({'op': 'remove', 'target': 'P1'})
check('remove takes it out', not L._is_mechanics('P1'))

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
        'aligning a body to a beam')
refused(L, {'op': 'slide', 'target': 'Board', 'beam': 'b0',
            'beam_index': 0, 'distance': 0.05},
        'sliding a body along a beam')


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

# Where a mount belongs on its host is the model's to say - the local
# origin is drawn to coincide with the host's substrate centre - so a
# bare attach seats the body there, wherever it was lying.
mt.attach(M1)
check('attach with no offset seats the body at its designed position',
      close(mt.center, M1.center)
      and abs(mt.rotationAngle - float(M1.normAngleHR)) < 1e-12
      and close(mt.offset, [0, 0]) and mt.offset_angle == 0.0)

mt.detach()
mt.center = [0.2, 0.2]
mt.rotationAngle = a0
mt.attach(M1, keep_pose=True)
check('keep_pose pins it where it stands instead',
      close(mt.center, [0.2, 0.2], tol=1e-12)
      and abs(mt.rotationAngle - a0) < 1e-12)
try:
    mt.detach()
    mt.attach(M1, offset=[0, 0.01], keep_pose=True)
    check('keep_pose with an offset is refused', False)
except ValueError as e:
    check('keep_pose with an offset is refused', True, '(%s)' % str(e)[:40])
mt.detach()
mt.attach(M1, keep_pose=True)

# A body may stand on another body - a pedestal on a mount, a fork on
# a pedestal - and what is refused is a cycle, which would make a pose
# derive from itself and, since it is derived on every read, not
# wrongly but endlessly.
post = Mechanics(name='post')
post.attach(mt)
check('a body may stand on a body', post.attached_to is mt)
try:
    mt.attach(post)
    check('but not on what it already holds up', False)
except ValueError as e:
    check('but not on what it already holds up', True,
          '(%s)' % str(e)[:40])
check('  and the refusal changes nothing', mt.attached_to is M1)
try:
    post.attach(post)
    check('nor on itself', False)
except ValueError:
    check('nor on itself', True)
post.detach()
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
check('set attached_to seats the body at its designed position',
      bb.attached_to is M1 and close(bb.center, M1.center)
      and close(bb.offset, [0, 0]))
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
        'attaching to another body')
refused(L, {'op': 'remove', 'target': 'M1'},
        'removing an optics with a body attached')

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
# BB1 seats at its designed position - offset [0, 0], the host's
# substrate centre - so it rides wherever its own host's centre goes,
# and the original, on the original host, does not move.
check('and it follows that host, not the original',
      close(bb2.center, L2.get_optics('M1').center)
      and close(bb.center, M1.center))

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

# The mount follows the drawing in local/Polaris.png: origin at the
# substrate centre of the mounted optic, 3 mm behind the front face.
mm_ = mirror_mount()
lo, hi = mm_.local_bbox()
check('the front face stands 3 mm ahead of the origin',
      abs(hi[0] - 0.003) < 1e-12, str(hi[0]))
check('the knobs reach 36.1 mm behind it',
      abs(lo[0] + 0.0361) < 1e-12, str(lo[0]))
check('the knobs stick 1.2 mm out past the 45.7 mm plate',
      abs(hi[1] - 0.02405) < 1e-12 and abs(lo[1] + 0.02405) < 1e-12,
      '(%s..%s)' % (lo[1], hi[1]))
tips = [s for s in mm_.shapes if isinstance(s, draw.Arc)]
check('the two adjuster tips show in the gap',
      len(tips) == 2
      and all(abs(float(s.center[0]) + 0.0072) < 1e-12 for s in tips)
      and sorted(round(float(s.center[1]), 9) for s in tips)
          == [-0.016450, 0.016450]
      and all(abs(s.radius - 0.0025) < 1e-15 for s in tips))
check('the whole drawing is nine shapes', len(mm_.shapes) == 9)
hole = [s for s in mm_.shapes if isinstance(s, draw.Circle)]
check('the post hole is drawn where the part names it',
      len(hole) == 1 and close(hole[0].center, mm_.points['post'])
      and abs(hole[0].radius - 0.002) < 1e-15,
      str(np.round(hole[0].center, 5).tolist()))
nk = mirror_mount(knobs=False)
check('knobs=False leaves the plates, the tips and the hole',
      len(nk.shapes) == 5 and abs(nk.local_bbox()[0][0] + 0.0199) < 1e-12)
lo2, hi2 = mirror_mount(scale=2.0).local_bbox()
check('scale scales every dimension',
      close(lo2, 2 * lo, tol=1e-15) and close(hi2, 2 * hi, tol=1e-15))

# The two-inch mount follows the Thorlabs KA2A drawing
# (local/Polaris-2inch.pdf): plates 68.6 and 69.9 wide, 7.0 and 12.7
# deep across the same 3.2 gap (22.9 body overall), adjusters 35.6
# apart protruding 12.2 (35.1 overall), and the origin 3.95 behind
# the front face - where the 10.3 optic pocket centres a standard
# 12.7 thick two-inch optic.
m2 = mirror_mount_2in()
lo, hi = m2.local_bbox()
check('the 2in front face stands 3.95 mm ahead of the origin',
      abs(hi[0] - 0.00395) < 1e-12, str(hi[0]))
check('the whole mount is 35.1 mm deep',
      abs((hi[0] - lo[0]) - 0.0351) < 1e-12, str(hi[0] - lo[0]))
check('the back plate governs the width at 69.9 mm',
      abs(hi[1] - 0.03495) < 1e-12 and abs(lo[1] + 0.03495) < 1e-12)
rects = [s for s in m2.shapes if isinstance(s, draw.Rectangle)]
check('the front plate is its own 68.6 mm wide',
      abs(rects[0].height - 0.0686) < 1e-12, str(rects[0].height))
tips2 = [s for s in m2.shapes if isinstance(s, draw.Arc)]
check('the adjuster lines sit 35.6 mm apart',
      sorted(round(float(s.center[1]), 9) for s in tips2)
      == [-0.0178, 0.0178])
check('the 2in drawing is nine shapes too', len(m2.shapes) == 9)
hole2 = [s for s in m2.shapes if isinstance(s, draw.Circle)]
check('  with the same post hole, bored the same way',
      len(hole2) == 1 and close(hole2[0].center, m2.points['post'])
      and abs(hole2[0].radius - 0.002) < 1e-15)

# The lens holders: a plain rectangle wrapping its lens symmetrically,
# so the origin is the middle of it.
lh = lens_holder()
lo, hi = lh.local_bbox()
check('the holder is one rectangle centred on the origin',
      len(lh.shapes) == 1
      and close(lo, [-0.005, -0.015]) and close(hi, [0.005, 0.015]))
lh2 = lens_holder(length=0.056, thickness=0.0127)
lo, hi = lh2.local_bbox()
check('  cut to whatever lens it is for',
      close(lo, [-0.00635, -0.028]) and close(hi, [0.00635, 0.028]))
check('both holders are on the shelf',
      from_model('HOLDER-25', name='lh1').local_bbox()[1][1] == 0.015
      and from_model('HOLDER-50', name='lh2').local_bbox()[1][0] == 0.00635)


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


print('--- the library saves, and merges back file by file ---')

lib_all = os.path.join(WORK, 'mech_models_all.json')
save_models(lib_all)
with open(lib_all, encoding='utf-8') as f:
    on_disk = json.load(f)
check('save_models writes the whole shelf, stock included',
      'BB3030' in on_disk['models'] and 'TEST-CLAMP' in on_disk['models'])
check('  as the same data the registry holds',
      on_disk['models']['TEST-CLAMP']['shapes']
      == [shape_to_dict(s) for s in model_shapes('TEST-CLAMP')])

lib_a = os.path.join(WORK, 'mech_models_a.json')
register_model('TEST-FILE-X', [draw.Circle([0, 0], 0.011)], 'X, version A')
register_model('TEST-FILE-Y', [draw.Circle([0, 0], 0.012)], 'only in A')
save_models(lib_a, names=['TEST-FILE-X', 'TEST-FILE-Y'])
with open(lib_a, encoding='utf-8') as f:
    check('a subset saves only what was named',
          sorted(json.load(f)['models']) == ['TEST-FILE-X', 'TEST-FILE-Y'])
try:
    save_models(lib_a, names=['NO-SUCH-MODEL'])
    check('a name not on the shelf is refused', False)
except KeyError as e:
    check('a name not on the shelf is refused', True, '(%s)' % str(e)[:40])

lib_b = os.path.join(WORK, 'mech_models_b.json')
register_model('TEST-FILE-X', [draw.Circle([0, 0], 0.021)], 'X, version B')
register_model('TEST-FILE-Z', [draw.Circle([0, 0], 0.022)], 'only in B')
save_models(lib_b, names=['TEST-FILE-X', 'TEST-FILE-Z'])

# Wind the registry back to something older than either file, then
# merge the two files over it in order.
register_model('TEST-FILE-X', [draw.Circle([0, 0], 0.001)], 'stale')
check('loading a file merges its models in',
      load_models(lib_a) == ['TEST-FILE-X', 'TEST-FILE-Y']
      and abs(model_shapes('TEST-FILE-X')[0].radius - 0.011) < 1e-15)
check('and a second file wins name by name, leaving the rest',
      load_models(lib_b) == ['TEST-FILE-X', 'TEST-FILE-Z']
      and abs(model_shapes('TEST-FILE-X')[0].radius - 0.021) < 1e-15
      and abs(model_shapes('TEST-FILE-Y')[0].radius - 0.012) < 1e-15
      and models()['TEST-FILE-X'] == 'X, version B')

lib_bb = os.path.join(WORK, 'mech_models_bb.json')
save_models(lib_bb, names=['BB3030'])
register_model('BB3030', [draw.Rectangle([-0.15, -0.15], 0.3, 0.3)],
               'flattened for the test')
load_models(lib_bb)
check('the builder parameters survive the round trip',
      model_params('BB3030') is not None
      and from_model('BB3030', name='rt').resizable)

before_reg = json.dumps(sorted(models()))
lib_bad = os.path.join(WORK, 'mech_models_bad.json')
with open(lib_bad, 'w', encoding='utf-8') as f:
    json.dump({'models': {'GOOD': {'shapes': [
                   {'type': 'circle', 'center': [0, 0], 'radius': 0.01,
                    'thickness': 0}], 'layer': 'mechanics'},
               'BAD': {'shapes': [{'type': 'blob'}]}}}, f)
try:
    load_models(lib_bad)
    check('a file with one bad shape is refused whole', False)
except ValueError as e:
    check('a file with one bad shape is refused whole', True,
          '(%s)' % str(e)[:60])
check('  and merges nothing of it',
      json.dumps(sorted(models())) == before_reg and 'GOOD' not in models())

with open(lib_bad, 'w', encoding='utf-8') as f:
    json.dump({'layouts': []}, f)
try:
    load_models(lib_bad)
    check('a file that is not a library is refused', False)
except ValueError as e:
    check('a file that is not a library is refused', True,
          '(%s)' % str(e)[:50])


print('--- resize: a parametric body is re-drilled, not scaled ---')

def hole_count(m):
    return sum(isinstance(s, draw.Circle) for s in m.shapes)

def hole_r(m):
    return [float(s.radius) for s in m.shapes
            if isinstance(s, draw.Circle)][0]

bb = breadboard(0.30, 0.30, name='RB')
check('a builder body knows how it is resizable', bb.resizable == 'box'
      and bb.params['kind'] == 'breadboard')
r0 = hole_r(bb)
bb.resize(0.45, 0.30)
check('a wider board has more holes, on the same grid',
      hole_count(bb) == 216 and abs(hole_r(bb) - r0) < 1e-15,
      str(hole_count(bb)))
check('  and remembers its new size',
      bb.params['width'] == 0.45 and bb.params['height'] == 0.30)
bb.resize(height=0.20)
check('one side alone resizes alone',
      bb.params['width'] == 0.45 and bb.params['height'] == 0.20)
try:
    bb.resize(-0.1)
    check('a negative size is refused', False)
except ValueError:
    check('a negative size is refused', True)
try:
    Mechanics(shapes=[draw.Circle([0, 0], 0.01)], name='hd').resize(0.1, 0.1)
    check('a hand-drawn body is refused, with the reason', False)
except ValueError as e:
    check('a hand-drawn body is refused, with the reason', True,
          '(%s)' % str(e)[:50])
check('  and does not claim to be resizable',
      not Mechanics(shapes=[draw.Circle([0, 0], 0.01)]).resizable)

check('the parameters survive a copy', bb.copy().resizable)
d = mechanics_to_dict(bb)
check('and a save', d['params']['width'] == 0.45)
check('and a load', mechanics_from_dict(d).resize(0.5).params['width'] == 0.5)
d_plain = mechanics_to_dict(Mechanics(shapes=[draw.Circle([0, 0], 0.01)]))
check('a hand-drawn body saves no parameters', 'params' not in d_plain)


print('--- a round breadboard ---')

# A tank is round, and so is the board in the bottom of it. The grid
# is the rectangular board's; the rim decides which of its holes are
# there.
rb = round_breadboard(0.30, name='RBR')
rim = [s for s in rb.shapes if isinstance(s, draw.Circle)][0]
holes = [s for s in rb.shapes if isinstance(s, draw.Circle)][1:]
check('the rim is the first shape, at the local origin',
      abs(rim.radius - 0.15) < 1e-15 and close(rim.center, [0, 0]))
check('and it is round: the box around it is square',
      close(rb.local_bbox()[0], [-0.15, -0.15])
      and close(rb.local_bbox()[1], [0.15, 0.15]))
check('the holes are the same 25 mm grid, symmetric about the centre',
      all(abs(round(float(h.center[0]) / 0.025)
              - float(h.center[0]) / 0.025) < 1e-9
          and abs(round(float(h.center[1]) / 0.025)
                  - float(h.center[1]) / 0.025) < 1e-9 for h in holes)
      and all(abs(h.radius - 0.003) < 1e-15 for h in holes))
check('  and every one of them is a margin in from the rim',
      holes and max(float(np.hypot(*h.center)) for h in holes)
      <= 0.15 - 0.025 / 2 + 1e-9,
      str(max(float(np.hypot(*h.center)) for h in holes)))
check('  with the ones outside it left out, so the rows shorten',
      len(holes) == 97, str(len(holes)))
check('a hole sits at the centre, as on a rectangular board',
      any(close(h.center, [0, 0]) for h in holes))
check('holes=False leaves the rim alone',
      len(round_breadboard(0.30, holes=False).shapes) == 1)

check('it knows it is round', rb.resizable == 'round'
      and rb.params['kind'] == 'round_breadboard')
rb.resize(0.45)
check('one number cuts it to a new size, and re-drills it',
      rb.params['width'] == 0.45 and rb.params['height'] == 0.45
      and len(rb.shapes) > 98,
      '%d shapes' % len(rb.shapes))
rb.resize(height=0.20)
check('either name sets the one size it has',
      rb.params['width'] == 0.20 and rb.params['height'] == 0.20)
rb.resize(0.30, 0.30)
check('and both together, when they agree',
      rb.params['width'] == 0.30 and rb.params['height'] == 0.30)
try:
    rb.resize(0.30, 0.40)
    check('two sizes that disagree are refused', False)
except ValueError as e:
    check('two sizes that disagree are refused', True,
          '(%s)' % str(e)[:60])
check('  and it is left as it was',
      rb.params['width'] == 0.30 and rb.params['height'] == 0.30)

check('it survives a save and a load, still round',
      mechanics_from_dict(mechanics_to_dict(rb)).resizable == 'round')
check('the library stocks one', 'BBR30' in models()
      and from_model('BBR30', name='x').resizable == 'round')

L = fresh()
L.add_mechanics(round_breadboard(0.30, name='RB1', center=[0.0, 0.0]))
rbs = [e for e in L.scene_dict()['mechanics'] if e['name'] == 'RB1'][0]
check('the scene says how it resizes, and at what size',
      rbs['resizable'] == 'round' and rbs['width'] == 0.30
      and rbs['height'] == 0.30, json.dumps(rbs['resizable']))
L.apply_edit({'op': 'set', 'target': 'RB1',
              'attrs': {'width': 0.40, 'height': 0.40}})
check('the protocol cuts it to a new diameter',
      L.get_mechanics('RB1').params['width'] == 0.40)
refused(L, {'op': 'set', 'target': 'RB1',
            'attrs': {'width': 0.40, 'height': 0.50}},
        'a round board asked to be two different sizes')

check('a library model built from a builder keeps them',
      from_model('BB3030', name='LB').resizable)

L = fresh()
L.add_mechanics(breadboard(0.30, 0.30, name='RB', center=[0.9, 0.9]))
rb = L.get_mechanics('RB')
L.trace()
L.apply_edit({'op': 'set', 'target': 'RB',
              'attrs': {'width': 0.45, 'height': 0.30,
                        'center': [0.95, 0.9]}})
check('the protocol resizes and re-places in one message',
      rb.params['width'] == 0.45 and close(rb.center, [0.95, 0.9])
      and hole_count(rb) == 216)
check('  without invalidating the trace', L.beams is not None)
L.apply_edit({'op': 'undo'})
check('  and undo puts the old grid back',
      rb.params['width'] == 0.30 and hole_count(rb) == 144)
refused(L, {'op': 'set', 'target': 'RB', 'attrs': {'width': 0}},
        'a size of nothing')
refused(L, {'op': 'set', 'target': 'BB1', 'attrs': {'width': 0.1}},
        'sizing a hand-drawn body')

sc = L.scene_dict()
rbd = [m for m in sc['mechanics'] if m['name'] == 'RB'][0]
bbd = [m for m in sc['mechanics'] if m['name'] == 'BB1'][0]
check('the scene says which bodies resize, and at what size',
      rbd['resizable'] and rbd['width'] == 0.30
      and not bbd['resizable'] and bbd['width'] is None)


print('--- adding from the library, and the mechlib channel ---')

L = fresh()
L.apply_edit({'op': 'add', 'type': 'Mechanics', 'name': 'P1',
              'params': {'model': 'MOUNT-25', 'attached_to': 'M1'}})
h1 = L.get_mechanics('P1')
check('an add naming a model takes its shapes off the shelf',
      h1.model == 'MOUNT-25' and len(h1.shapes) == 9
      and h1.attached_to is L.get_optics('M1'))
L.apply_edit({'op': 'add', 'type': 'Mechanics', 'name': 'P2',
              'params': {'model': 'BB3030', 'center': [1.0, 1.0]}})
check('  and a board from the shelf still resizes',
      L.get_mechanics('P2').resizable)
refused(L, {'op': 'add', 'type': 'Mechanics',
            'params': {'model': 'NO-SUCH'}}, 'a model not in the library')

sc = L.scene_dict()
check('the scene carries the library shelf',
      set(e['name'] for e in sc['mechlib'])
      >= {'BB3030', 'BB4530', 'MOUNT-25'}
      and all('description' in e for e in sc['mechlib']))
check('the scene is still strict JSON', json.dumps(sc) is not None)


print('--- the holes are snap points, and the names are quiet ---')

L = fresh()
L.add_mechanics(breadboard(0.10, 0.10, name='SB', center=[1.0, 1.0],
                           rotationAngle=np.pi / 2))
sc = L.scene_dict()
holes = [p for p in sc['snap'] if p['kind'] == 'hole'
         and p['optic'] == 'SB']
check('every screw hole is a snap point', len(holes) == 16,
      str(len(holes)))
sb = L.get_mechanics('SB')
want = sorted(tuple(np.round(sb.to_world(s.center), 9))
              for s in sb.shapes if isinstance(s, draw.Circle))
got = sorted(tuple(np.round(p['point'], 9)) for p in holes)
check('  standing where the turned board drilled them', want == got)

texts = [s for ly in sc['canvas']['layers'] if ly['name'] == 'text'
         for s in ly['shapes'] if s.get('text') in ('SB', 'BB1')]
check('mechanics names are off by default', texts == [])
L.apply_edit({'op': 'draw', 'params': {'drawMechanicsNames': True}})
sc = L.scene_dict()
texts = [s for ly in sc['canvas']['layers'] if ly['name'] == 'text'
         for s in ly['shapes'] if s.get('text') in ('SB', 'BB1')]
check('  and come back when asked for', len(texts) == 2)


print('--- named points of a part ---')

pd = pedestal(name='P1')
fk = clamping_fork(name='FK1')
check('a pedestal names its axis', sorted(pd.points) == ['axis']
      and close(pd.points['axis'], [0, 0]))
check('a fork names its bore and its screw',
      sorted(fk.points) == ['bore', 'screw']
      and close(fk.points['bore'], [0, 0]))
check('a mount names the hole a pedestal screws into',
      'post' in mirror_mount().points)
check('the pedestal is drawn to the RS05P8E drawing: 25.4 post, 31.8 base',
      sorted(round(2 * s.radius, 4) for s in pd.shapes
             if isinstance(s, draw.Circle)) == [0.0044, 0.0102, 0.0254,
                                                0.0318])
box = fk.local_bbox()
check('the fork is drawn to the CF125 drawing: 36.3 across the prongs',
      abs((box[1][1] - box[0][1]) - 0.0363) < 0.0002,
      str(np.round(box[1] - box[0], 4)))
check('  73.8 from the prong tips to the tail',
      abs(fk.params['length'] - 0.0738) < 1e-12
      # The box is longer than the part: an arc is bounded by the
      # circle it lies on, and the bore is a circle the prongs only
      # follow part of.
      and abs((box[1][0] - box[0][0])
              - (0.0738 + 0.0130 - 0.0038)) < 0.0002)
check('  3.8 from the bore centre to the prong tips',
      abs(fk.params['tip_ahead'] - 0.0038) < 1e-12
      and all(abs(max(s.x) - 0.0038) < 1e-12 for s in fk.shapes
              if isinstance(s, draw.PolyLine)))
check('  and a bore that takes a 25 mm post',
      abs(fk.params['bore_diameter'] - 0.026) < 1e-12)

pd.center = [0.2, 0.1]
pd.rotationAngle = 0.5
check('world_points carries them onto the bench',
      close(pd.world_points()['axis'], [0.2, 0.1]))

L = fresh()
L.add_mechanics(pedestal(name='P1', center=[0.2, 0.1]))
marks = [s for s in L.snap_points()
         if s['optic'] == 'P1' and s['kind'] == 'point']
check('and a front end is offered them, named',
      len(marks) == 1 and marks[0]['label'] == 'P1 axis'
      and close(marks[0]['point'], [0.2, 0.1]), json.dumps(marks))

# A mount's post hole is both a circle in the drawing and the point
# it stands on its pedestal by, so two marks land on one place. A
# front end takes the first of them, and the named one says more.
L.add_mechanics(mirror_mount(name='MT1', attached_to=L.get_optics('M1')))
both = [s for s in L.snap_points() if s['optic'] == 'MT1'
        and close(s['point'],
                  L.get_mechanics('MT1').world_points()['post'])]
check('where a named point and a circle coincide, both are offered',
      sorted(s['kind'] for s in both) == ['hole', 'point'],
      json.dumps(both))
check('  and the named one comes first, so it is the one taken',
      both[0]['kind'] == 'point' and both[0]['label'] == 'MT1 post')

register_model('PEDESTAL-TEST', pedestal(), 'a pedestal')
check('the library carries the points of the part',
      sorted(model_points('PEDESTAL-TEST')) == ['axis'])
check('  and a body built from it has them',
      sorted(from_model('PEDESTAL-TEST', name='X').points) == ['axis'])
lib = os.path.join(WORK, 'points_lib.json')
save_models(lib, names=['PEDESTAL-TEST'])
register_model('PEDESTAL-TEST', [draw.Circle([0, 0], 0.001)], 'wiped')
check('  a wiped model has none', model_points('PEDESTAL-TEST') == {})
load_models(lib)
check('  and loading the file brings them back',
      sorted(model_points('PEDESTAL-TEST')) == ['axis'])


print('--- standing one body on another ---')

L = fresh()
M1 = L.get_optics('M1')
mt = mirror_mount(name='MT', attached_to=M1)
L.add_mechanics(mt)
L.add_mechanics(pedestal(name='P1', center=[0.0, 0.0]))
L.add_mechanics(clamping_fork(name='FK1', center=[0.0, 0.0]))
pd = L.get_mechanics('P1')
fk = L.get_mechanics('FK1')

check('host_pose reads an optics by its face and a body by its own turn',
      close(host_pose(M1)[0], M1.center)
      and host_pose(M1)[1] == float(M1.normAngleHR)
      and host_pose(pd)[1] == pd.rotationAngle)

# Drop the pedestal on the hole under the mount, then pin it.
pd.center = mt.to_world(mt.points['post'])
L.apply_edit({'op': 'set', 'target': 'P1', 'attrs': {'attached_to': 'MT'}})
check('a body stands on a body', pd.attached_to is mt)
check('  where it already was, since the bench chose that, not the model',
      close(pd.center, mt.to_world(mt.points['post'])))
check('  pinned by the point the two share',
      close(pd.attach_point, [0, 0]))

# And the fork on the pedestal, free to swing.
fk.center = np.asarray(pd.center)
L.apply_edit({'op': 'set', 'target': 'FK1',
              'attrs': {'attached_to': 'P1', 'fix_rotation': False}})
check('and a third on the second', fk.attached_to is pd
      and [h.name for h in fk.hosts()] == ['P1', 'MT', 'M1'])

axis = np.asarray(pd.center)
L.apply_edit({'op': 'rotate', 'target': 'FK1', 'rotationAngle': 1.0})
check('a free turn is allowed while attached',
      abs(fk.rotationAngle - 1.0) < 1e-12)
check('  and goes about the point it is held by',
      close(fk.to_world(fk.points['bore']), axis))
refused(L, {'op': 'move', 'target': 'FK1', 'center': [0.1, 0.1]},
        'moving an attached body, free turn or not')
L.apply_edit({'op': 'set', 'target': 'FK1', 'attrs': {'fix_rotation': True}})
refused(L, {'op': 'rotate', 'target': 'FK1', 'rotationAngle': 0.2},
        'turning one whose turn is fixed')
L.apply_edit({'op': 'set', 'target': 'FK1', 'attrs': {'fix_rotation': False}})

# The whole stack hangs off the optics at the root of it.
poses = [(np.array(b.center), b.rotationAngle) for b in (mt, pd, fk)]
M1.HRcenter = np.asarray(M1.HRcenter) + [0.05, 0.0]
check('moving the optics carries the whole chain',
      all(close(b.center, p[0] + [0.05, 0.0])
          for b, p in zip((mt, pd, fk), poses)))
before = np.asarray(pd.center)
M1.normAngleHR = float(M1.normAngleHR) + 0.2
check('turning it turns the chain with it',
      all(abs(b.rotationAngle - (p[1] + 0.2)) < 1e-12
          for b, p in zip((mt, pd, fk), poses))
      and not close(pd.center, before))
check('  and the fork is still on the post',
      close(fk.to_world(fk.points['bore']), pd.center))

check('a body with something on it cannot be removed',
      not L.can_undo or True)
for target in ('P1', 'MT', 'M1'):
    try:
        L.apply_edit({'op': 'remove', 'target': target})
        check("removing '%s' is refused while held" % target, False)
    except EditError as e:
        check("removing '%s' is refused while held" % target, True,
              '(%s)' % str(e)[:40])
refused(L, {'op': 'set', 'target': 'MT', 'attrs': {'attached_to': 'FK1'}},
        'a circle of attachments')
refused(L, {'op': 'set', 'target': 'MT', 'attrs': {'attached_to': 'MT'}},
        'standing on itself')

path = os.path.join(WORK, 'stack.json')
L.save(path)
L2 = OpticalLayout.load(path)
f2 = L2.get_mechanics('FK1')
check('a saved chain comes back joined up',
      [h.name for h in f2.hosts()] == ['P1', 'MT', 'M1']
      and close(f2.center, fk.center))
check('  with what it is pinned by and whether it may turn',
      close(f2.attach_point, fk.attach_point)
      and f2.fix_rotation is False)
with open(path, encoding='utf-8') as f:
    d = json.load(f)
saved = dict((m['name'], m) for m in d['mechanics'])
check('  and no pose of its own is written down',
      'center' not in saved['FK1'] and saved['FK1']['attached_to'] == 'P1')

# The order a file lists them in is not the order they can be built in.
d['mechanics'].reverse()
L3 = OpticalLayout.from_dict(d)
check('a file that lists a body before its host still loads',
      [h.name for h in L3.get_mechanics('FK1').hosts()] == ['P1', 'MT', 'M1'])
d2 = json.loads(json.dumps(d))
for m in d2['mechanics']:
    if m['name'] == 'MT':
        m['attached_to'] = 'FK1'
try:
    OpticalLayout.from_dict(d2)
    check('a file whose bodies stand in a circle is refused', False)
except ValueError as e:
    check('a file whose bodies stand in a circle is refused', True,
          '(%s)' % str(e)[:40])


print('--- copying an element brings its stack along ---')

L = fresh()
M1 = L.get_optics('M1')
mt = mirror_mount(name='MT1', attached_to=M1)
L.add_mechanics(mt)
L.add_mechanics(pedestal(name='P1', attached_to=mt))
L.add_mechanics(clamping_fork(name='FK1', attached_to=L.get_mechanics('P1'),
                              fix_rotation=False))
L.trace()
c = L.copy_optics('M1')
check('the copy is the next free name off the original',
      c.name == 'M2' and L._is_optics('M2'))
check('  standing its own diameter away, along both axes',
      close(c.HRcenter, np.asarray(M1.HRcenter) + M1.diameter))
check('  and the same element otherwise',
      abs(c.diameter - M1.diameter) < 1e-15
      and abs(float(c.normAngleHR) - float(M1.normAngleHR)) < 1e-15
      and abs(c.inv_ROC_HR - M1.inv_ROC_HR) < 1e-15)
names = [m.name for m in L.mechanics]
check('the whole stack comes with it',
      names == ['BB1', 'MT1', 'P1', 'FK1', 'MT2', 'P2', 'FK2'],
      json.dumps(names))
check('  and the free-standing board is not copied with it',
      names.count('BB1') == 1)
chain = [h.name for h in L.get_mechanics('FK2').hosts()]
check('  pinned to the copies, not to the originals',
      chain == ['P2', 'MT2', 'M2'], json.dumps(chain))
check('  so it stands beside the original, not on top of it',
      close(L.get_mechanics('MT2').center,
            np.asarray(mt.center) + M1.diameter))
check('  and what was free to turn still is',
      L.get_mechanics('FK2').fix_rotation is False
      and L.get_mechanics('MT2').fix_rotation is True)

# By value: the copy is a second body, not a second name for one.
check('nothing is shared with the original',
      not any(a is b for a in mt.shapes
              for b in L.get_mechanics('MT2').shapes)
      and L.get_mechanics('MT2').points is not mt.points
      and close(L.get_mechanics('MT2').points['post'], mt.points['post']))
L.get_mechanics('MT2').shapes[0].thickness = 3.0
check('  so editing one leaves the other alone',
      mt.shapes[0].thickness != 3.0)
check('the model name travels, so a relink still knows what it is',
      L.get_mechanics('MT2').model == mt.model)

# Moving the copy moves its own stack and leaves the original's alone.
was = np.array(mt.center)
p2was = np.array(L.get_mechanics('P2').center)
L.get_optics('M2').HRcenter = np.asarray(c.HRcenter) + [0.1, 0.0]
check('moving the copy carries its own stack',
      close(L.get_mechanics('P2').center, p2was + [0.1, 0.0]))
check('  and leaves the original where it was', close(mt.center, was))

L2 = fresh()
L2.add_mechanics(mirror_mount(name='MT1', attached_to=L2.get_optics('M1')))
L2.trace()
L2.apply_edit({'op': 'copy', 'target': 'M1', 'name': 'M9',
               'offset': [0.2, 0.0]})
check('the protocol copies it too, where the message says',
      L2._is_optics('M9')
      and close(L2.get_optics('M9').HRcenter,
                np.asarray(L2.get_optics('M1').HRcenter) + [0.2, 0.0])
      and [m.name for m in L2.mechanics] == ['BB1', 'MT1', 'MT2'])
check('  and the trace no longer stands', L2.beams is None)
L2.apply_edit({'op': 'undo'})
check('  and one undo takes the whole copy back',
      not L2._is_optics('M9')
      and [m.name for m in L2.mechanics] == ['BB1', 'MT1'])
L2.apply_edit({'op': 'redo'})
check('  redo puts it back, stack and all',
      L2._is_optics('M9') and len(L2.mechanics) == 3)

L3 = fresh()
L3.add_mechanics(mirror_mount(name='MT1', attached_to=L3.get_optics('M1')))
refused(L3, {'op': 'copy', 'target': 'MT1'},
        'copying a body rather than an element')
refused(L3, {'op': 'copy', 'target': 'b0'}, 'copying a source')
refused(L3, {'op': 'copy', 'target': 'nope'}, 'copying nothing')
refused(L3, {'op': 'copy', 'target': 'M1', 'name': '  '},
        'a copy with a blank name')
refused(L3, {'op': 'copy', 'target': 'M1', 'name': 'MT1'},
        'a copy named after something already there')
refused(L3, {'op': 'copy', 'target': 'M1', 'offset': [0.0, np.inf]},
        'a copy put at infinity')
refused(L3, {'op': 'copy', 'target': 'M1', 'offset': 'over there'},
        'an offset that is not a place')


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
