'''
The shape editor behind Mechanics.edit(): the model, the protocol it
speaks, and what it refuses.

A part's geometry is a list of drawing primitives in local
coordinates, and until now the only way to lay one out was to write
the numbers in a cell. The editor is the same viewer handed a scene
of nothing but those shapes, drawn in the frame they are written in
with the origin marked.

Three things carry it and are checked hardest.

The first is that a shape is edited by taking it apart into the dict
shape_to_dict writes, changing what the message names, and building
it again. So there is no second list of rules about what a valid
circle is: the constructor is the rule, and a message that would make
no shape leaves the one that was there untouched.

The second is identity. The editor holds the Mechanics by reference -
the user's own object, which a layout may already be drawing - so an
edit lands there and nowhere else. That includes undo, which refills
the very list rather than replacing it.

The third is that an index names a place, not a thing. Removing a
shape renumbers the ones after it, exactly as a list does.
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
from gtrace.draw.serialize import shape_to_dict
from gtrace.layout import OpticalLayout, TraceRules, EditError, q_from_waist
from gtrace.mechanics import (Mechanics, breadboard, mirror_mount,
                              models, model_shapes, model_params,
                              from_model, register_model)
from gtrace.draw.viewer.editor import (ShapeEditor, NEW_SHAPES,
                                       EDITABLE_SHAPE_ATTRS, EDIT_LAYER,
                                       DUPLICATE_OFFSET)
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

def refused(ed, msg, why):
    '''
    Check that an edit is rejected without side effects.
    '''
    before = json.dumps([shape_to_dict(s) for s in ed.shapes])
    try:
        ed.apply_edit(msg)
    except EditError as e:
        check('refuses %s' % why, True, '(%s)' % str(e)[:60])
    except Exception as e:
        check('refuses %s' % why, False,
              '(raised %s instead)' % type(e).__name__)
        return
    else:
        check('refuses %s' % why, False, '(it went through)')
        return
    check('  and leaves the shapes alone',
          json.dumps([shape_to_dict(s) for s in ed.shapes]) == before)

def fresh():
    return ShapeEditor(Mechanics(
        shapes=[draw.Rectangle([-0.02, -0.01], 0.04, 0.02),
                draw.Circle([0.0, 0.0], 0.003)],
        center=[0.3, 0.1], name='P1'))


print('--- the scene an editor draws ---')

ed = fresh()
scene = ed.scene_dict()
check('the scene is strict JSON', json.dumps(scene) is not None)
check('it says it is an editor, and what of',
      scene['editor']['name'] == 'P1'
      and scene['editor']['layer'] == 'mechanics',
      json.dumps(scene['editor']))
check('the shapes are listed with the index a message names them by',
      [s['index'] for s in scene['shapes']] == [0, 1]
      and [s['type'] for s in scene['shapes']] == ['rectangle', 'circle'])
check('and drawn on a layer of their own',
      [ly['name'] for ly in scene['canvas']['layers']] == [EDIT_LAYER]
      and len(scene['canvas']['layers'][0]['shapes']) == 2)

# The frame is the local one: the body stands at [0.3, 0.1] and the
# drawing does not know it. A part is drawn around its origin, and
# where it will stand is the layout's business.
rect = [s for s in scene['canvas']['layers'][0]['shapes']
        if s['type'] == 'rectangle'][0]
check('drawn in the local frame, not where the body stands',
      np.allclose(rect['point'], [-0.02, -0.01]))

check('no beams, optics or bodies in it',
      scene['beams'] == [] and scene['optics'] == []
      and scene['mechanics'] == [])
kinds = set(p['kind'] for p in scene['snap'])
check('the origin is a snap point, with the shapes',
      'origin' in kinds and 'corner' in kinds and 'centre' in kinds,
      str(sorted(kinds)))
check('  and the origin is at zero',
      [p['point'] for p in scene['snap'] if p['kind'] == 'origin']
      == [[0.0, 0.0]])

# The middle of a straight edge is a place a part is drawn against,
# and the measuring tool reaches the same points a drag settles on.
mids = ShapeEditor(Mechanics(shapes=[
    draw.Rectangle([-0.02, -0.01], 0.04, 0.02),
    draw.PolyLine([0.0, 0.04, 0.04], [0.05, 0.05, 0.09]),
    draw.Line([-0.05, 0.03], [-0.01, 0.03]),
    draw.Circle([0.0, -0.05], 0.01)], name='P2')).snap_points()
def points_of(kind, index):
    return sorted([tuple(np.round(p['point'], 12)) for p in mids
                   if p['kind'] == kind and p['label'].split()[1] == index])
check('the four edges of a plate are marked at their middles',
      points_of('midpoint', '1')
      == [(-0.02, 0.0), (0.0, -0.01), (0.0, 0.01), (0.02, 0.0)],
      str(points_of('midpoint', '1')))
check('every segment of an outline is, too',
      points_of('midpoint', '2') == [(0.02, 0.05), (0.04, 0.07)],
      str(points_of('midpoint', '2')))
check('and a line is marked at its middle, between its two ends',
      points_of('midpoint', '3') == [(-0.03, 0.03)],
      str(points_of('midpoint', '3')))
check('a curve has none: the middle of an arc lines nothing up',
      points_of('midpoint', '4') == [])


print('--- adding, setting, copying, reordering, removing ---')

ed = fresh()
mech = ed.mechanics
check('the editor edits the body itself, by reference',
      ed.shapes is mech.shapes)

for kind in sorted(NEW_SHAPES):
    n = len(ed.shapes)
    ed.apply_edit({'op': 'add_shape', 'type': kind})
    check('a %s can be put down' % kind,
          len(ed.shapes) == n + 1
          and shape_to_dict(ed.shapes[-1])['type'] == kind)
check('  each one at the origin, at a size a bench would see',
      all(abs(v) <= 0.02
          for s in ed.shapes[2:]
          for k, val in shape_to_dict(s).items()
          if k in ('point', 'center', 'start', 'stop')
          for v in val))

ed = fresh()
ed.apply_edit({'op': 'set_shape', 'index': 1,
               'attrs': {'radius': 0.004, 'center': [0.01, -0.005]}})
c = ed.shapes[1]
check('a shape is set by naming its keys',
      c.radius == 0.004 and np.allclose(c.center, [0.01, -0.005]))
check('  and the others are left alone',
      ed.shapes[0].width == 0.04)

ed.apply_edit({'op': 'duplicate_shape', 'index': 1})
check('a copy lands just after the original',
      len(ed.shapes) == 3
      and np.allclose(ed.shapes[2].center,
                      [0.01 + DUPLICATE_OFFSET, -0.005 + DUPLICATE_OFFSET]))
check('  and is a shape of its own',
      ed.shapes[2] is not ed.shapes[1])

ed.apply_edit({'op': 'move_shape', 'index': 2, 'to': 0})
check('a shape can be drawn earlier or later',
      isinstance(ed.shapes[0], draw.Circle)
      and isinstance(ed.shapes[1], draw.Rectangle))

ed.apply_edit({'op': 'remove_shape', 'index': 0})
check('and taken away', len(ed.shapes) == 2
      and isinstance(ed.shapes[0], draw.Rectangle))


print('--- what the editor refuses ---')

ed = fresh()
refused(ed, {'op': 'add_shape', 'type': 'blob'}, 'a shape gtrace cannot draw')
refused(ed, {'op': 'add_shape'}, 'an add with no type')
refused(ed, {'op': 'set_shape', 'index': 0, 'attrs': {'radius': 0.01}},
        'a radius on a rectangle')
refused(ed, {'op': 'set_shape', 'index': 0, 'attrs': {'type': 'circle'}},
        'turning one kind of shape into another')
refused(ed, {'op': 'set_shape', 'index': 0, 'attrs': {'width': 'wide'}},
        'a width that is not a number')
refused(ed, {'op': 'set_shape', 'index': 0,
             'attrs': {'width': float('inf')}}, 'an infinite width')
refused(ed, {'op': 'set_shape', 'index': 0,
             'attrs': {'point': [float('nan'), 0.0]}}, 'a corner at nan')
refused(ed, {'op': 'set_shape', 'index': 9, 'attrs': {}}, 'a shape that is not there')
refused(ed, {'op': 'set_shape', 'index': -1, 'attrs': {}}, 'a negative index')
refused(ed, {'op': 'set_shape', 'index': '0', 'attrs': {}},
        'an index that is not a number')
refused(ed, {'op': 'remove_shape', 'index': 5}, 'removing what is not there')
refused(ed, {'op': 'move_shape', 'index': 0, 'to': 7}, 'a move past the end')
refused(ed, {'op': 'jiggle'}, 'an operation it does not have')
refused(ed, 'not a message', 'a message that is not a dict')

# A polyline is edited by its whole list of vertices - that is what
# the shape carries - so adding, moving and taking one away are all
# the same message with a different list in it.
ed.apply_edit({'op': 'add_shape', 'type': 'polyline',
               'params': {'x': [0.0, 0.01], 'y': [0.0, 0.02]}})
check('a polyline takes its vertices through the protocol',
      list(ed.shapes[-1].x) == [0.0, 0.01]
      and list(ed.shapes[-1].y) == [0.0, 0.02])
last = len(ed.shapes) - 1
refused(ed, {'op': 'set_shape', 'index': last,
             'attrs': {'x': [0.0, 0.01, 0.02]}},
        'a polyline whose x and y no longer match')
ed.apply_edit({'op': 'set_shape', 'index': last,
               'attrs': {'x': [0.0, 0.01, 0.02], 'y': [0.0, 0.02, 0.03]}})
check('a vertex put in is a longer list',
      len(ed.shapes[last].x) == 3 and ed.shapes[last].numpoints == 3)
ed.apply_edit({'op': 'set_shape', 'index': last,
               'attrs': {'x': [0.0, 0.02], 'y': [0.0, 0.03]}})
check('  and one taken out is a shorter one',
      len(ed.shapes[last].x) == 2 and ed.shapes[last].numpoints == 2)
# The constructor only asks that x and y be of the same length, so the
# floor is the editor's to hold - as the positive width is.
refused(ed, {'op': 'set_shape', 'index': last,
             'attrs': {'x': [0.01], 'y': [0.02]}},
        'a polyline of one vertex, which draws nothing')
refused(ed, {'op': 'set_shape', 'index': last,
             'attrs': {'x': [], 'y': []}},
        'a polyline of none at all')
check('  and the shape it would have replaced is untouched',
      len(ed.shapes[last].x) == 2)


print('--- turning a shape ---')

# A turn is the one edit that is not a set of attributes: what it
# means differs by kind, and a rectangle cannot carry one at all.
ed = ShapeEditor(Mechanics(shapes=[
    draw.Rectangle([-0.02, -0.01], 0.04, 0.02),
    draw.Circle([0.01, 0.0], 0.003),
    draw.Arc([0.0, 0.0], 0.01, 0.0, np.pi),
    draw.Text('hi', [0.01, 0.0], height=0.005),
    draw.Line([0.0, 0.0], [0.02, 0.0])], name='P3'))

ed.apply_edit({'op': 'rotate_shape', 'index': 4, 'angle': np.pi / 2,
               'pivot': [0.0, 0.0]})
check('a line turns about the point it is given',
      np.allclose(ed.shapes[4].start, [0.0, 0.0], atol=1e-15)
      and np.allclose(ed.shapes[4].stop, [0.0, 0.02], atol=1e-12),
      str(np.round(ed.shapes[4].stop, 6)))

# Nothing said about a pivot means the middle of the box the shape
# occupies, which is the box a front end draws around it.
ed.apply_edit({'op': 'rotate_shape', 'index': 1, 'angle': 1.0})
check('a circle turned about its own middle does not move at all',
      np.allclose(ed.shapes[1].center, [0.01, 0.0], atol=1e-15)
      and abs(ed.shapes[1].radius - 0.003) < 1e-15,
      str(np.round(ed.shapes[1].center, 6)))

ed.apply_edit({'op': 'rotate_shape', 'index': 2, 'angle': np.pi / 2})
check('an arc turns by turning both of its angles',
      np.allclose(ed.shapes[2].center, [0.0, 0.0], atol=1e-15)
      and abs(ed.shapes[2].startangle - np.pi / 2) < 1e-15
      and abs(ed.shapes[2].stopangle - 3 * np.pi / 2) < 1e-15,
      '%.4f .. %.4f' % (ed.shapes[2].startangle, ed.shapes[2].stopangle))

ed.apply_edit({'op': 'rotate_shape', 'index': 3, 'angle': np.pi / 2,
               'pivot': [0.0, 0.0]})
check('a text turns about the pivot and turns with it',
      np.allclose(ed.shapes[3].point, [0.0, 0.01], atol=1e-12)
      and abs(ed.shapes[3].rotation - np.pi / 2) < 1e-15,
      '%s %.4f' % (np.round(ed.shapes[3].point, 6), ed.shapes[3].rotation))

# The rectangle. Its corners are written out here rather than asked
# for: a shape turned by the code that turns it proves nothing.
ed.apply_edit({'op': 'rotate_shape', 'index': 0, 'angle': np.pi / 4})
turned = ed.shapes[0]
th = np.pi / 4
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
want = np.array([[-0.02, -0.01], [0.02, -0.01], [0.02, 0.01],
                 [-0.02, 0.01], [-0.02, -0.01]]) @ R.T
check('a turned rectangle comes back as an outline, closed',
      isinstance(turned, draw.PolyLine) and len(turned.x) == 5
      and turned.x[0] == turned.x[-1] and turned.y[0] == turned.y[-1],
      type(turned).__name__)
check('  of its four corners, turned about the middle of the shape',
      np.allclose(np.column_stack([turned.x, turned.y]), want, atol=1e-15),
      str(np.round(np.column_stack([turned.x, turned.y]), 6).tolist()))
check('  and it keeps its place in the list',
      len(ed.shapes) == 5 and isinstance(ed.shapes[1], draw.Circle))
ed.apply_edit({'op': 'undo'})
check('  undo puts the rectangle back',
      isinstance(ed.shapes[0], draw.Rectangle)
      and abs(ed.shapes[0].width - 0.04) < 1e-15)
ed.apply_edit({'op': 'rotate_shape', 'index': 0, 'angle': 0.0})
check('a turn of nothing leaves a rectangle a rectangle',
      isinstance(ed.shapes[0], draw.Rectangle))

refused(ed, {'op': 'rotate_shape', 'index': 0, 'angle': 'a lot'},
        'a turn that is not an angle')
refused(ed, {'op': 'rotate_shape', 'index': 0, 'angle': float('inf')},
        'a turn of infinity')
refused(ed, {'op': 'rotate_shape', 'index': 0, 'angle': 0.5,
             'pivot': [0.0, 'x']}, 'a pivot that is not a point')
refused(ed, {'op': 'rotate_shape', 'index': 0, 'angle': 0.5,
             'pivot': [0.0, 0.0, 0.0]}, 'a pivot of three numbers')
refused(ed, {'op': 'rotate_shape', 'index': 9, 'angle': 0.5},
        'turning a shape that is not there')


print('--- undo and redo ---')

ed = fresh()
mech = ed.mechanics
ed.apply_edit({'op': 'add_shape', 'type': 'circle'})
check('there is something to undo', ed.can_undo and not ed.can_redo)
ed.apply_edit({'op': 'undo'})
check('undo takes the shape back', len(ed.shapes) == 2)
check('  into the body\'s own list', ed.shapes is mech.shapes)
ed.apply_edit({'op': 'redo'})
check('redo puts it back', len(ed.shapes) == 3)

ed.apply_edit({'op': 'undo'})
ed.apply_edit({'op': 'set_shape', 'index': 0, 'attrs': {'width': 0.05}})
check('an edit after an undo drops the redo', not ed.can_redo)
ed.apply_edit({'op': 'undo'})
check('and undoing it restores the value',
      ed.shapes[0].width == 0.04)

before = len(ed._history)
try:
    ed.apply_edit({'op': 'set_shape', 'index': 0, 'attrs': {'width': -1}})
except EditError:
    pass
check('a refused edit costs no history', len(ed._history) == before)


print('--- saving to the library ---')

ed = fresh()
mech = ed.mechanics
depth = len(ed._history)
ed.apply_edit({'op': 'save_model', 'name': 'TEST-EDITED',
               'description': 'from the editor'})
check('the part goes on the shelf',
      'TEST-EDITED' in models()
      and models()['TEST-EDITED'] == 'from the editor')
check('  with the shapes as they stand',
      len(model_shapes('TEST-EDITED')) == 2)
check('  and the body remembers where they came from',
      mech.model == 'TEST-EDITED')
check('saving is not an edit to undo', len(ed._history) == depth)

ed.apply_edit({'op': 'set_shape', 'index': 1, 'attrs': {'radius': 0.008}})
check('the shelf keeps what was saved, not what came after',
      abs(model_shapes('TEST-EDITED')[1].radius - 0.003) < 1e-15)
ed.apply_edit({'op': 'save_model', 'name': 'TEST-EDITED'})
check('  until it is saved again',
      abs(model_shapes('TEST-EDITED')[1].radius - 0.008) < 1e-15)

refused(ed, {'op': 'save_model', 'name': '  '}, 'a blank model name')
refused(ed, {'op': 'save_model', 'name': 3}, 'a name that is not a string')
refused(ed, {'op': 'save_model', 'name': 'X', 'description': 7},
        'a description that is not text')

# A part built by a builder keeps knowing how it was built, so a
# breadboard edited by hand and saved is still a breadboard.
bed = ShapeEditor(breadboard(0.1, 0.1, name='B1'))
bed.apply_edit({'op': 'add_shape', 'type': 'circle'})
bed.apply_edit({'op': 'save_model', 'name': 'TEST-EDITED-BOARD'})
check('a builder part keeps its parameters through the editor',
      model_params('TEST-EDITED-BOARD') is not None
      and from_model('TEST-EDITED-BOARD', name='x').resizable)


print('--- the editor and a layout share the body ---')

L = OpticalLayout(name='ed', rules=TraceRules(order=2,
                                              power_threshold=1e-6))
L.add_optics(opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.pi, name='M1'))
L.add_source(GaussianBeam(pos=[0, 0], dirAngle=0.0,
                          q0=q_from_waist(0.2*mm, 0.0, 1064*nm),
                          wl=1064*nm, name='b0'))
mt = mirror_mount(name='MT1', attached_to=L.get_optics('M1'))
L.add_mechanics(mt)

ed = ShapeEditor(mt)
n0 = len(L.scene_dict()['canvas']['layers'][-1]['shapes'])
ed.apply_edit({'op': 'add_shape', 'type': 'circle'})
layers = dict((ly['name'], ly) for ly in L.scene_dict()['canvas']['layers'])
check('a shape added in the editor is in the layout at its next draw',
      len(layers['mechanics']['shapes']) == len(mt.shapes))
check('  and the body is still attached, still where it was',
      mt.attached_to is L.get_optics('M1')
      and np.allclose(mt.center, L.get_optics('M1').center))
ed.apply_edit({'op': 'undo'})
check('undoing in the editor reaches the layout too',
      len(L.get_mechanics('MT1').shapes) == len(mt.shapes))

check('the editor leaves the pose alone',
      np.allclose(mt.offset, [0, 0]) and mt.offset_angle == 0.0)


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
