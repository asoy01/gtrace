'''
gtrace.draw.viewer.editor

A shape editor for Mechanics: the front end behind Mechanics.edit().

A body's geometry is a list of drawing primitives in local
coordinates, and until now the only way to lay one out was to write
the numbers in a cell. That is fine for the shapes gtrace ships -
a plate and a grid of holes - and tiring for anything a real bench
has on it.

The editor is deliberately not a second viewer. It is the same one,
handed a scene of nothing but the shapes being edited, drawn in the
local frame with the origin marked: the origin is the point that
comes to sit at the host's substrate centre when the body is
attached, so seeing it is most of what makes a part right. Zoom, pan,
undo and the layer panel come along because they were never about
optics in the first place.

What it edits is the Mechanics it was opened on - by reference, like
everything else in gtrace - so a body already registered in a layout
is redrawn there as soon as the layout is drawn again. Saving to the
model library is one button, and the same register_model any cell
would call.
'''

#{{{ Import modules

import copy

import numpy as np

import gtrace.draw as draw
from gtrace.draw.serialize import (scene_to_dict, shape_to_dict,
                                   shape_from_dict, UnknownShapeError)
from gtrace.layout import EditError, UNDO_DEPTH
from gtrace.mechanics import (DEFAULT_LAYER, LAYER_COLOR, register_model,
                              rotate_shape, shape_centre)

#}}}

#{{{ Author and License Infomation

#Copyright (c) 2011-2026, Yoichi Aso
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

__author__ = "Yoichi Aso"
__copyright__ = "Copyright 2011-2026, Yoichi Aso"
__credits__ = ["Yoichi Aso"]
__license__ = "BSD"
__version__ = "0.4.0"
__maintainer__ = "Yoichi Aso"
__email__ = "asoy01@gmail.com"
__status__ = "Beta"

#}}}

#{{{ What a shape is made of

#: The keys of each kind of shape a front end may set, as
#: shape_to_dict spells them. A shape is edited by taking it apart
#: into that dict, changing what the message names and building it
#: again - so an edit that would make no shape is refused by the
#: constructor rather than by a second list of rules kept here.
#:
#: 'type' is not among them anywhere: a rectangle asked to become a
#: circle is a different shape, which is an add and a remove.
EDITABLE_SHAPE_ATTRS = {
    'line': frozenset(['start', 'stop', 'thickness']),
    'polyline': frozenset(['x', 'y', 'thickness']),
    'rectangle': frozenset(['point', 'width', 'height', 'thickness']),
    'circle': frozenset(['center', 'radius', 'thickness']),
    'arc': frozenset(['center', 'radius', 'startangle', 'stopangle',
                      'thickness']),
    'text': frozenset(['text', 'point', 'height', 'rotation']),
}

#: What each kind of shape looks like when it is first put down, in
#: metres about the origin. A part is drawn around the point that sits
#: at the host's substrate centre, so a new shape starts there and is
#: given a size a bench would recognise rather than one that has to be
#: found by zooming.
NEW_SHAPES = {
    'line': {'type': 'line', 'start': [-0.01, 0.0], 'stop': [0.01, 0.0],
             'thickness': 0.0},
    'polyline': {'type': 'polyline', 'x': [-0.01, 0.0, 0.01],
                 'y': [-0.01, 0.01, -0.01], 'thickness': 0.0},
    'rectangle': {'type': 'rectangle', 'point': [-0.01, -0.01],
                  'width': 0.02, 'height': 0.02, 'thickness': 0.0},
    'circle': {'type': 'circle', 'center': [0.0, 0.0], 'radius': 0.005,
               'thickness': 0.0},
    'arc': {'type': 'arc', 'center': [0.0, 0.0], 'radius': 0.01,
            'startangle': 0.0, 'stopangle': np.pi, 'thickness': 0.0},
    'text': {'type': 'text', 'text': 'label', 'point': [0.0, 0.0],
             'height': 0.005, 'rotation': 0.0},
}

#: How far a duplicate is put from the shape it was made from, in
#: metres. Enough to be seen and taken hold of, small enough that it is
#: obviously the same shape moved.
DUPLICATE_OFFSET = 0.005

#: The layer the shapes being edited are drawn on. Its own name rather
#: than the body's, since what is on screen is the drawing of a part
#: and not the part standing on a bench.
EDIT_LAYER = 'shapes'

#: Operations that change nothing, and so are not worth a snapshot.
_NOT_UNDOABLE = frozenset(['save_model', 'undo', 'redo'])

#}}}

#{{{ ShapeEditor

class ShapeEditor(object):
    '''
    An editor for the shapes of one Mechanics.

    The body is held by reference - it is the user's own object - so
    every edit lands on the body a layout may already be drawing, and
    the layout shows it at the next draw. What the editor does not
    touch is the pose: where a body stands is the layout's business,
    and an editor that moved things while a shape was being drawn
    would be answering a question nobody asked.

    Attributes
    ----------
    mechanics : gtrace.mechanics.Mechanics
        The body being edited.
    '''

    def __init__(self, mechanics):
        self.mechanics = mechanics
        #: Serialized shape lists from before each edit, oldest first.
        self._history = []
        self._future = []
        #: What the last save_model was told, so the panel can offer
        #: it again rather than asking twice.
        self.model_name = mechanics.model or mechanics.name

    @property
    def shapes(self):
        '''
        The shapes being edited: the body's own list.
        '''
        return self.mechanics.shapes

#{{{ Drawing and the scene

    def draw(self, canvas=None, **options):
        '''
        Draw the shapes into a canvas, in local coordinates.

        Not through Mechanics.draw: that carries the shapes onto the
        bench through the pose, and an editor works in the frame the
        shapes are written in. The two agree whenever the body stands
        at the origin unturned, which is exactly the case this makes
        of every body.
        '''
        if canvas is None:
            canvas = draw.Canvas()
            canvas.unit = 'm'
        canvas.add_layer(EDIT_LAYER, color=LAYER_COLOR)
        for s in self.shapes:
            canvas.add_shape(s, layername=EDIT_LAYER)
        return canvas

    def scene_dict(self, **kwargs):
        '''
        The scene an editing front end draws: the shapes, and the list
        of them it addresses by index.

        The optics and beam channels are empty and stay that way. A
        part has no beams through it - it is a drawing of a body, and
        the bench it will stand on is elsewhere.
        '''
        canvas = self.draw()
        scene = scene_to_dict(canvas, [], [])
        scene['dimensions'] = []
        scene['snap'] = self.snap_points()
        scene['sources'] = []
        scene['mechanics'] = []
        scene['mechlib'] = []
        scene['can_undo'] = self.can_undo
        scene['can_redo'] = self.can_redo
        # What tells a front end it is editing a part rather than
        # looking at a bench, and what it is editing.
        scene['editor'] = {
            'name': str(self.mechanics.name),
            'model': (None if self.mechanics.model is None
                      else str(self.mechanics.model)),
            'model_name': str(self.model_name),
            'layer': str(self.mechanics.layer)}
        scene['shapes'] = self.shapes_dict()
        return scene

    def shapes_dict(self):
        '''
        The shapes, each with the index an edit message names it by.

        The index is the position in the list, which is also the order
        they are drawn in. It is not an identity: removing a shape
        renumbers the ones after it, exactly as a list does, and a
        front end holding one across an edit is holding a place rather
        than a thing.
        '''
        out = []
        for i, s in enumerate(self.shapes):
            d = shape_to_dict(s)
            d['index'] = i
            out.append(d)
        return out

    def snap_points(self):
        '''
        Points worth snapping to while drawing: the origin, and the
        corners, centres and edge midpoints of the shapes already
        down.

        The origin is first because it is the one point every part is
        drawn around - it becomes the host's substrate centre - and
        the one a new shape most often wants to line up with.
        '''
        points = [{'point': [0.0, 0.0], 'kind': 'origin',
                   'label': 'origin', 'optic': ''}]
        for i, s in enumerate(self.shapes):
            for p, kind in _shape_points(s):
                points.append({'point': [float(p[0]), float(p[1])],
                               'kind': kind, 'optic': '',
                               'label': '%s %d %s'
                                        % (shape_to_dict(s)['type'], i + 1,
                                           kind)})
        return points

#}}}

#{{{ Editing

    def apply_edit(self, msg):
        '''
        Apply an edit message from an editing front end::

            {'op': 'add_shape',       'type': 'circle'}
            {'op': 'set_shape',       'index': 2,
                                      'attrs': {'radius': 0.004}}
            {'op': 'remove_shape',    'index': 2}
            {'op': 'duplicate_shape', 'index': 2}
            {'op': 'move_shape',      'index': 2, 'to': 0}
            {'op': 'rotate_shape',    'index': 2, 'angle': 0.7854,
                                      'pivot': [0.0, 0.0]}
            {'op': 'save_model',      'name': 'MY-PART',
                                      'description': 'one line'}
            {'op': 'undo'}
            {'op': 'redo'}

        A shape is edited by taking it apart into the dict
        shape_to_dict writes, changing what the message names, and
        building it again - so a value that describes no shape is
        refused by the constructor, and the shape it would have
        replaced is left alone.

        A turn is the one edit that is not a set of attributes: what
        turning means differs by kind - an arc's angles move, a text
        turns with its own rotation - and a rectangle, having its
        sides along the axes, comes back as the closed polyline of its
        corners. The angle is in radians, counterclockwise; ``pivot``
        defaults to the middle of the shape's bounding box, which is
        the box a front end draws around it.

        Raises
        ------
        EditError
            If the operation, the index or an attribute is not
            allowed.
        '''
        if not isinstance(msg, dict):
            raise EditError('An edit message must be a dict, not %s'
                            % type(msg).__name__)
        op = msg.get('op')
        if op == 'undo':
            return self.undo()
        if op == 'redo':
            return self.redo()

        snapshot = (None if op in _NOT_UNDOABLE else self._snapshot())
        result = self._apply_edit(msg)
        if snapshot is not None:
            self._history.append(snapshot)
            del self._history[:-UNDO_DEPTH]
            del self._future[:]
        return result

    def _apply_edit(self, msg):
        op = msg.get('op')

        if op == 'add_shape':
            kind = msg.get('type')
            if kind not in NEW_SHAPES:
                raise EditError('%r is not a shape gtrace can draw. It '
                                'draws %s.'
                                % (kind, ', '.join(sorted(NEW_SHAPES))))
            d = copy.deepcopy(NEW_SHAPES[kind])
            for key, value in (msg.get('params') or {}).items():
                if key not in EDITABLE_SHAPE_ATTRS[kind]:
                    raise EditError('%r is not a parameter of a %s.'
                                    % (key, kind))
                d[key] = value
            self.shapes.append(self._build(d))
            return self

        if op == 'set_shape':
            i = self._index(msg.get('index'))
            d = shape_to_dict(self.shapes[i])
            allowed = EDITABLE_SHAPE_ATTRS[d['type']]
            attrs = msg.get('attrs') or {}
            for key in attrs:
                if key not in allowed:
                    raise EditError('%r is not an editable attribute of a '
                                    '%s. It has %s.'
                                    % (key, d['type'],
                                       ', '.join(sorted(allowed))))
            d.update(attrs)
            # Built before it is put in place, so a refusal leaves the
            # shape that was there untouched.
            self.shapes[i] = self._build(d)
            return self

        if op == 'remove_shape':
            i = self._index(msg.get('index'))
            del self.shapes[i]
            return self

        if op == 'duplicate_shape':
            i = self._index(msg.get('index'))
            d = shape_to_dict(self.shapes[i])
            _offset_shape_dict(d, DUPLICATE_OFFSET)
            self.shapes.insert(i + 1, self._build(d))
            return self

        if op == 'move_shape':
            i = self._index(msg.get('index'))
            to = self._index(msg.get('to'))
            self.shapes.insert(to, self.shapes.pop(i))
            return self

        if op == 'rotate_shape':
            i = self._index(msg.get('index'))
            angle = msg.get('angle')
            if isinstance(angle, bool) or not isinstance(angle, (int, float)):
                raise EditError('A turn is an angle in radians, not %r.'
                                % (angle,))
            angle = float(angle)
            if not np.isfinite(angle):
                raise EditError('A shape cannot be turned by %r.' % (angle,))
            if msg.get('pivot') is None:
                pivot = shape_centre(self.shapes[i])
            else:
                try:
                    pivot = np.asarray(msg['pivot'],
                                       dtype='float64').reshape(2)
                except (TypeError, ValueError):
                    raise EditError('A shape is turned about a point '
                                    '[x, y], not %r.' % (msg['pivot'],))
                if not np.all(np.isfinite(pivot)):
                    raise EditError('A shape cannot be turned about %r.'
                                    % (msg['pivot'],))
            # Through the same door as every other edit: what comes
            # back is taken apart and built again, so the constructors
            # have the last word here too.
            self.shapes[i] = self._build(
                shape_to_dict(rotate_shape(self.shapes[i], angle, pivot)))
            return self

        if op == 'save_model':
            name = msg.get('name')
            if not isinstance(name, str) or not name.strip():
                raise EditError('A model name must be a non-empty string, '
                                'not %r.' % (name,))
            description = msg.get('description', '')
            if not isinstance(description, str):
                raise EditError('A description is a line of text, not %r.'
                                % (description,))
            register_model(name, self.mechanics, description)
            # The body now says where its shapes came from, which is
            # what relink_mechanics later goes by.
            self.mechanics.model = name
            self.model_name = name
            return self

        raise EditError('Unknown edit operation %r.' % (op,))

    def _index(self, i):
        '''
        A shape index from a message, or an EditError.
        '''
        if not isinstance(i, int) or isinstance(i, bool):
            raise EditError('A shape is named by its index, not %r.' % (i,))
        if i < 0 or i >= len(self.shapes):
            raise EditError('There is no shape %d; the part has %d.'
                            % (i, len(self.shapes)))
        return i

    def _build(self, d):
        '''
        A shape from its serialized form, with a refusal a front end
        can show. This is where a width of 'wide' or a circle of no
        radius is turned away.

        The constructors carry most of it - a polyline whose x and y
        no longer match says so itself - and what they do not is the
        arithmetic that has to hold for a shape to be drawable at all:
        a size is positive and every number is finite. Both are
        checked on the way out, against the shape as it came back,
        rather than against the message.
        '''
        try:
            shape = shape_from_dict(d)
        except UnknownShapeError as e:
            raise EditError(str(e))
        except draw.NumberOfElementError as e:
            raise EditError('That does not describe a %s: %s'
                            % (d.get('type'), e))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise EditError('That does not describe a %s (%s: %s).'
                            % (d.get('type'), type(e).__name__, e))

        out = shape_to_dict(shape)
        # A nan or an infinity would take the whole view with it the
        # first time anything was framed.
        for value in out.values():
            for v in (value if isinstance(value, list) else [value]):
                if isinstance(v, float) and not np.isfinite(v):
                    raise EditError('A %s cannot be drawn with %r in it.'
                                    % (out['type'], v))
        # A rectangle of no width, or of less than none, is not a
        # smaller rectangle: it is a shape SVG refuses to draw and a
        # bounding box that comes out inside out.
        for key in ('width', 'height', 'radius'):
            if key in out and not out[key] > 0:
                raise EditError('A %s needs a positive %s, not %r.'
                                % (out['type'], key, out[key]))
        # A polyline of one vertex draws nothing and has nothing to
        # take hold of; of none, not even a place. The constructor
        # only asks that x and y be of the same length, so this is
        # the same kind of arithmetic as a positive width.
        if out['type'] == 'polyline' and len(out['x']) < 2:
            raise EditError('A polyline needs at least two vertices, '
                            'not %d.' % len(out['x']))
        if out.get('thickness', 0.0) < 0:
            raise EditError('A %s cannot be drawn with a thickness of %r.'
                            % (out['type'], out['thickness']))
        return shape

#}}}

#{{{ Undo

    def _snapshot(self):
        return [shape_to_dict(s) for s in self.shapes]

    def _restore(self, snapshot):
        # The body's own list is refilled rather than replaced: it is
        # the user's object, and a layout may be holding it.
        self.shapes[:] = [shape_from_dict(d) for d in snapshot]
        return self

    def undo(self):
        '''
        Put the shapes back as they were before the last edit.
        '''
        if not self._history:
            raise EditError('There is nothing to undo.')
        self._future.append(self._snapshot())
        del self._future[:-UNDO_DEPTH]
        return self._restore(self._history.pop())

    def redo(self):
        '''
        Put back the state the last undo stepped out of.
        '''
        if not self._future:
            raise EditError('There is nothing to redo.')
        self._history.append(self._snapshot())
        del self._history[:-UNDO_DEPTH]
        return self._restore(self._future.pop())

    @property
    def can_undo(self):
        return len(self._history) > 0

    @property
    def can_redo(self):
        return len(self._future) > 0

#}}}

#{{{ The widget

    def widget(self, height=None, **kwargs):
        '''
        Return a Jupyter widget for editing this part.

        The same viewer the layout uses, handed a scene of nothing but
        the shapes: zoom, pan, undo and the layer panel are the same
        code, and what differs is the side bar, which offers shapes
        instead of optics.

        Requires anywidget, like any other notebook viewer.
        '''
        from gtrace.draw.viewer.widget import LayoutViewer
        return LayoutViewer(scene=self.scene_dict(), layout=self,
                            draw_kwargs={},
                            height=0 if height is None else height,
                            editable=True, **kwargs)

#}}}

#}}}

#{{{ Shape helpers

def _shape_points(s):
    '''
    The points of a shape worth snapping to, as (point, kind) pairs.

    The middle of every straight edge is among them, alongside the
    corners and the ends. It is not one of the shape's own numbers -
    no row of a panel sets it - but it is a place a part is drawn
    against: a plate centred on the edge of another, a hole on the
    middle of a side. A curve has none; the middle of an arc is not a
    place anything is lined up on.
    '''
    if isinstance(s, draw.Line):
        a = np.asarray(s.start, dtype='float64')
        b = np.asarray(s.stop, dtype='float64')
        return [(a, 'end'), (b, 'end'), ((a + b) / 2.0, 'midpoint')]
    if isinstance(s, draw.PolyLine):
        pts = [(np.array([x, y], dtype='float64'), 'vertex')
               for x, y in zip(s.x, s.y)]
        return pts + [((pts[i][0] + pts[i + 1][0]) / 2.0, 'midpoint')
                      for i in range(len(pts) - 1)]
    if isinstance(s, draw.Rectangle):
        p = np.asarray(s.point, dtype='float64')
        corners = [p,
                   p + [s.width, 0.0],
                   p + [s.width, s.height],
                   p + [0.0, s.height]]
        return ([(c, 'corner') for c in corners]
                + [(p + [s.width / 2.0, s.height / 2.0], 'centre')]
                + [((corners[i] + corners[(i + 1) % 4]) / 2.0, 'midpoint')
                   for i in range(4)])
    if isinstance(s, (draw.Circle, draw.Arc)):
        return [(s.center, 'centre')]
    if isinstance(s, draw.Text):
        return [(s.point, 'corner')]
    return []

def _offset_shape_dict(d, by):
    '''
    Move a serialized shape, for the copy a duplicate makes.
    '''
    for key in ('start', 'stop', 'point', 'center'):
        if key in d:
            d[key] = [d[key][0] + by, d[key][1] + by]
    if 'x' in d and 'y' in d:
        d['x'] = [v + by for v in d['x']]
        d['y'] = [v + by for v in d['y']]
    return d

#}}}
