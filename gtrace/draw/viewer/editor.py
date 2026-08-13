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
from gtrace.draw.serialize import (NEW_SHAPES, build_shape,
                                   scene_to_dict, shape_to_dict,
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
__version__ = "0.6.0"
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
    'rectangle': frozenset(['point', 'width', 'height', 'angle',
                            'pivot', 'thickness']),
    'circle': frozenset(['center', 'radius', 'thickness']),
    'arc': frozenset(['center', 'radius', 'startangle', 'stopangle',
                      'thickness']),
    'text': frozenset(['text', 'point', 'height', 'rotation']),
}

# NEW_SHAPES - what each kind of shape looks like when it is first put
# down - is imported above rather than defined here. It moved to
# gtrace.draw.serialize, with the rest of what a shape dict is, when a
# shape came to be put down in two places: here, into the part being
# drawn, and on the bench itself, where the viewer's + Shape makes a
# body of one. What a new circle is should not be answered twice. The
# name still reads from here, which is where it was.

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
        scene['points'] = self.points_dict()
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

    def points_dict(self):
        '''
        The points the part names for itself, each with the index an
        edit message names it by.

        A list, where the body keeps a dict: a panel needs a row to
        stay where it is while its name is being typed, and a name
        that is being typed is halfway to something else. So the
        index is the place in the list and the name is a value like
        any other - which is what lets one be renamed at all.
        '''
        return [{'name': str(k), 'point': [float(v[0]), float(v[1])],
                 'index': i}
                for i, (k, v) in enumerate(self.mechanics.points.items())]

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
        # The points the part names for itself. Last, as they are on
        # a bench: a corner of a plate is where the drawing happens to
        # go, and a named point is what the part is stood on something
        # by, so the label says which one.
        for name in sorted(self.mechanics.points):
            p = self.mechanics.points[name]
            points.append({'point': [float(p[0]), float(p[1])],
                           'kind': 'point', 'optic': '', 'label': name})
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
            {'op': 'set_points',      'points': [{'name': 'post',
                                                  'point': [-0.0135, 0.0]}]}
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
        turning means differs by kind - an arc's two angles move, a
        text turns with its own rotation, and a rectangle carries a
        turn of its own, so it takes the angle and the point it was
        taken about and stays a rectangle. The angle is in radians,
        counterclockwise; ``pivot`` defaults to the middle of the
        shape's bounding box, which is the box a front end draws
        around it.

        The named points are set as a whole list rather than one at a
        time. There is no index that survives a rename - a point is
        known by its name, and a name is the thing being edited - so
        every gesture that touches them, adding one, moving one,
        renaming one, taking one away, arrives as the list they leave
        behind. One message is also one step of undo, which is what a
        drag of a point should be.

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
            # A rectangle carries a turn of its own, so turning one
            # here sets that rather than taking the shape apart: the
            # drawing is the same either way - the corners land in the
            # same places - and what comes back is still a rectangle,
            # with a width and a height to go on editing.
            #
            # A body's turn is a different question and is not written
            # into the shape: see turned_shape.
            turned = self.shapes[i]
            if isinstance(turned, draw.Rectangle):
                turned = turned.turned(angle, pivot)
            else:
                turned = rotate_shape(turned, angle, pivot)
            # Through the same door as every other edit: what comes
            # back is taken apart and built again, so the constructors
            # have the last word here too.
            self.shapes[i] = self._build(shape_to_dict(turned))
            return self

        if op == 'set_points':
            # Built whole first, then the body's own dict is refilled
            # rather than replaced - the same care the shapes are
            # restored with, and for the same reason.
            self._fill_points(self._points(msg.get('points')))
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
            # What the bodies built from it are called, before the
            # number. Optional: a model that says nothing leaves the
            # naming to whoever is doing it.
            prefix = msg.get('prefix')
            if prefix is not None and (not isinstance(prefix, str)
                                       or not prefix.strip()):
                raise EditError('A name prefix is a non-empty string, '
                                'not %r.' % (prefix,))
            register_model(name, self.mechanics, description, prefix=prefix)
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

    def _points(self, entries):
        '''
        The named points of a part from a message, or an EditError.

        Built whole before anything is put in place, so a list with
        one bad entry in it leaves the part with the points it had.
        '''
        if not isinstance(entries, list):
            raise EditError('The named points are a list of '
                            "{'name': ..., 'point': [x, y]}, not %r."
                            % (entries,))
        out = {}
        for e in entries:
            if not isinstance(e, dict):
                raise EditError('A named point is a dict with a name and a '
                                'point in it, not %r.' % (e,))
            name = e.get('name')
            if not isinstance(name, str) or not name.strip():
                raise EditError('A point needs a name to be known by, and '
                                '%r is not one.' % (name,))
            name = name.strip()
            if name in out:
                raise EditError('Two points cannot both be called %r: a '
                                'part is stood on one of them by name.'
                                % (name,))
            try:
                p = np.asarray(e.get('point'), dtype='float64').reshape(2)
            except (TypeError, ValueError):
                raise EditError('The point %r is at an [x, y], not %r.'
                                % (name, e.get('point')))
            if not np.all(np.isfinite(p)):
                raise EditError('The point %r cannot be put at %r.'
                                % (name, e.get('point')))
            out[name] = p
        return out

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
        # The rules are build_shape's, which a layout editing the one
        # shape of a body on the bench comes through as well: what can
        # be drawn should not be answered twice.
        try:
            return build_shape(d, error=EditError)
        except UnknownShapeError as e:
            raise EditError(str(e))

#}}}

#{{{ Undo

    def _snapshot(self):
        return {'shapes': [shape_to_dict(s) for s in self.shapes],
                'points': self.points_dict()}

    def _restore(self, snapshot):
        # The body's own list is refilled rather than replaced: it is
        # the user's object, and a layout may be holding it.
        self.shapes[:] = [shape_from_dict(d)
                          for d in snapshot['shapes']]
        self._fill_points(
            dict((e['name'], np.array(e['point'], dtype='float64'))
                 for e in snapshot['points']))
        return self

    def _fill_points(self, points):
        '''
        Put these named points in the body's own dict, in this order.
        '''
        self.mechanics.points.clear()
        self.mechanics.points.update(points)
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
        corners = list(s.corners())
        return ([(c, 'corner') for c in corners]
                + [(sum(corners) / 4.0, 'centre')]
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
