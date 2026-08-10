'''
gtrace.mechanics

A module providing the Mechanics class: a named body on the bench that
takes no part in the trace - a breadboard, a mirror mount, the housing
of a beam dump.

A Mechanics is the counterpart of an Optics: something to draw on the
layout because it is physically there, without any optical role. It is
deliberately not derived from Optics - having no isHit and no hitFrom*
is the definition of the class, not an omission - and the trace never
sees it. Anything that is to stop or reflect light is an Optics (a beam
dump is a mirror of zero reflectivity), and anything that is only to be
seen is a Mechanics.

The shapes are held in local coordinates and carried onto the bench by
a pose (center and rotationAngle), so that moving the body is a change
of two numbers rather than of every vertex, and so that the pose and
the geometry cannot fall out of step - the position is written down
once. The same reasoning put the AR surface right in 2026-08-03: a
second description of where something is, is a place for the two to
disagree.
'''

#{{{ Import modules

import numpy as np

import gtrace.draw as draw
from gtrace.draw.serialize import UnknownShapeError

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

#{{{ Constants

#: The layer a Mechanics is drawn on unless told otherwise. A layer of
#: its own, because a layer is exactly the mechanism CAD offers for
#: something you want to be able to switch off: the hardware can be
#: hidden without touching the optics or the beams.
DEFAULT_LAYER = 'hardware'

#: Color of that layer: a grey, so the hardware reads as background to
#: the beams and optics rather than competing with them.
LAYER_COLOR = (110, 110, 110)

#}}}

#{{{ point_in_polygon

def point_in_polygon(point, polygon):
    '''
    Whether a point lies inside a polygon, by ray casting.

    This exists because a Mechanics is picked by its area: a breadboard
    is a large rectangle, and the enclosing-circle test the optics use
    would cover the whole bench around it.

    Parameters
    ----------
    point : array-like
        The point, of shape (2,).
    polygon : sequence of array-like
        The vertices, in order, each of shape (2,). The polygon is
        closed implicitly; the first vertex need not be repeated.

    Returns
    -------
    bool
    '''
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside

#}}}

#{{{ Local bounding box

def _shape_bbox_points(s):
    '''
    Points that bound one shape, in its own coordinates. An arc is
    bounded by the full circle it lies on - looser than the arc itself,
    but never smaller, which is what a bound must guarantee.
    '''
    if isinstance(s, draw.Line):
        return [s.start, s.stop]
    if isinstance(s, draw.PolyLine):
        return [[x, y] for x, y in zip(s.x, s.y)]
    if isinstance(s, draw.Rectangle):
        p = np.asarray(s.point, dtype='float64')
        return [p, p + [s.width, s.height]]
    if isinstance(s, (draw.Circle, draw.Arc)):
        c = np.asarray(s.center, dtype='float64')
        return [c - s.radius, c + s.radius]
    if isinstance(s, draw.Text):
        return [s.point]
    return []

#}}}

#{{{ Mechanics

class Mechanics(object):
    '''
    A named body on the bench that takes no part in the trace.

    The geometry is a list of drawing primitives (gtrace.draw shapes)
    in local coordinates, placed on the bench by a pose: ``center`` is
    where the local origin lands and ``rotationAngle`` how far the body
    is turned about it. The shapes themselves never change when the
    body moves.

    Attributes
    ----------
    name : str
        Name of the body, unique within a layout.
    center : numpy.ndarray
        Where the local origin stands on the bench, of shape (2,).
    rotationAngle : float
        How far the body is turned about the local origin, in radians,
        counterclockwise.
    shapes : list of gtrace.draw.Shape
        The geometry, in local coordinates.
    layer : str
        The layer the body is drawn on. Defaults to 'hardware'.
    model : str or None
        The catalogue model the shapes came from, if any. A label, not
        a reference: the shapes saved with a layout are the truth, and
        the model name only records where they were taken from, so a
        library that has moved on cannot silently redraw an old layout.
    '''

    def __init__(self, shapes=None, center=[0.0, 0.0], rotationAngle=0.0,
                 name='Hardware', layer=DEFAULT_LAYER, model=None):
        self.name = name
        self.center = np.array(center, dtype='float64')
        self.rotationAngle = float(rotationAngle)
        self.shapes = list(shapes) if shapes is not None else []
        self.layer = str(layer)
        self.model = model if model is None else str(model)

#{{{ Coordinate transform

    def to_world(self, p):
        '''
        Carry a local point onto the bench: turned by rotationAngle and
        stood at center.

        Parameters
        ----------
        p : array-like
            A local point of shape (2,), or an array of them of shape
            (N, 2).

        Returns
        -------
        numpy.ndarray
        '''
        p = np.asarray(p, dtype='float64')
        ca = np.cos(self.rotationAngle)
        sa = np.sin(self.rotationAngle)
        R = np.array([[ca, -sa], [sa, ca]])
        return p @ R.T + self.center

#}}}

#{{{ Bounding box and outline

    def local_bbox(self):
        '''
        The bounding box of the shapes, in local coordinates.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The lower-left and upper-right corners, each of shape (2,).
            A body with no shapes is a point at the local origin.
        '''
        points = []
        for s in self.shapes:
            points.extend(_shape_bbox_points(s))
        if not points:
            z = np.zeros(2)
            return z, z.copy()
        pts = np.array([[float(p[0]), float(p[1])] for p in points])
        return pts.min(axis=0), pts.max(axis=0)

    def outline(self):
        '''
        The bounding box of the shapes, carried onto the bench: four
        corners in world coordinates, counterclockwise.

        This is the polygon the body is picked by. A box rather than
        the exact silhouette, because a grab handle wants to be simple
        and a little generous; the exact shape belongs to the drawing.

        Returns
        -------
        numpy.ndarray
            Of shape (4, 2).
        '''
        lo, hi = self.local_bbox()
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                            [hi[0], hi[1]], [lo[0], hi[1]]])
        return self.to_world(corners)

    def contains(self, point):
        '''
        Whether a world point falls on the body - inside its outline().
        '''
        return point_in_polygon(point, self.outline())

#}}}

#{{{ World shapes and drawing

    def world_shapes(self):
        '''
        The shapes as they stand on the bench: new primitives in world
        coordinates. The local ones are left untouched.

        A rectangle survives as a rectangle only while the body is not
        turned; a Rectangle has no orientation of its own, so a turned
        one comes out as the closed polyline of its corners.
        '''
        out = []
        for s in self.shapes:
            out.append(self._world_shape(s))
        return out

    def _world_shape(self, s):
        if isinstance(s, draw.Line):
            return draw.Line(self.to_world(s.start), self.to_world(s.stop),
                             thickness=s.thickness)
        if isinstance(s, draw.PolyLine):
            pts = self.to_world(np.column_stack([s.x, s.y]))
            return draw.PolyLine(x=pts[:, 0], y=pts[:, 1],
                                 thickness=s.thickness)
        if isinstance(s, draw.Rectangle):
            p = np.asarray(s.point, dtype='float64')
            if self.rotationAngle == 0.0:
                return draw.Rectangle(self.to_world(p), s.width, s.height,
                                      thickness=s.thickness)
            corners = np.array([p, p + [s.width, 0.0],
                                p + [s.width, s.height], p + [0.0, s.height],
                                p])
            pts = self.to_world(corners)
            return draw.PolyLine(x=pts[:, 0], y=pts[:, 1],
                                 thickness=s.thickness)
        if isinstance(s, draw.Circle):
            return draw.Circle(self.to_world(s.center), s.radius,
                               thickness=s.thickness)
        if isinstance(s, draw.Arc):
            return draw.Arc(self.to_world(s.center), s.radius,
                            s.startangle + self.rotationAngle,
                            s.stopangle + self.rotationAngle,
                            thickness=s.thickness)
        if isinstance(s, draw.Text):
            return draw.Text(s.text, self.to_world(s.point), height=s.height,
                             rotation=s.rotation + self.rotationAngle)
        raise UnknownShapeError(
            'Shape not supported: %s' % type(s).__name__)

    def draw(self, cv, drawName=False):
        '''
        Draw itself into a canvas, on its own layer.

        Parameters
        ----------
        cv : gtrace.draw.Canvas
            The canvas to draw into.
        drawName : bool, optional
            Whether to write the name at the center, on the 'text'
            layer like an optics name. Defaults to False.
        '''
        if self.layer not in cv.layers:
            cv.add_layer(self.layer, color=LAYER_COLOR)
        for s in self.world_shapes():
            cv.add_shape(s, layername=self.layer)

        if drawName:
            # Sized against the body, as a mirror's name is sized
            # against its thickness: a drawing has no fixed scale.
            lo, hi = self.local_bbox()
            size = min(hi[0] - lo[0], hi[1] - lo[1])
            height = size / 8.0 if size > 0 else 0.01
            width = height * len(self.name)
            point = (self.to_world((lo + hi) / 2.0)
                     - np.array([width / 2.0, height / 2.0]))
            cv.add_shape(draw.Text(text=self.name, point=point,
                                   height=height),
                         layername='text')

#}}}

#{{{ Move, rotate, copy

    def translate(self, trVect):
        '''
        Move the body by a vector, like Optics.translate.
        '''
        self.center = self.center + np.array(trVect, dtype='float64')

    def rotate(self, angle, center=None):
        '''
        Turn the body, like Optics.rotate.

        Parameters
        ----------
        angle : float
            How far to turn, in radians, counterclockwise.
        center : array-like or None, optional
            The point to turn about. None - the default - turns the
            body about its own center, which then stays put.
        '''
        self.rotationAngle = self.rotationAngle + float(angle)
        if center is not None:
            pivot = np.array(center, dtype='float64')
            offset = self.center - pivot
            ca, sa = np.cos(angle), np.sin(angle)
            self.center = pivot + np.array(
                [offset[0] * ca - offset[1] * sa,
                 offset[0] * sa + offset[1] * ca])

    def copy(self):
        '''
        A new Mechanics with the same pose and a copy of the shape
        list. The shapes themselves are shared: nothing in gtrace
        mutates a drawing primitive, and the copy is what save, undo
        and the clipboard-of-the-future need.
        '''
        m = Mechanics(shapes=list(self.shapes),
                      center=self.center.copy(),
                      rotationAngle=self.rotationAngle,
                      name=self.name, layer=self.layer, model=self.model)
        return m

#}}}

#}}}
