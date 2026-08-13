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

import json

import numpy as np

import gtrace.draw as draw
from gtrace.draw.serialize import (shape_to_dict, shape_from_dict,
                                   UnknownShapeError)

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

#{{{ Constants

#: The layer a Mechanics is drawn on unless told otherwise. A layer of
#: its own, because a layer is exactly the mechanism CAD offers for
#: something you want to be able to switch off: the bodies can be
#: hidden without touching the optics or the beams.
DEFAULT_LAYER = 'mechanics'

#: Color of that layer: a grey, so a body reads as background to the
#: beams and optics rather than competing with them.
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
        # Its corners rather than the two it is written from: a turned
        # rectangle reaches past both of them.
        return [c for c in s.corners()]
    if isinstance(s, (draw.Circle, draw.Arc)):
        c = np.asarray(s.center, dtype='float64')
        return [c - s.radius, c + s.radius]
    if isinstance(s, draw.Text):
        return [s.point]
    return []

#}}}

#{{{ Turning a shape

def turned_shape(s, angle, offset=(0.0, 0.0)):
    '''
    A copy of one shape turned about the origin and carried.

    The shape itself is never touched: a new primitive comes back,
    with the same thickness.

    A ``Rectangle`` survives as a ``Rectangle`` only while the turn is
    nothing at all - a carry alone keeps it, pivot and all. Turned, it
    comes back as the closed polyline of its four corners, which is
    what a DXF would write of it either way. A rectangle turned about
    a pivot of its own is turned about the pivot carried with it, so
    the corners are its own, not those of the box it was written
    from.

    Parameters
    ----------
    s : gtrace.draw shape
        The shape to turn.
    angle : float
        How far to turn it, counterclockwise, in radians.
    offset : array-like, optional
        Where to carry it afterwards. Default the origin.

    Returns
    -------
    A new shape, of the same class except for a turned rectangle.

    Raises
    ------
    gtrace.draw.serialize.UnknownShapeError
        If the shape is of a kind this does not know.
    '''
    ca = np.cos(angle)
    sa = np.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    off = np.asarray(offset, dtype='float64')

    def carry(p):
        return np.asarray(p, dtype='float64') @ R.T + off

    if isinstance(s, draw.Line):
        return draw.Line(carry(s.start), carry(s.stop),
                         thickness=s.thickness)
    if isinstance(s, draw.PolyLine):
        pts = carry(np.column_stack([s.x, s.y]))
        return draw.PolyLine(x=pts[:, 0], y=pts[:, 1],
                             thickness=s.thickness)
    if isinstance(s, draw.Rectangle):
        p = np.asarray(s.point, dtype='float64')
        if angle == 0.0:
            # A pivot of None is the middle of the rectangle, and the
            # middle travels with it: it is left as None rather than
            # written out, so a carried rectangle still turns about
            # itself.
            return draw.Rectangle(carry(p), s.width, s.height,
                                  thickness=s.thickness, angle=s.angle,
                                  pivot=(None if s.pivot is None
                                         else carry(s.pivot)))
        # The first corner again at the end: an open polyline of four
        # would be a rectangle with a side missing.
        corners = np.vstack([s.corners(), s.corners()[:1]])
        pts = carry(corners)
        return draw.PolyLine(x=pts[:, 0], y=pts[:, 1],
                             thickness=s.thickness)
    if isinstance(s, draw.Circle):
        return draw.Circle(carry(s.center), s.radius, thickness=s.thickness)
    if isinstance(s, draw.Arc):
        return draw.Arc(carry(s.center), s.radius,
                        s.startangle + angle, s.stopangle + angle,
                        thickness=s.thickness)
    if isinstance(s, draw.Text):
        return draw.Text(s.text, carry(s.point), height=s.height,
                         rotation=s.rotation + angle)
    raise UnknownShapeError(
        'Shape not supported: %s' % type(s).__name__)

def rotate_shape(s, angle, pivot=(0.0, 0.0)):
    '''
    A copy of one shape turned about a point.

    Turning about a point is turning about the origin and carrying
    the result back, which is why this and the pose of a body are the
    same arithmetic. A turned rectangle comes back as the closed
    polyline of its corners - see :func:`turned_shape`.

    Parameters
    ----------
    s : gtrace.draw shape
    angle : float
        Counterclockwise, in radians.
    pivot : array-like, optional
        The point to turn about. Default the local origin.

    Returns
    -------
    A new shape.
    '''
    ca = np.cos(angle)
    sa = np.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    pv = np.asarray(pivot, dtype='float64')
    return turned_shape(s, angle, pv - R @ pv)

def shape_centre(s):
    '''
    The middle of the box a shape occupies, in its own coordinates.

    What a shape is turned about when nothing else is said: it is the
    point the editor already draws a box around, so a turn about it is
    the one turn that can be seen coming.
    '''
    pts = np.asarray(_shape_bbox_points(s), dtype='float64')
    if not len(pts):
        return np.zeros(2)
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0

#}}}

#{{{ The frame a body attaches to

def host_pose(host):
    '''
    Where a host stands and which way it is turned.

    An optics is turned by the normal of its HR face and a body by its
    own angle. That is the only difference between standing on a mirror
    and standing on a pedestal, so it is written down once here rather
    than at each of the three places that attach, derive and detach.

    Parameters
    ----------
    host : gtrace.optcomp.Optics or Mechanics

    Returns
    -------
    (numpy.ndarray, float)
        The host's centre, and its angle in radians.

    Raises
    ------
    ValueError
        If the object has no pose to attach to.
    '''
    if isinstance(host, Mechanics):
        return (np.asarray(host.center, dtype='float64'),
                float(host.rotationAngle))
    if hasattr(host, 'center') and hasattr(host, 'normAngleHR'):
        return (np.asarray(host.center, dtype='float64'),
                float(host.normAngleHR))
    raise ValueError('%r has no pose to attach to.' % (host,))

def _turn(p, angle):
    '''
    A local vector carried into a frame turned by angle.
    '''
    ca, sa = np.cos(angle), np.sin(angle)
    p = np.asarray(p, dtype='float64')
    return np.array([p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca])

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

    A body can be **attached** to an optics, which is what a mirror
    mount is: something that stands where its mirror stands. An
    attached body has no pose of its own - ``center`` and
    ``rotationAngle`` are derived, on every read, from the host's pose
    and the attachment offset. There is no callback to miss and no
    stored copy to go stale: the 2026-08-03 bugs were all a
    notification not arriving, and a derived value cannot fail to be
    notified. The price is that an attached body cannot be moved on
    its own - which is what "attached" means. ``detach()`` bakes the
    derived pose in and frees it.

    Attributes
    ----------
    name : str
        Name of the body, unique within a layout.
    center : numpy.ndarray
        Where the local origin stands on the bench, of shape (2,).
        Derived from the host while attached, and read-only then.
    rotationAngle : float
        How far the body is turned about the local origin, in radians,
        counterclockwise. Derived while attached, like center.
    shapes : list of gtrace.draw.Shape
        The geometry, in local coordinates.
    layer : str
        The layer the body is drawn on. Defaults to 'mechanics'.
    model : str or None
        The catalogue model the shapes came from, if any. A label, not
        a reference: the shapes saved with a layout are the truth, and
        the model name only records where they were taken from, so a
        library that has moved on cannot silently redraw an old layout.
    attached_to : gtrace.optcomp.Optics or None
        The optics this body stands on, held by reference, or None for
        a body standing on its own. The constructor also accepts a
        name, which OpticalLayout resolves when the body is
        registered; until then the pose cannot be read.
    offset : numpy.ndarray
        Where the local origin stands in the host's frame: the host's
        substrate centre, with x along its HR normal. ``[0, 0]`` puts
        the body's origin at the host's centre.
    offset_angle : float
        How far the body is turned relative to the host, in radians.
    '''

    def __init__(self, shapes=None, center=None, rotationAngle=None,
                 name='Mechanics', layer=DEFAULT_LAYER, model=None,
                 attached_to=None, offset=None, offset_angle=0.0,
                 params=None, points=None, attach_point=None,
                 fix_rotation=True):
        self.name = name
        self.shapes = list(shapes) if shapes is not None else []
        self.layer = str(layer)
        self.model = model if model is None else str(model)
        #: How the shapes were built, when a builder built them: a
        #: JSON-compatible dict with a 'kind' and that kind's
        #: parameters. What it buys is resize(): a breadboard cut to a
        #: new size is re-drilled from its parameters, where scaling
        #: the shapes would scale the holes and the grid with them.
        #: None for a body drawn by hand, which has no parameters to
        #: rebuild from.
        self.params = None if params is None else dict(params)

        #: Named points of the part, in local coordinates: the screw
        #: hole under a mount, the axis of a pedestal, the bore of a
        #: clamping fork. A drag settles on them and a measurement
        #: reaches them, and they are what one part is stood on
        #: another by - so they are a property of the part rather than
        #: of the layout, and travel with the model.
        self.points = {}
        for k, v in (points or {}).items():
            self.points[str(k)] = np.array(v, dtype='float64')

        #: Which point of this body is pinned to the host, in local
        #: coordinates. The origin by default, which is the convention
        #: every mount is drawn to; a pedestal clamped by a fork is
        #: pinned by the bore the fork closes around instead, and that
        #: is also the point it turns about.
        self.attach_point = (np.zeros(2) if attach_point is None
                             else np.array(attach_point, dtype='float64'))

        #: Whether the turn relative to the host is frozen. True - the
        #: default - is a mount bolted to its mirror: it faces where
        #: the host faces and there is nothing to edit. False is a
        #: clamping fork, which may be swung about the point it is
        #: pinned by. Either way the body turns *with* the host: what
        #: this decides is who may change the relative angle, not how
        #: the pose is derived.
        self.fix_rotation = bool(fix_rotation)

        #: The host object, or None. A name given instead is kept in
        #: _attach_name until a layout resolves it; the pose refuses to
        #: be read in between, which is louder than being wrong.
        self.attached_to = None
        self._attach_name = None
        self.offset = np.zeros(2)
        self.offset_angle = 0.0
        self._center = np.zeros(2)
        self._rotationAngle = 0.0

        if attached_to is not None:
            if center is not None or rotationAngle is not None:
                raise ValueError(
                    'An attached body has no pose of its own: give '
                    'offset and offset_angle instead of center and '
                    'rotationAngle.')
            if isinstance(attached_to, str):
                self._attach_name = attached_to
                self.offset = (np.zeros(2) if offset is None
                               else np.array(offset, dtype='float64'))
                self.offset_angle = float(offset_angle)
            else:
                self.attach(attached_to, offset=(offset if offset is not None
                                                 else [0.0, 0.0]),
                            offset_angle=offset_angle)
        else:
            self._center = np.array([0.0, 0.0] if center is None else center,
                                    dtype='float64')
            self._rotationAngle = float(0.0 if rotationAngle is None
                                        else rotationAngle)

#{{{ Pose: derived while attached, held while free

    def _require_link(self):
        if self._attach_name is not None:
            raise ValueError(
                "'%s' is attached to '%s' by name only. Register it in a "
                'layout holding that optics to resolve the link.'
                % (self.name, self._attach_name))

    @property
    def center(self):
        self._require_link()
        if self.attached_to is None:
            return self._center
        hc, ha = host_pose(self.attached_to)
        # The attach point lands at the offset, in the host's frame;
        # the local origin is wherever that leaves it. With the
        # default attach point of [0, 0] the second term vanishes and
        # this is the rule every mount was already drawn to.
        return (hc + _turn(self.offset, ha)
                - _turn(self.attach_point, ha + self.offset_angle))

    @center.setter
    def center(self, value):
        if self.attached_to is not None or self._attach_name is not None:
            raise ValueError(
                "'%s' is attached: it goes where its host goes. Detach it "
                'first, or change the offset.' % self.name)
        self._center = np.array(value, dtype='float64')

    @property
    def rotationAngle(self):
        self._require_link()
        if self.attached_to is None:
            return self._rotationAngle
        return host_pose(self.attached_to)[1] + self.offset_angle

    @rotationAngle.setter
    def rotationAngle(self, value):
        if self.attached_to is not None or self._attach_name is not None:
            if self.fix_rotation:
                raise ValueError(
                    "'%s' is attached with its turn fixed: it faces where "
                    "its host faces. Set fix_rotation=False to swing it, "
                    'or detach it.' % self.name)
            # Free to swing, about the point it is pinned by: the
            # relative angle is the thing that is really being set,
            # and the position follows from it.
            self._require_link()
            self.offset_angle = (float(value)
                                 - host_pose(self.attached_to)[1])
            return
        self._rotationAngle = float(value)

    def attach(self, host, offset=None, offset_angle=None,
               keep_pose=None, attach_point=None, fix_rotation=None):
        '''
        Stand this body on an optics. From here on its pose is derived
        from the host's - move the mirror and the mount comes along,
        with no notification to miss.

        Where it stands on the host is the model's to say, not the
        drop point's: a mirror mount is built around its optic, so the
        right relative position is unique and drawn into the shapes.
        The convention that carries it is the local origin - every
        builder and library model draws its shapes so that the origin
        is the point meant to coincide with the host's substrate
        centre - so attaching with no offset seats the body there,
        wherever it happened to be lying beforehand.

        Parameters
        ----------
        host : gtrace.optcomp.Optics or Mechanics
            What to stand on. A body may stand on another body - a
            pedestal on a mount, a fork on a pedestal - and the chain
            follows the optics at the root of it. A cycle is refused.
        offset : array-like or None, optional
            Where the local origin stands in the host's frame (its
            substrate centre, x along the HR normal). None - the
            default - is ``[0, 0]``: the designed position.
        offset_angle : float or None, optional
            The turn relative to the host. None is 0: squarely on it.
        keep_pose : bool or None, optional
            Derive the offset and the angle from where the body stands
            now instead, so that attaching changes what moves it and
            not where it is; offset and offset_angle must be left None
            with it. None - the default - decides by the host: a body
            standing on an **optics** goes to the model's designed
            place, since a mount's position on its mirror is unique
            and drawn into the shapes; a body standing on another
            **body** keeps where it is, since which hole of a mount a
            pedestal sits in is a choice made on the bench and not by
            the library.
        attach_point : array-like or None, optional
            Which point of this body is pinned, in local coordinates.
            None keeps whatever it already had (the origin, unless
            something else was said). This is also the point the body
            turns about when its turn is free.
        fix_rotation : bool or None, optional
            Whether the turn relative to the host is frozen. None
            keeps the current setting.

        Returns
        -------
        self : Mechanics

        Raises
        ------
        ValueError
            If the host has no pose, or if standing on it would make a
            cycle.
        '''
        hc, ha = host_pose(host)
        self._refuse_cycle(host)
        if attach_point is not None:
            self.attach_point = np.array(attach_point, dtype='float64')
        if fix_rotation is not None:
            self.fix_rotation = bool(fix_rotation)
        if keep_pose is None:
            # An offset given explicitly is an answer to the same
            # question, and the explicit one wins.
            keep_pose = (isinstance(host, Mechanics)
                         and offset is None and offset_angle is None)

        if keep_pose:
            if offset is not None or offset_angle is not None:
                raise ValueError(
                    'keep_pose derives the offset from where the body '
                    'stands; giving one as well is two answers to one '
                    'question.')
            # The current world pose, read before the switch. It works
            # whether the body is free or already attached to
            # something else. What is pinned is the attach point, so
            # that is the point whose place is kept.
            here = self.center + _turn(self.attach_point, self.rotationAngle)
            offset = _turn(here - hc, -ha)
            offset_angle = self.rotationAngle - ha
        else:
            if offset is None:
                offset = [0.0, 0.0]
            if offset_angle is None:
                offset_angle = 0.0

        self.attached_to = host
        self._attach_name = None
        self.offset = np.array(offset, dtype='float64')
        self.offset_angle = float(offset_angle)
        return self

    def _refuse_cycle(self, host):
        '''
        Refuse standing on something that already stands on this body.

        A cycle would make the pose derive from itself, and since it
        is derived on every read that is not a wrong answer but an
        unbounded one.
        '''
        seen = host
        while isinstance(seen, Mechanics):
            if seen is self:
                raise ValueError(
                    "'%s' cannot stand on '%s': it already holds it up."
                    % (self.name, getattr(host, 'name', host)))
            seen = seen.attached_to

    def hosts(self):
        '''
        What this body stands on, nearest first, ending at the optics
        the chain hangs from. Empty for a free body.
        '''
        out = []
        h = self.attached_to
        while h is not None:
            out.append(h)
            h = getattr(h, 'attached_to', None) if isinstance(h, Mechanics) \
                else None
        return out

    def world_points(self):
        '''
        The named points of this part, where they stand on the bench.

        Returns
        -------
        dict
            name -> numpy array of shape (2,).
        '''
        return dict((k, self.to_world(v)) for k, v in self.points.items())

    def detach(self):
        '''
        Free this body, leaving it exactly where it stands: the derived
        pose is baked into center and rotationAngle. A body that is not
        attached is left alone.

        Returns
        -------
        self : Mechanics
        '''
        if self._attach_name is not None:
            # Attached by name and never resolved: there is no derived
            # pose to bake, so the free pose it was built with stands.
            self._attach_name = None
            return self
        if self.attached_to is None:
            return self
        c = self.center
        a = self.rotationAngle
        self.attached_to = None
        self._center = c
        self._rotationAngle = a
        return self

#}}}

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
        turned; a rectangle carries an angle of its own, but a body's
        turn is not written into it, so a turned one comes out as the
        closed polyline of its corners.
        '''
        out = []
        for s in self.shapes:
            out.append(self._world_shape(s))
        return out

    def _world_shape(self, s):
        return turned_shape(s, self.rotationAngle, self.center)

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

    def resize(self, width=None, height=None):
        '''
        Rebuild a parametric body at a new size.

        Only a body a builder made knows how to do this: a breadboard
        cut to a new size is re-drilled from its parameters, with the
        same pitch and the same holes, where scaling the shapes would
        scale the holes and the grid along with the plate. A body
        drawn by hand has no parameters, and is refused - its shapes
        are all anyone knows about it.

        A round body has one size, not two. Either name sets its
        diameter, and a message that gives both is taken at its word:
        two that disagree are refused rather than resolved by picking
        one, since a round board asked to be 300 by 400 is a
        misunderstanding and not a size.

        Parameters
        ----------
        width, height : float or None, optional
            The new size, in metres. None keeps that side as it is.

        Returns
        -------
        self : Mechanics
        '''
        kind = self.params.get('kind') if self.params else None
        if kind not in _RESIZABLE:
            raise ValueError(
                "'%s' is not a resizable body: it was drawn by hand, so "
                'edit its shapes instead.' % self.name)
        p = dict(self.params)
        if kind in _ROUND:
            if (width is not None and height is not None
                    and float(width) != float(height)):
                raise ValueError(
                    "'%s' is round, so it has one size: %g and %g cannot "
                    'both be it.' % (self.name, width, height))
            if width is None:
                width = height
            height = width
        if width is not None:
            p['width'] = float(width)
        if height is not None:
            p['height'] = float(height)
        if p['width'] <= 0 or p['height'] <= 0:
            raise ValueError('A size must be positive, not %g x %g.'
                             % (p['width'], p['height']))
        self.shapes = _RESIZABLE[kind](p)
        self.params = p
        return self

    @property
    def resizable(self):
        '''
        How resize() can rebuild this body, or None.

        ``'round'`` for a body with one size - a round board is cut to
        a diameter - and ``'box'`` for one with two. A front end needs
        the difference: the rows it offers are Diameter or Width and
        Height, and a corner dragged on a disc has no opposite corner
        to hold still.
        '''
        kind = self.params.get('kind') if self.params else None
        if kind not in _RESIZABLE:
            return None
        return 'round' if kind in _ROUND else 'box'

    def edit(self, height=None, **kwargs):
        '''
        Open a shape editor on this body, in a notebook.

        The editor works in the local frame - the one the shapes are
        written in, with the origin marked, since the origin is what
        comes to sit at the host's substrate centre - and edits this
        body by reference: a layout already holding it draws the new
        shapes at its next draw.

        Returns the widget, so a cell ending in ``mt.edit()`` shows
        it. ``ShapeEditor`` itself, for driving without a browser, is
        in ``gtrace.draw.viewer.editor``.

        Parameters
        ----------
        height : int or None, optional
            Height of the editor in pixels, as for a layout widget.
        **kwargs
            Passed to the widget.

        Returns
        -------
        widget : anywidget.AnyWidget
        '''
        from gtrace.draw.viewer.editor import ShapeEditor
        return ShapeEditor(self).widget(height=height, **kwargs)

    def copy(self):
        '''
        A new Mechanics with the same pose and a copy of the shape
        list. The shapes themselves are shared: nothing in gtrace
        mutates a drawing primitive, and the copy is what save, undo
        and the clipboard-of-the-future need. An attached body copies
        as attached, standing on the same host.
        '''
        common = dict(shapes=list(self.shapes), name=self.name,
                      layer=self.layer, model=self.model,
                      params=self.params, points=self.points,
                      attach_point=self.attach_point.copy(),
                      fix_rotation=self.fix_rotation)
        if self.attached_to is not None or self._attach_name is not None:
            return Mechanics(attached_to=(self.attached_to
                                          if self.attached_to is not None
                                          else self._attach_name),
                             offset=self.offset.copy(),
                             offset_angle=self.offset_angle, **common)
        return Mechanics(center=self._center.copy(),
                         rotationAngle=self._rotationAngle, **common)

#}}}

#}}}

#{{{ Builders

def breadboard(width, height, pitch=0.025, hole_diameter=0.006,
               margin=None, holes=True, **kwargs):
    '''
    A breadboard: a rectangular plate with a grid of mounting holes,
    its local origin at the centre of the plate.

    The hole grid is laid out the way real boards drill it: symmetric
    about the centre, the outermost rows a margin in from the edges,
    everything else on the pitch. The defaults are the metric standard
    - M6 clearance holes on a 25 mm grid, half a pitch in from the
    edge.

    Parameters
    ----------
    width, height : float
        Size of the plate, in metres.
    pitch : float, optional
        Hole spacing. Defaults to 25 mm.
    hole_diameter : float, optional
        Diameter of a hole. Defaults to 6 mm.
    margin : float or None, optional
        Distance from an edge to the first hole row. None - the
        default - is half a pitch, which is where the standard grid
        falls.
    holes : bool, optional
        Whether to draw the grid at all. A board across a whole bench
        is sometimes wanted as just its outline.
    **kwargs
        Passed to Mechanics: name, center, rotationAngle, layer,
        attached_to and the rest.

    Returns
    -------
    Mechanics
    '''
    params = {'kind': 'breadboard',
              'width': float(width), 'height': float(height),
              'pitch': float(pitch),
              'hole_diameter': float(hole_diameter),
              'margin': None if margin is None else float(margin),
              'holes': bool(holes)}
    return Mechanics(shapes=_breadboard_shapes(params), params=params,
                     **kwargs)

def _breadboard_shapes(p):
    '''
    The shapes of a breadboard, from its parameters. Split out so that
    resize() can re-drill an existing board the same way breadboard()
    drilled it.
    '''
    width, height, pitch = p['width'], p['height'], p['pitch']
    shapes = []
    shapes.append(draw.Rectangle([-width / 2.0, -height / 2.0],
                                 width, height))
    m = pitch / 2.0 if p['margin'] is None else p['margin']
    if p['holes'] and width >= 2 * m and height >= 2 * m:
        r = p['hole_diameter'] / 2.0
        # The +1e-9 keeps a span that is an exact number of pitches
        # from losing its last row to floating point.
        nx = int(np.floor((width - 2 * m) / pitch + 1e-9)) + 1
        ny = int(np.floor((height - 2 * m) / pitch + 1e-9)) + 1
        x0 = -(nx - 1) * pitch / 2.0
        y0 = -(ny - 1) * pitch / 2.0
        for i in range(nx):
            for j in range(ny):
                shapes.append(draw.Circle([x0 + i * pitch,
                                           y0 + j * pitch], r))
    return shapes

def round_breadboard(diameter, pitch=0.025, hole_diameter=0.006,
                     margin=None, holes=True, **kwargs):
    '''
    A round breadboard: a disc with the same grid of mounting holes,
    its local origin at the centre.

    A vacuum tank is round, and so is the board that goes in the
    bottom of it. The grid is the one a rectangular board is drilled
    with - symmetric about the centre, on the pitch - and the edge
    decides which of those holes exist: a hole is drilled where it
    lies a margin in from the rim, and left out where it does not, so
    the outermost rows come out shorter towards the edge the way a
    real disc is drilled.

    Parameters
    ----------
    diameter : float
        Across the disc, in metres.
    pitch : float, optional
        Hole spacing. Defaults to 25 mm, as on a rectangular board.
    hole_diameter : float, optional
        Diameter of a hole. Defaults to 6 mm.
    margin : float or None, optional
        How far in from the rim a hole must be to be drilled. None -
        the default - is half a pitch, which is the rectangular
        board's rule read round.
    holes : bool, optional
        Whether to draw the grid at all.
    **kwargs
        Passed to Mechanics: name, center, rotationAngle, layer,
        attached_to and the rest.

    Returns
    -------
    Mechanics
    '''
    # Width and height rather than a diameter of its own: a body's
    # size is one thing, asked for in one place - the scene, the
    # panel, resize() - and a round board is a body whose two sides
    # are the same. resize() is what keeps them so.
    params = {'kind': 'round_breadboard',
              'width': float(diameter), 'height': float(diameter),
              'pitch': float(pitch),
              'hole_diameter': float(hole_diameter),
              'margin': None if margin is None else float(margin),
              'holes': bool(holes)}
    return Mechanics(shapes=_round_breadboard_shapes(params), params=params,
                     **kwargs)

def _round_breadboard_shapes(p):
    '''
    The shapes of a round breadboard, from its parameters. The rim,
    and the holes of the same grid that fall inside it.
    '''
    radius = p['width'] / 2.0
    pitch = p['pitch']
    shapes = [draw.Circle([0.0, 0.0], radius)]
    m = pitch / 2.0 if p['margin'] is None else p['margin']
    reach = radius - m
    if p['holes'] and reach > 0:
        r = p['hole_diameter'] / 2.0
        # The grid is the same one a rectangular board carries -
        # symmetric about the centre, on the pitch - and the rim only
        # decides which of its holes are there.
        n = int(np.floor(reach / pitch + 1e-9))
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x, y = i * pitch, j * pitch
                if np.hypot(x, y) <= reach + 1e-9:
                    shapes.append(draw.Circle([x, y], r))
    return shapes

#: The parametric kinds resize() can rebuild, and how. A mount is
#: deliberately not here: what its width means to the plate and the
#: knobs is not a corner-drag's to decide.
_RESIZABLE = {'breadboard': _breadboard_shapes,
              'round_breadboard': _round_breadboard_shapes}

#: The kinds that have one size rather than two, because they are
#: round. A drag on one of these sets a diameter, and a pair of sizes
#: that disagree is refused rather than resolved by picking one.
_ROUND = frozenset(['round_breadboard'])

def mirror_mount(scale=1.0, knobs=True, **kwargs):
    '''
    A one-inch kinematic mirror mount seen from above, drawn after a
    Polaris KA1 style footprint: the front plate the optic sits in, the
    back plate across the adjustment gap with the two adjuster tips
    showing in it, the two adjuster knobs on their stems out of the
    back, and the hole it is bolted down through.

    The local origin is the substrate centre of the mounted optic -
    the point marked on the drawing this is taken from - which sits
    3 mm behind the front face of the front plate. ``attached_to``
    with no offset therefore seats the mount so that a 6 mm thick
    optic ends flush with that face. The dimensions are the
    drawing's, in millimetres, times ``scale``:

    - plate 45.7 wide, front plate 7 deep
    - adjustment gap 3.2, with the two 5 mm adjuster tips showing,
      6.4 in from either edge
    - back plate 12.7 deep
    - knobs 15.2 x 8.6 on 6.4-wide, 7.6-long stems, on the adjuster
      lines
    - the post hole, 4 across, 13.5 behind the origin - the point
      named 'post', which is also where the circle is drawn

    Parameters
    ----------
    scale : float, optional
        Multiplies every dimension. 1 - the default - is the one-inch
        mount; for a two-inch optic see mirror_mount_2in, which is
        measured rather than scaled.
    knobs : bool, optional
        Whether to draw the adjuster knobs and their stems.
    **kwargs
        Passed to Mechanics.

    Returns
    -------
    Mechanics
    '''
    u = 0.001 * float(scale)
    kwargs.setdefault('points', {'post': [_MOUNT_POST_X * u, 0.0]})
    shapes = _mount_shapes(u, knobs,
                           front_w=45.7, front_d=7.0, gap=3.2,
                           back_w=45.7, back_d=12.7,
                           face_ahead=3.0, tip_span=16.45, tip_r=2.5,
                           stem_d=7.6, stem_w=6.4,
                           knob_d=8.6, knob_w=15.2)
    return Mechanics(shapes=shapes, **kwargs)

def mirror_mount_2in(scale=1.0, knobs=True, **kwargs):
    '''
    A two-inch kinematic mirror mount seen from above: the same
    schematic as mirror_mount, with its dimensions taken from the
    Thorlabs KA2A drawing (a three-adjuster 2" mount).

    Measured off that drawing: front plate 68.6 wide and 7.0 deep,
    the 3.2 adjustment gap (the 22.9 overall body minus the two
    plates), back plate 69.9 wide and 12.7 deep. The adjusters
    protrude 12.2 behind the back plate (35.1 overall); how that
    splits into stem and knob, and the knob width, are drawn in the
    one-inch mount's proportions, since the drawing does not dimension
    them.

    **The adjuster lines are 2.1 inch (53.34 mm) apart**, which is
    measured on the mount rather than read off the drawing - the
    figure the drawing gives puts the knobs closer together than they
    are. The knobs sit on those lines, so it is knob to knob.

    The local origin is the substrate centre of the mounted optic.
    The drawing's optic pocket is 10.3 deep, so a standard 12.7 thick
    two-inch optic seated against the stop centres 3.95 behind the
    front face - which is where ``attached_to`` with no offset puts
    the host's substrate centre. The post hole is drawn as the
    one-inch mount's is: 4 across, 13.5 behind the origin, at the
    point named 'post'.

    Parameters
    ----------
    scale : float, optional
        Multiplies every dimension.
    knobs : bool, optional
        Whether to draw the adjuster knobs and their stems.
    **kwargs
        Passed to Mechanics.

    Returns
    -------
    Mechanics
    '''
    u = 0.001 * float(scale)
    kwargs.setdefault('points', {'post': [_MOUNT_POST_X * u, 0.0]})
    shapes = _mount_shapes(u, knobs,
                           front_w=68.6, front_d=7.0, gap=3.2,
                           back_w=69.9, back_d=12.7,
                           face_ahead=3.95, tip_span=26.67, tip_r=2.5,
                           stem_d=5.7, stem_w=6.4,
                           knob_d=6.5, knob_w=12.7)
    return Mechanics(shapes=shapes, **kwargs)

def lens_holder(length=0.030, thickness=0.010, **kwargs):
    '''
    A lens holder seen from above: a plain rectangle, centred on the
    substrate centre of the optic it holds.

    A holder wraps its lens symmetrically, so the local origin is the
    middle of the rectangle - ``attached_to`` with no offset centres
    it on the host's substrate.

    Parameters
    ----------
    length : float, optional
        Across the beam, in metres. Defaults to 30 mm, which suits a
        one-inch optic.
    thickness : float, optional
        Along the beam. Defaults to 10 mm.
    **kwargs
        Passed to Mechanics.

    Returns
    -------
    Mechanics
    '''
    shapes = []
    shapes.append(draw.Rectangle([-thickness / 2.0, -length / 2.0],
                                 thickness, length))
    return Mechanics(shapes=shapes, **kwargs)

def pedestal(post_diameter=0.0254, base_diameter=0.0318,
             relief_diameter=0.0102, bore_diameter=0.0044, **kwargs):
    '''
    A pedestal pillar post seen from above: concentric circles.

    A pedestal is what a mount is bolted to and what a clamping fork
    holds down, so it is the middle of a stack rather than either end
    of one. Its named point is ``'axis'``, at the local origin: the
    tapped hole a mount screws into from above, the post a fork closes
    around, and the point the whole thing turns about.

    The dimensions default to Thorlabs' RS05P8E drawing - a 1 inch
    post on a 1.25 inch base, with the relief cut and the #8-32
    tapped hole - since that is the pedestal the design was drawn
    against. Measure yours and pass the numbers if it differs.

    Parameters
    ----------
    post_diameter : float, optional
        The pillar, in metres. Defaults to 25.4 mm.
    base_diameter : float, optional
        The flange a clamping fork bears on. Defaults to 31.8 mm.
    relief_diameter : float, optional
        The relief cut around the tapped hole. Defaults to 10.2 mm;
        zero leaves it out.
    bore_diameter : float, optional
        The tapped hole through the middle. Defaults to 4.4 mm, which
        is a #8-32 thread.
    **kwargs
        Passed to Mechanics.

    Returns
    -------
    Mechanics
    '''
    shapes = [draw.Circle([0.0, 0.0], base_diameter / 2.0),
              draw.Circle([0.0, 0.0], post_diameter / 2.0)]
    if relief_diameter:
        shapes.append(draw.Circle([0.0, 0.0], relief_diameter / 2.0))
    if bore_diameter:
        shapes.append(draw.Circle([0.0, 0.0], bore_diameter / 2.0))
    kwargs.setdefault('points', {'axis': [0.0, 0.0]})
    kwargs.setdefault('params', {'kind': 'pedestal',
                                 'post_diameter': post_diameter,
                                 'base_diameter': base_diameter,
                                 'relief_diameter': relief_diameter,
                                 'bore_diameter': bore_diameter})
    return Mechanics(shapes=shapes, **kwargs)

def clamping_fork(bore_diameter=0.0260, length=0.0738, width=0.0363,
                  tail_width=0.0234, tip_ahead=0.0038, slot_span=0.0315,
                  slot_width=0.0070, slot_near=0.0268, **kwargs):
    '''
    A clamping fork seen from above: the U that closes on a pedestal,
    the tapering shank, and the slot its screw goes through.

    The origin is the **bore centre** - the point that comes to sit on
    the pedestal it holds - and that is its named point, ``'bore'``.
    The shank runs from there along -x, so a fork attached with no
    turn of its own lies behind the post it clamps. ``'screw'`` is the
    middle of the slot, which is where the bolt into the bench
    roughly goes.

    The smooth waist of the real part is drawn as a straight taper and
    two arcs: this is a drawing of what the part occupies, not a
    reproduction of it. The dimensions default to Thorlabs' CF125
    drawing.

    Parameters
    ----------
    bore_diameter : float, optional
        The U that closes on the post. Defaults to 26.0 mm, which
        takes a 25 mm pedestal.
    length : float, optional
        Prong tips to tail, in metres. Defaults to 73.8 mm.
    width : float, optional
        Across the prongs. Defaults to 36.3 mm.
    tail_width : float, optional
        Across the rounded tail. Defaults to 23.4 mm.
    tip_ahead : float, optional
        From the bore centre to the prong tips, along the fork.
        Defaults to 3.8 mm. Less than the bore radius, or the U does
        not close on anything.
    slot_span : float, optional
        Between the two ends of the slot, centre to centre. Defaults
        to 31.5 mm.
    slot_width : float, optional
        Across the slot. Defaults to 7 mm, which passes a 1/4-20 cap
        screw.
    slot_near : float, optional
        From the bore centre to the near end of the slot. Defaults to
        26.8 mm.
    **kwargs
        Passed to Mechanics.

    Returns
    -------
    Mechanics
    '''
    r = bore_diameter / 2.0
    half = width / 2.0
    tail_half = tail_width / 2.0
    # The prong tips stand a little beyond the bore centre, so the U
    # wraps more than half of the post - which is what makes it a fork
    # rather than a hook - and they meet the bore where the circle has
    # got that far along.
    tip = tip_ahead
    edge = np.sqrt(max(r * r - tip * tip, 0.0))
    tail = tip - length
    waist = tail + tail_half

    shapes = []
    # The U, from one prong tip round the back to the other.
    a = np.arctan2(edge, tip)
    shapes.append(draw.Arc([0.0, 0.0], r, a, 2 * np.pi - a))
    # The outline, in two runs: the tail is an arc, and one polyline
    # through it would draw a chord straight across it.
    shapes.append(draw.PolyLine(x=[tip, tip, waist],
                                y=[edge, half, tail_half]))
    shapes.append(draw.PolyLine(x=[waist, tip, tip],
                                y=[-tail_half, -half, -edge]))
    # The rounded tail.
    shapes.append(draw.Arc([waist, 0.0], tail_half,
                           np.pi / 2.0, 3 * np.pi / 2.0))
    # The slot, as two caps and two sides.
    near, far = -slot_near, -slot_near - slot_span
    sr = slot_width / 2.0
    shapes.append(draw.Arc([near, 0.0], sr, -np.pi / 2.0, np.pi / 2.0))
    shapes.append(draw.Arc([far, 0.0], sr, np.pi / 2.0, 3 * np.pi / 2.0))
    shapes.append(draw.Line([near, sr], [far, sr]))
    shapes.append(draw.Line([near, -sr], [far, -sr]))

    kwargs.setdefault('points', {'bore': [0.0, 0.0],
                                 'screw': [(near + far) / 2.0, 0.0]})
    kwargs.setdefault('params', {'kind': 'clamping_fork',
                                 'bore_diameter': bore_diameter,
                                 'length': length, 'width': width,
                                 'tail_width': tail_width,
                                 'tip_ahead': tip_ahead,
                                 'slot_span': slot_span,
                                 'slot_width': slot_width,
                                 'slot_near': slot_near})
    return Mechanics(shapes=shapes, **kwargs)

#: Where the post hole of a kinematic mount sits, in millimetres
#: behind the optic it holds, along the face normal. A mount is bolted
#: to its pedestal from underneath, so the hole is in no top view at
#: all - what is drawn from a top view is where it is. Measured on the
#: bench: 13.5 mm behind the substrate centre, on the axis, in the
#: back plate. Pass ``points={'post': [x, y]}`` to a builder to say
#: otherwise.
_MOUNT_POST_X = -13.5

#: How wide that hole is drawn, in millimetres. A tapped hole seen
#: through the plate above it: the drawing says a mount is bolted
#: down there, which is part of what a top view of a bench is read
#: for.
_MOUNT_POST_D = 4.0

def _mount_shapes(u, knobs, front_w, front_d, gap, back_w, back_d,
                  face_ahead, tip_span, tip_r, stem_d, stem_w,
                  knob_d, knob_w):
    '''
    The shapes every kinematic mount is drawn from, with the
    dimensions in millimetres and ``u`` carrying them to metres.

    The frame is the attachment frame: the origin is the substrate
    centre of the mounted optic, +x the host's HR normal. The front
    face stands ``face_ahead`` of the origin; the plates stack
    backwards from it, the adjuster tips bulge out of the back plate
    into the gap on the two lines ``tip_span`` either side of the
    centre, and the knobs hang off their stems out of the back. The
    hole the mount is bolted down through is drawn last, over the
    plate it is bored in.
    '''
    front_hi = face_ahead * u
    front_lo = front_hi - front_d * u
    back_hi = front_lo - gap * u
    back_lo = back_hi - back_d * u
    ty = tip_span * u

    shapes = []
    shapes.append(draw.Rectangle([front_lo, -front_w * u / 2],
                                 front_hi - front_lo, front_w * u))
    shapes.append(draw.Rectangle([back_lo, -back_w * u / 2],
                                 back_hi - back_lo, back_w * u))
    for s in (-1.0, 1.0):
        # The adjuster tips: half circles on the +x side of their
        # centres.
        shapes.append(draw.Arc([back_hi, s * ty], tip_r * u,
                               -np.pi / 2, np.pi / 2))
    if knobs:
        stem_lo = back_lo - stem_d * u
        knob_lo = stem_lo - knob_d * u
        for s in (-1.0, 1.0):
            cy = s * ty
            shapes.append(draw.Rectangle([stem_lo, cy - stem_w * u / 2],
                                         stem_d * u, stem_w * u))
            shapes.append(draw.Rectangle([knob_lo, cy - knob_w * u / 2],
                                         knob_d * u, knob_w * u))
    # The post hole, at the point the mount names 'post'. Last, so it
    # is drawn over the back plate it goes through rather than under
    # it.
    shapes.append(draw.Circle([_MOUNT_POST_X * u, 0.0],
                              _MOUNT_POST_D * u / 2))
    return shapes

#}}}

#{{{ Model library

#: The model definitions: name -> {'shapes': [shape dicts], 'layer',
#: 'description'}. Data, not code - a definition is exactly what a
#: saved layout carries, so registering, saving and relinking all
#: speak the same format. Seeded below with a few generic parts;
#: vendor models are the user's to register from measured footprints,
#: rather than shipped here with dimensions gtrace would be guessing
#: at.
_MODEL_REGISTRY = {}

def register_model(name, source, description='', prefix=None):
    '''
    Put a model into the library, by value.

    This is the whole of "save it to the library": build a shape in a
    cell - by hand, or starting from breadboard() or mirror_mount() -
    look at it in the viewer, and register what you settled on. The
    shapes are copied out as data at this moment, so editing the
    original afterwards does not quietly edit the library.

    A name already registered is overwritten: that is how a definition
    is updated, and relink_mechanics() is how a layout then chooses to
    follow it.

    Parameters
    ----------
    name : str
        The model name, e.g. 'POLARIS-K1'.
    source : Mechanics or sequence of gtrace.draw.Shape
        Where the shapes come from. A Mechanics contributes its layer
        as well; its pose is ignored - a model is a shape, not a
        place.
    description : str, optional
        One line for models() to show.
    prefix : str or None, optional
        What to call the bodies built from this model, before the
        number: 'MT' gives MT1, MT2. A part is known by what it is - a
        mount is MT1 whatever catalogue it came from - so the model
        says it rather than every layout choosing again, and a front
        end that adds one has the name to hand. None leaves it to
        whoever is naming, which is 'H' in the viewer.

    Returns
    -------
    name : str
    '''
    if isinstance(source, Mechanics):
        shapes = source.shapes
        layer = source.layer
        params = None if source.params is None else dict(source.params)
        points = source.points
    else:
        shapes = list(source)
        layer = DEFAULT_LAYER
        params = None
        points = {}
    _MODEL_REGISTRY[str(name)] = {
        'shapes': [shape_to_dict(s) for s in shapes],
        'layer': str(layer),
        'description': str(description),
        # What bodies built from it are called. Part of the definition
        # rather than of the layout: the same part dropped into two
        # layouts should come out with the same kind of name, and a
        # model of one's own says what its parts are called the same
        # way the stock does.
        'prefix': None if prefix is None else str(prefix),
        # The named points of the part - the hole a pedestal screws
        # into, the bore a fork closes on. They are what one part is
        # stood on another by, so they belong to the model rather than
        # to the body that happens to be built from it.
        'points': dict((str(k), [float(v[0]), float(v[1])])
                       for k, v in points.items()),
        # Carried so that a body built from the model keeps whatever
        # the source knew about itself - a breadboard from the library
        # is still a breadboard, and still resizes.
        'params': params}
    return str(name)

def models():
    '''
    The library: model name -> description, sorted by name.
    '''
    return dict((k, v['description'])
                for k, v in sorted(_MODEL_REGISTRY.items()))

def model_shapes(name):
    '''
    Fresh shape objects for a registered model.

    Raises
    ------
    KeyError
        If the library has no such model.
    '''
    d = _MODEL_REGISTRY.get(str(name))
    if d is None:
        raise KeyError('No model named %r in the library. '
                       'mechanics.models() lists what there is.' % (name,))
    return [shape_from_dict(s) for s in d['shapes']]

def model_points(name):
    '''
    The named points of a registered model, in local coordinates.

    Raises
    ------
    KeyError
        If the library has no such model.
    '''
    d = _MODEL_REGISTRY.get(str(name))
    if d is None:
        raise KeyError('No model named %r in the library. '
                       'mechanics.models() lists what there is.' % (name,))
    return dict((k, np.array(v, dtype='float64'))
                for k, v in (d.get('points') or {}).items())

def model_prefix(name):
    '''
    What bodies built from a registered model are called, before the
    number - 'MT' for a mount, 'FK' for a fork - or None where the
    model does not say.

    Raises
    ------
    KeyError
        If the library has no such model.
    '''
    d = _MODEL_REGISTRY.get(str(name))
    if d is None:
        raise KeyError('No model named %r in the library. '
                       'mechanics.models() lists what there is.' % (name,))
    return d.get('prefix')

def model_params(name):
    '''
    The builder parameters of a registered model, or None for one
    registered from hand-drawn shapes. A copy; the registry is not
    handed out to be edited in place.

    Raises
    ------
    KeyError
        If the library has no such model.
    '''
    d = _MODEL_REGISTRY.get(str(name))
    if d is None:
        raise KeyError('No model named %r in the library. '
                       'mechanics.models() lists what there is.' % (name,))
    return None if d.get('params') is None else dict(d['params'])

def from_model(model, **kwargs):
    '''
    A Mechanics built from a library model, carrying the model name as
    its label.

    The label is a label: the layout saves the shapes by value, and a
    library that changes afterwards changes nothing until
    relink_mechanics() is asked to.

    Parameters
    ----------
    model : str
        The model name.
    **kwargs
        Passed to Mechanics: name, center, rotationAngle, layer,
        attached_to and the rest. The layer defaults to the model's
        own.

    Returns
    -------
    Mechanics
    '''
    d = _MODEL_REGISTRY.get(str(model))
    if d is None:
        raise KeyError('No model named %r in the library. '
                       'mechanics.models() lists what there is.' % (model,))
    kwargs.setdefault('layer', d['layer'])
    kwargs.setdefault('points', d.get('points') or {})
    return Mechanics(shapes=[shape_from_dict(s) for s in d['shapes']],
                     model=str(model), params=d.get('params'), **kwargs)

def save_models(filename, names=None):
    '''
    Write library models to a JSON file.

    The file carries exactly what the registry holds - the serialized
    shapes, the layer, the description and the builder parameters -
    under a single 'models' object, so a saved library is the same
    data a saved layout embeds and load_models() can take it back
    without a conversion in between.

    Parameters
    ----------
    filename : str
        Name of the file to write.
    names : sequence of str or None, optional
        Which models to save. None - the default - saves the whole
        library, the built-in stock included: they are values like any
        other, and loading them back changes nothing.

    Returns
    -------
    filename : str

    Raises
    ------
    KeyError
        If a name in ``names`` is not in the library.
    '''
    if names is None:
        picked = dict(_MODEL_REGISTRY)
    else:
        picked = {}
        for n in names:
            d = _MODEL_REGISTRY.get(str(n))
            if d is None:
                raise KeyError('No model named %r in the library. '
                               'mechanics.models() lists what there is.'
                               % (n,))
            picked[str(n)] = d
    payload = {'models': dict(
        (k, {'shapes': [dict(s) for s in v['shapes']],
             'layer': str(v['layer']),
             'description': str(v.get('description', '')),
             'prefix': (None if v.get('prefix') is None
                        else str(v['prefix'])),
             'points': dict((str(pk), [float(pv[0]), float(pv[1])])
                            for pk, pv in (v.get('points') or {}).items()),
             'params': (None if v.get('params') is None
                        else dict(v['params']))})
        for k, v in picked.items())}
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=1)
    return filename

def load_models(filename):
    '''
    Merge the models of a JSON file into the library.

    Name by name, the file wins - exactly the rule register_model
    already has, since loading a definition is one more way of
    registering it. Models the file does not mention are left alone,
    so building a library out of several files is just calling this
    once per file.

    Everything is checked before anything is merged: a file with one
    unreadable shape in it changes nothing, rather than leaving the
    library half updated.

    Parameters
    ----------
    filename : str
        A file written by save_models().

    Returns
    -------
    list of str
        The names that were merged, in the file's order.

    Raises
    ------
    ValueError
        If the file is not a model library, or an entry of it does
        not describe one.
    '''
    with open(filename, 'r') as f:
        data = json.load(f)
    models_in = data.get('models') if isinstance(data, dict) else None
    if not isinstance(models_in, dict):
        raise ValueError("%s is not a model library file gtrace can read "
                         "(no 'models' object in it)." % filename)

    staged = {}
    for name, d in models_in.items():
        if not isinstance(d, dict):
            raise ValueError('Model %r of %s is not a definition.'
                             % (name, filename))
        try:
            # Through the shape constructors and back: what comes out
            # is both validated and normalized, so a hand-edited file
            # either loads cleanly or refuses loudly here - not on the
            # first scene built from it.
            shapes = [shape_to_dict(shape_from_dict(s))
                      for s in d.get('shapes', [])]
        except UnknownShapeError as e:
            raise ValueError('Model %r of %s: %s' % (name, filename, e))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise ValueError('Model %r of %s has a malformed shape '
                             '(%s: %s).' % (name, filename,
                                            type(e).__name__, e))
        layer = d.get('layer', DEFAULT_LAYER)
        if not isinstance(layer, str) or not layer.strip():
            raise ValueError('Model %r of %s has no usable layer: %r.'
                             % (name, filename, layer))
        points = d.get('points') or {}
        if not isinstance(points, dict):
            raise ValueError('Model %r of %s has malformed points: %r.'
                             % (name, filename, points))
        try:
            points = dict((str(k), [float(v[0]), float(v[1])])
                          for k, v in points.items())
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError('Model %r of %s has a point that is not '
                             '[x, y].' % (name, filename))
        params = d.get('params')
        if params is not None and not isinstance(params, dict):
            raise ValueError('Model %r of %s has malformed parameters: %r.'
                             % (name, filename, params))
        prefix = d.get('prefix')
        if prefix is not None and (not isinstance(prefix, str)
                                   or not prefix.strip()):
            raise ValueError('Model %r of %s has no usable name prefix: '
                             '%r.' % (name, filename, prefix))
        staged[str(name)] = {
            'shapes': shapes,
            'layer': str(layer),
            'description': str(d.get('description', '')),
            # A library written before models said what their parts
            # are called has none, and none is what it meant.
            'prefix': None if prefix is None else str(prefix),
            'points': points,
            'params': None if params is None else dict(params)}

    _MODEL_REGISTRY.update(staged)
    return list(staged)

# The generic stock: a few breadboards and mounts under names that say
# what they are and no more. Registered through the same door a user's
# models come through, so the built-ins prove the door works.
# Each says what its parts are called: a mount is MT1 and a fork FK1,
# whatever catalogue the footprint came from. 'PD' is deliberately not
# among them - a photodetector is what that reads as - which is why a
# pedestal is P.
register_model('BB3030', breadboard(0.30, 0.30),
               '300 x 300 mm breadboard, 25 mm grid', prefix='BB')
register_model('BB4530', breadboard(0.45, 0.30),
               '450 x 300 mm breadboard, 25 mm grid', prefix='BB')
register_model('BB6045', breadboard(0.60, 0.45),
               '600 x 450 mm breadboard, 25 mm grid', prefix='BB')
register_model('BBR30', round_breadboard(0.30),
               '300 mm round breadboard, 25 mm grid', prefix='BB')
register_model('BBR45', round_breadboard(0.45),
               '450 mm round breadboard, 25 mm grid', prefix='BB')
register_model('MOUNT-25', mirror_mount(),
               'kinematic mount for a 1 inch optic', prefix='MT')
register_model('MOUNT-50', mirror_mount_2in(),
               'kinematic mount for a 2 inch optic (KA2A footprint)',
               prefix='MT')
register_model('HOLDER-25', lens_holder(length=0.030, thickness=0.010),
               'lens holder for a 1 inch optic, 30 x 10 mm', prefix='HLD')
register_model('PEDESTAL-25', pedestal(),
               '1 inch pedestal post, 31.8 mm base (RS05P8E footprint)',
               prefix='P')
register_model('FORK-125', clamping_fork(),
               'clamping fork for a 1 inch pedestal (CF125 footprint)',
               prefix='FK')
register_model('HOLDER-50', lens_holder(length=0.056, thickness=0.0127),
               'lens holder for a 2 inch optic, 56 x 12.7 mm', prefix='HLD')

#}}}
