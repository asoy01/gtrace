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
        The layer the body is drawn on. Defaults to 'hardware'.
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
                 name='Hardware', layer=DEFAULT_LAYER, model=None,
                 attached_to=None, offset=None, offset_angle=0.0,
                 params=None):
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
        host = self.attached_to
        a = float(host.normAngleHR)
        ca, sa = np.cos(a), np.sin(a)
        off = self.offset
        return (np.asarray(host.center, dtype='float64')
                + np.array([off[0] * ca - off[1] * sa,
                            off[0] * sa + off[1] * ca]))

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
        return float(self.attached_to.normAngleHR) + self.offset_angle

    @rotationAngle.setter
    def rotationAngle(self, value):
        if self.attached_to is not None or self._attach_name is not None:
            raise ValueError(
                "'%s' is attached: it turns as its host turns. Detach it "
                'first, or change the offset angle.' % self.name)
        self._rotationAngle = float(value)

    def attach(self, host, offset=None, offset_angle=None,
               keep_pose=False):
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
        host : gtrace.optcomp.Optics
            The optics to stand on. Only an optics: a chain of
            hardware standing on hardware is a graph to walk and a
            cycle to forbid, and nothing on a bench has needed it.
        offset : array-like or None, optional
            Where the local origin stands in the host's frame (its
            substrate centre, x along the HR normal). None - the
            default - is ``[0, 0]``: the designed position.
        offset_angle : float or None, optional
            The turn relative to the host. None is 0: squarely on it.
        keep_pose : bool, optional
            Derive the offset and the angle from where the body stands
            now instead, so that attaching changes what moves it and
            not where it is. For pinning something that was placed by
            eye; offset and offset_angle must be left None with it.

        Returns
        -------
        self : Mechanics
        '''
        if isinstance(host, Mechanics):
            raise ValueError(
                'A mechanics attaches to an optics, not to another '
                'mechanics: the attachment is one-way by design.')
        if not (hasattr(host, 'center') and hasattr(host, 'normAngleHR')):
            raise ValueError('%r has no pose to attach to.' % (host,))

        if keep_pose:
            if offset is not None or offset_angle is not None:
                raise ValueError(
                    'keep_pose derives the offset from where the body '
                    'stands; giving one as well is two answers to one '
                    'question.')
            # The current world pose, read before the switch. It works
            # whether the body is free or already attached to
            # something else.
            ha = float(host.normAngleHR)
            d = self.center - np.asarray(host.center, dtype='float64')
            ca, sa = np.cos(-ha), np.sin(-ha)
            offset = [d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca]
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

    def resize(self, width=None, height=None):
        '''
        Rebuild a parametric body at a new size.

        Only a body a builder made knows how to do this: a breadboard
        cut to a new size is re-drilled from its parameters, with the
        same pitch and the same holes, where scaling the shapes would
        scale the holes and the grid along with the plate. A body
        drawn by hand has no parameters, and is refused - its shapes
        are all anyone knows about it.

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
        Whether resize() knows how to rebuild this body.
        '''
        return bool(self.params and self.params.get('kind') in _RESIZABLE)

    def copy(self):
        '''
        A new Mechanics with the same pose and a copy of the shape
        list. The shapes themselves are shared: nothing in gtrace
        mutates a drawing primitive, and the copy is what save, undo
        and the clipboard-of-the-future need. An attached body copies
        as attached, standing on the same host.
        '''
        if self.attached_to is not None or self._attach_name is not None:
            m = Mechanics(shapes=list(self.shapes),
                          name=self.name, layer=self.layer, model=self.model,
                          attached_to=(self.attached_to
                                       if self.attached_to is not None
                                       else self._attach_name),
                          offset=self.offset.copy(),
                          offset_angle=self.offset_angle,
                          params=self.params)
            return m
        m = Mechanics(shapes=list(self.shapes),
                      center=self._center.copy(),
                      rotationAngle=self._rotationAngle,
                      name=self.name, layer=self.layer, model=self.model,
                      params=self.params)
        return m

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

#: The parametric kinds resize() can rebuild, and how. A mount is
#: deliberately not here: what its width means to the plate and the
#: knobs is not a corner-drag's to decide.
_RESIZABLE = {'breadboard': _breadboard_shapes}

def mirror_mount(scale=1.0, knobs=True, **kwargs):
    '''
    A one-inch kinematic mirror mount seen from above, drawn after a
    Polaris-style footprint: the front plate the optic sits in, the
    back plate across the adjustment gap with the two adjuster tips
    showing in it, and the two adjuster knobs on their stems out of
    the back.

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
    plates), back plate 69.9 wide and 12.7 deep, and the adjuster
    lines 35.6 apart. The adjusters protrude 12.2 behind the back
    plate (35.1 overall); how that splits into stem and knob, and the
    knob width, are drawn in the one-inch mount's proportions, since
    the drawing does not dimension them.

    The local origin is the substrate centre of the mounted optic.
    The drawing's optic pocket is 10.3 deep, so a standard 12.7 thick
    two-inch optic seated against the stop centres 3.95 behind the
    front face - which is where ``attached_to`` with no offset puts
    the host's substrate centre.

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
    shapes = _mount_shapes(u, knobs,
                           front_w=68.6, front_d=7.0, gap=3.2,
                           back_w=69.9, back_d=12.7,
                           face_ahead=3.95, tip_span=17.8, tip_r=2.5,
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
    centre, and the knobs hang off their stems out of the back.
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

def register_model(name, source, description=''):
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

    Returns
    -------
    name : str
    '''
    if isinstance(source, Mechanics):
        shapes = source.shapes
        layer = source.layer
        params = None if source.params is None else dict(source.params)
    else:
        shapes = list(source)
        layer = DEFAULT_LAYER
        params = None
    _MODEL_REGISTRY[str(name)] = {
        'shapes': [shape_to_dict(s) for s in shapes],
        'layer': str(layer),
        'description': str(description),
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
    return Mechanics(shapes=[shape_from_dict(s) for s in d['shapes']],
                     model=str(model), params=d.get('params'), **kwargs)

# The generic stock: a few breadboards and mounts under names that say
# what they are and no more. Registered through the same door a user's
# models come through, so the built-ins prove the door works.
register_model('BB3030', breadboard(0.30, 0.30),
               '300 x 300 mm breadboard, 25 mm grid')
register_model('BB4530', breadboard(0.45, 0.30),
               '450 x 300 mm breadboard, 25 mm grid')
register_model('BB6045', breadboard(0.60, 0.45),
               '600 x 450 mm breadboard, 25 mm grid')
register_model('MOUNT-25', mirror_mount(),
               'kinematic mount for a 1 inch optic')
register_model('MOUNT-50', mirror_mount_2in(),
               'kinematic mount for a 2 inch optic (KA2A footprint)')
register_model('HOLDER-25', lens_holder(length=0.030, thickness=0.010),
               'lens holder for a 1 inch optic, 30 x 10 mm')
register_model('HOLDER-50', lens_holder(length=0.056, thickness=0.0127),
               'lens holder for a 2 inch optic, 56 x 12.7 mm')

#}}}
