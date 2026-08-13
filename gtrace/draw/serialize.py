'''
Serialization of gtrace drawing canvases and beams into
JSON-compatible dictionaries.

This module provides an alternative "renderer" that, instead of
writing a DXF file, converts a Canvas (and optionally a list of
GaussianBeams) into plain Python dictionaries composed only of
JSON-serializable types (dict, list, str, float, int, bool, None).
The output is consumed by the HTML/JS viewer and other GUI front ends.

All coordinates are kept in the unit of the canvas (usually meters).
Complex numbers are represented as [real, imag] pairs.
'''

#{{{ Import modules

import copy

import numpy as np

import gtrace.draw as draw

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

#{{{ Helper functions

class UnknownShapeError(Exception):
    '''
    Raised when a shape cannot be serialized or rebuilt.

    Derived from Exception, not BaseException: the copy of this class
    in renderer.py was fixed the same way on 2026-08-04, after a
    BaseException sailed through every 'except Exception' between the
    renderer and the user - including the one that would have shown the
    failure in the GUI.
    '''
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def _vec(v):
    '''
    Convert a 2D vector (numpy array, list or tuple) to [float, float].
    '''
    a = np.asarray(v, dtype='float64')
    return [float(a[0]), float(a[1])]

def _floats(v):
    '''
    Convert a sequence of numbers to a list of floats.
    '''
    return [float(x) for x in np.asarray(v, dtype='float64')]

def _complex(c):
    '''
    Convert a complex number to a [real, imag] pair.
    '''
    c = complex(c)
    return [c.real, c.imag]

#}}}

#{{{ NEW_SHAPES

#: What each kind of shape looks like when it is first put down, in
#: metres about the origin, as the dict shape_from_dict builds from.
#:
#: A shape is put down in two places - into the part a shape editor is
#: drawing, and onto the bench itself, where the viewer makes a body of
#: one - so it is defined here, with the rest of what a shape dict is,
#: rather than in either of them. It is drawn about the origin because
#: that is the point a part is built around: the one that comes to sit
#: at a host's substrate centre.
#:
#: The sizes are what a bench would recognise rather than ones that
#: have to be found by zooming. A front end that works at some other
#: scale - a layout is kilometres across where a part is millimetres -
#: is expected to scale them to what it is showing.
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

def new_shapes():
    '''
    A copy of NEW_SHAPES, kind by kind.

    A copy, because what comes back is handed to a front end and to
    whoever asks: a caller that scaled the dict it was given would
    otherwise resize every shape put down after it.

    Returns
    -------
    dict
    '''
    return dict((k, copy.deepcopy(v)) for k, v in NEW_SHAPES.items())

#}}}

#{{{ shape_to_dict

def shape_to_dict(s):
    '''
    Convert a single draw.Shape into a JSON-compatible dict.

    The type of the shape is stored in the 'type' key.
    '''
    if isinstance(s, draw.Line):
        return {'type': 'line',
                'start': _vec(s.start),
                'stop': _vec(s.stop),
                'thickness': float(s.thickness)}
    elif isinstance(s, draw.PolyLine):
        return {'type': 'polyline',
                'x': _floats(s.x),
                'y': _floats(s.y),
                'thickness': float(s.thickness)}
    elif isinstance(s, draw.Rectangle):
        # The pivot is written as it is kept: null where the rectangle
        # turns about its own middle, so that a rectangle moved after
        # a load still turns about itself.
        return {'type': 'rectangle',
                'point': _vec(s.point),
                'width': float(s.width),
                'height': float(s.height),
                'angle': float(s.angle),
                'pivot': None if s.pivot is None else _vec(s.pivot),
                'thickness': float(s.thickness)}
    elif isinstance(s, draw.Circle):
        return {'type': 'circle',
                'center': _vec(s.center),
                'radius': float(s.radius),
                'thickness': float(s.thickness)}
    elif isinstance(s, draw.Arc):
        return {'type': 'arc',
                'center': _vec(s.center),
                'radius': float(s.radius),
                'startangle': float(s.startangle),
                'stopangle': float(s.stopangle),
                'thickness': float(s.thickness)}
    elif isinstance(s, draw.Text):
        return {'type': 'text',
                'text': str(s.text),
                'point': _vec(s.point),
                'height': float(s.height),
                'rotation': float(s.rotation)}
    else:
        raise UnknownShapeError('Shape not supported: %s' % type(s).__name__)

#}}}

#{{{ shape_from_dict

def shape_from_dict(d):
    '''
    Rebuild a draw.Shape from a dict produced by shape_to_dict().

    The inverse exists because a Mechanics carries its geometry as
    shapes: a saved layout writes them out with shape_to_dict, and
    loading it - or receiving them in an edit message - has to build
    the primitives back.

    Raises
    ------
    UnknownShapeError
        If the dict does not describe a shape gtrace can draw. A
        malformed dict (a missing key, a value of the wrong kind) comes
        back as whatever the constructor raised.
    '''
    if not isinstance(d, dict):
        raise UnknownShapeError('A shape must be a dict, not %s'
                                % type(d).__name__)
    kind = d.get('type')
    thickness = float(d.get('thickness', 0.0))
    if kind == 'line':
        return draw.Line(list(d['start']), list(d['stop']),
                         thickness=thickness)
    elif kind == 'polyline':
        return draw.PolyLine([float(v) for v in d['x']],
                             [float(v) for v in d['y']],
                             thickness=thickness)
    elif kind == 'rectangle':
        # A file written before a rectangle could be turned has
        # neither key, and squarely on the axes is what it meant.
        pivot = d.get('pivot')
        return draw.Rectangle(list(d['point']), float(d['width']),
                              float(d['height']), thickness=thickness,
                              angle=float(d.get('angle', 0.0)),
                              pivot=None if pivot is None else list(pivot))
    elif kind == 'circle':
        return draw.Circle(list(d['center']), float(d['radius']),
                           thickness=thickness)
    elif kind == 'arc':
        return draw.Arc(list(d['center']), float(d['radius']),
                        float(d['startangle']), float(d['stopangle']),
                        thickness=thickness)
    elif kind == 'text':
        return draw.Text(str(d['text']), list(d['point']),
                         height=float(d.get('height', 1.0)),
                         rotation=float(d.get('rotation', 0.0)))
    else:
        raise UnknownShapeError('Shape not supported: %r' % (kind,))

#}}}

#{{{ build_shape

def build_shape(d, error=None):
    '''
    Build a shape from a dict and refuse one that cannot be drawn.

    :func:`shape_from_dict` is the constructor's door: what describes
    no shape is refused by the constructor itself, so there is no
    second list of rules about what a circle is. What it cannot catch
    is a shape that builds and then cannot be drawn - a rectangle of
    no width, a coordinate at infinity - because those are numbers a
    constructor has no opinion about. They are caught here.

    Both places a shape is edited come through this: the shape editor
    drawing a part, and a layout editing the one shape of a body drawn
    on the bench. A rule kept in one of them would be a rule the other
    did not have.

    Parameters
    ----------
    d : dict
        A shape as :func:`shape_to_dict` writes it.
    error : type or None, optional
        What to raise instead of ValueError, for a caller whose
        protocol has an exception of its own.

    Returns
    -------
    A drawing primitive.

    Raises
    ------
    ValueError, or whatever ``error`` names
        If the dict describes no shape, or one that cannot be drawn.
    '''
    fail = error or ValueError
    try:
        shape = shape_from_dict(d)
    except UnknownShapeError:
        raise
    except draw.NumberOfElementError as e:
        raise fail('That does not describe a %s: %s' % (d.get('type'), e))
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise fail('That does not describe a %s (%s: %s).'
                   % (d.get('type'), type(e).__name__, e))

    out = shape_to_dict(shape)
    # A nan or an infinity would take the whole view with it the first
    # time anything was framed.
    for value in out.values():
        for v in (value if isinstance(value, list) else [value]):
            if isinstance(v, float) and not np.isfinite(v):
                raise fail('A %s cannot be drawn with %r in it.'
                           % (out['type'], v))
    # A rectangle of no width, or of less than none, is not a smaller
    # rectangle: it is a shape SVG refuses to draw and a bounding box
    # that comes out inside out.
    for key in ('width', 'height', 'radius'):
        if key in out and not out[key] > 0:
            raise fail('A %s needs a positive %s, not %r.'
                       % (out['type'], key, out[key]))
    # A polyline of one vertex draws nothing and has nothing to take
    # hold of; of none, not even a place. The constructor only asks
    # that x and y be of the same length, so this is the same kind of
    # arithmetic as a positive width.
    if out['type'] == 'polyline' and len(out['x']) < 2:
        raise fail('A polyline needs at least two vertices, not %d.'
                   % len(out['x']))
    if out.get('thickness', 0.0) < 0:
        raise fail('A %s cannot be drawn with a thickness of %r.'
                   % (out['type'], out['thickness']))
    return shape

#}}}

#{{{ canvas_to_dict

def canvas_to_dict(canvas):
    '''
    Convert a draw.Canvas into a JSON-compatible dict.

    The structure of the returned dict is::

        {'unit': 'm',
         'layers': [{'name': str,
                     'color': [r, g, b],
                     'shapes': [shape dict, ...]},
                    ...]}

    Coordinates are kept in the unit of the canvas (no scaling
    is applied, unlike renderDXF which converts to mm).
    '''
    layers = []
    for ly in canvas.layers.values():
        layers.append({'name': ly.name,
                       'color': [int(c) for c in ly.color],
                       'shapes': [shape_to_dict(s) for s in ly.shapes]})
    return {'unit': canvas.unit, 'layers': layers}

#}}}

#{{{ beam_to_dict

def beam_to_dict(b):
    '''
    Convert a GaussianBeam into a JSON-compatible dict of its
    physical parameters.

    This is the metadata channel that DXF export does not have.
    It allows a viewer to compute the beam parameters (q, width,
    ROC, Gouy phase, etc.) at an arbitrary point along the beam.

    The q-parameters are given at the origin of the beam ('pos').
    The parameters at a distance d along the beam are obtained
    by q -> q + d.
    '''
    pos = np.asarray(b.pos, dtype='float64')
    dirVect = np.asarray(b.dirVect, dtype='float64')
    return {'name': str(b.name),
            'layer': str(b.layer),
            'pos': _vec(pos),
            'end': _vec(pos + dirVect * float(b.length)),
            'dirVect': _vec(dirVect),
            'dirAngle': float(b.dirAngle),
            'length': float(b.length),
            'wl': float(b.wl),
            'n': float(b.n),
            'P': float(b.P),
            'qx': _complex(b.qx),
            'qy': _complex(b.qy),
            'wx': float(b.wx),
            'wy': float(b.wy),
            'Gouyx': float(b.Gouyx),
            'Gouyy': float(b.Gouyy),
            'optDist': float(b.optDist),
            'stray_order': int(b.stray_order)}

#}}}

#{{{ beams_to_dict

def beams_to_dict(beamList):
    '''
    Convert a list of GaussianBeams into a list of JSON-compatible
    dicts. See beam_to_dict for the format of each element.
    '''
    return [beam_to_dict(b) for b in beamList]

#}}}

#{{{ optic_to_dict

#: Attributes of an optics passed to the viewer, when present.
_OPTIC_SCALARS = ['diameter', 'thickness', 'wedgeAngle', 'inv_ROC_HR',
                  'inv_ROC_AR', 'n', 'Refl_HR', 'Trans_HR', 'Refl_AR',
                  'Trans_AR']
_OPTIC_POINTS = ['HRcenter', 'ARcenter', 'center']
_OPTIC_ANGLES = ['normAngleHR', 'normAngleAR']
_OPTIC_FLAGS = ['HRtransmissive', 'HRreflective', 'term_on_HR',
                'term_on_HR_transmits']
_OPTIC_INTS = ['term_on_HR_order']

def optic_to_dict(o):
    '''
    Convert an Optics into a JSON-compatible dict of the attributes a
    viewer needs in order to point at it: where it is, how big it is and
    which way it faces.

    Unlike the shapes on the canvas, this keeps the identity of the
    optics, which is what lets a GUI say "the user grabbed M1".
    Attributes that a particular class does not have are left out.
    '''
    d = {'name': str(o.name), 'type': type(o).__name__}
    for k in _OPTIC_POINTS:
        if hasattr(o, k):
            d[k] = _vec(getattr(o, k))
    for k in _OPTIC_ANGLES + _OPTIC_SCALARS:
        if hasattr(o, k):
            d[k] = float(getattr(o, k))
    for k in _OPTIC_FLAGS:
        if hasattr(o, k):
            d[k] = bool(getattr(o, k))
    for k in _OPTIC_INTS:
        if hasattr(o, k):
            d[k] = int(getattr(o, k))
    if hasattr(o, 'curve_direction'):
        d['curve_direction'] = str(o.curve_direction)
    if hasattr(o, 'anchor_point'):
        d['anchor_point'] = str(o.anchor_point)
    if hasattr(o, 'f'):
        # The power rather than the focal length itself. A substrate
        # with no power left in it focuses at infinity, and JSON has no
        # infinity to carry that with; the viewer inverts this back,
        # exactly as it already does with the curvatures.
        fl = float(o.f)
        d['inv_f'] = 0.0 if not np.isfinite(fl) else 1.0/fl
    if hasattr(o, 'max_stray_order'):
        d['max_stray_order'] = (None if o.max_stray_order is None
                                else int(o.max_stray_order))
    # What it follows, if anything, and the joint it follows by. A
    # front end needs it to know that the pose rows are the host's
    # doing rather than the element's - and that it is not to be
    # dragged.
    host = getattr(o, 'assembled_to', None)
    d['assembled_to'] = None if host is None else str(host.name)
    if host is not None:
        d['assembly_offset'] = _vec(o.assembly_offset)
        d['assembly_angle'] = float(o.assembly_angle)
        d['fix_rotation'] = bool(o.fix_rotation)
    return d

def optics_to_dict(opticsList):
    '''
    Convert a list of Optics into a list of JSON-compatible dicts.
    See optic_to_dict for the format of each element.
    '''
    return [optic_to_dict(o) for o in opticsList]

#}}}

#{{{ scene_to_dict

def scene_to_dict(canvas, beamList=None, opticsList=None, display=None):
    '''
    Convert a canvas, an optional list of beams and an optional list of
    optics into a single JSON-compatible dict:

    {'canvas': canvas dict,
     'beams': [beam dict, ...],
     'optics': [optic dict, ...],
     'display': {...}}

    This is the top-level data structure consumed by the HTML/JS
    viewer. The 'optics' entry is what an editing front end addresses
    when the user drags an element. 'display' tells the front end how
    the canvas was drawn - the envelope width and which transverse
    direction it shows - so that its controls can show the choice in
    force rather than guessing.

    OpticalLayout.scene_dict adds channels of its own on top of these -
    the dimensions noted on the layout, the points a measurement may
    snap to, and whether undo and redo have anything to work with. They
    are added there rather than here because they belong to the layout
    rather than to a drawing: a canvas and a list of beams, which is all
    this function is given, know nothing of them.
    '''
    return {'canvas': canvas_to_dict(canvas),
            'beams': beams_to_dict(beamList) if beamList is not None else [],
            'optics': (optics_to_dict(opticsList)
                       if opticsList is not None else []),
            'display': dict(display) if display else {}}

#}}}
