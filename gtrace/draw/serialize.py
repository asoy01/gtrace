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
__version__ = "0.3.1"
__maintainer__ = "Yoichi Aso"
__email__ = "asoy01@gmail.com"
__status__ = "Beta"

#}}}

#{{{ Helper functions

class UnknownShapeError(BaseException):
    def __init__(self, message):
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
        return {'type': 'rectangle',
                'point': _vec(s.point),
                'width': float(s.width),
                'height': float(s.height),
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
_OPTIC_FLAGS = ['HRtransmissive', 'term_on_HR']
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
