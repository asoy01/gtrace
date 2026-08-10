'''
gtrace.layout

A module providing the OpticalLayout class: a container representing
a whole optical system (optics, source beams and tracing rules).

An OpticalLayout is the model object that GUI front ends interact
with. The intended workflow is:

1. Construct and align the optics with ordinary Python code
   (using sequential tracing, cavity eigenmode solving, etc.).
2. Register the finished optics, the source beam(s) and the tracing
   rules into an OpticalLayout.
3. Use layout.trace() to run the non-sequential trace and
   layout.draw() / layout.scene_dict() to visualize the result.

The layout holds the registered optics and sources BY REFERENCE
(no copies). Changing an attribute of a registered mirror (e.g.
M.HRcenter) and calling layout.trace() again gives the updated
result. This is the basis of the future bidirectional GUI: an edit
made in the GUI is applied to the corresponding object in the
layout, which is the same object the user has in their code.
'''

#{{{ Import modules

import json
import os
import sys
import tempfile
import webbrowser

import numpy as np

import gtrace.optcomp as optcomp
from gtrace.beam import GaussianBeam
from gtrace.mechanics import Mechanics
from gtrace.nonsequential import non_seq_trace
from gtrace.draw.tools import drawAllOptics
from gtrace.draw.serialize import (scene_to_dict, shape_to_dict,
                                   shape_from_dict, UnknownShapeError)
from gtrace.draw.viewer import renderHTML
import gtrace.draw as draw
import gtrace.draw.renderer as renderer

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

#{{{ TraceRules

class TraceRules(object):
    '''
    Rules controlling the non-sequential trace of an OpticalLayout.

    Attributes
    ----------
    order : int
        Number of internal reflections computed when a beam hits an
        optics. Defaults to 10.
    power_threshold : float
        Beams with power below this threshold are not propagated
        further. Defaults to 0.1.
    open_beam_length : float
        Drawn length of beams that do not hit anything.
        Defaults to 1.0.

    Note that the order can be overridden for an individual element
    through its max_stray_order attribute. That setting lives on the
    optics rather than here, next to term_on_HR, because how deep an
    element's ghosts are worth chasing is a property of the element.
    '''

    def __init__(self, order=10, power_threshold=0.1, open_beam_length=1.0):
        self.order = order
        self.power_threshold = power_threshold
        self.open_beam_length = open_beam_length

    def to_dict(self):
        '''
        Convert to a JSON-compatible dict.
        '''
        return {'order': int(self.order),
                'power_threshold': float(self.power_threshold),
                'open_beam_length': float(self.open_beam_length)}

    @classmethod
    def from_dict(cls, d):
        '''
        Construct a TraceRules from a dict produced by to_dict().
        '''
        return cls(order=d.get('order', 10),
                   power_threshold=d.get('power_threshold', 0.1),
                   open_beam_length=d.get('open_beam_length', 1.0))

#}}}

#{{{ Dimension

class Dimension(object):
    '''
    A distance measured between two points of a layout.

    A dimension is a note about the system rather than a part of it: it
    takes part in no trace and changes no beam. It is registered in the
    layout all the same, because a measurement worth taking is worth
    keeping - it is saved with the layout, comes back with it, and can
    be taken back with undo, like anything else the layout holds.

    The two points are plain coordinates. A dimension does not hold on
    to the element a point was taken from: an element that then moves
    leaves the measurement where it was made, which is what a note
    should do. Measuring the same thing again after a change is a matter
    of drawing it again.

    Attributes
    ----------
    name : str
        Name of the dimension, unique within the layout.
    p1, p2 : numpy.ndarray
        The two ends, of shape (2,), in global coordinates.
    offset : float
        How far to one side of the two ends the dimension line is
        drawn, in metres, positive to the left of the direction from
        p1 to p2. Zero puts the line straight between them.

        This is where a drawing puts the line, not what is being
        measured: what a bench wants measured usually runs along a beam
        or through an element, which is exactly where a line drawn on
        top of it cannot be read. Extension lines carry the ends out to
        wherever there is room, as they do on any engineering drawing.
    '''

    def __init__(self, p1, p2, name='D', offset=0.0):
        self.name = name
        self.p1 = np.array(p1, dtype='float64')
        self.p2 = np.array(p2, dtype='float64')
        self.offset = float(offset)

    @property
    def length(self):
        '''
        The distance between the two ends.

        The offset does not come into it: the dimension line is moved
        aside to be read, not to measure something else.
        '''
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def normal(self):
        '''
        The unit vector the offset is measured along: to the left of the
        direction from p1 to p2. Zero for a dimension of no length.
        '''
        seg = self.p2 - self.p1
        L = np.linalg.norm(seg)
        if L == 0.0:
            return np.zeros(2)
        return np.array([-seg[1], seg[0]], dtype='float64') / L

    def line_ends(self):
        '''
        The two ends of the dimension line itself - the ends carried out
        by the offset. Equal to p1 and p2 when the offset is zero.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
        '''
        n = self.normal * self.offset
        return self.p1 + n, self.p2 + n

    def measure(self, optics=()):
        '''
        What this dimension comes to, against a list of optics.

        The optical distance is reported only when the whole span runs
        inside one substrate, where it is the physical distance times
        the refractive index. A span that crosses in and out of glass
        has an optical length too, but it is not a dimension of anything
        - it depends on where the ends happen to fall - so it is left
        out rather than written next to a number it would be mistaken
        for.

        Parameters
        ----------
        optics : sequence, optional
            The optics to test the span against.

        Returns
        -------
        dict
            ``{'length': float, 'optical': float or None,
            'inside': str or None, 'n': float or None}``
        '''
        d = {'length': self.length, 'optical': None,
             'inside': None, 'n': None}
        if d['length'] == 0.0:
            return d
        for o in optics:
            if not hasattr(o, 'contains_segment'):
                continue
            if o.contains_segment(self.p1, self.p2):
                d['inside'] = str(o.name)
                d['n'] = float(o.n)
                d['optical'] = d['n'] * d['length']
                break
        return d

    def copy(self):
        '''
        A new Dimension with the same ends, offset and name.
        '''
        return Dimension(self.p1, self.p2, name=self.name,
                         offset=self.offset)

#}}}

#{{{ Edit protocol

#: Attributes of an optics that a front end is allowed to change.
#: Edit messages arrive from a browser, so the set is explicit rather
#: than "anything setattr accepts". The same list is used by the
#: notebook widget and, later, by the live server.
EDITABLE_OPTIC_ATTRS = frozenset([
    'HRcenter', 'ARcenter', 'center',
    'normAngleHR', 'normVectHR',
    'diameter', 'thickness', 'wedgeAngle',
    'inv_ROC_HR', 'inv_ROC_AR', 'n',
    'Refl_HR', 'Trans_HR', 'Refl_AR', 'Trans_AR',
    'HRtransmissive', 'HRreflective', 'term_on_HR', 'term_on_HR_order',
    'max_stray_order', 'curve_direction', 'anchor_point',
    # Only a Lens has a focal length, and assigning to it re-solves both
    # curvatures. The check that the target is a lens is in
    # _set_optic_attr: a whitelist can say which names are allowed, not
    # which classes carry them.
    'f',
])

#: Values an attribute is restricted to. A whitelist of names keeps a
#: front end from reaching attributes it should not; this keeps it from
#: putting nonsense into the ones it may.
ATTR_CHOICES = {'curve_direction': ('h', 'v'),
                'anchor_point': ('HRcenter', 'center'),
                'width_mode': ('x', 'y', 'avg')}

#: How a layout is drawn, and what each option defaults to. These are
#: display choices, not part of the model: changing one redraws but does
#: not re-trace.
DRAW_OPTIONS = {
    #: Font size of the annotations, or False for the gtrace default.
    'fontSize': False,
    #: Whether to draw the width envelope of the main / stray beams.
    'drawMainWidth': True,
    'drawStrayWidth': True,
    #: Width of the drawn envelope, in units of the 1/e^2 radius.
    #: 2.7 is the aperture at which the diffraction loss is 1 ppm.
    'sigma_main': 2.7,
    'sigma_stray': 2.7,
    #: Which transverse direction the envelope shows: 'x', 'y', or
    #: 'avg' for the mean of the two. A beam is not round in general,
    #: so this is a choice the drawing cannot make for the user.
    'width_mode': 'x',
    #: Whether to annotate each beam with its name and power. Off by
    #: default: the labels of neighbouring beams overlap badly, and the
    #: viewer reports the same values for whichever beam is clicked.
    'drawBeamLabels': False,
    #: Whether to annotate each optics with its name. On, since optics
    #: carry no other label.
    'drawOpticsNames': True,
}

#: Drawing options a front end may change.
EDITABLE_DRAW_OPTIONS = frozenset([
    'sigma_main', 'sigma_stray', 'width_mode',
    'drawMainWidth', 'drawStrayWidth', 'drawBeamLabels', 'drawOpticsNames',
])

#: Attributes of the tracing rules that a front end may change.
EDITABLE_RULE_ATTRS = frozenset([
    'order', 'power_threshold', 'open_beam_length',
])

#: Constructor parameters a new optics may be given. These are the
#: arguments of Mirror.__init__ that make sense to set from a GUI.
CREATABLE_OPTIC_PARAMS = frozenset([
    'HRcenter', 'normAngleHR', 'normVectHR',
    'diameter', 'thickness', 'wedgeAngle',
    'inv_ROC_HR', 'inv_ROC_AR', 'n',
    'Refl_HR', 'Trans_HR', 'Refl_AR', 'Trans_AR',
    'HRtransmissive', 'HRreflective', 'term_on_HR', 'max_stray_order',
    'curve_direction',
])

#: Parameters a new optics copies from the optics already in the layout,
#: so that a mirror added to a system of 10 cm optics is a 10 cm optics
#: rather than whatever the class default happens to be.
_INHERITED_PARAMS = ['diameter', 'thickness', 'wedgeAngle', 'n',
                     'Refl_HR', 'Trans_HR', 'Refl_AR', 'Trans_AR']

#: Parameters a new Lens may be given on top of CREATABLE_OPTIC_PARAMS.
#: A lens is ordered by its focal length, which nothing else has.
CREATABLE_LENS_PARAMS = frozenset(['f', 'shape', 'ROC_HR'])

#: Focal length of a lens added without one being asked for. A front end
#: offering an "add a lens" button has to put something in the layout,
#: and 500 mm is an unremarkable lens off a catalogue shelf.
DEFAULT_LENS_F = 0.5

#: Types that can be created from an edit message.
CREATABLE_OPTIC_TYPES = {'Mirror': 'Mirror', 'CyMirror': 'CyMirror',
                         'Lens': 'Lens', 'CyLens': 'CyLens'}

#: The lens types among them: the ones ordered by focal length and
#: built by _lens_from_params rather than from the mirror around them.
_LENS_TYPES = frozenset(['Lens', 'CyLens'])

#: Attributes of a source beam a front end may change.
#:
#: The waist is not among them in the model's own terms - a GaussianBeam
#: holds q-parameters - but it is what a laser is ordered and specified
#: by, so it is offered here under the four derived names below and
#: converted in _set_source_attr. The conversion is on this side because
#: what a waist means is the model's to say: beam.waist() is already the
#: statement of it, and a second one in a browser is the sort of double
#: description that put the AR surface a sagitta out of place.
EDITABLE_SOURCE_ATTRS = frozenset([
    'pos', 'dirAngle', 'dirVect', 'length', 'wl', 'P', 'n',
    'qx', 'qy',
    'waist_size_x', 'waist_size_y', 'waist_pos_x', 'waist_pos_y',
])

#: The derived ones among them, and which axis and part of q each names.
#: A waist size is the 1/e^2 radius there, and a waist position is the
#: distance from the origin of the beam forward to the waist - positive
#: downstream, exactly as GaussianBeam.waist() reports it.
_SOURCE_WAIST_ATTRS = {'waist_size_x': ('x', 'size'),
                       'waist_size_y': ('y', 'size'),
                       'waist_pos_x': ('x', 'pos'),
                       'waist_pos_y': ('y', 'pos')}

#: The order in which a batch of attributes is applied to a source.
#: Applying them is not commutative: the index of refraction scales the
#: distance to the waist, so a message that sets both it and a waist
#: position has to set the index first, or the distance that arrives is
#: measured in the old medium. A raw q-parameter comes after the light
#: it is to be read against, and before the waist rows, which are the
#: more specific way of saying the same thing. Anything not named here
#: sorts last, by name.
_SOURCE_ATTR_ORDER = ['wl', 'n', 'P',
                      'qx', 'qy',
                      'waist_size_x', 'waist_size_y',
                      'waist_pos_x', 'waist_pos_y',
                      'dirAngle', 'dirVect', 'pos', 'length']

_SOURCE_ATTR_RANK = dict((k, i) for i, k in enumerate(_SOURCE_ATTR_ORDER))

#: Constructor parameters a new source may be given. 'waist_size' and
#: 'waist_pos' are the pair a laser is described by, and stand for both
#: axes at once: a source added from a button is round until it is told
#: otherwise.
CREATABLE_SOURCE_PARAMS = frozenset([
    'pos', 'dirAngle', 'dirVect', 'length', 'wl', 'P', 'n',
    'waist_size', 'waist_pos',
])

#: The type an 'add' message names to create a source beam. Kept out of
#: CREATABLE_OPTIC_TYPES for the same reason a dimension is: a source is
#: not an optics. Nothing is traced through it - it is where the trace
#: starts - and it is registered in a list of its own.
SOURCE_TYPE = 'Source'

#: What a source added without a beam being described looks like: a
#: 1064 nm beam of 1 W whose waist is 0.2 mm and lies at the laser
#: itself. A front end offering an "add a source" button has to put
#: something in the layout, and this is an unremarkable bench laser.
DEFAULT_SOURCE_WL = 1064e-9
DEFAULT_SOURCE_WAIST = 0.2e-3

#: Attributes of a dimension a front end may change. A dimension is two
#: points, where its line is drawn, and a name; the name has its own
#: operation, so this is the whole of it.
EDITABLE_DIMENSION_ATTRS = frozenset(['p1', 'p2', 'offset'])

#: The ones that are points rather than numbers.
_DIMENSION_POINTS = frozenset(['p1', 'p2'])

#: How a dimension is drawn into a DXF, as fractions of its own length.
#: A drawing has no fixed scale, so a tick and a label sized in
#: millimetres would be invisible across a bench and enormous across a
#: substrate; sized against the measurement they are legible at both.
_DXF_TICK = 0.02
_DXF_TEXT = 0.08

#: Formats to write on a dimension line in an exported drawing. The
#: viewer picks an SI prefix to suit; a drawing wants one unit
#: throughout, and millimetres is what an optical bench drawing uses.
def _fmt_mm(metres):
    return '%.3f mm' % (metres * 1000.0)

#: The type an 'add' message names to create a dimension. It is kept out
#: of CREATABLE_OPTIC_TYPES because a dimension is not an optics: it is
#: not traced, it is not drawn among the elements, and it takes ends
#: rather than construction parameters.
DIMENSION_TYPE = 'Dimension'

#: The type an 'add' message names to create a Mechanics. Kept out of
#: CREATABLE_OPTIC_TYPES for the same reason a dimension is: a
#: Mechanics is not an optics - the trace never sees it - and it takes
#: shapes rather than construction parameters.
MECHANICS_TYPE = 'Mechanics'

#: Attributes of a Mechanics a front end may change: its pose - or,
#: attached, its attachment - and nothing else. The shapes are the body
#: itself: they come from Python or from a saved layout, and a front
#: end moves the body rather than redrawing it. 'attached_to' takes an
#: optics name, or null to detach, which bakes the derived pose in and
#: leaves the body standing where it was.
EDITABLE_MECHANICS_ATTRS = frozenset(['center', 'rotationAngle',
                                      'attached_to', 'offset',
                                      'offset_angle'])

#: The pose half of those: what an attached body does not have.
_MECHANICS_POSE_ATTRS = frozenset(['center', 'rotationAngle'])

#: Parameters a new Mechanics may be given. 'shapes' is a list of
#: serialized primitives, exactly as a saved layout carries them.
CREATABLE_MECHANICS_PARAMS = frozenset(['center', 'rotationAngle',
                                        'shapes', 'layer', 'model',
                                        'attached_to', 'offset',
                                        'offset_angle'])

#: How many edits back undo can go. A snapshot is the serialized
#: layout, so the cost is a few tens of kilobytes each for a system of
#: any size, and a bound is what keeps a long session from growing
#: without limit. It bounds the redo side as well, which is only ever
#: filled by undoing and so cannot outgrow it.
UNDO_DEPTH = 50

#: Operations that leave the layout as it was, and so are not worth a
#: snapshot: two write a file, and the others walk the history rather
#: than adding to it. Being on this list also means the operation does
#: not discard the redo stack - writing a file in the middle of stepping
#: back and forth is not a change of mind.
_NOT_UNDOABLE = frozenset(['save', 'export', 'undo', 'redo'])

#: Formats 'export' can write.
EXPORT_FORMATS = frozenset(['dxf'])

class EditError(ValueError):
    '''
    Raised when an edit message cannot be applied: unknown operation,
    unknown target or an attribute that is not editable.
    '''
    pass

#: The order in which a batch of attributes is applied to an optics.
#: Applying them is not commutative: the anchor decides what a curvature
#: then does to the substrate, and the position handlers work from the
#: orientation, so a position set before an orientation lands off the
#: old one. A message is a JSON object and a saved layout is a JSON
#: file, and the key order of neither is something to rest on.
_ATTR_ORDER = ['anchor_point',
               'diameter', 'thickness', 'wedgeAngle', 'n',
               'Refl_HR', 'Trans_HR', 'Refl_AR', 'Trans_AR',
               'inv_ROC_HR', 'inv_ROC_AR', 'f',
               'HRtransmissive', 'HRreflective', 'term_on_HR',
               'term_on_HR_order', 'max_stray_order', 'curve_direction',
               'normAngleHR', 'normVectHR',
               'HRcenter', 'ARcenter', 'center']

_ATTR_RANK = dict((k, i) for i, k in enumerate(_ATTR_ORDER))

def _ordered(attrs):
    '''
    The items of an attribute dict, in the order they must be applied.

    Anything not in _ATTR_ORDER sorts last, by name, so that an
    attribute added to the whitelist without a thought about ordering
    still lands somewhere reproducible.
    '''
    return sorted(attrs.items(),
                  key=lambda kv: (_ATTR_RANK.get(kv[0], len(_ATTR_ORDER)),
                                  kv[0]))

def _check_choice(key, value):
    '''
    Reject a value outside the set an attribute allows.
    '''
    choices = ATTR_CHOICES.get(key)
    if choices is not None and value not in choices:
        raise EditError('%r must be one of %s, not %r.'
                        % (key, ', '.join(repr(c) for c in choices), value))

def _as_point(value, key):
    '''
    A pair of finite coordinates from an edit message, or an EditError.

    Points arrive from a browser as JSON arrays, where a missing number
    is null and a runaway one is a string; neither is a place on a
    bench, and both would otherwise settle quietly into a numpy array as
    nan.
    '''
    try:
        p = np.array([float(value[0]), float(value[1])], dtype='float64')
    except (TypeError, ValueError, IndexError, KeyError):
        raise EditError('%r must be a pair of coordinates, not %r.'
                        % (key, value))
    if not np.all(np.isfinite(p)):
        raise EditError('%r must be a pair of finite coordinates, not %r.'
                        % (key, value))
    return p

def _as_distance(value, key):
    '''
    A finite distance in metres from an edit message, or an EditError.
    '''
    try:
        d = float(value)
    except (TypeError, ValueError):
        raise EditError('%r must be a distance in metres, not %r.'
                        % (key, value))
    if not np.isfinite(d):
        raise EditError('%r must be a finite distance, not %r.' % (key, value))
    return d

#: The largest trace depth a front end may ask for. Each order is
#: another round of internal reflections at every element, so the work
#: grows quickly; a typed digit too many should come back as a refusal
#: rather than as a kernel that stops answering.
MAX_RULE_ORDER = 200

def _as_rule_value(key, value):
    '''
    A tracing rule value from an edit message, or an EditError.
    '''
    if key == 'order':
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise EditError("'order' must be a whole number of internal "
                            'reflections, not %r.' % (value,))
        if isinstance(value, float) and n != value:
            raise EditError("'order' must be a whole number of internal "
                            'reflections, not %r.' % (value,))
        if n < 0:
            raise EditError("'order' must not be negative, and %r is."
                            % (value,))
        if n > MAX_RULE_ORDER:
            raise EditError("'order' is limited to %d; each order is another "
                            'round of reflections at every element. Got %r.'
                            % (MAX_RULE_ORDER, value))
        return n
    if key == 'power_threshold':
        p = _as_distance(value, key)
        if p < 0:
            raise EditError("'power_threshold' must not be negative, and "
                            '%r is.' % (value,))
        return p
    return _as_positive(value, key, 'a length in metres')

def _set_optic_attr(optics, key, value):
    '''
    Set one whitelisted attribute of an optics on behalf of a front end.

    Two things the whitelist cannot express happen here. An attribute
    may exist on only some classes, and a model that accepts the name
    may still refuse the value - assigning a focal length re-solves both
    curvatures, and there are focal lengths a given blank cannot have.
    Both come back as EditError, so that a front end sees one kind of
    refusal however the model spelt it, and in both cases the optics is
    left exactly as it was.
    '''
    if key == 'f' and not isinstance(optics, optcomp.Lens):
        raise EditError("Only a lens has a focal length, and '%s' is a %s. "
                        'Set inv_ROC_HR or inv_ROC_AR instead.'
                        % (optics.name, type(optics).__name__))
    try:
        setattr(optics, key, value)
    except EditError:
        raise
    except ValueError as e:
        raise EditError("Cannot set %s on '%s' - %s: %s"
                        % (key, optics.name, type(e).__name__, e))

#}}}

#{{{ Front end selection

def _in_notebook():
    '''
    Whether we are running inside a Jupyter kernel.

    Looks the module up in sys.modules rather than importing it, so that
    gtrace does not pull IPython in when it is not already there.
    '''
    ipython = sys.modules.get('IPython')
    if ipython is None:
        return False
    shell = ipython.get_ipython()
    return shell is not None and type(shell).__name__ == 'ZMQInteractiveShell'

#}}}

#{{{ Serialization helpers for optics and sources

def optic_to_dict(m):
    '''
    Convert a Mirror, CyMirror, Lens or CyLens to a JSON-compatible
    dict of its construction parameters.

    A Lens is written out by its curvatures, like any other optics, and
    not by its focal length: the focal length is derived from them, and
    a lens whose radii were edited after it was built is that lens, not
    the one originally ordered.
    '''
    d = {'type': type(m).__name__,
         'name': str(m.name),
         'HRcenter': [float(x) for x in np.asarray(m.HRcenter)],
         'normAngleHR': float(m.normAngleHR),
         'diameter': float(m.diameter),
         'thickness': float(m.thickness),
         'wedgeAngle': float(m.wedgeAngle),
         'inv_ROC_HR': float(m.inv_ROC_HR),
         'inv_ROC_AR': float(m.inv_ROC_AR),
         'Refl_HR': float(m.Refl_HR),
         'Trans_HR': float(m.Trans_HR),
         'Refl_AR': float(m.Refl_AR),
         'Trans_AR': float(m.Trans_AR),
         'n': float(m.n),
         'HRtransmissive': bool(m.HRtransmissive),
         'HRreflective': bool(m.HRreflective),
         'term_on_HR': bool(m.term_on_HR),
         'term_on_HR_order': int(m.term_on_HR_order),
         'max_stray_order': (None if m.max_stray_order is None
                             else int(m.max_stray_order))}
    if isinstance(m, optcomp.CyMirror):
        d['curve_direction'] = str(m.curve_direction)
    if hasattr(m, 'anchor_point'):
        d['anchor_point'] = str(m.anchor_point)
    return d

def optic_from_dict(d):
    '''
    Construct a Mirror, CyMirror, Lens or CyLens from a dict produced
    by optic_to_dict().
    '''
    kwargs = {'HRcenter': d['HRcenter'],
              'normAngleHR': d['normAngleHR'],
              'diameter': d['diameter'],
              'thickness': d['thickness'],
              'wedgeAngle': d['wedgeAngle'],
              'inv_ROC_HR': d['inv_ROC_HR'],
              'inv_ROC_AR': d['inv_ROC_AR'],
              'Refl_HR': d['Refl_HR'],
              'Trans_HR': d['Trans_HR'],
              'Refl_AR': d['Refl_AR'],
              'Trans_AR': d['Trans_AR'],
              'n': d['n'],
              'name': d['name'],
              'HRtransmissive': d.get('HRtransmissive', False),
              #Absent from an older file, the class default stands:
              #True for a mirror, False for a lens.
              'HRreflective': d.get('HRreflective',
                                    d['type'] not in ('Lens', 'CyLens')),
              'term_on_HR': d.get('term_on_HR', False),
              'max_stray_order': d.get('max_stray_order', None)}
    if d['type'] == 'CyMirror':
        m = optcomp.CyMirror(curve_direction=d.get('curve_direction', 'h'),
                             **kwargs)
    elif d['type'] == 'Lens':
        #f=None: build from the curvatures written out above rather
        #than re-solving, which would reshape a lens whose radii had
        #been edited since.
        m = optcomp.Lens(f=None, **kwargs)
    elif d['type'] == 'CyLens':
        #From the curvatures, like a Lens, and with the direction,
        #like a CyMirror.
        m = optcomp.CyLens(f=None,
                           curve_direction=d.get('curve_direction', 'h'),
                           **kwargs)
    elif d['type'] == 'Mirror':
        m = optcomp.Mirror(**kwargs)
    else:
        raise ValueError('Unknown optics type: %s' % d['type'])
    m.term_on_HR_order = d.get('term_on_HR_order', 0)
    #Absent from an older file, or from one written before the anchor
    #existed, the class default stands: 'HRcenter' for a mirror,
    #'center' for a lens.
    if 'anchor_point' in d:
        m.anchor_point = d['anchor_point']
    return m

def source_to_dict(b):
    '''
    Convert a source GaussianBeam to a JSON-compatible dict of its
    construction parameters.
    '''
    return {'name': str(b.name),
            'layer': str(b.layer),
            'pos': [float(x) for x in np.asarray(b.pos)],
            'dirAngle': float(b.dirAngle),
            'length': float(b.length),
            'wl': float(b.wl),
            'P': float(b.P),
            'n': float(b.n),
            'qx': [complex(b.qx).real, complex(b.qx).imag],
            'qy': [complex(b.qy).real, complex(b.qy).imag]}

def dimension_to_dict(d):
    '''
    Convert a Dimension to a JSON-compatible dict.

    Only the ends are written. What the dimension comes to - the
    distance, and whether it runs inside a substrate - is derived from
    them and from the optics around it, and is recomputed on the way
    into a scene rather than stored, so that it cannot go stale.
    '''
    return {'type': 'Dimension',
            'name': str(d.name),
            'p1': [float(x) for x in np.asarray(d.p1)],
            'p2': [float(x) for x in np.asarray(d.p2)],
            'offset': float(d.offset)}

def dimension_from_dict(d):
    '''
    Construct a Dimension from a dict produced by dimension_to_dict().
    '''
    #Absent from a file written before the dimension line could be
    #carried aside, which is a line drawn straight between the ends.
    return Dimension(d['p1'], d['p2'], name=d['name'],
                     offset=d.get('offset', 0.0))

def _update_dimension(dim, d):
    '''
    Apply a serialized dimension to an existing one, in place.
    '''
    for key in ['p1', 'p2']:
        if key in d:
            setattr(dim, key, np.array(d[key], dtype='float64'))
    if 'offset' in d:
        dim.offset = float(d['offset'])

def mechanics_to_dict(m):
    '''
    Convert a Mechanics to a JSON-compatible dict.

    The shapes are written out by value, not by model name: a saved
    layout is complete in itself, the way a written HTML page is, and a
    library that has moved on since cannot silently redraw it. The
    model name travels alongside as a label.

    An attached body writes its host's name and its offset, and no
    pose: the pose is derived, and a stored copy of a derived value is
    exactly the second description this design exists to avoid. The
    name is read from the host object at save time, so a host renamed
    since the attachment saves under its current name.
    '''
    d = {'type': 'Mechanics',
         'name': str(m.name),
         'layer': str(m.layer),
         'model': None if m.model is None else str(m.model),
         'shapes': [shape_to_dict(s) for s in m.shapes]}
    host = m.attached_to.name if m.attached_to is not None else m._attach_name
    if host is not None:
        d['attached_to'] = str(host)
        d['offset'] = [float(x) for x in np.asarray(m.offset)]
        d['offset_angle'] = float(m.offset_angle)
    else:
        d['center'] = [float(x) for x in np.asarray(m.center)]
        d['rotationAngle'] = float(m.rotationAngle)
    return d

def mechanics_from_dict(d):
    '''
    Construct a Mechanics from a dict produced by mechanics_to_dict().

    An attached body comes back attached by name: the host is another
    entry of the same file, so only the layout that is loading both can
    join them up - which OpticalLayout does when the body is registered
    or merged. Until then the pose refuses to be read.
    '''
    if d.get('attached_to') is not None:
        return Mechanics(shapes=[shape_from_dict(s)
                                 for s in d.get('shapes', [])],
                         name=d['name'],
                         layer=d.get('layer', 'hardware'),
                         model=d.get('model', None),
                         attached_to=str(d['attached_to']),
                         offset=d.get('offset', [0.0, 0.0]),
                         offset_angle=d.get('offset_angle', 0.0))
    return Mechanics(shapes=[shape_from_dict(s)
                             for s in d.get('shapes', [])],
                     center=d.get('center', [0.0, 0.0]),
                     rotationAngle=d.get('rotationAngle', 0.0),
                     name=d['name'],
                     layer=d.get('layer', 'hardware'),
                     model=d.get('model', None))

def _update_mechanics(m, d):
    '''
    Apply a serialized Mechanics to an existing one, in place.

    The attachment is settled first, because it decides what the pose
    keys then mean: a body the file says is attached takes an offset
    and no pose, and one the file says is free takes a pose. The host
    name is left pending for the layout to resolve against its own
    optics - see OpticalLayout._link_mechanics.
    '''
    if d.get('attached_to') is not None:
        m.attached_to = None
        m._attach_name = str(d['attached_to'])
        m.offset = np.array(d.get('offset', [0.0, 0.0]), dtype='float64')
        m.offset_angle = float(d.get('offset_angle', 0.0))
    else:
        m.attached_to = None
        m._attach_name = None
        if 'center' in d:
            m.center = np.array(d['center'], dtype='float64')
        if 'rotationAngle' in d:
            m.rotationAngle = float(d['rotationAngle'])
    if 'layer' in d:
        m.layer = str(d['layer'])
    if 'model' in d:
        m.model = None if d['model'] is None else str(d['model'])
    if 'shapes' in d:
        m.shapes = [shape_from_dict(s) for s in d['shapes']]

def mechanics_scene_dict(m):
    '''
    Convert a Mechanics into the dict a viewer addresses it by: its
    pose, and the polygon it is picked with.

    The shapes are not here - they are drawn through the canvas, on the
    body's own layer, like the substrates of the optics. What a front
    end needs on top of the drawing is the identity ("the user grabbed
    the breadboard") and the outline, which is what a click is tested
    against and what a drag previews.
    '''
    outline = m.outline()
    center = m.center
    return {'name': str(m.name),
            'type': 'Mechanics',
            'center': [float(center[0]), float(center[1])],
            'rotationAngle': float(m.rotationAngle),
            'layer': str(m.layer),
            'model': None if m.model is None else str(m.model),
            # The name of the host, or null. What a front end needs to
            # know is that the pose is not this body's own: the panel
            # says whose it is, and the drag lets the host be the thing
            # that is dragged.
            'attached_to': (None if m.attached_to is None
                            else str(m.attached_to.name)),
            'outline': [[float(p[0]), float(p[1])] for p in outline]}

def mechanics_snap_points(m):
    '''
    The points of a Mechanics worth snapping a measurement to: the four
    corners of its outline, and its center. The distance from a mirror
    to the edge of the breadboard it stands on is exactly the kind of
    thing the measuring tool is for.
    '''
    points = []
    for i, c in enumerate(m.outline()):
        points.append({'point': [float(c[0]), float(c[1])],
                       'optic': str(m.name), 'kind': 'corner',
                       'label': '%s corner %d' % (m.name, i + 1)})
    points.append({'point': [float(m.center[0]), float(m.center[1])],
                   'optic': str(m.name), 'kind': 'centre',
                   'label': '%s centre' % m.name})
    return points

#: What a snap point on an optics is called, and where it is. The corner
#: entries are filled in from get_corners(), which is where the wedge and
#: the sagitta of a curved face are accounted for.
_SNAP_FACE_POINTS = [('HRcenter', 'HR'), ('ARcenter', 'AR'),
                     ('center', 'centre')]

def optic_snap_points(o):
    '''
    The points of an optics worth snapping a measurement to: the four
    corners of the substrate, the apex of each face, and the middle.

    These come from Python rather than being worked out by a front end
    because they are geometry: a corner is where the wedge and the
    sagitta of a curved face put it, and there is no reason for a second
    description of that to exist in a browser. Beam ends are a different
    matter - they are already carried literally in the scene - so they
    are not here.

    Returns
    -------
    list of dict
        ``{'point': [x, y], 'optic': name, 'kind': str, 'label': str}``
    '''
    points = []
    corners = o.get_corners() if hasattr(o, 'get_corners') else []
    for i, c in enumerate(corners):
        points.append({'point': [float(c[0]), float(c[1])],
                       'optic': str(o.name), 'kind': 'corner',
                       'label': '%s corner %d' % (o.name, i + 1)})
    for attr, what in _SNAP_FACE_POINTS:
        if not hasattr(o, attr):
            continue
        p = np.asarray(getattr(o, attr), dtype='float64')
        points.append({'point': [float(p[0]), float(p[1])],
                       'optic': str(o.name),
                       'kind': 'centre' if attr == 'center' else 'face',
                       'label': '%s %s' % (o.name, what)})
    return points

def _merge_by_name(registered, specs, build, update):
    '''
    Rebuild a list of registered objects from their serialized form,
    reusing an object whenever the name and the class still match.

    This is what lets a load keep the identity of the elements that
    survived it, which matters because the whole design rests on the
    layout holding the user's own objects by reference.
    '''
    by_name = {}
    for obj in registered:
        by_name[(obj.name, type(obj).__name__)] = obj

    result = []
    for spec in specs:
        key = (spec.get('name'), spec.get('type', type(None).__name__))
        existing = by_name.get(key)
        if existing is None and 'type' not in spec:
            # Sources carry no type; match on the name alone.
            existing = next((o for o in registered
                             if o.name == spec.get('name')), None)
        if existing is not None:
            update(existing, spec)
            result.append(existing)
        else:
            result.append(build(spec))
    return result

def _update_optic(m, d):
    '''
    Apply a serialized optics to an existing one, in place.
    '''
    # _ATTR_ORDER puts the anchor before the curvatures it governs and
    # the orientation before the position that is measured from it. The
    # keys a serialized optics carries that are not attributes to set -
    # its type and its name - are not in that list, and so are skipped.
    for key in _ATTR_ORDER:
        if key in d and hasattr(m, key):
            setattr(m, key, d[key])

def _update_source(b, d):
    '''
    Apply a serialized source beam to an existing one, in place.
    '''
    for key in ['pos', 'dirAngle', 'length', 'wl', 'P', 'n', 'layer']:
        if key in d:
            setattr(b, key, d[key])
    if 'qx' in d:
        b.qx = complex(d['qx'][0], d['qx'][1])
    if 'qy' in d:
        b.qy = complex(d['qy'][0], d['qy'][1])

def source_from_dict(d):
    '''
    Construct a source GaussianBeam from a dict produced by
    source_to_dict().
    '''
    b = GaussianBeam(pos=d['pos'], dirAngle=d['dirAngle'],
                     length=d.get('length', 1.0), wl=d['wl'],
                     P=d.get('P', 1.0), n=d.get('n', 1.0),
                     name=d['name'], layer=d.get('layer', 'main_beam'))
    b.qx = complex(d['qx'][0], d['qx'][1])
    b.qy = complex(d['qy'][0], d['qy'][1])
    return b

def rayleigh_range(w0, wl, n=1.0):
    '''
    The Rayleigh range of a beam of waist size w0.

    Parameters
    ----------
    w0 : float
        Waist size: the radius at which the power falls to 1/e^2 there,
        which is the convention GaussianBeam.width uses throughout.
    wl : float
        Vacuum wavelength.
    n : float, optional
        Index of refraction of the medium. It is folded into the
        q-parameter the same way GaussianBeam does it, so that the two
        agree. Defaults 1.0.

    Returns
    -------
    float
    '''
    return np.pi * n * w0 * w0 / wl

def q_from_waist(w0, d, wl, n=1.0):
    '''
    The q-parameter, at the origin of a beam, of light whose waist is of
    size w0 and lies a distance d further along.

    This is the inverse of what GaussianBeam.waist() reports, and exists
    so that a front end can offer the pair a laser is actually specified
    by without having to hold a second description of what a waist is.
    The distance runs the same way waist() reports it: positive means
    the waist is downstream of the origin.

    Returns
    -------
    complex
    '''
    return complex(-d, rayleigh_range(w0, wl, n))

def source_waist(b):
    '''
    The waist of a source beam, as the pair a front end shows: sizes and
    distances, each in x and y.

    Returns
    -------
    dict
        ``{'waist_size': (wx, wy), 'waist_pos': (dx, dy)}``
    '''
    w = b.waist()
    return {'waist_size': tuple(float(x) for x in w['Waist Size']),
            'waist_pos': tuple(float(x) for x in w['Waist Position'])}

def source_scene_dict(b):
    '''
    Convert a source beam into the dict a viewer draws and addresses it
    by: where the laser stands, which way it points, and what light it
    puts out.

    This is not beam_to_dict. That describes a beam of the trace - a
    segment with two ends, whose q is a result - and every beam in the
    scene is one of those, including the copy of this source that the
    trace began from. What is missing there is the source itself: which
    of the beams the user put in the layout, and which are consequences.
    A front end that cannot tell them apart cannot offer to edit the one
    and not the others, and cannot draw the laser at all.

    The waist travels alongside the q-parameters rather than instead of
    them. The q is what the model holds and what a saved layout carries;
    the waist is what the panel shows, and working it out here keeps the
    formula on the side that owns the beam.
    '''
    pos = np.asarray(b.pos, dtype='float64')
    dirVect = np.asarray(b.dirVect, dtype='float64')
    w = source_waist(b)
    # The width where the light leaves, asked of the beam rather than
    # worked out again from q. A front end drawing something at that
    # point has to know how wide the beam already is there: the aperture
    # a laser is drawn with cannot be narrower than the beam coming out
    # of it, however far the view is zoomed in.
    width = b.width(0.0)
    return {'name': str(b.name),
            'layer': str(b.layer),
            'pos': [float(pos[0]), float(pos[1])],
            'dirVect': [float(dirVect[0]), float(dirVect[1])],
            'dirAngle': float(b.dirAngle),
            'length': float(b.length),
            'wl': float(b.wl),
            'P': float(b.P),
            'n': float(b.n),
            'qx': [complex(b.qx).real, complex(b.qx).imag],
            'qy': [complex(b.qy).real, complex(b.qy).imag],
            'width': [float(width[0]), float(width[1])],
            'waist_size': [w['waist_size'][0], w['waist_size'][1]],
            'waist_pos': [w['waist_pos'][0], w['waist_pos'][1]]}

def _ordered_source(attrs):
    '''
    The items of a source attribute dict, in the order they must be
    applied. See _SOURCE_ATTR_ORDER for why the order matters.
    '''
    return sorted(attrs.items(),
                  key=lambda kv: (_SOURCE_ATTR_RANK.get(kv[0],
                                                        len(_SOURCE_ATTR_ORDER)),
                                  kv[0]))

def _as_positive(value, key, what='a positive number'):
    '''
    A finite positive number from an edit message, or an EditError.
    '''
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise EditError('%r must be %s, not %r.' % (key, what, value))
    if not np.isfinite(v) or v <= 0:
        raise EditError('%r must be %s, not %r.' % (key, what, value))
    return v

def _as_complex(value, key):
    '''
    A q-parameter from an edit message, or an EditError.

    JSON has no complex numbers, so one arrives as a pair. A q whose
    imaginary part is not positive is not a beam - the Rayleigh range is
    what gives it a width - so it is refused here rather than left to
    surface as a nan the first time the width is asked for.
    '''
    try:
        if isinstance(value, complex):
            q = value
        elif isinstance(value, (list, tuple)):
            q = complex(float(value[0]), float(value[1]))
        else:
            q = complex(value)
    except (TypeError, ValueError, IndexError, KeyError):
        raise EditError('%r must be a q-parameter as [real, imag], '
                        'not %r.' % (key, value))
    if not (np.isfinite(q.real) and np.isfinite(q.imag)):
        raise EditError('%r must be finite, not %r.' % (key, value))
    if q.imag <= 0:
        raise EditError('%r must have a positive imaginary part - that is '
                        'the Rayleigh range, and a beam without one has no '
                        'width. Got %r.' % (key, value))
    return q

def _set_source_attr(b, key, value):
    '''
    Set one whitelisted attribute of a source beam on behalf of a front
    end.

    The four waist names are not attributes of the beam at all: each
    stands for one half of one q-parameter, and is converted here. The
    other half is left as it was, so that setting a waist size does not
    also move the waist, and moving it does not resize it.
    '''
    if key in _SOURCE_WAIST_ATTRS:
        axis, part = _SOURCE_WAIST_ATTRS[key]
        qname = 'qx' if axis == 'x' else 'qy'
        q = complex(getattr(b, qname))
        if part == 'size':
            w0 = _as_positive(value, key, 'a waist size in metres')
            q = complex(q.real, rayleigh_range(w0, b.wl, b.n))
        else:
            d = _as_distance(value, key)
            q = complex(-d, q.imag)
        setattr(b, qname, q)
        return

    if key in ('qx', 'qy'):
        setattr(b, key, _as_complex(value, key))
        return
    if key == 'pos':
        b.pos = _as_point(value, key)
        return
    if key == 'dirVect':
        v = _as_point(value, key)
        if not np.any(v):
            raise EditError("'dirVect' must point somewhere; [0, 0] does not.")
        b.dirVect = v
        return
    if key == 'dirAngle':
        b.dirAngle = _as_distance(value, key)
        return
    if key in ('wl', 'n', 'length'):
        # A wavelength or an index of zero would divide the width
        # formula by nothing, and a beam of no length is not drawn.
        what = {'wl': 'a wavelength in metres',
                'n': 'an index of refraction',
                'length': 'a length in metres'}[key]
        v = _as_positive(value, key, what)
        if key == 'wl':
            # A q-parameter says nothing on its own: what width it comes
            # to depends on the wavelength. So changing the wavelength
            # has to keep one of the two, and the waist is the one to
            # keep - it is what the laser is specified by, and the
            # divergence is what follows from the light. The model
            # already does exactly this for the index of refraction,
            # whose handler holds the reduced q fixed and so preserves
            # the waist size; without this the two would disagree.
            #
            # Assigning b.wl directly in Python is untouched, and keeps
            # the q-parameters instead. This is the edit protocol, which
            # deals in the waist throughout.
            w = source_waist(b)
            b.wl = v
            b.qx = q_from_waist(w['waist_size'][0], w['waist_pos'][0], v, b.n)
            b.qy = q_from_waist(w['waist_size'][1], w['waist_pos'][1], v, b.n)
        else:
            setattr(b, key, v)
        if key in ('wl', 'n'):
            # Neither leaves the cached widths right: there is no
            # handler for the wavelength at all, and the one for the
            # index rescales the q-parameters with the notification
            # turned off. Ask the beam itself rather than repeating the
            # formula here.
            b.wx, b.wy = b.width(0.0)
        return
    if key == 'P':
        # Power alone may be zero: a beam is still a beam when it is off,
        # and the trace keeps its geometry.
        p = _as_distance(value, key)
        if p < 0:
            raise EditError("'P' must not be negative, and %r is." % (value,))
        b.P = p
        return
    # Nothing else reaches here: the whitelist is checked first.
    raise EditError('%r is not an editable attribute of a source.' % (key,))

#}}}

#{{{ OpticalLayout

class OpticalLayout(object):
    '''
    A container representing a whole optical system: optics,
    source beams and tracing rules.

    Attributes
    ----------
    name : str
        Name of the layout.
    optics : list of gtrace.optcomp.Optics
        Registered optics (held by reference). All optics must have
        unique names.
    sources : list of gtrace.beam.GaussianBeam
        Registered source beams (held by reference). All sources must
        have unique names.
    rules : TraceRules
        Rules for the non-sequential trace.
    beams : list of gtrace.beam.GaussianBeam or None
        The result of the last trace(). None if trace() has not been
        run yet.
    beams_by_source : dict or None
        Mapping from a source name to the list of beams originating
        from that source, from the last trace().
    '''

    def __init__(self, optics=None, sources=None, rules=None, name='Layout',
                 dimensions=None, mechanics=None):
        self.name = name
        self.optics = []
        self.sources = []
        #: Registered Dimensions - measurements noted on the layout.
        #: They take no part in the trace; see the Dimension class.
        self.dimensions = []
        #: Registered Mechanics - bodies on the bench the trace never
        #: sees. Drawn, saved, edited and undone like everything else.
        self.mechanics = []
        self.rules = rules if rules is not None else TraceRules()
        #: Overrides for DRAW_OPTIONS, as chosen by a front end. Display
        #: settings, so changing one redraws but does not re-trace.
        self.draw_options = {}
        self.beams = None
        self.beams_by_source = None
        #: Serialized states from before each edit, oldest first. The
        #: history is kept here rather than in a front end so that undo
        #: means the same thing however the edit arrived - through a
        #: browser, or from a cell calling apply_edit directly.
        self._history = []
        #: States undone but not yet given up on, newest last. Filled
        #: only by undo() and emptied by the next edit that goes
        #: through: once the layout takes a different turn, the branch
        #: that was stepped out of is no longer somewhere to return to.
        self._future = []

        if optics is not None:
            for m in optics:
                self.add_optics(m)
        if sources is not None:
            for b in sources:
                self.add_source(b)
        if dimensions is not None:
            for d in dimensions:
                self.add_dimension(d)
        if mechanics is not None:
            for m in mechanics:
                self.add_mechanics(m)

#{{{ Registration

    def add_optics(self, m):
        '''
        Register an optics. The optics is held by reference.
        Its name must be unique within the layout.
        '''
        self._check_name_free(m.name)
        self.optics.append(m)

    def add_source(self, b):
        '''
        Register a source beam. The beam is held by reference.
        Its name must be unique within the layout.
        '''
        self._check_name_free(b.name)
        self.sources.append(b)

    def add_dimension(self, d):
        '''
        Register a Dimension. It is held by reference, and its name must
        be unique within the layout.
        '''
        self._check_name_free(d.name)
        self.dimensions.append(d)

    def add_mechanics(self, m):
        '''
        Register a Mechanics. It is held by reference, and its name must
        be unique within the layout.

        A body attached by name (built as ``Mechanics(attached_to='M1')``
        or loaded from a file) is joined to the registered optics of
        that name here; a name nothing answers to is refused, since a
        body with an unreadable pose is not something to hold.
        '''
        self._check_name_free(m.name)
        if m._attach_name is not None:
            try:
                m.attach(self.get_optics(m._attach_name),
                         offset=m.offset, offset_angle=m.offset_angle)
            except KeyError:
                raise ValueError(
                    "'%s' is attached to '%s', and no optics of that name "
                    'is registered.' % (m.name, m._attach_name))
        self.mechanics.append(m)

    def _link_mechanics(self):
        '''
        Resolve every attachment left pending by name against the
        registered optics.

        Loading and restoring build the bodies before the joins can be
        made - the host is another entry of the same file or snapshot -
        so the names are kept until everything is in place and then
        resolved here, in one pass, after the optics list is settled.
        '''
        for m in self.mechanics:
            if m._attach_name is not None:
                try:
                    m.attach(self.get_optics(m._attach_name),
                             offset=m.offset, offset_angle=m.offset_angle)
                except KeyError:
                    raise ValueError(
                        "'%s' is attached to '%s', and no optics of that "
                        'name is in the layout.' % (m.name, m._attach_name))

    def _check_name_free(self, name):
        '''
        Refuse a name already taken by an optics, a source, a dimension
        or a mechanics.

        The four share a namespace because a front end points at all of
        them the same way - an edit message names its target and nothing
        else - and a name that meant one thing in one message and
        another in the next would be a trap. Sources joined that
        namespace when they became editable; before that they were only
        ever addressed as a list. Mechanics joined it on arrival.
        '''
        if name in [o.name for o in self.optics]:
            raise ValueError("An optics named '%s' is already registered."
                             % name)
        if name in [s.name for s in self.sources]:
            raise ValueError("A source named '%s' is already registered."
                             % name)
        if name in [d.name for d in self.dimensions]:
            raise ValueError("A dimension named '%s' is already registered."
                             % name)
        if name in [m.name for m in self.mechanics]:
            raise ValueError("A mechanics named '%s' is already registered."
                             % name)

    def remove_optics(self, name):
        '''
        Remove the optics with the given name from the layout.

        An optics with hardware attached is refused: the mounts would
        be left standing on something no longer there, with a pose
        derived from a ghost. Detaching them - which leaves each one
        exactly where it stands - or removing them first says what is
        actually meant.
        '''
        target = self.get_optics(name)
        attached = [m.name for m in self.mechanics
                    if m.attached_to is target]
        if attached:
            raise ValueError(
                "Cannot remove '%s': %s attached to it. Detach or remove "
                '%s first.'
                % (name, ' and '.join("'%s'" % n for n in attached),
                   'it' if len(attached) == 1 else 'them'))
        self.optics.remove(target)

    def remove_source(self, name):
        '''
        Remove the source with the given name from the layout.
        '''
        self.sources.remove(self.get_source(name))

    def remove_dimension(self, name):
        '''
        Remove the dimension with the given name from the layout.
        '''
        self.dimensions.remove(self.get_dimension(name))

    def remove_mechanics(self, name):
        '''
        Remove the mechanics with the given name from the layout.
        '''
        self.mechanics.remove(self.get_mechanics(name))

    def get_optics(self, name):
        '''
        Return the registered optics with the given name.
        '''
        for m in self.optics:
            if m.name == name:
                return m
        raise KeyError("No optics named '%s' in the layout." % name)

    def get_source(self, name):
        '''
        Return the registered source with the given name.
        '''
        for b in self.sources:
            if b.name == name:
                return b
        raise KeyError("No source named '%s' in the layout." % name)

    def get_dimension(self, name):
        '''
        Return the registered dimension with the given name.
        '''
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError("No dimension named '%s' in the layout." % name)

    def get_mechanics(self, name):
        '''
        Return the registered mechanics with the given name.
        '''
        for m in self.mechanics:
            if m.name == name:
                return m
        raise KeyError("No mechanics named '%s' in the layout." % name)

    def unique_optics_name(self, prefix='M'):
        '''
        Return a name of the form prefix + number that nothing in the
        layout uses. Front ends need a name before they can talk about
        the element they are asking for.
        '''
        taken = set(o.name for o in self.optics)
        taken.update(s.name for s in self.sources)
        taken.update(d.name for d in self.dimensions)
        taken.update(m.name for m in self.mechanics)
        i = 1
        while '%s%d' % (prefix, i) in taken:
            i += 1
        return '%s%d' % (prefix, i)

    def unique_dimension_name(self, prefix='D'):
        '''
        Return a name of the form prefix + number that nothing in the
        layout uses. The same namespace as unique_optics_name, since
        edit messages resolve a target across both.
        '''
        return self.unique_optics_name(prefix)

    def unique_source_name(self, prefix='S'):
        '''
        Return a name of the form prefix + number that nothing in the
        layout uses. The same namespace again.
        '''
        return self.unique_optics_name(prefix)

    def unique_mechanics_name(self, prefix='H'):
        '''
        Return a name of the form prefix + number that nothing in the
        layout uses. The same namespace again; H for hardware, since M
        already means a mirror.
        '''
        return self.unique_optics_name(prefix)

#}}}

#{{{ trace

    def trace(self):
        '''
        Run the non-sequential trace from all the registered sources
        through the registered optics, according to the rules.

        The registered source beams are not modified: a copy of each
        source is used for tracing.

        The result is stored in self.beams (a flat list) and
        self.beams_by_source (a dict keyed by source name), and
        self.beams is also returned.
        '''
        self.beams = []
        self.beams_by_source = {}
        for src in self.sources:
            beams = non_seq_trace(self.optics, src.copy(),
                                  order=self.rules.order,
                                  power_threshold=self.rules.power_threshold,
                                  open_beam_length=self.rules.open_beam_length)
            self.beams_by_source[src.name] = beams
            self.beams.extend(beams)
        return self.beams

#}}}

#{{{ apply_edit

    def apply_edit(self, msg):
        '''
        Apply an edit message coming from a front end.

        The message is a plain dict, so the same protocol travels over
        the notebook widget's comm and, later, over a websocket::

            {'op': 'move',   'target': 'M1', 'HRcenter': [0.52, 0.0]}
            {'op': 'rotate', 'target': 'M1', 'normAngleHR': 2.3}
            {'op': 'set',    'target': 'M1', 'attrs': {'diameter': 0.15}}
            {'op': 'draw',   'params': {'sigma_main': 1.0,
                                        'width_mode': 'y'}}
            {'op': 'save',   'path': 'layout.json'}
            {'op': 'load',   'path': 'layout.json'}
            {'op': 'export', 'format': 'dxf', 'path': 'layout.dxf'}
            {'op': 'rename', 'target': 'M1', 'name': 'PRM'}
            {'op': 'add',    'type': 'Mirror', 'name': 'M4',
                             'params': {'HRcenter': [0.3, 0.2]}}
            {'op': 'add',    'type': 'Lens', 'name': 'L1',
                             'params': {'f': 0.3, 'shape': 'plano-convex'}}
            {'op': 'add',    'type': 'Source', 'name': 'S1',
                             'params': {'pos': [0.0, 0.0],
                                        'waist_size': 0.0002}}
            {'op': 'move',   'target': 'S1', 'pos': [0.1, 0.0]}
            {'op': 'rotate', 'target': 'S1', 'dirAngle': 0.0}
            {'op': 'set',    'target': 'S1',
                             'attrs': {'waist_size_x': 0.0003,
                                       'waist_pos_x': 0.05}}
            {'op': 'remove', 'target': 'M4'}
            {'op': 'align',  'target': 'L1', 'beam': 'b0',
                             'beam_index': 0, 'point': [0.4, 0.02]}
            {'op': 'slide',  'target': 'L1', 'beam': 'b0',
                             'beam_index': 0, 'distance': 0.05}
            {'op': 'rules',  'rules': {'power_threshold': 1e-6}}
            {'op': 'add',    'type': 'Dimension', 'name': 'D1',
                             'params': {'p1': [0.0, 0.0],
                                        'p2': [0.5, 0.0],
                                        'offset': 0.05}}
            {'op': 'set',    'target': 'D1', 'attrs': {'p2': [0.6, 0.0]}}
            {'op': 'add',    'type': 'Mechanics', 'name': 'BB1',
                             'params': {'center': [0.0, -0.1],
                                        'shapes': [{'type': 'rectangle',
                                                    'point': [-0.15, -0.15],
                                                    'width': 0.3,
                                                    'height': 0.3,
                                                    'thickness': 0}]}}
            {'op': 'move',   'target': 'BB1', 'center': [0.2, 0.1]}
            {'op': 'rotate', 'target': 'BB1', 'rotationAngle': 0.1}
            {'op': 'add',    'type': 'Mechanics', 'name': 'MT1',
                             'params': {'attached_to': 'M1',
                                        'offset': [0.0, 0.0],
                                        'shapes': [...]}}
            {'op': 'set',    'target': 'MT1', 'attrs': {'attached_to': None}}
            {'op': 'undo'}
            {'op': 'redo'}

        A source, a dimension and a mechanics are named and addressed
        exactly as an optics is: 'remove', 'rename' and 'set' resolve
        their target across all four, which share one namespace. What the operations
        mean differs where the things differ - 'move' on a source names
        where the laser stands rather than the middle of a substrate,
        and a source takes a waist where an optics takes a curvature -
        so the attribute whitelists are separate.

        The edit is applied to the registered object itself, which is
        the same object the user holds in their own code. The trace
        result is invalidated so that the next draw() or scene_dict()
        re-traces.

        The state before the edit is kept, so that undo() can put it
        back; see there for what that costs and what it restores. An
        edit that is refused changes nothing and is not recorded. An
        edit that goes through discards whatever undo() had put aside
        for redo(): the layout has taken a different turn.

        Parameters
        ----------
        msg : dict
            The edit message.

        Returns
        -------
        self : OpticalLayout

        Raises
        ------
        EditError
            If the operation, the target or an attribute is not allowed.
        '''
        if not isinstance(msg, dict):
            raise EditError('An edit message must be a dict, not %s'
                            % type(msg).__name__)

        if msg.get('op') == 'undo':
            return self.undo()
        if msg.get('op') == 'redo':
            return self.redo()

        # Taken before the edit and kept only if the edit goes through:
        # a refused message leaves the layout alone, and an undo step
        # that restores what is already there is one press wasted.
        snapshot = (None if msg.get('op') in _NOT_UNDOABLE
                    else self._snapshot())
        result = self._apply_edit(msg)
        if snapshot is not None:
            self._history.append(snapshot)
            del self._history[:-UNDO_DEPTH]
            # An edit made after stepping back is a new branch, and the
            # one that was stepped out of has no way back to it: the
            # states in there describe elements this edit may have just
            # renamed, removed or moved on from.
            del self._future[:]
        return result

    def _snapshot(self):
        '''
        Capture the current state, for undo() or redo() to put back.

        The objects are kept alongside their serialized form. Restoring
        is then a matter of putting the values back onto the very
        objects that held them, rather than matching them up by name the
        way loading a file has to - so a rename comes back without
        swapping the object it named, and an element that was removed
        comes back as itself rather than as a copy.
        '''
        return (self.to_dict(), list(self.optics), list(self.sources),
                list(self.dimensions), list(self.mechanics))

    def _restore(self, snapshot):
        '''
        Put back a state captured by _snapshot().
        '''
        d, optics, sources, dimensions, mechanics = snapshot

        # The object lists and the serialized ones were made from the
        # same lists at the same moment, so they line up entry for
        # entry. The name is set apart from the rest: it is not an
        # editable attribute, so _update_optic does not carry it.
        self.optics = list(optics)
        self.sources = list(sources)
        self.dimensions = list(dimensions)
        self.mechanics = list(mechanics)
        for m, spec in zip(self.optics, d.get('optics', [])):
            m.name = spec['name']
            _update_optic(m, spec)
        for b, spec in zip(self.sources, d.get('sources', [])):
            b.name = spec['name']
            _update_source(b, spec)
        for dim, spec in zip(self.dimensions, d.get('dimensions', [])):
            dim.name = spec['name']
            _update_dimension(dim, spec)
        for mech, spec in zip(self.mechanics, d.get('mechanics', [])):
            mech.name = spec['name']
            _update_mechanics(mech, spec)
        # After every name is back: an attachment is stored by the
        # host's name, and the host may itself have just been renamed
        # by this restore.
        self._link_mechanics()

        self.rules = TraceRules.from_dict(d.get('rules', {}))
        self.draw_options = dict(d.get('draw_options', {}))
        self.name = d.get('name', self.name)
        self.beams = None
        self.beams_by_source = None
        return self

    def undo(self):
        '''
        Put the layout back as it was before the last edit.

        A step of the history holds the elements themselves as well as
        their values, so undoing restores those values onto those same
        objects. The ``M1`` of the user's own code, and the selection of
        a front end, go on naming the right thing - through a rename,
        and even through a removal, since an element taken out of the
        layout is put back as itself rather than as a copy. This is
        stronger than what loading a file can offer, which has only
        names to match objects up by.

        The state being left is put aside for redo(), and stays there
        until an edit goes through.

        Only edits applied through apply_edit are recorded. Assigning to
        an optics directly in Python is not an edit the layout ever sees,
        and is not undone - though it is captured by the snapshot of the
        next edit that does go through, and so is restored by undoing
        that one.

        Returns
        -------
        self : OpticalLayout

        Raises
        ------
        EditError
            If there is nothing left to undo.
        '''
        if not self._history:
            raise EditError('There is nothing to undo.')
        # Taken before the restore, so that redo() puts back the state
        # this undo is stepping out of.
        self._future.append(self._snapshot())
        del self._future[:-UNDO_DEPTH]
        return self._restore(self._history.pop())

    def redo(self):
        '''
        Put back the state that the last undo() stepped out of.

        Redo undoes an undo, and restores as exactly as undo() does: the
        same elements, so an element that undoing a removal brought back
        goes away again as itself rather than as a copy.

        Only undoing fills the redo stack, and the next edit that goes
        through empties it - once the layout has taken a different turn
        there is no branch left to return to.

        Returns
        -------
        self : OpticalLayout

        Raises
        ------
        EditError
            If there is nothing left to redo.
        '''
        if not self._future:
            raise EditError('There is nothing to redo.')
        self._history.append(self._snapshot())
        del self._history[:-UNDO_DEPTH]
        return self._restore(self._future.pop())

    @property
    def can_undo(self):
        '''
        Whether there is an edit left to undo.
        '''
        return len(self._history) > 0

    @property
    def can_redo(self):
        '''
        Whether there is an undo left to take back.
        '''
        return len(self._future) > 0

    def _apply_edit(self, msg):
        '''
        Apply one edit message. See apply_edit, which wraps this with
        the history that makes an edit undoable.
        '''
        op = msg.get('op')
        # 'name' is deliberately not in EDITABLE_OPTIC_ATTRS: renaming
        # changes the identity the layout resolves edits by, so it has
        # its own operation with a uniqueness check.
        if op == 'set' and self._is_dimension(msg.get('target')):
            self._set_dimension_attrs(self.get_dimension(msg['target']),
                                      msg.get('attrs') or {})
            # Moving an end of a measurement changes the measurement and
            # nothing else. No beam has moved, so the trace still stands.
            return self

        elif op in ('move', 'rotate', 'set') and self._is_mechanics(
                msg.get('target')):
            self._edit_mechanics(self.get_mechanics(msg['target']), op, msg)
            # A mechanics takes no part in the trace: moving one changes
            # the picture and nothing about the beams, so the trace
            # still stands.
            return self

        elif op in ('move', 'rotate', 'set') and self._is_source(
                msg.get('target')):
            # A source is not an optics and is not addressed like one:
            # it has no substrate to be held by and no face to be
            # squared with, so 'move' names where the laser stands and
            # 'rotate' which way it fires. Its own branch rather than a
            # widened optics branch, since the two share no attribute
            # names at all.
            self._edit_source(self.get_source(msg['target']), op, msg)

        elif op in ('move', 'rotate', 'set'):
            name = msg.get('target')
            try:
                optics = self.get_optics(name)
            except KeyError:
                if self._is_dimension(name):
                    # A dimension is two points, not a body: there is
                    # nothing to turn, and either end moves on its own.
                    raise EditError("%r is a dimension, which has no %s. "
                                    "Set 'p1' or 'p2' instead."
                                    % (name, 'orientation'
                                       if op == 'rotate' else 'position'))
                raise EditError("No optics named %r in the layout." % (name,))

            if op == 'set':
                attrs = msg.get('attrs') or {}
            else:
                # move and rotate are spellings of a one-attribute set;
                # they exist so that a front end says what it means.
                keys = (['HRcenter', 'center'] if op == 'move'
                        else ['normAngleHR', 'normVectHR'])
                attrs = {k: msg[k] for k in keys if k in msg}
                if not attrs:
                    raise EditError("A '%s' message needs one of %s."
                                    % (op, ' or '.join(keys)))

            for key, value in _ordered(attrs):
                if key not in EDITABLE_OPTIC_ATTRS:
                    raise EditError('%r is not an editable attribute of an '
                                    'optics.' % (key,))
                _check_choice(key, value)
                _set_optic_attr(optics, key, value)

        elif op == 'rename':
            old = msg.get('target')
            new = msg.get('name')
            target = self._resolve_target(old)
            if not isinstance(new, str) or not new.strip():
                raise EditError('A name must be a non-empty string, '
                                'not %r.' % (new,))
            if new != old:
                try:
                    self._check_name_free(new)
                except ValueError as e:
                    raise EditError(str(e))
            # Nothing else is keyed by the name: the per-optics tracing
            # settings live on the optics itself, so they travel with it.
            target.name = new
            if isinstance(target, (Dimension, Mechanics)):
                return self

        elif op in ('align', 'slide'):
            name = msg.get('target')
            try:
                optics = self.get_optics(name)
            except KeyError:
                raise EditError("No optics named %r in the layout." % (name,))
            if op == 'align':
                self._align_to_beam(optics, msg)
            else:
                self._slide_along_beam(optics, msg)

        elif op == 'add':
            if msg.get('type') == MECHANICS_TYPE:
                mech = self._mechanics_from_message(msg)
                try:
                    self.add_mechanics(mech)
                except ValueError as e:
                    raise EditError(str(e))
                # The trace never sees a mechanics; the picture grew,
                # the beams did not move.
                return self
            if msg.get('type') == DIMENSION_TYPE:
                dim = self._dimension_from_message(msg)
                try:
                    self.add_dimension(dim)
                except ValueError as e:
                    raise EditError(str(e))
                # A dimension is a note on the layout, not a part of it:
                # nothing about the trace has changed.
                return self
            if msg.get('type') == SOURCE_TYPE:
                src = self._source_from_message(msg)
                try:
                    self.add_source(src)
                except ValueError as e:
                    raise EditError(str(e))
            else:
                optics = self._optics_from_message(msg)
                try:
                    self.add_optics(optics)
                except ValueError as e:
                    raise EditError(str(e))

        elif op == 'remove':
            name = msg.get('target')
            if self._is_dimension(name):
                self.remove_dimension(name)
                return self
            if self._is_mechanics(name):
                self.remove_mechanics(name)
                return self
            if self._is_source(name):
                # Taking the last source out leaves a layout with
                # nothing to trace, which is a picture of the optics and
                # no beams. That is a state to be able to reach - it is
                # where a layout starts - so it is not refused.
                self.remove_source(name)
            else:
                try:
                    self.remove_optics(name)
                except KeyError:
                    raise EditError("No optics named %r in the layout."
                                    % (name,))
                except ValueError as e:
                    # An optics with hardware attached; the message
                    # already says what to do about it.
                    raise EditError(str(e))

        elif op == 'rules':
            rules = msg.get('rules') or {}
            for key in rules:
                if key not in EDITABLE_RULE_ATTRS:
                    raise EditError('%r is not an editable tracing rule.'
                                    % (key,))
            # Checked before anything is set, and checked at all now
            # that these have a control of their own: a depth of 'lots'
            # or a threshold of nothing would be taken quietly and only
            # show up as a trace that never returns.
            for key, value in sorted(rules.items()):
                setattr(self.rules, key, _as_rule_value(key, value))

        elif op in ('save', 'load'):
            # The path comes from the front end. In a notebook the page
            # and the kernel belong to the same user, so this is no more
            # than that user naming a file. A front end reachable over a
            # network would have to confine it instead.
            path = msg.get('path')
            if not isinstance(path, str) or not path.strip():
                raise EditError('A file name is needed, not %r.' % (path,))
            try:
                if op == 'save':
                    self.save(path)
                    # Saving changes nothing about the layout.
                    return self
                self.update_from_file(path)
            except EditError:
                raise
            except OSError as e:
                raise EditError('%s: %s' % (type(e).__name__, e))
            except (ValueError, KeyError, TypeError) as e:
                raise EditError('%s is not a layout file gtrace can read '
                                '(%s: %s).' % (path, type(e).__name__, e))
            return self

        elif op == 'export':
            # Its own branch rather than joining save/load: sharing
            # their catch-all would report a bad drawing option as
            # 'not a layout file gtrace can read'.
            path = msg.get('path')
            if not isinstance(path, str) or not path.strip():
                raise EditError('A file name is needed, not %r.' % (path,))
            fmt = msg.get('format', 'dxf')
            if fmt not in EXPORT_FORMATS:
                raise EditError('%r is not a format gtrace can write. It '
                                'writes %s.'
                                % (fmt, ', '.join(sorted(EXPORT_FORMATS))))
            dims = msg.get('dimensions', True)
            if not isinstance(dims, bool):
                raise EditError("'dimensions' is yes or no, not %r." % (dims,))
            try:
                self.export_dxf(path, dimensions=dims)
            except EditError:
                raise
            except OSError as e:
                raise EditError('%s: %s' % (type(e).__name__, e))
            except Exception as e:
                raise EditError('Could not write %s (%s: %s).'
                                % (path, type(e).__name__, e))
            # Writing a file changes nothing about the layout.
            return self

        elif op == 'draw':
            for key, value in (msg.get('params') or {}).items():
                if key not in EDITABLE_DRAW_OPTIONS:
                    raise EditError('%r is not an editable drawing option.'
                                    % (key,))
                _check_choice(key, value)
                self.draw_options[key] = value
            # A drawing option changes the picture, not the physics, so
            # the trace stands and is deliberately not invalidated.
            return self

        else:
            raise EditError('Unknown edit operation %r.' % (op,))

        # The trace no longer matches the layout.
        self.beams = None
        self.beams_by_source = None
        return self

    def _is_dimension(self, name):
        '''
        Whether a target name belongs to a registered dimension.
        '''
        return any(d.name == name for d in self.dimensions)

    def _is_source(self, name):
        '''
        Whether a target name belongs to a registered source.
        '''
        return any(s.name == name for s in self.sources)

    def _is_mechanics(self, name):
        '''
        Whether a target name belongs to a registered mechanics.
        '''
        return any(m.name == name for m in self.mechanics)

    def _resolve_target(self, name):
        '''
        The optics, source, dimension or mechanics a message names.

        The four share a namespace, so a message can say 'remove D1',
        'remove S1' or 'remove M1' without also having to say which kind
        of thing it is: a front end has a name under the cursor, not a
        class.
        '''
        for m in self.optics:
            if m.name == name:
                return m
        for s in self.sources:
            if s.name == name:
                return s
        for d in self.dimensions:
            if d.name == name:
                return d
        for h in self.mechanics:
            if h.name == name:
                return h
        raise EditError('Nothing named %r in the layout.' % (name,))

    def _edit_mechanics(self, m, op, msg):
        '''
        Apply a 'move', 'rotate' or 'set' to a mechanics.

        As everywhere else, 'move' and 'rotate' are spellings of a
        one-attribute 'set'. A free body takes a pose; an attached one
        has no pose of its own - it takes an offset instead, and a
        message trying to move it is refused with the reason. The
        attachment itself is edited through 'attached_to': an optics
        name attaches the body where it stands, and null detaches it,
        baking the derived pose in.

        Everything is checked before anything is set - the host looked
        up, the numbers converted - so a message half right leaves the
        body exactly as it was.
        '''
        if op == 'set':
            attrs = msg.get('attrs') or {}
        else:
            keys = ['center'] if op == 'move' else ['rotationAngle']
            attrs = {k: msg[k] for k in keys if k in msg}
            if not attrs:
                raise EditError("A '%s' message for a mechanics needs %s."
                                % (op, ' or '.join(keys)))
        for key in attrs:
            if key not in EDITABLE_MECHANICS_ATTRS:
                raise EditError('%r is not an editable attribute of a '
                                'mechanics.' % (key,))

        # What the attachment will be once this message is applied,
        # which is what decides whether a pose or an offset makes
        # sense in the same message.
        detaching = 'attached_to' in attrs and attrs['attached_to'] is None
        attaching = 'attached_to' in attrs and attrs['attached_to'] is not None
        will_be_attached = attaching or (m.attached_to is not None
                                         and not detaching)

        pose = {k: attrs[k] for k in _MECHANICS_POSE_ATTRS if k in attrs}
        if pose and will_be_attached:
            host = (attrs['attached_to'] if attaching
                    else m.attached_to.name)
            raise EditError(
                "'%s' is attached to '%s': it goes where its host goes. "
                "Move the optics, change the offset, or detach it first "
                "(set attached_to to null)." % (m.name, host))
        offs = {k: attrs[k] for k in ('offset', 'offset_angle')
                if k in attrs}
        if offs and not will_be_attached:
            raise EditError(
                "'%s' is not attached to anything, so it has no offset. "
                'Set center and rotationAngle instead.' % m.name)

        # Convert and look everything up before touching the body.
        new_host = None
        if attaching:
            name = attrs['attached_to']
            if not isinstance(name, str):
                raise EditError("'attached_to' is an optics name or null, "
                                'not %r.' % (name,))
            try:
                new_host = self.get_optics(name)
            except KeyError:
                raise EditError("Cannot attach '%s' to %r: no optics of "
                                'that name in the layout.' % (m.name, name))
        if 'offset' in offs:
            offs['offset'] = _as_point(offs['offset'], 'offset')
        if 'offset_angle' in offs:
            offs['offset_angle'] = _as_distance(offs['offset_angle'],
                                                'offset_angle')
        if 'center' in pose:
            pose['center'] = _as_point(pose['center'], 'center')
        if 'rotationAngle' in pose:
            pose['rotationAngle'] = _as_distance(pose['rotationAngle'],
                                                 'rotationAngle')

        # The attachment first: it decides what the rest lands on.
        if detaching:
            m.detach()
        elif attaching:
            # Attaching where it stands, unless the message also says
            # where on the host to stand.
            m.attach(new_host, offset=offs.pop('offset', None),
                     offset_angle=offs.pop('offset_angle', None))
        for key, value in sorted(offs.items()):
            setattr(m, key, value)
        for key, value in sorted(pose.items()):
            setattr(m, key, value)

    def _mechanics_from_message(self, msg):
        '''
        Build a Mechanics from an 'add' message.

        The shapes arrive serialized, exactly as a saved layout writes
        them; a shape gtrace cannot draw, or one missing a coordinate,
        comes back as a refusal rather than as a body that fails the
        first time the scene is built.
        '''
        params = msg.get('params') or {}
        for key in params:
            if key not in CREATABLE_MECHANICS_PARAMS:
                raise EditError('%r is not a parameter a new mechanics '
                                'takes.' % (key,))
        name = msg.get('name') or self.unique_mechanics_name()
        if not isinstance(name, str) or not name.strip():
            raise EditError('A name must be a non-empty string, not %r.'
                            % (name,))

        shapes_in = params.get('shapes', [])
        if not isinstance(shapes_in, (list, tuple)):
            raise EditError("'shapes' must be a list of serialized shapes, "
                            'not %r.' % (shapes_in,))
        try:
            shapes = [shape_from_dict(s) for s in shapes_in]
        except UnknownShapeError as e:
            raise EditError(str(e))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise EditError('A shape in the message is malformed (%s: %s).'
                            % (type(e).__name__, e))

        layer = params.get('layer', 'hardware')
        if not isinstance(layer, str) or not layer.strip():
            raise EditError('A layer must be a non-empty string, not %r.'
                            % (layer,))
        model = params.get('model', None)
        if model is not None and not isinstance(model, str):
            raise EditError('A model is a name or nothing, not %r.'
                            % (model,))

        if params.get('attached_to') is not None:
            host_name = params['attached_to']
            if not isinstance(host_name, str):
                raise EditError("'attached_to' is an optics name, not %r."
                                % (host_name,))
            if 'center' in params or 'rotationAngle' in params:
                raise EditError('An attached body has no pose of its own: '
                                "give 'offset' and 'offset_angle' instead "
                                "of 'center' and 'rotationAngle'.")
            try:
                host = self.get_optics(host_name)
            except KeyError:
                raise EditError("Cannot attach '%s' to %r: no optics of "
                                'that name in the layout.'
                                % (name, host_name))
            offset = _as_point(params.get('offset', [0.0, 0.0]), 'offset')
            offset_angle = _as_distance(params.get('offset_angle', 0.0),
                                        'offset_angle')
            return Mechanics(shapes=shapes, name=name, layer=layer,
                             model=model, attached_to=host, offset=offset,
                             offset_angle=offset_angle)

        center = _as_point(params.get('center', [0.0, 0.0]), 'center')
        angle = _as_distance(params.get('rotationAngle', 0.0),
                             'rotationAngle')
        return Mechanics(shapes=shapes, center=center, rotationAngle=angle,
                         name=name, layer=layer, model=model)

    def _edit_source(self, b, op, msg):
        '''
        Apply a 'move', 'rotate' or 'set' to a source beam.

        As with an optics, 'move' and 'rotate' are spellings of a
        one-attribute 'set'; they exist so that a front end says what it
        means. What they name is different, because a beam is a ray from
        a point rather than a body: its position is where it starts and
        its orientation is where it is aimed.
        '''
        if op == 'set':
            attrs = msg.get('attrs') or {}
        else:
            keys = ['pos'] if op == 'move' else ['dirAngle', 'dirVect']
            attrs = {k: msg[k] for k in keys if k in msg}
            if not attrs:
                raise EditError("A '%s' message for a source needs one of %s."
                                % (op, ' or '.join(keys)))
        # Every name is checked before any value is set, so that a
        # message naming one attribute the source does not have leaves
        # the source exactly as it was rather than half changed.
        for key in attrs:
            if key not in EDITABLE_SOURCE_ATTRS:
                raise EditError('%r is not an editable attribute of a '
                                'source.' % (key,))
        for key, value in _ordered_source(attrs):
            _set_source_attr(b, key, value)

    def _source_from_message(self, msg):
        '''
        Build a source beam from an 'add' message.

        Unlike a new optics, a new source copies nothing from the
        sources already in the layout. A laser is not cut to match the
        one next to it, and inheriting a q-parameter would be worse than
        useless: it describes a waist measured from a point the new
        source does not stand at.
        '''
        params = msg.get('params') or {}
        for key in params:
            if key not in CREATABLE_SOURCE_PARAMS:
                raise EditError('%r is not a parameter a new source takes.'
                                % (key,))
        name = msg.get('name') or self.unique_source_name()
        if not isinstance(name, str) or not name.strip():
            raise EditError('A name must be a non-empty string, not %r.'
                            % (name,))

        wl = _as_positive(params.get('wl', DEFAULT_SOURCE_WL), 'wl',
                          'a wavelength in metres')
        n = _as_positive(params.get('n', 1.0), 'n', 'an index of refraction')
        length = _as_positive(params.get('length', 1.0), 'length',
                              'a length in metres')
        w0 = _as_positive(params.get('waist_size', DEFAULT_SOURCE_WAIST),
                          'waist_size', 'a waist size in metres')
        d = _as_distance(params.get('waist_pos', 0.0), 'waist_pos')
        P = _as_distance(params.get('P', 1.0), 'P')
        if P < 0:
            raise EditError("'P' must not be negative, and %r is."
                            % (params['P'],))
        pos = _as_point(params.get('pos', [0.0, 0.0]), 'pos')

        b = GaussianBeam(q0=q_from_waist(w0, d, wl, n), pos=pos,
                         length=length, wl=wl, P=P, n=n, name=name)
        # After the constructor, which has taken the index into account
        # already: setting either of these here would only be undone by
        # the other, and dirVect is the one to win if both are given
        # since it says the same thing without a wrap.
        if 'dirVect' in params:
            _set_source_attr(b, 'dirVect', params['dirVect'])
        elif 'dirAngle' in params:
            _set_source_attr(b, 'dirAngle', params['dirAngle'])
        return b

    def _set_dimension_attrs(self, dim, attrs):
        '''
        Move the ends of a dimension, or the line drawn between them.
        '''
        for key, value in attrs.items():
            if key not in EDITABLE_DIMENSION_ATTRS:
                raise EditError('%r is not an editable attribute of a '
                                'dimension.' % (key,))
        ends = {'p1': np.asarray(dim.p1, dtype='float64'),
                'p2': np.asarray(dim.p2, dtype='float64')}
        for key, value in attrs.items():
            if key in _DIMENSION_POINTS:
                ends[key] = _as_point(value, key)
        if np.array_equal(ends['p1'], ends['p2']):
            raise EditError('A dimension needs two different ends; both '
                            'would be at %s.'
                            % ([float(x) for x in ends['p1']],))
        offset = (_as_distance(attrs['offset'], 'offset')
                  if 'offset' in attrs else dim.offset)
        dim.p1 = ends['p1']
        dim.p2 = ends['p2']
        dim.offset = offset

    def _dimension_from_message(self, msg):
        '''
        Build a Dimension from an 'add' message.
        '''
        params = msg.get('params') or {}
        for key in params:
            if key not in EDITABLE_DIMENSION_ATTRS:
                raise EditError('%r is not a parameter of a dimension.'
                                % (key,))
        for key in ('p1', 'p2'):
            if key not in params:
                raise EditError("Adding a dimension needs both ends; %r is "
                                "missing." % (key,))
        p1 = _as_point(params['p1'], 'p1')
        p2 = _as_point(params['p2'], 'p2')
        if np.array_equal(p1, p2):
            raise EditError('A dimension needs two different ends; both '
                            'would be at %s.' % ([float(x) for x in p1],))
        #Where the line goes is a drawing choice, and a line straight
        #between the ends is a sound default for a caller with no view
        #to place it against.
        offset = _as_distance(params.get('offset', 0.0), 'offset')

        name = msg.get('name')
        if name is None:
            name = self.unique_dimension_name()
        if not isinstance(name, str) or not name.strip():
            raise EditError('A dimension name must be a non-empty string, '
                            'not %r.' % (name,))
        return Dimension(p1, p2, name=name, offset=offset)

    def _beam_from_message(self, msg):
        '''
        The beam a message names, from the last trace.

        Identified by its index, with the name as a check. The front end
        is looking at a scene built from that trace, so the index is
        exact; but two beams can share a name and a stale scene would
        make an index alone point quietly at the wrong one, so the name
        has to agree, and is what the search falls back to when the
        index does not fit.
        '''
        beams = self.beams
        if beams is None:
            beams = self.trace()

        name = msg.get('beam')
        index = msg.get('beam_index')
        b = None
        if isinstance(index, int) and 0 <= index < len(beams):
            if name is None or beams[index].name == name:
                b = beams[index]
        if b is None and name is not None:
            b = next((x for x in beams if x.name == name), None)
        if b is None:
            raise EditError('The beam named in the message (%r, index %r) is '
                            'not in the last trace.' % (name, index))
        return b

    def _slide_along_beam(self, optics, msg):
        '''
        Move an optics along a beam's axis by a given distance, in
        metres, positive in the direction the beam travels.

        Nothing else about the optics changes: not its orientation, not
        its offset across the beam. An element already square on the
        beam therefore stays square and simply slides along it, which is
        the one degree of freedom left after aligning, and the one that
        wants a number rather than a drag - a lens goes where the mode
        matching says, not where the mouse happens to land.

        This is a translation of the whole substrate, so which point of
        it is nominally being moved makes no difference to the result.
        '''
        b = self._beam_from_message(msg)

        distance = msg.get('distance')
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            raise EditError('A slide message needs a distance in metres, '
                            'not %r.' % (distance,))
        if not np.isfinite(distance):
            raise EditError('A slide of %r is not a distance.' % (distance,))

        step = np.asarray(b.dirVect, dtype='float64') * distance
        if optics.anchor_point == 'center':
            optics.center = np.asarray(optics.center, dtype='float64') + step
        else:
            optics.HRcenter = (np.asarray(optics.HRcenter, dtype='float64')
                               + step)

    def _align_to_beam(self, optics, msg):
        '''
        Turn an optics square to a beam and slide it onto that beam's
        axis, leaving it at the point along the beam it was dropped.

        Almost every element on a bench is meant to sit square across a
        beam with the beam through its middle. Dragging one gets it
        approximately there and no closer, and the two things a drag
        cannot say - the exact angle and the exact offset across the
        beam - are the two the bench does not leave to chance. The one
        the user does mean to choose, where along the beam it sits, is
        the one kept.

        Which point of the element lands on the axis is what
        ``anchor_point`` already names: the apex of the front face for a
        mirror, since that is where the beam stops, and the middle of
        the substrate for a lens, since the beam goes through. At normal
        incidence the two are on the same line anyway, so the choice
        only shifts the element along the beam - which is the direction
        that was left approximate to begin with.

        The beam is identified by its index in the last trace, with its
        name as a check: the viewer is looking at a scene built from
        that trace, and an index alone would quietly align to the wrong
        beam if the two had drifted apart.
        '''
        b = self._beam_from_message(msg)

        point = msg.get('point')
        try:
            p = np.asarray([float(point[0]), float(point[1])])
        except (TypeError, ValueError, IndexError, KeyError):
            raise EditError('An align message needs a point on the beam, '
                            'not %r.' % (point,))

        pos = np.asarray(b.pos, dtype='float64')
        direction = np.asarray(b.dirVect, dtype='float64')
        # Clamped to the drawn segment: past either end the beam does
        # not exist, and the element would be aligned to a continuation
        # of it that nothing in the layout says anything about.
        along = float(np.dot(p - pos, direction))
        along = min(max(along, 0.0), float(b.length))
        foot = pos + direction * along

        # Facing back down the beam is what "square to it" means: the
        # front face normal points at where the light is coming from.
        optics.normAngleHR = float(np.arctan2(-direction[1], -direction[0]))
        if optics.anchor_point == 'center':
            optics.center = foot
        else:
            optics.HRcenter = foot

    def _optics_from_message(self, msg):
        '''
        Build the optics described by an 'add' message.

        Parameters not given are taken from the optics already in the
        layout, so that an element added to a system of 10 cm mirrors
        comes out a 10 cm mirror instead of a 25 cm one. The surfaces
        are flat unless asked otherwise: a curvature copied from a
        neighbour would be a surprise rather than a convenience.

        A lens is the exception, and inherits nothing. Its coatings, its
        aperture and its wedge are the lens's own: a lens given a
        mirror's 99% front face is one the main beam does not go
        through, and an aperture taken from a large mirror is a focal
        length the blank cannot be ground to. A lens comes off the
        catalogue shelf instead, at DEFAULT_LENS_F unless the message
        says otherwise.
        '''
        kind = msg.get('type', 'Mirror')
        if kind not in CREATABLE_OPTIC_TYPES:
            raise EditError('Cannot create an optics of type %r. Known '
                            'types are %s.'
                            % (kind, ', '.join(sorted(CREATABLE_OPTIC_TYPES))))

        allowed = CREATABLE_OPTIC_PARAMS
        if kind in _LENS_TYPES:
            allowed = allowed | CREATABLE_LENS_PARAMS
        params = msg.get('params') or {}
        for key, value in params.items():
            if key not in allowed:
                raise EditError('%r is not a parameter a new %s may be '
                                'given.' % (key, kind))
            _check_choice(key, value)

        # A missing name means "pick one for me"; a name that is present
        # but unusable is a mistake, not a request to invent one.
        name = msg.get('name')
        if name is None:
            name = self.unique_optics_name()
        elif not isinstance(name, str) or not name.strip():
            raise EditError('An optics name must be a non-empty string, '
                            'not %r.' % (name,))

        if kind in _LENS_TYPES:
            return self._lens_from_params(params, name, kind)

        kwargs = {'inv_ROC_HR': 0.0, 'inv_ROC_AR': 0.0}
        if self.optics:
            template = self.optics[-1]
            for key in _INHERITED_PARAMS:
                if hasattr(template, key):
                    kwargs[key] = getattr(template, key)
        kwargs.update(params)
        kwargs['name'] = name

        if kind == 'CyMirror':
            return optcomp.CyMirror(**kwargs)
        kwargs.pop('curve_direction', None)   # CyMirror only
        return optcomp.Mirror(**kwargs)

    def _lens_from_params(self, params, name, kind='Lens'):
        '''
        Build the lens described by an 'add' message.

        Raises
        ------
        EditError
            If the lens asked for cannot be made. The solver says why -
            a blank too thin for the faces, a shape that contradicts the
            sign of f - and that reason is what the front end shows.
        '''
        kwargs = dict(params)
        if kind == 'CyLens':
            cls = optcomp.CyLens
        else:
            cls = optcomp.Lens
            kwargs.pop('curve_direction', None)   # cylindrical types only
        # Curvatures given outright describe the lens completely, and
        # Lens refuses a focal length on top of them rather than
        # silently preferring one. Only fill in the default when there
        # is nothing else to go on.
        if 'f' not in kwargs and not ('inv_ROC_HR' in kwargs
                                      or 'inv_ROC_AR' in kwargs):
            kwargs['f'] = DEFAULT_LENS_F
        kwargs['name'] = name
        try:
            return cls(**kwargs)
        except ValueError as e:
            raise EditError('Cannot make that lens - %s: %s'
                            % (type(e).__name__, e))

#}}}

#{{{ draw

    def draw(self, canvas=None, **options):
        '''
        Draw the optics and the result of the last trace into a canvas.

        If trace() has not been run yet, it is run automatically.

        Parameters
        ----------
        canvas : draw.Canvas or None, optional
            The canvas to draw into. If None, a new canvas is created.
        **options
            Any of the keys of DRAW_OPTIONS, which documents what each
            one does and what it defaults to. Options not given here
            fall back to self.draw_options, then to those defaults, so
            a front end can change how the layout is drawn without
            every caller having to pass the choice along.

        Returns
        -------
        canvas : draw.Canvas
        '''
        opt = self.resolve_draw_options(**options)

        if self.beams is None:
            self.trace()

        if canvas is None:
            canvas = draw.Canvas()
            canvas.unit = 'm'

        canvas.add_layer("main_beam", color=(255,0,0))
        canvas.add_layer("main_beam_width", color=(255,0,255))
        canvas.add_layer("stray_beam", color=(0,255,0))
        canvas.add_layer("stray_beam_width", color=(0,255,255))

        for b in self.beams:
            if b.stray_order > 0:
                b.layer = 'stray_beam'
                sigma = opt['sigma_stray']
                drawWidth = opt['drawStrayWidth']
            else:
                b.layer = 'main_beam'
                sigma = opt['sigma_main']
                drawWidth = opt['drawMainWidth']

            b.draw(canvas, sigma=sigma, mode=opt['width_mode'],
                   drawWidth=drawWidth,
                   drawPower=opt['drawBeamLabels'],
                   drawName=opt['drawBeamLabels'],
                   fontSize=opt['fontSize'])

        drawAllOptics(canvas, self.optics, drawName=opt['drawOpticsNames'])

        # The hardware, on its own layer, so that CAD - and the layer
        # panel of the viewer - can switch it off as one thing. Its
        # names follow the same option as the optics names: both label
        # the elements of the bench.
        for m in self.mechanics:
            m.draw(canvas, drawName=opt['drawOpticsNames'])

        return canvas

    def resolve_draw_options(self, **options):
        '''
        The drawing options in force: the defaults, overridden by
        self.draw_options, overridden by the arguments given here.

        Raises
        ------
        TypeError
            If an option is not one of DRAW_OPTIONS. Accepting a
            misspelt key silently would mean the setting is ignored
            without anything saying so.
        '''
        for key in list(options) + list(self.draw_options):
            if key not in DRAW_OPTIONS:
                raise TypeError('Unknown drawing option %r. Known options '
                                'are %s.'
                                % (key, ', '.join(sorted(DRAW_OPTIONS))))
        resolved = dict(DRAW_OPTIONS)
        resolved.update(self.draw_options)
        resolved.update(options)
        return resolved

#}}}

#{{{ scene_dict

    def scene_dict(self, **kwargs):
        '''
        Return the JSON-compatible scene dict {'canvas': ..., 'beams': ...}
        of this layout, for consumption by the GUI viewer.

        If trace() has not been run yet, it is run automatically.

        Parameters
        ----------
        **kwargs
            Passed to draw().
        '''
        canvas = self.draw(**kwargs)
        scene = scene_to_dict(canvas, self.beams, self.optics,
                              display=self.resolve_draw_options(**kwargs))
        # Not part of the drawing, and not a property of any element:
        # whether the front end's Undo and Redo have anything to work
        # with. They travel with the scene because the scene is what a
        # front end is handed after every edit, which is exactly when
        # the answers can have changed.
        scene['can_undo'] = self.can_undo
        scene['can_redo'] = self.can_redo
        scene['dimensions'] = self.dimensions_dict()
        scene['snap'] = self.snap_points()
        # Which of the beams the user put there, as against which the
        # trace produced. The two are indistinguishable in 'beams' - a
        # source is traced from a copy of itself, so its own beam is in
        # there like any other - and a front end that cannot tell them
        # apart can neither draw the laser nor offer to edit it.
        scene['sources'] = self.sources_dict()
        # The hardware. Its shapes are already in the canvas, drawn on
        # their own layer; this channel is what lets a front end point
        # at a body - pick it by its outline, and edit its pose.
        scene['mechanics'] = self.mechanics_dict()
        # How deep the trace went, which is not a property of any
        # element but decides how much of the picture there is.
        scene['rules'] = self.rules.to_dict()
        return scene

    def mechanics_dict(self):
        '''
        The registered mechanics, as a front end addresses them. See
        mechanics_scene_dict for what each carries. The outline is
        worked out here rather than stored, like everything else that
        is derived: a stored copy could only go stale.
        '''
        return [mechanics_scene_dict(m) for m in self.mechanics]

    def sources_dict(self):
        '''
        The registered sources, as a front end draws and addresses them.

        See source_scene_dict for what each carries. The waist is worked
        out here rather than stored, for the same reason a dimension's
        length is: it is derived from the q-parameters, and a stored
        copy could only go stale.
        '''
        return [source_scene_dict(b) for b in self.sources]

    def dimensions_dict(self):
        '''
        The registered dimensions, each with what it comes to.

        The measurement is worked out here rather than stored on the
        dimension, so that moving the optics a span runs through cannot
        leave a stale number behind: the answer is recomputed every time
        the scene is built, which is after every edit.

        Returns
        -------
        list of dict
            The keys of dimension_to_dict, plus ``line`` - the two ends
            of the dimension line itself, carried aside by the offset -
            and ``length``, ``optical``, ``inside`` and ``n``. See
            Dimension.measure.
        '''
        out = []
        for dim in self.dimensions:
            d = dimension_to_dict(dim)
            a, b = dim.line_ends()
            d['line'] = [[float(a[0]), float(a[1])],
                         [float(b[0]), float(b[1])]]
            d.update(dim.measure(self.optics))
            out.append(d)
        return out

    def snap_points(self):
        '''
        The points of the optics a front end may snap a measurement to.

        Beam ends are not here: the scene already carries the ends of
        every beam literally, so a front end can offer those without
        anything being worked out twice. See optic_snap_points for what
        each optics contributes.
        '''
        points = []
        for o in self.optics:
            points.extend(optic_snap_points(o))
        for m in self.mechanics:
            points.extend(mechanics_snap_points(m))
        return points

#}}}

#{{{ DXF export

    def draw_dimensions(self, canvas, layername='dimensions'):
        '''
        Draw the registered dimensions into a canvas, on a layer of
        their own.

        A layer, because a dimension is a note about the system rather
        than part of it: a layer is exactly the mechanism CAD offers for
        something you want to be able to switch off, so an exported
        drawing carries the measurements without imposing them.

        This is deliberately not part of draw(). That method means "the
        physical picture", and the viewer draws dimensions itself from
        the scene, so putting them in draw() would draw them twice
        there.

        Parameters
        ----------
        canvas : draw.Canvas
            The canvas to draw into.
        layername : str, optional
            Layer the dimensions go on. Defaults to 'dimensions'.

        Returns
        -------
        canvas : draw.Canvas
        '''
        if not self.dimensions:
            return canvas
        canvas.add_layer(layername, color=(0, 128, 64))
        for dim in self.dimensions:
            a, b = dim.line_ends()
            canvas.add_shape(draw.Line(a, b), layername=layername)
            # Extension lines back to the points actually measured, and
            # a tick across each end of the dimension line. Drawn in
            # scene units here rather than in screen pixels as the
            # viewer does: a DXF has no screen to be read at.
            span = np.linalg.norm(b - a)
            tick = dim.normal * (span * _DXF_TICK)
            for end, foot in ((a, dim.p1), (b, dim.p2)):
                canvas.add_shape(draw.Line(end - tick, end + tick),
                                 layername=layername)
                if not np.allclose(end, foot):
                    canvas.add_shape(draw.Line(foot, end),
                                     layername=layername)

            m = dim.measure(self.optics)
            text = _fmt_mm(m['length'])
            if m['optical'] is not None:
                text += '  (%s optical)' % _fmt_mm(m['optical'])
            mid = (a + b) / 2
            height = span * _DXF_TEXT
            # Above the line and along it, kept the right way up, as in
            # the viewer.
            angle = np.arctan2(b[1] - a[1], b[0] - a[0])
            if angle > np.pi/2 or angle < -np.pi/2:
                angle += np.pi
            up = np.array([-np.sin(angle), np.cos(angle)])
            along = np.array([np.cos(angle), np.sin(angle)])
            point = mid + up * (height * 0.4) - along * (height * len(text) * 0.3)
            canvas.add_shape(draw.Text(text, point, height=height,
                                       rotation=angle),
                             layername=layername)
        return canvas

    def export_dxf(self, filename, dimensions=True, **kwargs):
        '''
        Write the layout to a DXF file. The companion of render_html.

        If trace() has not been run yet, it is run automatically.

        Parameters
        ----------
        filename : str
            Name of the DXF file to write.
        dimensions : bool, optional
            Whether the dimensions noted on the layout are drawn, on a
            layer of their own. Defaults to True; see draw_dimensions.
        **kwargs
            Passed to draw(), e.g. sigma_main or drawMainWidth.

        Returns
        -------
        filename : str
        '''
        canvas = self.draw(**kwargs)
        if dimensions:
            self.draw_dimensions(canvas)
        renderer.renderDXF(canvas, filename)
        return filename

#}}}

#{{{ HTML viewer

    def render_html(self, filename, title=None, **kwargs):
        '''
        Write the layout to a self-contained HTML file that can be
        opened in any browser, with zoom, pan and click readout of the
        beam parameters.

        If trace() has not been run yet, it is run automatically.

        Parameters
        ----------
        filename : str
            Name of the HTML file to write.
        title : str or None, optional
            Title shown in the browser tab and in the viewer.
            Defaults to the name of the layout.
        **kwargs
            Passed to draw(), e.g. sigma_main or drawMainWidth.

        Returns
        -------
        filename : str
        '''
        # Go through scene_dict so that the file gets exactly what the
        # widget gets - optics channel included, or clicking an element
        # in the static page would find nothing to show.
        return renderHTML(None, None, filename,
                          title=title if title is not None else self.name,
                          scene=self.scene_dict(**kwargs))

    def widget(self, height=None, editable=True,
               path='layout.json', dxf_path=None, **kwargs):
        '''
        Return a Jupyter widget showing this layout.

        The widget carries the scene in a traitlet, so a re-trace can be
        pushed into a view that is already on screen:

            w = layout.widget()
            w                       # displays the viewer
            M1.HRcenter = [0.6, 0]
            w.update()              # re-traces and redraws in place

        The loop also runs the other way: dragging an optics in the
        viewer sends an edit message, which this layout applies to the
        registered object before re-tracing and pushing the result back.
        Since the optics are held by reference, M1 in your own code is
        the object that moved.

        Requires anywidget. Use render_html() or show(backend='html')
        if it is not installed.

        Parameters
        ----------
        height : int or None, optional
            Height of the viewer in pixels. Defaults to None, which
            lets the front end make it as tall as it is wide - the
            width of the output area, capped to the window, both of
            which only the browser knows. Dragging the viewer's bottom
            edge settles it on a height of its own.
        editable : bool, optional
            Whether the optics can be dragged in the viewer.
            Defaults to True.
        path : str, optional
            File the Save and Load buttons start on, relative to where
            the kernel is running. Defaults to 'layout.json'.
        dxf_path : str or None, optional
            File the Export button starts on. Defaults to None, which
            names it after path with a .dxf extension - the two are
            usually wanted together, and it is still the user's to
            change in the panel.
        **kwargs
            Passed to draw(), and remembered for the redraws that
            follow an edit.

        Returns
        -------
        widget : anywidget.AnyWidget
        '''
        from gtrace.draw.viewer.widget import LayoutViewer
        return LayoutViewer(scene=self.scene_dict(**kwargs), layout=self,
                            draw_kwargs=kwargs,
                            height=0 if height is None else height,
                            editable=editable,
                            layout_path=path,
                            dxf_path=dxf_path if dxf_path is not None else '')

    def show(self, filename=None, browser=True, title=None, backend=None,
             **kwargs):
        '''
        Show the layout in the viewer.

        This is the front end entry point of the layout. In a notebook
        it returns a widget that renders in the output cell; anywhere
        else it writes a self-contained HTML file and opens it in the
        default browser. Both drive the same viewer core.

        Parameters
        ----------
        filename : str or None, optional
            Name of the HTML file to write. If None, a temporary file
            is created (and left behind for the browser to read).
            Ignored by the widget backend.
        browser : bool, optional
            Whether to open the file in the default browser.
            Defaults to True. Ignored by the widget backend.
        title : str or None, optional
            Title of the page, shown in the browser tab. Ignored by the
            widget backend, which has no tab to name.
        backend : {'widget', 'html'} or None, optional
            Which front end to use. Defaults to None, which picks the
            widget inside a Jupyter kernel with anywidget installed and
            HTML otherwise.
        **kwargs
            Passed to draw(), and to widget() for the widget backend.

        Returns
        -------
        widget or filename
            The widget for the 'widget' backend, the name of the file
            that was written for the 'html' one.
        '''
        if backend is None:
            from gtrace.draw.viewer.widget import widget_available
            backend = ('widget' if _in_notebook() and widget_available()
                       else 'html')

        if backend == 'widget':
            return self.widget(**kwargs)
        if backend != 'html':
            raise ValueError("backend must be 'widget', 'html' or None, "
                             'not %r' % (backend,))

        if filename is None:
            fd, filename = tempfile.mkstemp(prefix='gtrace_', suffix='.html')
            os.close(fd)

        self.render_html(filename, title=title, **kwargs)

        if browser:
            url = 'file:///' + os.path.abspath(filename).replace('\\', '/')
            webbrowser.open(url)

        return filename

#}}}

#{{{ Persistence

    def to_dict(self):
        '''
        Convert the layout (optics, sources, dimensions, rules and
        drawing options) to a JSON-compatible dict. The trace result is
        not included; it can be regenerated with trace().
        '''
        return {'name': str(self.name),
                'optics': [optic_to_dict(m) for m in self.optics],
                'sources': [source_to_dict(b) for b in self.sources],
                'dimensions': [dimension_to_dict(d) for d in self.dimensions],
                'mechanics': [mechanics_to_dict(m) for m in self.mechanics],
                'rules': self.rules.to_dict(),
                'draw_options': dict(self.draw_options)}

    @classmethod
    def from_dict(cls, d):
        '''
        Construct an OpticalLayout from a dict produced by to_dict().
        '''
        layout = cls._from_dict_parts(d)
        layout.draw_options = dict(d.get('draw_options', {}))
        return layout

    @classmethod
    def _from_dict_parts(cls, d):
        return cls(optics=[optic_from_dict(x) for x in d.get('optics', [])],
                   sources=[source_from_dict(x) for x in d.get('sources', [])],
                   #Absent from a file written before dimensions existed,
                   #which is a layout with no measurements noted on it.
                   dimensions=[dimension_from_dict(x)
                               for x in d.get('dimensions', [])],
                   #Absent from a file written before mechanics existed.
                   mechanics=[mechanics_from_dict(x)
                              for x in d.get('mechanics', [])],
                   rules=TraceRules.from_dict(d.get('rules', {})),
                   name=d.get('name', 'Layout'))

    def save(self, filename):
        '''
        Save the layout to a JSON file.
        '''
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=1)
        return filename

    @classmethod
    def load(cls, filename):
        '''
        Load a layout from a JSON file created by save().

        Returns a new layout. Use update_from_file() to load into an
        existing one instead.
        '''
        with open(filename, 'r') as f:
            return cls.from_dict(json.load(f))

    def update_from_dict(self, d):
        '''
        Replace the contents of this layout with the ones described by
        d, in place.

        An optics of the file that matches a registered one by name and
        by class is updated rather than replaced, so a variable holding
        it - the M1 of the user's own code, or the selection of a front
        end - keeps pointing at the right object. Anything else is
        built afresh, and registered elements the file does not mention
        are dropped.

        Loading a genuinely different layout therefore leaves the
        variables that named the old elements pointing at objects no
        longer registered; get_optics() gives the current ones.
        '''
        self.optics = _merge_by_name(self.optics, d.get('optics', []),
                                     optic_from_dict, _update_optic)
        self.sources = _merge_by_name(self.sources, d.get('sources', []),
                                      source_from_dict, _update_source)
        self.dimensions = _merge_by_name(self.dimensions,
                                         d.get('dimensions', []),
                                         dimension_from_dict,
                                         _update_dimension)
        self.mechanics = _merge_by_name(self.mechanics,
                                        d.get('mechanics', []),
                                        mechanics_from_dict,
                                        _update_mechanics)
        # The merges are done and the optics list is settled, so the
        # attachments stored by name can be joined to their hosts.
        self._link_mechanics()
        self.rules = TraceRules.from_dict(d.get('rules', {}))
        self.draw_options = dict(d.get('draw_options', {}))
        self.name = d.get('name', self.name)
        self.beams = None
        self.beams_by_source = None
        return self

    def update_from_file(self, filename):
        '''
        Load a JSON file created by save() into this layout, in place.
        See update_from_dict for what happens to the objects already
        registered.
        '''
        with open(filename, 'r') as f:
            return self.update_from_dict(json.load(f))

#}}}

#}}}
