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
from gtrace.nonsequential import non_seq_trace
from gtrace.draw.tools import drawAllOptics
from gtrace.draw.serialize import scene_to_dict
from gtrace.draw.viewer import renderHTML
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
    'HRtransmissive', 'term_on_HR', 'term_on_HR_order', 'max_stray_order',
    'curve_direction',
])

#: Values an attribute is restricted to. A whitelist of names keeps a
#: front end from reaching attributes it should not; this keeps it from
#: putting nonsense into the ones it may.
ATTR_CHOICES = {'curve_direction': ('h', 'v'),
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
    'HRtransmissive', 'term_on_HR', 'max_stray_order', 'curve_direction',
])

#: Parameters a new optics copies from the optics already in the layout,
#: so that a mirror added to a system of 10 cm optics is a 10 cm optics
#: rather than whatever the class default happens to be.
_INHERITED_PARAMS = ['diameter', 'thickness', 'wedgeAngle', 'n',
                     'Refl_HR', 'Trans_HR', 'Refl_AR', 'Trans_AR']

#: Types that can be created from an edit message.
CREATABLE_OPTIC_TYPES = {'Mirror': 'Mirror', 'CyMirror': 'CyMirror'}

class EditError(ValueError):
    '''
    Raised when an edit message cannot be applied: unknown operation,
    unknown target or an attribute that is not editable.
    '''
    pass

def _check_choice(key, value):
    '''
    Reject a value outside the set an attribute allows.
    '''
    choices = ATTR_CHOICES.get(key)
    if choices is not None and value not in choices:
        raise EditError('%r must be one of %s, not %r.'
                        % (key, ', '.join(repr(c) for c in choices), value))

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
    Convert a Mirror or CyMirror to a JSON-compatible dict of its
    construction parameters.
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
         'term_on_HR': bool(m.term_on_HR),
         'term_on_HR_order': int(m.term_on_HR_order),
         'max_stray_order': (None if m.max_stray_order is None
                             else int(m.max_stray_order))}
    if isinstance(m, optcomp.CyMirror):
        d['curve_direction'] = str(m.curve_direction)
    return d

def optic_from_dict(d):
    '''
    Construct a Mirror or CyMirror from a dict produced by
    optic_to_dict().
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
              'term_on_HR': d.get('term_on_HR', False),
              'max_stray_order': d.get('max_stray_order', None)}
    if d['type'] == 'CyMirror':
        m = optcomp.CyMirror(curve_direction=d.get('curve_direction', 'h'),
                             **kwargs)
    elif d['type'] == 'Mirror':
        m = optcomp.Mirror(**kwargs)
    else:
        raise ValueError('Unknown optics type: %s' % d['type'])
    m.term_on_HR_order = d.get('term_on_HR_order', 0)
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
    for key in ['diameter', 'thickness', 'wedgeAngle', 'n',
                'Refl_HR', 'Trans_HR', 'Refl_AR', 'Trans_AR',
                'inv_ROC_HR', 'inv_ROC_AR',
                'HRtransmissive', 'term_on_HR', 'term_on_HR_order',
                'max_stray_order', 'curve_direction']:
        if key in d and hasattr(m, key):
            setattr(m, key, d[key])
    # Orientation before position: the position handler works from the
    # normal vector, so setting them the other way round would leave
    # the substrate placed off the old orientation.
    if 'normAngleHR' in d:
        m.normAngleHR = d['normAngleHR']
    if 'HRcenter' in d:
        m.HRcenter = d['HRcenter']

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

    def __init__(self, optics=None, sources=None, rules=None, name='Layout'):
        self.name = name
        self.optics = []
        self.sources = []
        self.rules = rules if rules is not None else TraceRules()
        #: Overrides for DRAW_OPTIONS, as chosen by a front end. Display
        #: settings, so changing one redraws but does not re-trace.
        self.draw_options = {}
        self.beams = None
        self.beams_by_source = None

        if optics is not None:
            for m in optics:
                self.add_optics(m)
        if sources is not None:
            for b in sources:
                self.add_source(b)

#{{{ Registration

    def add_optics(self, m):
        '''
        Register an optics. The optics is held by reference.
        Its name must be unique within the layout.
        '''
        if m.name in [o.name for o in self.optics]:
            raise ValueError("An optics named '%s' is already registered."
                             % m.name)
        self.optics.append(m)

    def add_source(self, b):
        '''
        Register a source beam. The beam is held by reference.
        Its name must be unique within the layout.
        '''
        if b.name in [s.name for s in self.sources]:
            raise ValueError("A source named '%s' is already registered."
                             % b.name)
        self.sources.append(b)

    def remove_optics(self, name):
        '''
        Remove the optics with the given name from the layout.
        '''
        self.optics.remove(self.get_optics(name))

    def remove_source(self, name):
        '''
        Remove the source with the given name from the layout.
        '''
        self.sources.remove(self.get_source(name))

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

    def unique_optics_name(self, prefix='M'):
        '''
        Return a name of the form prefix + number that no registered
        optics uses. Front ends need a name before they can talk about
        the element they are asking for.
        '''
        taken = set(o.name for o in self.optics)
        i = 1
        while '%s%d' % (prefix, i) in taken:
            i += 1
        return '%s%d' % (prefix, i)

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
            {'op': 'rename', 'target': 'M1', 'name': 'PRM'}
            {'op': 'add',    'type': 'Mirror', 'name': 'M4',
                             'params': {'HRcenter': [0.3, 0.2]}}
            {'op': 'remove', 'target': 'M4'}
            {'op': 'rules',  'rules': {'power_threshold': 1e-6}}

        The edit is applied to the registered object itself, which is
        the same object the user holds in their own code. The trace
        result is invalidated so that the next draw() or scene_dict()
        re-traces.

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

        op = msg.get('op')
        # 'name' is deliberately not in EDITABLE_OPTIC_ATTRS: renaming
        # changes the identity the layout resolves edits by, so it has
        # its own operation with a uniqueness check.
        if op in ('move', 'rotate', 'set'):
            name = msg.get('target')
            try:
                optics = self.get_optics(name)
            except KeyError:
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

            for key, value in attrs.items():
                if key not in EDITABLE_OPTIC_ATTRS:
                    raise EditError('%r is not an editable attribute of an '
                                    'optics.' % (key,))
                _check_choice(key, value)
                setattr(optics, key, value)

        elif op == 'rename':
            old = msg.get('target')
            new = msg.get('name')
            try:
                optics = self.get_optics(old)
            except KeyError:
                raise EditError("No optics named %r in the layout." % (old,))
            if not isinstance(new, str) or not new.strip():
                raise EditError('An optics name must be a non-empty string, '
                                'not %r.' % (new,))
            if new != old and any(o.name == new for o in self.optics):
                raise EditError("An optics named '%s' is already registered."
                                % new)
            # Nothing else is keyed by the name: the per-optics tracing
            # settings live on the optics itself, so they travel with it.
            optics.name = new

        elif op == 'add':
            optics = self._optics_from_message(msg)
            try:
                self.add_optics(optics)
            except ValueError as e:
                raise EditError(str(e))

        elif op == 'remove':
            name = msg.get('target')
            try:
                self.remove_optics(name)
            except KeyError:
                raise EditError("No optics named %r in the layout." % (name,))

        elif op == 'rules':
            for key, value in (msg.get('rules') or {}).items():
                if key not in EDITABLE_RULE_ATTRS:
                    raise EditError('%r is not an editable tracing rule.'
                                    % (key,))
                setattr(self.rules, key, value)

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

    def _optics_from_message(self, msg):
        '''
        Build the optics described by an 'add' message.

        Parameters not given are taken from the optics already in the
        layout, so that an element added to a system of 10 cm mirrors
        comes out a 10 cm mirror instead of a 25 cm one. The surfaces
        are flat unless asked otherwise: a curvature copied from a
        neighbour would be a surprise rather than a convenience.
        '''
        kind = msg.get('type', 'Mirror')
        if kind not in CREATABLE_OPTIC_TYPES:
            raise EditError('Cannot create an optics of type %r. Known '
                            'types are %s.'
                            % (kind, ', '.join(sorted(CREATABLE_OPTIC_TYPES))))

        params = msg.get('params') or {}
        for key, value in params.items():
            if key not in CREATABLE_OPTIC_PARAMS:
                raise EditError('%r is not a parameter a new optics may be '
                                'given.' % (key,))
            _check_choice(key, value)

        kwargs = {'inv_ROC_HR': 0.0, 'inv_ROC_AR': 0.0}
        if self.optics:
            template = self.optics[-1]
            for key in _INHERITED_PARAMS:
                if hasattr(template, key):
                    kwargs[key] = getattr(template, key)
        kwargs.update(params)

        # A missing name means "pick one for me"; a name that is present
        # but unusable is a mistake, not a request to invent one.
        name = msg.get('name')
        if name is None:
            name = self.unique_optics_name()
        elif not isinstance(name, str) or not name.strip():
            raise EditError('An optics name must be a non-empty string, '
                            'not %r.' % (name,))
        kwargs['name'] = name

        if kind == 'CyMirror':
            return optcomp.CyMirror(**kwargs)
        kwargs.pop('curve_direction', None)   # CyMirror only
        return optcomp.Mirror(**kwargs)

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
        return scene_to_dict(canvas, self.beams, self.optics,
                             display=self.resolve_draw_options(**kwargs))

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

    def widget(self, title=None, height=520, editable=True,
               path='layout.json', **kwargs):
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
        title : str or None, optional
            Title shown in the side bar. Defaults to the layout name.
        height : int, optional
            Height of the viewer in pixels. Defaults to 520.
        editable : bool, optional
            Whether the optics can be dragged in the viewer.
            Defaults to True.
        path : str, optional
            File the Save and Load buttons start on, relative to where
            the kernel is running. Defaults to 'layout.json'.
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
                            title=title if title is not None else self.name,
                            height=height, editable=editable,
                            layout_path=path)

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
            Title shown in the browser tab and in the viewer.
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
            return self.widget(title=title, **kwargs)
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
        Convert the layout (optics, sources, rules and drawing options)
        to a JSON-compatible dict. The trace result is not included;
        it can be regenerated with trace().
        '''
        return {'name': str(self.name),
                'optics': [optic_to_dict(m) for m in self.optics],
                'sources': [source_to_dict(b) for b in self.sources],
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
