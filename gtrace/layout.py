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
__version__ = "0.2.4"
__maintainer__ = "Yoichi Aso"
__email__ = "yoichi.aso@nao.ac.jp"
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
    per_optic_order : dict
        A dictionary mapping an optics name to the internal
        reflection order used for that optics, overriding the
        global order. Defaults to {}.
    '''

    def __init__(self, order=10, power_threshold=0.1, open_beam_length=1.0,
                 per_optic_order=None):
        self.order = order
        self.power_threshold = power_threshold
        self.open_beam_length = open_beam_length
        self.per_optic_order = dict(per_optic_order) if per_optic_order else {}

    def to_dict(self):
        '''
        Convert to a JSON-compatible dict.
        '''
        return {'order': int(self.order),
                'power_threshold': float(self.power_threshold),
                'open_beam_length': float(self.open_beam_length),
                'per_optic_order': {str(k): int(v)
                                    for k, v in self.per_optic_order.items()}}

    @classmethod
    def from_dict(cls, d):
        '''
        Construct a TraceRules from a dict produced by to_dict().
        '''
        return cls(order=d.get('order', 10),
                   power_threshold=d.get('power_threshold', 0.1),
                   open_beam_length=d.get('open_beam_length', 1.0),
                   per_optic_order=d.get('per_optic_order', None))

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
    'HRtransmissive', 'term_on_HR', 'term_on_HR_order',
])

#: Attributes of the tracing rules that a front end may change.
EDITABLE_RULE_ATTRS = frozenset([
    'order', 'power_threshold', 'open_beam_length', 'per_optic_order',
])

class EditError(ValueError):
    '''
    Raised when an edit message cannot be applied: unknown operation,
    unknown target or an attribute that is not editable.
    '''
    pass

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
         'term_on_HR_order': int(m.term_on_HR_order)}
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
              'term_on_HR': d.get('term_on_HR', False)}
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
                                  open_beam_length=self.rules.open_beam_length,
                                  per_optic_order=self.rules.per_optic_order)
            self.beams_by_source[src.name] = beams
            self.beams.extend(beams)
        return self.beams

#}}}

#{{{ apply_edit

    def apply_edit(self, msg):
        '''
        Apply an edit message coming from a front end.

        The message is a plain dict, so the same protocol travels over
        the notebook widget's comm and, later, over a websocket:

            {'op': 'move',   'target': 'M1', 'HRcenter': [0.52, 0.0]}
            {'op': 'rotate', 'target': 'M1', 'normAngleHR': 2.3}
            {'op': 'set',    'target': 'M1', 'attrs': {'diameter': 0.15}}
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
                setattr(optics, key, value)

        elif op == 'rules':
            for key, value in (msg.get('rules') or {}).items():
                if key not in EDITABLE_RULE_ATTRS:
                    raise EditError('%r is not an editable tracing rule.'
                                    % (key,))
                setattr(self.rules, key, value)

        else:
            raise EditError('Unknown edit operation %r.' % (op,))

        # The trace no longer matches the layout.
        self.beams = None
        self.beams_by_source = None
        return self

#}}}

#{{{ draw

    def draw(self, canvas=None, fontSize=False, drawMainWidth=True,
             drawStrayWidth=True, sigma_main=2.7, sigma_stray=2.7,
             drawBeamLabels=False, drawOpticsNames=True):
        '''
        Draw the optics and the result of the last trace into a canvas.

        If trace() has not been run yet, it is run automatically.

        Parameters
        ----------
        canvas : draw.Canvas or None, optional
            The canvas to draw into. If None, a new canvas is created.
        fontSize : float or False, optional
            Font size for the annotations.
        drawMainWidth : bool, optional
            Whether to draw the width envelope of the main beams.
            Defaults to True.
        drawStrayWidth : bool, optional
            Whether to draw the width envelope of the stray beams.
            Defaults to True. Set it to False to reproduce what
            drawOptSys draws, which leaves the 'stray_beam_width' layer
            empty.
        sigma_main : float, optional
            Width of the drawn envelope of main beams, in units of the
            1/e^2 radius of the beam. Defaults to 2.7, the aperture at
            which the diffraction loss is 1 ppm.
        sigma_stray : float, optional
            Same for stray beams. Defaults to 2.7 as well, so that every
            envelope in the drawing means the same thing.
        drawBeamLabels : bool, optional
            Whether to annotate each beam with its name and power.
            Defaults to False: the labels of neighbouring beams overlap
            badly, and the viewer reports the same values (and more) for
            whichever beam is clicked. Set it to True for a DXF export,
            where there is nothing to click.
        drawOpticsNames : bool, optional
            Whether to annotate each optics with its name. Defaults to
            True: unlike beams, optics are not clickable.

        Returns
        -------
        canvas : draw.Canvas
        '''
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
                sigma = sigma_stray
                drawWidth = drawStrayWidth
            else:
                b.layer = 'main_beam'
                sigma = sigma_main
                drawWidth = drawMainWidth

            b.draw(canvas, sigma=sigma, drawWidth=drawWidth,
                   drawPower=drawBeamLabels, drawName=drawBeamLabels,
                   fontSize=fontSize)

        drawAllOptics(canvas, self.optics, drawName=drawOpticsNames)

        return canvas

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
        return scene_to_dict(canvas, self.beams, self.optics)

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
        canvas = self.draw(**kwargs)
        return renderHTML(canvas, self.beams, filename,
                          title=title if title is not None else self.name)

    def widget(self, title=None, height=520, editable=True, **kwargs):
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
                            height=height, editable=editable)

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
        Convert the layout (optics, sources and rules) to a
        JSON-compatible dict. The trace result is not included;
        it can be regenerated with trace().
        '''
        return {'name': str(self.name),
                'optics': [optic_to_dict(m) for m in self.optics],
                'sources': [source_to_dict(b) for b in self.sources],
                'rules': self.rules.to_dict()}

    @classmethod
    def from_dict(cls, d):
        '''
        Construct an OpticalLayout from a dict produced by to_dict().
        '''
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

    @classmethod
    def load(cls, filename):
        '''
        Load a layout from a JSON file created by save().
        '''
        with open(filename, 'r') as f:
            return cls.from_dict(json.load(f))

#}}}

#}}}
