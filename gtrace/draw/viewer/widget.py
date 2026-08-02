'''
Jupyter widget front end for the gtrace viewer (Stage 2).

LayoutViewer wraps the same dependency-free viewer core that the static
HTML output uses, so a notebook gets zoom, pan and click readout without
leaving the cell. The scene travels over the widget's traitlets, which
means a re-trace can be pushed into a live view:

    w = layout.widget()
    w                      # displays the viewer
    M1.HRcenter = [0.6, 0]
    w.update()             # re-traces and redraws in place

anywidget is an optional dependency: importing gtrace does not require
it, and the HTML output keeps working without it.
'''

#{{{ Import modules

from gtrace.draw.viewer import viewer_js, viewer_css, _read_asset

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
__version__ = "0.3.0"
__maintainer__ = "Yoichi Aso"
__email__ = "yoichi.aso@nao.ac.jp"
__status__ = "Beta"

#}}}

#{{{ Availability

class WidgetNotAvailable(ImportError):
    '''
    Raised when the notebook viewer is asked for but anywidget is
    not installed.
    '''
    pass

def widget_available():
    '''
    Whether the notebook viewer can be used, i.e. whether anywidget is
    importable.
    '''
    try:
        import anywidget  # noqa: F401
    except ImportError:
        return False
    return True

def _require_anywidget():
    try:
        import anywidget
        import traitlets
    except ImportError:
        raise WidgetNotAvailable(
            'The notebook viewer needs anywidget. Install it with '
            "'pip install anywidget' (or 'pixi add anywidget'), or use "
            'layout.show(backend="html") to open the viewer in a browser '
            'instead.')
    return anywidget, traitlets

#}}}

#{{{ ESM module

def widget_js():
    '''
    Return the source of the anywidget ESM wrapper as a string.
    '''
    return _read_asset('widget.js')

def widget_esm():
    '''
    Return the ESM module served to the notebook front end.

    It is the viewer core followed by the anywidget wrapper. The core is
    an IIFE that publishes GTraceViewer on globalThis, so concatenating
    the two is enough to bring it into scope; there is no build step and
    no module resolution to go wrong, which was the lesson of the
    dxf-viewer experiment.
    '''
    return viewer_js() + '\n' + widget_js()

#}}}

#{{{ LayoutViewer

_LAYOUT_VIEWER = None

def _build_class():
    '''
    Define the widget class on first use, so that importing this module
    does not require anywidget.
    '''
    anywidget, traitlets = _require_anywidget()

    class LayoutViewer(anywidget.AnyWidget):
        '''
        Notebook viewer for a gtrace scene.

        Parameters
        ----------
        scene : dict
            A scene dict as returned by serialize.scene_to_dict().
        title : str
            Title shown in the side bar.
        height : int
            Height of the viewer in pixels.
        '''

        _esm = widget_esm()
        _css = viewer_css()

        scene = traitlets.Dict().tag(sync=True)
        title = traitlets.Unicode('gtrace').tag(sync=True)
        height = traitlets.Int(520).tag(sync=True)
        #: Whether the front end may send edit messages back.
        editable = traitlets.Bool(True).tag(sync=True)
        #: Last edit failure, shown by the front end. Empty when fine.
        error = traitlets.Unicode('').tag(sync=True)
        #: Confirmation of something that leaves no visible trace, such
        #: as a save. Empty when there is nothing to say.
        notice = traitlets.Unicode('').tag(sync=True)
        #: File the Save and Load buttons start on.
        layout_path = traitlets.Unicode('layout.json').tag(sync=True)

        def __init__(self, scene=None, layout=None, draw_kwargs=None,
                     **kwargs):
            super().__init__(scene=scene if scene is not None else {},
                             **kwargs)
            # Held so that update() can re-trace. Not a traitlet: the
            # layout is a Python object, not something to synchronize.
            self._layout = layout
            self._draw_kwargs = dict(draw_kwargs or {})
            self._edit_log = []
            if layout is None:
                self.editable = False
            # Note the name: Widget._handle_custom_msg(content, buffers)
            # is ipywidgets' own dispatcher, and overriding it with a
            # callback signature breaks every custom message.
            self.on_msg(self._on_edit_msg)

        def update(self, **kwargs):
            '''
            Re-trace the layout this viewer came from and redraw, keeping
            the current zoom, pan and layer visibility.

            Parameters
            ----------
            **kwargs
                Passed to OpticalLayout.draw(). Remembered, so that
                edits coming from the front end draw the same way.
            '''
            if self._layout is None:
                raise ValueError('This viewer is not attached to a layout; '
                                 'assign to .scene directly instead.')
            if kwargs:
                self._draw_kwargs = dict(kwargs)
            self._layout.beams = None
            self.scene = self._layout.scene_dict(**self._draw_kwargs)
            return self

#{{{ Edits from the front end

        @property
        def edits(self):
            '''
            The edit messages received so far, oldest first.
            '''
            return list(self._edit_log)

        def apply_edit(self, msg):
            '''
            Apply one edit message to the layout and push the new scene.

            This is what the front end reaches through the widget's
            comm, and is also callable directly, which is how the tests
            drive it. A rejected edit leaves the layout untouched and
            reports itself through the 'error' traitlet rather than
            raising into the comm machinery, where nothing would see it.

            Parameters
            ----------
            msg : dict
                An edit message, as described by
                OpticalLayout.apply_edit.

            Returns
            -------
            ok : bool
                Whether the edit was applied.
            '''
            if self._layout is None or not self.editable:
                self.error = 'This viewer is read-only.'
                return False
            try:
                self._layout.apply_edit(msg)
            except Exception as e:
                self.error = '%s: %s' % (type(e).__name__, e)
                return False
            self._edit_log.append(msg)
            self.error = ''

            # A save leaves the drawing exactly as it was, so without a
            # word from us the user cannot tell it happened.
            if msg.get('op') == 'save':
                self.notice = 'Saved to %s' % (msg.get('path'),)
                return True
            self.notice = ('Loaded %s' % (msg.get('path'),)
                           if msg.get('op') == 'load' else '')
            self.scene = self._layout.scene_dict(**self._draw_kwargs)
            return True

        def _on_edit_msg(self, widget, content, buffers):
            '''
            Callback registered with on_msg. ipywidgets calls it as
            (widget, content, buffers) for every custom message the
            front end sends, so ignore anything that is not an edit.
            '''
            if isinstance(content, dict) and 'op' in content:
                self.apply_edit(content)

#}}}

    return LayoutViewer

def LayoutViewer(*args, **kwargs):
    '''
    Construct the notebook viewer widget.

    This is a function rather than the class itself so that anywidget is
    imported only when a widget is actually created.
    '''
    global _LAYOUT_VIEWER
    if _LAYOUT_VIEWER is None:
        _LAYOUT_VIEWER = _build_class()
    return _LAYOUT_VIEWER(*args, **kwargs)

#}}}
