'''
gtrace.draw.viewer

Self-contained HTML output for gtrace scenes (Stage 1 of the GUI).

renderHTML() writes a single HTML file containing the scene as embedded
JSON plus an inline copy of the dependency-free JavaScript viewer
(viewer.js / viewer.css, which live next to this module). The file has
no external references at all, so it can be opened by double-clicking,
sent to a collaborator or archived, and it still provides zoom, pan and
click readout of the beam parameters.

Typical use, replacing the DXF workflow:

    from gtrace.draw.viewer import renderHTML
    renderHTML(canvas, beams, 'trace.html')

or, from an OpticalLayout:

    layout.render_html('trace.html')
    layout.show()
'''

#{{{ Import modules

import json
import os

from gtrace.draw.serialize import scene_to_dict

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
__email__ = "asoy01@gmail.com"
__status__ = "Beta"

#}}}

#{{{ Asset loading

_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

def _read_asset(name):
    with open(os.path.join(_ASSET_DIR, name), 'r', encoding='utf-8') as f:
        return f.read()

def viewer_js():
    '''
    Return the source of the JavaScript viewer core as a string.
    '''
    return _read_asset('viewer.js')

def viewer_css():
    '''
    Return the style sheet of the viewer as a string.
    '''
    return _read_asset('viewer.css')

def viewer_template():
    '''
    Return the HTML template used by renderHTML() as a string.
    '''
    return _read_asset('template.html')

#}}}

#{{{ Helpers

def _escape_html(s):
    '''
    Escape a string for inclusion in HTML text content.
    '''
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;'))

def _js_literal(obj):
    '''
    Encode an object as a JavaScript literal that is safe to embed in a
    <script> element.

    json.dumps with ensure_ascii=True already escapes every non-ASCII
    character. The only remaining hazard is the sequence '</' (as in
    '</script>') appearing inside a string value, which would terminate
    the script element; inside a JavaScript string literal '<\\/' is an
    equivalent, harmless spelling.
    '''
    return json.dumps(obj, ensure_ascii=True).replace('</', '<\\/')

#}}}

#{{{ renderHTML

def renderHTML(canvas, beams=None, filename='gtrace.html', title=None,
               optics=None, scene=None):
    '''
    Render a canvas and the corresponding beams into a self-contained
    HTML file.

    Unlike renderDXF, the output carries the physical parameters of the
    beams as well as the drawing, so the viewer can report the beam
    parameters (q, radius, ROC, waist, Gouy phase) at an arbitrary point
    clicked along a beam.

    Parameters
    ----------
    canvas : draw.Canvas
        The canvas to render. Ignored if scene is given.
    beams : list of GaussianBeam or None, optional
        The beams whose parameters are made available to the viewer.
        If None, the viewer only shows the drawing (no click readout).
    filename : str, optional
        Name of the HTML file to write. Defaults to 'gtrace.html'.
    title : str or None, optional
        Title shown in the browser tab and in the viewer side bar.
        Defaults to the base name of filename.
    optics : list of Optics or None, optional
        The optics of the system. Without them the viewer draws the
        elements but cannot say which is which, so clicking one shows
        nothing. Pass them to get the properties panel.
    scene : dict or None, optional
        A scene dict as returned by serialize.scene_to_dict(). If given,
        it is used directly and canvas / beams / optics are ignored.
        This is the entry point for callers that already hold a
        serialized scene.

    Returns
    -------
    filename : str
        The name of the file that was written.
    '''
    if scene is None:
        scene = scene_to_dict(canvas, beams, optics)

    if title is None:
        title = os.path.splitext(os.path.basename(filename))[0]

    # The scene is substituted last: it is the only payload whose content
    # is arbitrary user data, so no placeholder can survive inside it.
    html = viewer_template()
    html = html.replace('__GTRACE_TITLE_JSON__', _js_literal(title))
    html = html.replace('<!--__GTRACE_TITLE_HTML__-->', _escape_html(title))
    html = html.replace('/*__GTRACE_CSS__*/', viewer_css())
    html = html.replace('/*__GTRACE_VIEWER_JS__*/', viewer_js())
    html = html.replace('__GTRACE_SCENE__', _js_literal(scene))

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)

    return filename

#}}}

#{{{ html_render_func

def html_render_func(beams=None, optics=None, **kwargs):
    '''
    Return a render_func(canvas, filename) suitable for
    gtrace.draw.tools.drawOptSys, producing HTML instead of DXF.

        drawOptSys(optList, beamList, 'trace.html',
                   render_func=html_render_func(beamList, optList))

    The beams and the optics have to be passed explicitly because
    drawOptSys only hands the canvas to its renderer, while the viewer
    also needs the physical parameters of the beams and the identity of
    the elements.
    '''
    def _render(canvas, filename):
        return renderHTML(canvas, beams, filename, optics=optics, **kwargs)
    return _render

#}}}
