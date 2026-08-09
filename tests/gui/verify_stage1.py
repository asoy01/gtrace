'''
Stage 1 verification: renderHTML + the JavaScript viewer core.

Builds a small optical system, renders it to HTML, checks that the file
is self-contained, and dumps reference values computed by gtrace so that
verify_stage1.js can compare them with what viewer.js computes.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json
import os
import re
import sys

import numpy as np

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.layout import OpticalLayout, TraceRules
from gtrace.draw.viewer import renderHTML, html_render_func
from gtrace.draw.tools import drawOptSys
from gtrace.unit import *

OUT = WORK

npass = 0
nfail = 0

def check(name, cond, detail=''):
    global npass, nfail
    if cond:
        npass += 1
        print('  PASS  %s %s' % (name, detail))
    else:
        nfail += 1
        print('  FAIL  %s %s' % (name, detail))

#{{{ Build a test system

b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                       pos=[0, 0], dirAngle=0, name='b0')

M1 = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=deg2rad(180-45),
                diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                inv_ROC_HR=0.0, Refl_HR=0.99, Trans_HR=0.01,
                Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45, name='M1')
M2 = opt.Mirror(HRcenter=[0.5, 0.4], normAngleHR=deg2rad(-90+45),
                diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                inv_ROC_HR=1.0/2.0, Refl_HR=0.99, Trans_HR=0.01,
                Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45, name='M2')
M3 = opt.Mirror(HRcenter=[0.9, 0.4], normAngleHR=deg2rad(180),
                diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                inv_ROC_HR=0.0, Refl_HR=0.9, Trans_HR=0.1,
                Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45, name='M3')

layout = OpticalLayout(optics=[M1, M2, M3], sources=[b0],
                       rules=TraceRules(order=5, power_threshold=1e-4),
                       name='Verify Stage 1')

#}}}

print('--- renderHTML ---')

html_file = os.path.join(OUT, 'stage1_view.html')
ret = layout.render_html(html_file)
check('render_html returns the filename', ret == html_file)
check('HTML file written', os.path.exists(html_file),
      '(%d bytes)' % os.path.getsize(html_file))

with open(html_file, encoding='utf-8') as f:
    html = f.read()

check('has doctype', html.lstrip().startswith('<!DOCTYPE html>'))
check('title is the layout name', '<title>Verify Stage 1</title>' in html)
check('viewer.js inlined', 'GTraceViewer' in html and 'beamParamsAt' in html)
check('viewer.css inlined', '.gt-root' in html)
check('scene JSON embedded', '"beams"' in html and '"layers"' in html)
check('no placeholder left', '__GTRACE_' not in html)

# Self-containment: no external resource may be referenced.
ext = re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', html)
check('no external src/href', ext == [], str(ext))
# The only permitted URL is the SVG namespace, which is an identifier
# rather than something the browser fetches.
urls = [u for u in re.findall(r'https?://[^\s"\'<>)]+', html)
        if u != 'http://www.w3.org/2000/svg']
check('no fetched URL', urls == [], str(urls[:3]))

# The '</' escaping must not have injected or broken a script element.
check('exactly the 3 script elements of the template',
      html.count('<script>') == 3 and html.count('</script>') == 3,
      '(%d open, %d close)' % (html.count('<script>'), html.count('</script>')))

print('--- renderHTML: other entry points ---')

f2 = os.path.join(OUT, 'stage1_func.html')
renderHTML(layout.draw(), layout.beams, f2, title='Direct call')
check('renderHTML(canvas, beams, filename)', os.path.exists(f2))
with open(f2, encoding='utf-8') as f:
    check('explicit title used', '<title>Direct call</title>' in f.read())

f3 = os.path.join(OUT, 'stage1_drawoptsys.html')
drawOptSys([M1, M2, M3], layout.beams, f3,
           render_func=html_render_func(layout.beams))
check('drawOptSys with html_render_func', os.path.exists(f3))

f4 = os.path.join(OUT, 'stage1_nobeams.html')
renderHTML(layout.draw(), None, f4)
check('renderHTML without beams', os.path.exists(f4))
with open(f4, encoding='utf-8') as f:
    check('empty beam list when beams=None', '"beams": []' in f.read())

f5 = os.path.join(OUT, 'stage1_scene.html')
renderHTML(None, None, f5, scene=layout.scene_dict(), title='From scene')
check('renderHTML(scene=...)', os.path.exists(f5))

# drawStrayWidth=False reproduces what drawOptSys draws and leaves
# stray_beam_width empty; keep that page for the empty-layer checks.
f7 = os.path.join(OUT, 'stage1_emptylayer.html')
layout.render_html(f7, drawStrayWidth=False)
check('drawStrayWidth=False page written', os.path.exists(f7))

def layer_counts(**kwargs):
    return {ly.name: len(ly.shapes)
            for ly in layout.draw(**kwargs).layers.values()}

empty_scene = layer_counts(drawStrayWidth=False)
full_scene = layer_counts()

# By default the drawing carries the optics names and nothing else: the
# beam name and power are read from the panel instead.
def texts(**kwargs):
    out = []
    for ly in layout.draw(**kwargs).layers.values():
        out += [s.text for s in ly.shapes if hasattr(s, 'text')]
    return out

plain = texts()
labelled = texts(drawBeamLabels=True)
check('no beam label by default',
      not any('P=' in t for t in plain), str(plain))
check('optics names are kept',
      sorted(t.strip() for t in plain) == ['M1', 'M2', 'M3'], str(plain))
check('drawBeamLabels=True restores them',
      sum(1 for t in labelled if 'P=' in t) == len(layout.beams),
      '(%d labels for %d beams)'
      % (sum(1 for t in labelled if 'P=' in t), len(layout.beams)))
check('drawOpticsNames=False drops the optics names',
      texts(drawOpticsNames=False) == [], str(texts(drawOpticsNames=False)))
check('drawStrayWidth=False leaves stray_beam_width empty',
      empty_scene.get('stray_beam_width') == 0, str(empty_scene))
check('the default fills stray_beam_width',
      full_scene.get('stray_beam_width', 0) > 0, str(full_scene))
check('only the stray width layer differs',
      {k: v for k, v in full_scene.items() if k != 'stray_beam_width'}
      == {k: v for k, v in empty_scene.items() if k != 'stray_beam_width'})

# show() must not need a browser to write its file.
f6 = os.path.join(OUT, 'stage1_show.html')
check('show(browser=False)', layout.show(f6, browser=False) == f6
      and os.path.exists(f6))

print('--- the optics channel reaches the file ---')
# The static page draws the elements from the canvas, but it can only
# say which is which - and so show properties on a click - if the optics
# channel travelled with it.
def embedded_scene(path):
    with open(path, encoding='utf-8') as f:
        doc = f.read()
    m = re.search(r'var GTRACE_SCENE = (\{.*?\});\n', doc, re.S)
    return json.loads(m.group(1).replace('<\\/', '</')) if m else None

emb = embedded_scene(html_file)
check('the embedded scene parses', emb is not None)
check('it has all the channels',
      set(emb.keys()) == {'canvas', 'beams', 'optics', 'display',
                          'sources', 'rules',
                          'can_undo', 'can_redo', 'dimensions', 'snap'},
      str(list(emb.keys())))
# A written page has no Python behind it, so there is nothing it could
# undo and no button offering to.
check('a static page reports no history', emb['can_undo'] is False,
      str(emb['can_undo']))
check('nor anything to redo', emb['can_redo'] is False,
      str(emb['can_redo']))
check('and says how it was drawn',
      emb['display'].get('width_mode') == 'x'
      and emb['display'].get('sigma_main') == 2.7,
      str(emb['display']))
check('render_html carries the optics',
      [o['name'] for o in emb['optics']] == ['M1', 'M2', 'M3'],
      str([o['name'] for o in emb['optics']]))
check('with what the panel needs',
      all(k in emb['optics'][0] for k in
          ['center', 'normAngleHR', 'diameter', 'thickness', 'inv_ROC_HR',
           'HRtransmissive', 'term_on_HR', 'term_on_HR_order',
           'max_stray_order']),
      str(sorted(emb['optics'][0].keys())))

f8 = os.path.join(OUT, 'stage1_optics.html')
renderHTML(layout.draw(), layout.beams, f8, optics=layout.optics)
check('renderHTML(optics=...) carries them too',
      len(embedded_scene(f8)['optics']) == 3)

f9 = os.path.join(OUT, 'stage1_drawoptsys_optics.html')
drawOptSys([M1, M2, M3], layout.beams, f9,
           render_func=html_render_func(layout.beams, [M1, M2, M3]))
check('so does html_render_func',
      len(embedded_scene(f9)['optics']) == 3)

check('and without them the channel is simply empty',
      embedded_scene(f3)['optics'] == [])

print('--- shape coverage ---')

scene = layout.scene_dict()
types = set()
for ly in scene['canvas']['layers']:
    for s in ly['shapes']:
        types.add(s['type'])
check('shape types present', types >= {'line', 'polyline', 'text'}, str(sorted(types)))

# The viewer must know every shape type that the serializer can emit.
import gtrace.draw.viewer as viewermod
js = viewermod.viewer_js()
for t in ['line', 'polyline', 'rectangle', 'circle', 'arc', 'text']:
    check("viewer handles '%s'" % t, "'%s'" % t in js)

print('--- gtrace self-consistency of the propagation convention ---')

# propagate(d), width(d) and R(d) all take a geometric distance and must
# advance q by that same d. This is what the viewer implements in JS.
tb = beam.GaussianBeam(q0=gauss.Rw2q(2.0, 1*mm), wl=1064*nm,
                       pos=[0, 0], dirAngle=0, n=1.45)
d = 0.03
tp = tb.copy()
tp.propagate(d)
check('propagate(d) advances q by d',
      abs(complex(tp.qx) - (complex(tb.qx) + d)) < 1e-15)
check('width(d) == propagate(d).width(0)',
      abs(tb.width(d)[0] - tp.width(0)[0]) < 1e-15)
check('R(d) == propagate(d).R(0)',
      abs(tb.R(d)[0] - tp.R(0)[0]) < 1e-12,
      '(%.9f vs %.9f)' % (tb.R(d)[0], tp.R(0)[0]))
check('R(d) == q2R(q + d)',
      abs(tb.R(d)[0] - gauss.q2R(tb.qx + d)) < 1e-12)
check('ROC does not depend on n for a given q',
      abs(beam.GaussianBeam(q0=gauss.Rw2q(2.0, 1*mm), wl=1064*nm,
                            pos=[0, 0], dirAngle=0, n=1.0).R(d)[0]
          - tb.R(d)[0]) < 1e-12)

print('--- reference values for the JS check ---')

# Sample points along a few beams; the JS side recomputes these from the
# embedded scene and must agree with gtrace.
def jnum(v):
    '''
    JSON.parse has no Infinity literal, so send it as a string and let
    the JS side check for a non-finite value.
    '''
    v = float(v)
    if np.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return v

samples = []
for i, b in enumerate(layout.beams):
    for frac in [0.0, 0.13, 0.5, 0.87, 1.0]:
        d = b.length * frac
        wx, wy = b.width(d)
        bb = b.copy()
        bb.propagate(d)
        # Take the ROC from the public API. q2R raises when the point is
        # exactly at a waist (Re(1/q) == 0), where the ROC is infinite.
        try:
            Rx, Ry = b.R(d)
        except ZeroDivisionError:
            Rx, Ry = np.inf, np.inf
        samples.append({
            'index': i,
            'd': float(d),
            'wx': jnum(wx), 'wy': jnum(wy),
            'Rx': jnum(Rx), 'Ry': jnum(Ry),
            'qx': [float(np.real(bb.qx)), float(np.imag(bb.qx))],
            'qy': [float(np.real(bb.qy)), float(np.imag(bb.qy))],
            'Gouyx': float(bb.Gouyx), 'Gouyy': float(bb.Gouyy),
            'optDist': float(bb.optDist),
        })

# A few off-beam points to check the hit test projection.
picks = []
rng = np.random.RandomState(0)
for i, b in enumerate(layout.beams[:8]):
    pos = np.asarray(b.pos, dtype=float)
    dv = np.asarray(b.dirVect, dtype=float)
    nv = np.array([-dv[1], dv[0]])
    for frac, off in [(0.3, 0.01), (0.7, -0.02), (-0.5, 0.0), (1.4, 0.005)]:
        p = pos + dv * (b.length * frac) + nv * off
        d_expected = min(max(b.length * frac, 0.0), b.length)
        foot = pos + dv * d_expected
        picks.append({'index': i,
                      'point': [float(p[0]), float(p[1])],
                      'd': float(d_expected),
                      'dist': float(np.hypot(*(p - foot)))})

with open(os.path.join(OUT, 'stage1_reference.json'), 'w') as f:
    json.dump({'scene': scene, 'samples': samples, 'picks': picks}, f)
print('  wrote stage1_reference.json (%d samples, %d picks)'
      % (len(samples), len(picks)))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
