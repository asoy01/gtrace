'''
Stage 2 verification, Python side: the anywidget front end.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json
import os
import sys

import numpy as np

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.layout import OpticalLayout, TraceRules, _in_notebook
from gtrace.draw.viewer import widget as wmod
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

#{{{ Test layout

b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                       pos=[0, 0], dirAngle=0, name='b0')

def M(name, c, a, roc=0.0, rh=0.99, th=0.01):
    return opt.Mirror(HRcenter=c, normAngleHR=a, diameter=10*cm,
                      thickness=5*cm, wedgeAngle=deg2rad(0.25),
                      inv_ROC_HR=roc, Refl_HR=rh, Trans_HR=th,
                      Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45, name=name)

M1 = M('M1', [0.5, 0.0], deg2rad(135))
M2 = M('M2', [0.5, 0.4], deg2rad(-45), 1.0/2.0)
M3 = M('M3', [0.9, 0.4], deg2rad(180), 0.0, 0.9, 0.1)

layout = OpticalLayout(optics=[M1, M2, M3], sources=[b0],
                       rules=TraceRules(order=5, power_threshold=1e-4),
                       name='Stage 2 layout')

#}}}

print('--- availability ---')
check('anywidget is available', wmod.widget_available())
check('not detected as a notebook when run as a script', not _in_notebook())

print('--- ESM module ---')
esm = wmod.widget_esm()
check('ESM carries the viewer core', 'GTraceViewer' in esm
      and 'beamParamsAt' in esm)
check('ESM carries the widget wrapper', 'function render(' in esm)
check('ESM has a default export', esm.rstrip().endswith('export default { render };'))
check('only one export statement', esm.count('export default') == 1)
check('no bare import to resolve', '\nimport ' not in esm)
check('CSS available', '.gt-root' in wmod.viewer_css())

with open(os.path.join(OUT, 'stage2_widget.mjs'), 'w', encoding='utf-8') as f:
    f.write(esm)

print('--- widget construction ---')
w = layout.widget()
check('widget created', w is not None, type(w).__name__)
check('title defaults to the layout name', w.title == 'Stage 2 layout', w.title)
check('height default', w.height == 520, str(w.height))
check('_esm set on the class', len(w._esm) > 1000, '(%d chars)' % len(w._esm))
check('_css set on the class', len(w._css) > 500, '(%d chars)' % len(w._css))

scene = w.scene
check('scene has the expected keys',
      set(scene.keys()) == {'canvas', 'beams', 'optics', 'display'},
      str(list(scene.keys())))
check('scene carries the beams', len(scene['beams']) == len(layout.beams),
      '(%d)' % len(scene['beams']))

# The traitlet is synchronized as JSON, so it must survive a round trip.
try:
    round_tripped = json.loads(json.dumps(scene))
    ok = round_tripped == scene
except (TypeError, ValueError) as e:
    ok = False
check('scene is JSON serializable', ok)

print('--- update() pushes a new scene ---')
before = w.scene
n_before = len(before['beams'])
first_len_before = before['beams'][0]['length']

M1.HRcenter = [0.6, 0.0]          # as a GUI drag would
w.update()
after = w.scene
check('scene object replaced', after is not before)
check('geometry followed the mirror',
      abs(after['beams'][0]['length'] - 0.6) < 1e-9,
      '(%.4f -> %.4f)' % (first_len_before, after['beams'][0]['length']))
# Moved to x = 0.6 the reflected beam misses M2, which sits at x = 0.5,
# so the whole downstream branch disappears. The trace is re-run, not
# patched, so the new scene must show that.
check('the trace was really re-run',
      0 < len(after['beams']) < n_before,
      '(%d -> %d beams)' % (n_before, len(after['beams'])))

M1.HRcenter = [0.5, 0.0]
w.update()
check('and back again', abs(w.scene['beams'][0]['length'] - 0.5) < 1e-9
      and len(w.scene['beams']) == n_before,
      '(%d beams)' % len(w.scene['beams']))

check('update() forwards draw kwargs',
      len([ly for ly in w.update(drawStrayWidth=False)
           .scene['canvas']['layers']
           if ly['name'] == 'stray_beam_width' and not ly['shapes']]) == 1)
w.update()

print('--- widget without a layout ---')
detached = wmod.LayoutViewer(scene=layout.scene_dict(), title='detached')
try:
    detached.update()
    raised = False
except ValueError:
    raised = True
check('update() on a detached widget raises', raised)
check('assigning .scene still works',
      (setattr(detached, 'scene', layout.scene_dict()) or True))

print('--- show() backend selection ---')
check('show(backend="widget") returns a widget',
      type(layout.show(backend='widget')).__name__ == type(w).__name__)

f_html = os.path.join(OUT, 'stage2_show.html')
check('show(backend="html") returns the filename',
      layout.show(f_html, browser=False, backend='html') == f_html
      and os.path.exists(f_html))

check('auto backend is html outside a notebook',
      layout.show(os.path.join(OUT, 'stage2_auto.html'),
                  browser=False) == os.path.join(OUT, 'stage2_auto.html'))

try:
    layout.show(backend='bogus')
    raised = False
except ValueError:
    raised = True
check('an unknown backend raises', raised)

print('--- data for the browser check ---')
scene_a = layout.scene_dict()
M1.HRcenter = [0.7, 0.0]
layout.beams = None
scene_b = layout.scene_dict()
M1.HRcenter = [0.5, 0.0]
layout.beams = None
with open(os.path.join(OUT, 'stage2_scenes.json'), 'w') as f:
    json.dump({'a': scene_a, 'b': scene_b}, f)
print('  wrote stage2_scenes.json (%d and %d beams)'
      % (len(scene_a['beams']), len(scene_b['beams'])))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
