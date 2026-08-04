'''
Stage 2 verification, browser side.

Loads the widget's ESM module the way anywidget does - as a real module,
via a blob URL, so the export itself is exercised - drives it with a
stand-in for the anywidget model, and checks what render() built and
what it does when the scene traitlet changes.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

import json
import os
import re
import subprocess
import sys

SP = WORK
CHROME = require_chrome()

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

from gtrace.draw.viewer import viewer_css

with open(os.path.join(SP, 'stage2_widget.mjs'), encoding='utf-8') as f:
    esm = f.read()
with open(os.path.join(SP, 'stage2_scenes.json')) as f:
    scenes = json.load(f)

def js(obj):
    return json.dumps(obj, ensure_ascii=True).replace('</', '<\\/')

PAGE = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html, body { margin: 0; height: 100%; }
#host { width: 900px; height: 600px; }
__CSS__
</style></head>
<body>
<div id="host"></div>
<div id="host2"></div>
<div id="out" style="display:none"></div>
<script>
var ESM_SRC = __ESM__;
var SCENES = __SCENES__;
</script>
<script type="module">
(async function () {
    var out = {error: null};
    function snapshot(el) {
        var v = el.gtraceViewer;
        return {
            root: el.querySelectorAll('.gt-root').length,
            svg: el.querySelectorAll('svg').length,
            geom: el.querySelectorAll('.gt-scene line, .gt-scene polyline, ' +
                                      '.gt-scene path, .gt-scene circle, ' +
                                      '.gt-scene rect').length,
            layerRows: el.querySelectorAll('.gt-layerrow').length,
            readoutRows: el.querySelectorAll('.gt-readout tr').length,
            heading: el.querySelectorAll('.gt-title').length,
            beams: v ? v.scene.beams.length : null,
            scale: v ? v.scale : null,
            cx: v ? v.cx : null,
            cy: v ? v.cy : null,
            listeners: v && v._listeners ? v._listeners.length : null
        };
    }
    try {
        // anywidget imports the ESM as a module; do the same.
        var url = URL.createObjectURL(
            new Blob([ESM_SRC], {type: 'text/javascript'}));
        var mod = await import(url);
        out.hasDefault = !!mod.default;
        out.hasRender = mod.default && typeof mod.default.render === 'function';

        // Stand-in for the anywidget model.
        var state = {scene: SCENES.a, height: 600};
        var handlers = {};
        var model = {
            get: function (k) { return state[k]; },
            set: function (k, v) {
                state[k] = v;
                (handlers['change:' + k] || []).forEach(function (f) { f(); });
            },
            on: function (ev, fn) {
                (handlers[ev] = handlers[ev] || []).push(fn);
            },
            off: function (ev, fn) {
                handlers[ev] = (handlers[ev] || []).filter(function (f) {
                    return f !== fn;
                });
            },
            // The real model has one; a stand-in that lacks it
            // would let a missing call pass unnoticed.
            save_changes: function () {}
        };

        var el = document.getElementById('host');
        var cleanup = mod.default.render({model: model, el: el});
        out.cleanupIsFunction = typeof cleanup === 'function';
        out.afterRender = snapshot(el);
        out.hostHeight = (el.querySelector('.gt-widget') || {}).style
            ? el.querySelector('.gt-widget').style.height : null;

        // Move the view, then push a new scene: the viewer must redraw
        // but keep where the user was looking.
        var v = el.gtraceViewer;
        v.scale *= 3; v.cx += 0.05; v.cy -= 0.02;
        v._applyTransform();
        var moved = snapshot(el);
        model.set('scene', SCENES.b);
        out.beforePush = moved;
        out.afterPush = snapshot(el);


        // Tear down.
        cleanup();
        out.afterCleanup = {
            root: el.querySelectorAll('.gt-root').length,
            listeners: v._listeners ? v._listeners.length : null
        };

        // --- mounting the way a notebook does ---
        // anywidget calls render() before the output area has been laid
        // out, so the element measures zero. Everything above mounts
        // into an element that is already in the document and sized,
        // which is why it never saw this: the initial fit was worked
        // out from a 1x1 view and left three orders of magnitude out.
        var det = document.createElement('div');
        det.style.width = '900px';
        det.style.height = '600px';
        var st2 = {scene: SCENES.a, height: 600};
        var h2 = {};
        var model2 = {
            get: function (k) { return st2[k]; },
            set: function (k, val) {
                st2[k] = val;
                (h2['change:' + k] || []).forEach(function (f) { f(); });
            },
            on: function (ev, fn) { (h2[ev] = h2[ev] || []).push(fn); },
            off: function () {},
            save_changes: function () {}
        };
        mod.default.render({model: model2, el: det});
        var v2 = det.gtraceViewer;
        out.detached = {width: v2.width, height: v2.height,
                        scale: v2.scale, pending: !!v2.fitPending};

        document.getElementById('host2').appendChild(det);
        await new Promise(function (r) { setTimeout(r, 300); });

        var bb2 = v2.bbox();
        var want = Math.min(v2.width / (bb2.maxx - bb2.minx),
                            v2.height / (bb2.maxy - bb2.miny)) * (1 - 2 * 0.06);
        out.attachedLater = {
            width: v2.width, height: v2.height,
            scale: v2.scale, wantScale: want,
            cx: v2.cx, cy: v2.cy,
            wantCx: (bb2.minx + bb2.maxx) / 2,
            wantCy: (bb2.miny + bb2.maxy) / 2,
            pending: !!v2.fitPending
        };
    } catch (e) {
        out.error = String((e && e.stack) || e);
    }
    document.getElementById('out').textContent = JSON.stringify(out);
})();
</script>
</body></html>
'''

page = PAGE.replace('__CSS__', viewer_css()) \
           .replace('__ESM__', js(esm)) \
           .replace('__SCENES__', js(scenes))

path = os.path.join(SP, 'stage2_page.html')
with open(path, 'w', encoding='utf-8') as f:
    f.write(page)

p = subprocess.run(
    [CHROME, '--headless=new', '--disable-gpu', '--window-size=1200,800',
     '--virtual-time-budget=6000', '--enable-logging=stderr', '--v=0',
     '--dump-dom', 'file:///' + path.replace('\\', '/')],
    capture_output=True, text=True, encoding='utf-8', errors='replace',
    timeout=120)

errs = [l.strip() for l in (p.stderr or '').splitlines()
        if 'CONSOLE' in l and ('Uncaught' in l or 'Error' in l)]
check('no console error', errs == [], '\n        '.join(errs[:4]))

m = re.search(r'<div id="out"[^>]*>(.*?)</div>', p.stdout or '', re.S)
if not m:
    print('  FAIL  the module produced no output')
    sys.exit(1)
payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
           .replace('&lt;', '<').replace('&gt;', '>'))
res = json.loads(payload)
check('module ran without exception', res['error'] is None,
      str(res['error'])[:300])

print('--- the ESM module ---')
check('has a default export', res.get('hasDefault'))
check('default export has render()', res.get('hasRender'))
check('render() returns a cleanup function', res.get('cleanupIsFunction'))

print('--- what render() built ---')
a = res.get('afterRender') or {}
check('viewer mounted in el', a.get('root') == 1, str(a.get('root')))
check('svg built', a.get('svg') == 1)
check('geometry drawn', (a.get('geom') or 0) > 20, '(%s)' % a.get('geom'))
check('layer list built', (a.get('layerRows') or 0) >= 5,
      '(%s)' % a.get('layerRows'))
check('readout table built', (a.get('readoutRows') or 0) >= 15,
      '(%s)' % a.get('readoutRows'))
# The side bar has no heading: in a notebook the layout is labelled by
# the cell that made it, so a line repeating its name is a line not
# spent on the readout.
check('no heading in the side bar', a.get('heading') == 0,
      str(a.get('heading')))
check('height taken from the model', res.get('hostHeight') == '600px',
      str(res.get('hostHeight')))
check('scene loaded', a.get('beams') == len(scenes['a']['beams']),
      '(%s of %d)' % (a.get('beams'), len(scenes['a']['beams'])))

print('--- pushing a new scene ---')
before, after = res.get('beforePush') or {}, res.get('afterPush') or {}
check('the new scene is shown',
      after.get('beams') == len(scenes['b']['beams']),
      '(%s -> %s beams)' % (before.get('beams'), after.get('beams')))
check('the drawing was rebuilt', after.get('geom') != before.get('geom'),
      '(%s -> %s shapes)' % (before.get('geom'), after.get('geom')))
check('zoom is preserved', after.get('scale') == before.get('scale'),
      '(%s -> %s)' % (before.get('scale'), after.get('scale')))
check('pan is preserved',
      after.get('cx') == before.get('cx') and after.get('cy') == before.get('cy'),
      '(%s,%s -> %s,%s)' % (before.get('cx'), before.get('cy'),
                            after.get('cx'), after.get('cy')))
check('the layer list was rebuilt, not duplicated',
      after.get('layerRows') == before.get('layerRows'),
      '(%s -> %s)' % (before.get('layerRows'), after.get('layerRows')))
check('listeners were not re-registered',
      after.get('listeners') == before.get('listeners'),
      '(%s -> %s)' % (before.get('listeners'), after.get('listeners')))

print('--- cleanup ---')
c = res.get('afterCleanup') or {}
check('the viewer DOM is removed', c.get('root') == 0, str(c.get('root')))
check('every listener is released', c.get('listeners') == 0,
      str(c.get('listeners')))

print('--- rendered before the view has a size (what a notebook does) ---')
d = res.get('detached') or {}
l = res.get('attachedLater') or {}
check('the view really does measure nothing at render time',
      d.get('width') == 1 and d.get('height') == 1,
      '(%sx%s)' % (d.get('width'), d.get('height')))
check('the fit is deferred rather than done against nothing',
      d.get('pending') is True)
check('the view has a size once it is in the document',
      (l.get('width') or 0) > 100 and (l.get('height') or 0) > 100,
      '(%sx%s)' % (l.get('width'), l.get('height')))
check('the deferred fit ran', l.get('pending') is False)

scale, want = l.get('scale'), l.get('wantScale')
check('and the scene is framed',
      scale is not None and want and abs(scale / want - 1) < 1e-6,
      '(scale %s, a fit would give %s)' % (scale, want))
check('centred on the scene',
      l.get('cx') is not None
      and abs(l['cx'] - l['wantCx']) < 1e-9
      and abs(l['cy'] - l['wantCy']) < 1e-9,
      '((%s,%s) vs (%s,%s))' % (l.get('cx'), l.get('cy'),
                                l.get('wantCx'), l.get('wantCy')))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
