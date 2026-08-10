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

        // --- a height of zero means "work one out" ---
        // A cell output is a letterbox and a bench drawing is not, so
        // with nothing said the viewer makes itself as tall as it is
        // wide. Only the browser knows the width, and on the first pass
        // it does not know it either: this mounts detached, exactly as
        // a notebook does, so the height has to be settled by the same
        // observer that settles the fit.
        function autoCase(width, name) {
            var box = document.createElement('div');
            box.style.width = width + 'px';
            var st = {scene: SCENES.a, height: 0};
            var hh = {};
            var m = {
                get: function (k) { return st[k]; },
                set: function (k, val) {
                    st[k] = val;
                    (hh['change:' + k] || []).forEach(function (f) { f(); });
                },
                on: function (ev, fn) { (hh[ev] = hh[ev] || []).push(fn); },
                off: function () {},
                save_changes: function () {}
            };
            var stop = mod.default.render({model: m, el: box});
            var before = box.querySelector('.gt-widget').style.height;
            document.getElementById('host2').appendChild(box);
            return {box: box, model: m, state: st, stop: stop,
                    detachedHeight: before, name: name};
        }
        var narrow = autoCase(420, 'narrow');
        var wide = autoCase(1600, 'wide');
        await new Promise(function (r) { setTimeout(r, 300); });

        function heightOf(c) {
            return parseFloat(
                c.box.querySelector('.gt-widget').style.height) || 0;
        }
        out.auto = {
            viewport: globalThis.innerHeight,
            floor: globalThis.GTraceViewer.MIN_HEIGHT,
            narrowDetached: narrow.detachedHeight,
            narrow: heightOf(narrow),
            wide: heightOf(wide),
            // Nothing is written back: the height is still "work it
            // out", so a pane that is resized later squares up again.
            narrowModel: narrow.state.height,
            wideModel: wide.state.height
        };
        // Narrower again, and it follows.
        // Following a later resize is the same resolver run again. The
        // observer that runs it is the one that resolved the height
        // above, so what is left to check is what it comes to - and
        // that is driven directly here rather than waited on: this
        // headless run delivers no further resize callbacks once the
        // page has settled, as the probe records.
        var probe = {fired: 0};
        if (globalThis.ResizeObserver) {
            var ro = new ResizeObserver(function () { probe.fired++; });
            ro.observe(narrow.box);
        }
        out.auto.observed = typeof narrow.box.gtraceApplyHeight === 'function';
        narrow.box.style.width = '300px';
        narrow.box.gtraceApplyHeight();
        out.auto.narrowed = heightOf(narrow);
        out.auto.narrowedWidth = narrow.box.getBoundingClientRect().width;
        out.auto.probe = probe;

        // Until something sets a height, which settles it. That is what
        // a drag of the grip does.
        narrow.model.set('height', 640);
        out.auto.afterSet = heightOf(narrow);
        narrow.box.style.width = '900px';
        narrow.box.gtraceApplyHeight();
        out.auto.afterSetThenResized = heightOf(narrow);

        narrow.stop();
        wide.stop();
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

print('--- a height of zero means "work one out" ---')
a = res.get('auto') or {}
# The height is taken from the width of the *drawing*: the side panel is
# a fixed 380 px of the widget and is a column of numbers, not part of
# the picture, so squaring the whole thing would make the drawing itself
# taller than it is wide. 520 is the height the widget used to be fixed
# at, and the floor the cap never goes below: an embedder reporting a
# nonsense viewport should leave the widget the size it used to be
# rather than at nothing.
SIDE, NARROW = 380, 700
cap = max((a.get('viewport') or 0) * 0.70, 520)
def want(width):
    drawing = width - SIDE if width > NARROW else width
    return max(240, min(drawing, cap))
# The first pass measures nothing - anywidget renders before the output
# area is laid out - so the height cannot be worked out there either.
check('the height it used to be fixed at stands in until then',
      a.get('narrowDetached') == '520px', str(a.get('narrowDetached')))
# Below the breakpoint the side panel stacks under the drawing rather
# than standing beside it, so there the drawing has the whole width.
check('a narrow output, where the panel stacks, is as tall as it is wide',
      abs((a.get('narrow') or 0) - want(420)) < 1.5,
      '%s for a width of 420 (cap %.0f)' % (a.get('narrow'), cap))
# A maximized cell is wider than the window is tall, and a viewer whose
# bottom edge is below the fold is unusable in its own way.
check('a wide one is capped to the window rather than the width',
      abs((a.get('wide') or 0) - cap) < 1.5,
      '%s for a width of 1600 (cap %.0f)' % (a.get('wide'), cap))
check('  which is shorter than the width it would have taken',
      (a.get('wide') or 0) < 1600 - SIDE, str(a.get('wide')))
# The height above was resolved by the widget's own observer, which is
# the wiring that matters: anywidget renders before the output area is
# laid out, so that run is the only one that can settle it. What a later
# resize comes to is the same resolver run again, driven directly here -
# this headless run delivers no further resize callbacks once the page
# has settled, which the probe records rather than leaves to be guessed.
check('the resolver is reachable', a.get('observed') is True)
check('and it follows the width when the pane is resized',
      abs((a.get('narrowed') or 0) - want(300)) < 1.5,
      '%s for a measured width of %s' % (a.get('narrowed'),
                                         a.get('narrowedWidth')))
check('  never going below the floor the grip respects',
      (a.get('narrowed') or 0) >= (a.get('floor') or 240),
      '%s vs %s' % (a.get('narrowed'), a.get('floor')))
# Nothing is written back while it is being worked out. The height a
# drag settles on is written back by the grip, and that is what stops it.
check('nothing is written back while it is working it out',
      a.get('narrowModel') == 0 and a.get('wideModel') == 0,
      '%s / %s' % (a.get('narrowModel'), a.get('wideModel')))
check('a height that is set is used as it stands',
      a.get('afterSet') == 640, str(a.get('afterSet')))
check('  and settles it: the width no longer decides',
      a.get('afterSetThenResized') == 640,
      str(a.get('afterSetThenResized')))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
