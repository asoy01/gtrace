'''
Stage 1 verification, interaction path.

Takes the generated HTML, appends a driver script that exercises the
viewer the way a user would (hover over a beam, click to pin, toggle a
layer) and dumps the resulting readout back into the DOM. Headless
Chrome then runs it and this script checks the values against gtrace.
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

import numpy as np

import gtrace.optics.gaussian as gauss

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

#{{{ Pick the test targets from the scene

with open(os.path.join(SP, 'stage1_reference.json')) as f:
    scene = json.load(f)['scene']

beams = scene['beams']

def probe_point(b, frac):
    return [b['pos'][0] + b['dirVect'][0] * b['length'] * frac,
            b['pos'][1] + b['dirVect'][1] * b['length'] * frac]

def overlapping(i, frac):
    '''
    How many beams pass within 1 mm of the probe point of beam i.
    Counter-propagating beams share a line, so this is often > 1.
    '''
    p = probe_point(beams[i], frac)
    n = 0
    for b in beams:
        pos = np.array(b['pos'])
        dv = np.array(b['dirVect'])
        t = np.clip(np.dot(np.array(p) - pos, dv), 0.0, b['length'])
        if np.hypot(*(np.array(p) - (pos + dv * t))) < 1e-3:
            n += 1
    return n

# Probe beams at points where they are the only beam, so that the hover
# test is unambiguous.
targets = []
for i, b in enumerate(beams):
    if len(targets) >= 4:
        break
    if b['length'] > 0.1 and overlapping(i, 0.4) == 1:
        targets.append({'index': i, 'frac': 0.4})
check('found unambiguous probe targets', len(targets) == 4,
      '(%d)' % len(targets))

# And find a point where beams DO overlap, to test the click cycling.
bundle = None
for i, b in enumerate(beams):
    if b['length'] > 0.1 and overlapping(i, 0.4) > 1:
        bundle = {'index': i, 'frac': 0.4, 'count': overlapping(i, 0.4)}
        break
check('found a bundle of overlapping beams', bundle is not None,
      str(bundle))

tests = json.dumps(targets)
bundle_js = json.dumps(bundle)

#}}}

#{{{ Build the instrumented page

with open(os.path.join(SP, 'stage1_view.html'), encoding='utf-8') as f:
    html = f.read()

driver = '''
<div id="gt-test-out" style="display:none"></div>
<script>
(function () {
    var v = window.gtraceViewer;
    var TESTS = ''' + tests + ''';
    var BUNDLE = ''' + bundle_js + ''';
    var out = {hover: [], pin: null, layer: null, cycle: null, error: null};

    function readRows() {
        var rows = {};
        for (var k in v.cells) {
            rows[k] = v.cells[k].map(function (c) { return c.textContent; });
        }
        return rows;
    }
    function screenOf(b, d) {
        return v.sceneToScreen(b.pos[0] + b.dirVect[0] * d,
                               b.pos[1] + b.dirVect[1] * d);
    }
    /* Tip and base midpoint of the direction arrowhead, in screen px. */
    function arrowVector() {
        var m = (v.arrow.getAttribute('d') || '').match(/-?[\\d.]+/g);
        if (!m || m.length < 6) { return null; }
        var n = m.map(Number);
        var tip = [n[0], n[1]];
        var base = [(n[2] + n[4]) / 2, (n[3] + n[5]) / 2];
        return {tip: tip, base: base,
                dx: tip[0] - base[0], dy: tip[1] - base[1],
                shown: v.arrow.style.display !== 'none'};
    }

    try {
        // --- hover over each target beam ---
        TESTS.forEach(function (t) {
            var b = v.scene.beams[t.index];
            var d = b.length * t.frac;
            var p = screenOf(b, d);
            v._onHover(p[0], p[1]);
            out.hover.push({index: t.index, wantD: d,
                            beam: v.hover ? v.hover.beam.name : null,
                            gotD: v.hover ? v.hover.d : null,
                            layer: v.hover ? v.hover.beam.layer : null,
                            markerShown: v.marker.style.display !== 'none',
                            arrow: arrowVector(),
                            dirVect: b.dirVect,
                            rows: readRows()});
        });

        // --- click to pin, then hover far away: the readout must stay ---
        var b0 = v.scene.beams[TESTS[0].index];
        var d0 = b0.length * 0.25;
        var p0 = screenOf(b0, d0);
        v._onClick(p0[0], p0[1]);
        var pinnedRows = readRows();
        v._onHover(5, 5);                       // empty corner of the canvas
        out.pin = {pinnedBeam: v.pinned ? v.pinned.beam.name : null,
                   pinnedD: v.pinned ? v.pinned.d : null,
                   wantD: d0,
                   rowsAfterPin: pinnedRows,
                   rowsAfterHoverAway: readRows(),
                   label: v.pinLabel.textContent};
        v._onClick(5, 5);                       // click empty space to unpin
        out.pin.unpinned = v.pinned === null;

        // --- repeated clicks must cycle through overlapping beams ---
        var bb = v.scene.beams[BUNDLE.index];
        var pb = screenOf(bb, bb.length * BUNDLE.frac);
        var seen = [], labels = [], arrows = [], dirs = [];
        for (var i = 0; i < BUNDLE.count + 1; i++) {
            v._onClick(pb[0], pb[1]);
            seen.push(v.pinned ? v.pinned.beam.name : null);
            labels.push(v.pinLabel.textContent);
            arrows.push(arrowVector());
            dirs.push(v.pinned ? v.pinned.beam.dirVect : null);
        }
        out.cycle = {wantBeam: bb.name, wantCount: BUNDLE.count,
                     seen: seen, labels: labels,
                     arrows: arrows, dirs: dirs,
                     reportedCount: v.pinned ? v.pinned.count : null};
        // A click somewhere else must restart the cycle.
        v._onClick(5, 5);
        v._onClick(pb[0], pb[1]);
        out.cycle.afterReset = v.pinned ? v.pinned.beam.name : null;

        // --- hiding a layer must remove its beams from the hit test ---
        var lay = b0.layer;
        v.setLayerVisible(lay, false);
        var p1 = screenOf(b0, d0);
        v._onHover(p1[0], p1[1]);
        out.layer = {name: lay, hitWhileHidden: v.hover ? v.hover.beam.name : null,
                     geomHidden: v.layerGroups[lay].geom.style.display === 'none'};
        v.setLayerVisible(lay, true);
        v._onHover(p1[0], p1[1]);
        out.layer.hitAfterShow = v.hover ? v.hover.beam.name : null;

        // --- zoom about the cursor must keep that scene point fixed ---
        var probe = [300, 300];
        var before = v.screenToScene(probe[0], probe[1]);
        var ev = {clientX: probe[0], clientY: probe[1], deltaY: -240,
                  preventDefault: function () {}};
        v.svg.dispatchEvent(new WheelEvent('wheel',
            {clientX: probe[0], clientY: probe[1], deltaY: -240,
             bubbles: true, cancelable: true}));
        var after = v.screenToScene(probe[0], probe[1]);
        out.zoom = {before: before, after: after, scale: v.scale};

        // --- swatch colors and empty-layer flags ---
        out.swatches = {};
        out.emptyRows = {};
        Array.prototype.forEach.call(
            v.layerBody.querySelectorAll('.gt-layerrow'), function (row) {
                var n = row.querySelector('.gt-layername').textContent;
                out.swatches[n] = row.querySelector('.gt-swatch').style.background;
                out.emptyRows[n] = row.className.indexOf('gt-layer-empty') >= 0;
            });
    } catch (e) {
        out.error = String(e && e.stack || e);
    }

    document.getElementById('gt-test-out').textContent = JSON.stringify(out);
})();
</script>
</body>'''

html = html.replace('</body>', driver)
inst = os.path.join(SP, 'stage1_interact.html')
with open(inst, 'w', encoding='utf-8') as f:
    f.write(html)

#}}}

#{{{ Run it

cmd = [CHROME, '--headless=new', '--disable-gpu', '--window-size=1400,900',
       '--virtual-time-budget=5000', '--enable-logging=stderr', '--v=0',
       '--dump-dom', 'file:///' + inst.replace('\\', '/')]
p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                   errors='replace', timeout=120)

errs = [l.strip() for l in (p.stderr or '').splitlines()
        if 'CONSOLE' in l and ('Uncaught' in l or 'Error' in l)]
check('no console error', errs == [], '\n        '.join(errs[:4]))

m = re.search(r'<div id="gt-test-out"[^>]*>(.*?)</div>', p.stdout or '', re.S)
if not m:
    print('  FAIL  driver produced no output')
    sys.exit(1)

# The DOM dump HTML-escapes the JSON payload.
payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
           .replace('&lt;', '<').replace('&gt;', '>'))
res = json.loads(payload)
check('driver ran without exception', res['error'] is None, str(res['error'])[:200])

#}}}

print('--- hover readout ---')
for h in res['hover']:
    b = beams[h['index']]
    tag = "beam '%s'" % b['name']
    check(tag + ' picked', h['beam'] == b['name'], '(got %s)' % h['beam'])
    check(tag + ' distance', h['gotD'] is not None
          and abs(h['gotD'] - h['wantD']) < 1e-6 * max(1.0, h['wantD']),
          '(got %s want %s)' % (h['gotD'], h['wantD']))
    check(tag + ' marker shown', h['markerShown'])

    # The arrowhead must point the way the beam travels. The screen has
    # y pointing down, so the expected screen direction is (dx, -dy).
    a = h['arrow']
    check(tag + ' arrow shown', a is not None and a['shown'])
    if a:
        norm = np.hypot(a['dx'], a['dy'])
        want = np.array([h['dirVect'][0], -h['dirVect'][1]])
        got = np.array([a['dx'], a['dy']]) / norm if norm else np.zeros(2)
        check(tag + ' arrow points along the beam',
              norm > 1.0 and np.dot(got, want) > 0.9999,
              '(cos = %.6f)' % float(np.dot(got, want)))

    rows = h['rows']
    check(tag + ' beam name in panel', rows['beam'][0] == b['name'])
    check(tag + ' layer in panel', rows['layer'][0] == b['layer'])
    check(tag + ' no empty field',
          all(v != '-' for k, vs in rows.items() for v in vs),
          str([k for k, vs in rows.items() if any(v == '-' for v in vs)]))

    # The displayed radius must match gtrace, to the printed precision.
    d = h['gotD']
    q = complex(b['qx'][0] + d, b['qx'][1])
    k = 2 * np.pi * b['n'] / b['wl']
    wx = np.sqrt(-2.0 / (k * np.imag(1.0 / q)))
    shown = rows['w'][0]
    unit = {'m': 1.0, 'mm': 1e-3, 'µm': 1e-6, 'nm': 1e-9, 'km': 1e3}
    val, un = shown.split(' ')
    check(tag + ' radius shown matches gtrace',
          abs(float(val) * unit[un] - wx) <= 1e-4 * wx,
          '(panel %s vs %.6e m)' % (shown, wx))

    Rshown = rows['R'][0]
    if Rshown not in ('∞', '-∞'):
        Rval, Run = Rshown.split(' ')
        Rx = 1.0 / np.real(1.0 / q)
        check(tag + ' ROC shown matches gtrace',
              abs(float(Rval) * unit[Run] - Rx) <= 1e-4 * abs(Rx),
              '(panel %s vs %.6e m)' % (Rshown, Rx))

print('--- pin / unpin ---')
pin = res['pin']
check('click pins a beam', pin['pinnedBeam'] is not None, str(pin['pinnedBeam']))
check('pinned distance', pin['pinnedD'] is not None
      and abs(pin['pinnedD'] - pin['wantD']) < 1e-6 * max(1.0, pin['wantD']),
      '(got %s want %s)' % (pin['pinnedD'], pin['wantD']))
check('readout frozen while pinned',
      pin['rowsAfterPin'] == pin['rowsAfterHoverAway'])
check("panel shows 'pinned'", pin['label'] == 'pinned', repr(pin['label']))
check('click on empty space unpins', pin['unpinned'])

print('--- cycling through overlapping beams ---')
cy = res['cycle']
check('reports how many beams share the point',
      cy['reportedCount'] == cy['wantCount'],
      '(got %s want %s)' % (cy['reportedCount'], cy['wantCount']))
check('each beam of the bundle is reachable',
      len(set(cy['seen'][:cy['wantCount']])) == cy['wantCount'],
      str(cy['seen']))
check('the probed beam is among them', cy['wantBeam'] in cy['seen'],
      '(%s not in %s)' % (cy['wantBeam'], cy['seen']))
check('the cycle wraps around', cy['seen'][cy['wantCount']] == cy['seen'][0],
      str(cy['seen']))
check('panel shows the position in the bundle',
      all(re.search(r'\d+/%d$' % cy['wantCount'], s) for s in cy['labels']),
      str(cy['labels']))
check('clicking elsewhere restarts the cycle',
      cy['afterReset'] == cy['seen'][0],
      '(got %s want %s)' % (cy['afterReset'], cy['seen'][0]))

# The arrow is what makes overlapping beams distinguishable, so it has
# to track the beam that is actually selected at each step.
for i, (a, dv) in enumerate(zip(cy['arrows'], cy['dirs'])):
    norm = np.hypot(a['dx'], a['dy']) if a else 0.0
    got = np.array([a['dx'], a['dy']]) / norm if norm else np.zeros(2)
    want = np.array([dv[0], -dv[1]])
    check('cycle step %d: arrow follows the selected beam' % i,
          norm > 1.0 and np.dot(got, want) > 0.9999,
          '(cos = %.6f)' % float(np.dot(got, want)))

# Counter-propagating beams must give opposite arrows.
d0 = np.array(cy['dirs'][0])
opposed = [i for i, d in enumerate(cy['dirs'][:cy['wantCount']])
           if np.dot(d0, np.array(d)) < -0.99]
check('the bundle holds a counter-propagating beam', opposed != [],
      str(cy['dirs'][:cy['wantCount']]))
if opposed:
    a0, a1 = cy['arrows'][0], cy['arrows'][opposed[0]]
    v0 = np.array([a0['dx'], a0['dy']]) / np.hypot(a0['dx'], a0['dy'])
    v1 = np.array([a1['dx'], a1['dy']]) / np.hypot(a1['dx'], a1['dy'])
    check('the arrow flips for it', np.dot(v0, v1) < -0.9999,
          '(cos = %.6f)' % float(np.dot(v0, v1)))

print('--- layer visibility ---')
lay = res['layer']
check('hidden layer is not hit tested', lay['hitWhileHidden'] is None,
      str(lay['hitWhileHidden']))
check('hidden layer geometry is hidden', lay['geomHidden'])
check('beam is hit again after re-showing', lay['hitAfterShow'] is not None)

print('--- zoom about the cursor ---')
z = res['zoom']
check('wheel changed the scale', z['scale'] > 0)
check('scene point under the cursor is preserved',
      abs(z['after'][0] - z['before'][0]) < 1e-9
      and abs(z['after'][1] - z['before'][1]) < 1e-9,
      '(%s -> %s)' % (z['before'], z['after']))

print('--- layer colors on the white background ---')
sw = res['swatches']
check('all layers listed', len(sw) == len(scene['canvas']['layers']),
      '(%d of %d)' % (len(sw), len(scene['canvas']['layers'])))

def luminance(css):
    m = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', css)
    if not m:
        return None
    def lin(c):
        c = int(c) / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = [lin(x) for x in m.groups()]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# Every drawn color must reach at least 3:1 against white, i.e. have a
# relative luminance of 0.30 or less. Rounding back to 8-bit sRGB can
# overshoot the target by a fraction of one step, hence the margin.
lums = {k: luminance(v) for k, v in sw.items()}
check('every layer color is readable on white',
      all(l is not None and l <= 0.305 for l in lums.values()),
      str({k: round(v, 4) for k, v in lums.items() if v is not None}))

# Colors that were already dark enough must pass through untouched.
by_name = {ly['name']: ly['color'] for ly in scene['canvas']['layers']}
check('black stays black', sw.get('Mirrors') == 'rgb(0, 0, 0)',
      str(sw.get('Mirrors')))
check('red is not altered', sw.get('main_beam') == 'rgb(255, 0, 0)',
      str(sw.get('main_beam')))
check('pure green is darkened',
      by_name.get('stray_beam') == [0, 255, 0]
      and sw.get('stray_beam') != 'rgb(0, 255, 0)',
      str(sw.get('stray_beam')))

print('--- empty layers ---')
empty = [ly['name'] for ly in scene['canvas']['layers'] if not ly['shapes']]
check('the empty layer is flagged in the list',
      all(res['emptyRows'].get(n) for n in empty),
      'empty=%s flagged=%s' % (empty, res['emptyRows']))
check('non-empty layers are not flagged',
      not any(v for k, v in res['emptyRows'].items() if k not in empty),
      str(res['emptyRows']))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
