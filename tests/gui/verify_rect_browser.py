'''
A turned rectangle in the page that is actually written: the SVG.

Everything else about the turn is checked where the numbers are -
gtrace works the corners out, verify_rect.js holds viewer.js to the
same ones. None of that reaches the last step, which is an SVG <rect>
with a transform on it: a rect is drawn from a corner and two sizes,
so the turn is the one part of the shape the element cannot carry by
itself, and a transform that turned the wrong way, or about the wrong
point, would draw a rectangle in the wrong place with every number
right.

So this loads the page renderHTML writes and reads the corners back
out of the browser's own matrix - getScreenCTM, which is what the
rectangle is actually painted through - and holds them against the
corners gtrace says the shape has, mapped to the screen by the
viewer's own sceneToScreen. Two ways of asking where the drawing is,
which have to agree.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

import json
import re
import subprocess

import numpy as np

import gtrace.draw as draw
from gtrace.layout import OpticalLayout
from gtrace.mechanics import Mechanics

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

def run_chrome(html_path, extra):
    url = 'file:///' + html_path.replace('\\', '/')
    cmd = [CHROME, '--headless=new', '--disable-gpu', '--window-size=1400,900',
           '--virtual-time-budget=4000', '--enable-logging=stderr', '--v=0'] \
        + extra + [url]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=120)

def console_errors(stderr):
    bad = []
    for line in (stderr or '').splitlines():
        if 'CONSOLE' in line and ('Uncaught' in line or 'Error' in line):
            bad.append(line.strip())
        if 'ERR_FILE_NOT_FOUND' in line or 'ERR_CONNECTION' in line:
            bad.append(line.strip())
    return bad

# Three rectangles, far enough apart that each is the only one anywhere
# near it: square to the axes, turned about its own middle, and turned
# about a point somewhere else.
SHAPES = [
    draw.Rectangle([0.0, 0.0], 0.20, 0.10),
    draw.Rectangle([1.0, 0.0], 0.20, 0.10, angle=0.6),
    draw.Rectangle([2.0, 0.0], 0.20, 0.10, angle=-0.9, pivot=[2.30, 0.25]),
]

L = OpticalLayout(optics=[], sources=[], name='RectPage')
L.add_mechanics(Mechanics(shapes=SHAPES, name='R1'))
page = os.path.join(WORK, 'rect_page.html')
L.render_html(page)

want = [[[float(v) for v in c] for c in s.corners()] for s in SHAPES]

driver = '''
<div id="gt-test-out" style="display:none"></div>
<script>
(function () {
    var out = {error: null};
    try {
        var v = window.gtraceViewer;
        var want = %s;
        var rects = [].slice.call(
            v.sceneGroup.querySelectorAll('rect'));
        out.count = rects.length;
        out.transforms = rects.map(function (r) {
            return r.getAttribute('transform');
        });
        // Where the browser paints each corner, through the matrix the
        // element is actually drawn with.
        out.painted = rects.map(function (r) {
            var m = r.getScreenCTM();
            var x = parseFloat(r.getAttribute('x'));
            var y = parseFloat(r.getAttribute('y'));
            var w = parseFloat(r.getAttribute('width'));
            var h = parseFloat(r.getAttribute('height'));
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                .map(function (p) {
                    var q = v.svg.createSVGPoint();
                    q.x = p[0]; q.y = p[1];
                    q = q.matrixTransform(m);
                    return [q.x, q.y];
                });
        });
        // Where gtrace says those corners are, put on the screen by the
        // viewer itself.
        var box = v.svg.getBoundingClientRect();
        out.wanted = want.map(function (cs) {
            return cs.map(function (p) {
                var q = v.sceneToScreen(p[0], p[1]);
                return [q[0] + box.left, q[1] + box.top];
            });
        });
    } catch (e) { out.error = String((e && e.stack) || e); }
    document.getElementById('gt-test-out').textContent = JSON.stringify(out);
})();
</script>
</body>''' % json.dumps(want)

with open(page, encoding='utf-8') as f:
    html = f.read().replace('</body>', driver)
inst = os.path.join(WORK, 'rect_page_driven.html')
with open(inst, 'w', encoding='utf-8') as f:
    f.write(html)

p = run_chrome(inst, ['--dump-dom'])
check('no console error', console_errors(p.stderr) == [],
      '\n        '.join(console_errors(p.stderr)[:3]))

m = re.search(r'<div id="gt-test-out"[^>]*>(.*?)</div>', p.stdout or '', re.S)
check('the driver ran', m is not None)
if m is None:
    print()
    print('%d passed, %d failed' % (npass, nfail))
    sys.exit(1)

payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
           .replace('&lt;', '<').replace('&gt;', '>'))
res = json.loads(payload)
check('  without an error', res['error'] is None, str(res['error'])[:300])
check('every rectangle is drawn as a rect', res.get('count') == len(SHAPES),
      '(%s)' % res.get('count'))

tf = res.get('transforms') or []
check('a rectangle square to the axes carries no transform',
      len(tf) == 3 and tf[0] is None, str(tf[0]))
check('a turned one carries a rotate about its pivot',
      len(tf) == 3 and tf[1] is not None and tf[1].startswith('rotate(')
      and tf[2] is not None and tf[2].startswith('rotate('),
      str(tf[1:]))

painted = res.get('painted') or []
wanted = res.get('wanted') or []
for i, s in enumerate(SHAPES):
    if i >= len(painted) or i >= len(wanted):
        check('rectangle %d was painted' % i, False)
        continue
    got = np.array(painted[i], dtype='float64')
    exp = np.array(wanted[i], dtype='float64')
    off = np.abs(got - exp).max()
    check('rectangle %d is painted where gtrace says it is (angle %.2f)'
          % (i, s.angle), off < 0.5, 'worst corner off by %.3f px' % off)

# A turn that went the wrong way would still land the corners of a
# square rectangle somewhere sensible, so the two turned ones are also
# held to being somewhere the untouched box is not.
if len(painted) == 3:
    plain = np.array(painted[0], dtype='float64')
    for i in (1, 2):
        got = np.array(painted[i], dtype='float64')
        check('rectangle %d really is turned on the screen' % i,
              np.abs(got - (plain + (got.mean(axis=0)
                                     - plain.mean(axis=0)))).max() > 5.0)

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
