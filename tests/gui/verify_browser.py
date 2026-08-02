'''
Stage 1 verification, browser side.

Loads the generated HTML in headless Chrome, fails on any console error,
and checks the resulting DOM: the viewer must have built the SVG scene,
the layer list and the readout panel.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

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

def run_chrome(html_path, extra):
    url = 'file:///' + html_path.replace('\\', '/')
    cmd = [CHROME, '--headless=new', '--disable-gpu', '--window-size=1400,900',
           '--virtual-time-budget=4000', '--enable-logging=stderr', '--v=0'] \
        + extra + [url]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                       errors='replace', timeout=120)
    return p

def console_errors(stderr):
    bad = []
    for line in (stderr or '').splitlines():
        if 'CONSOLE' in line and ('Uncaught' in line or 'Error' in line):
            bad.append(line.strip())
        # Chrome reports failed subresource loads as network errors.
        if 'ERR_FILE_NOT_FOUND' in line or 'ERR_CONNECTION' in line:
            bad.append(line.strip())
    return bad

for name in ['stage1_view.html', 'stage1_nobeams.html', 'stage1_drawoptsys.html']:
    path = os.path.join(SP, name)
    if not os.path.exists(path):
        check(name + ' exists', False)
        continue

    print('--- %s ---' % name)
    p = run_chrome(path, ['--dump-dom'])
    errs = console_errors(p.stderr)
    check('no console error', errs == [], '\n        '.join(errs[:4]))

    dom = p.stdout or ''
    check('viewer mounted', 'class="gt-root"' in dom)
    check('svg scene built', '<svg' in dom and 'gt-scene' in dom)
    check('geometry drawn',
          len(re.findall(r'<(line|polyline|path|circle|rect)\b', dom)) > 20,
          '(%d elements)' % len(re.findall(r'<(line|polyline|path|circle|rect)\b', dom)))
    check('labels rendered', '<text' in dom)
    check('layer list built', dom.count('gt-layerrow') >= 4,
          '(%d rows)' % dom.count('gt-layerrow'))
    check('readout table built', 'gt-readout' in dom and 'Radius w' in dom)
    check('scene group transformed',
          re.search(r'class="gt-scene" transform="translate\([-\d.]', dom) is not None)

print('--- empty layer marking ---')
# A layer that exists but holds no shape must be marked in the list, so
# that it does not read as a rendering failure.
p = run_chrome(os.path.join(SP, 'stage1_emptylayer.html'), ['--dump-dom'])
dom = p.stdout or ''
check('empty layer flagged', dom.count('gt-layerrow gt-layer-empty') == 1,
      '(%d rows)' % dom.count('gt-layerrow gt-layer-empty'))
check("'empty' note shown", 'class="gt-note">empty<' in dom)

p = run_chrome(os.path.join(SP, 'stage1_view.html'), ['--dump-dom'])
check('no layer flagged when all are filled',
      (p.stdout or '').count('gt-layerrow gt-layer-empty') == 0)

print('--- clicking an optics in the real static page ---')
# Everything else here is driven through the widget's ESM. This drives
# the file renderHTML actually writes, which is how the missing optics
# channel got past the checks the first time.
driver = '''
<div id="gt-test-out" style="display:none"></div>
<script>
(function () {
    var v = window.gtraceViewer;
    var out = {error: null};
    function mouse(target, type, x, y) {
        target.dispatchEvent(new MouseEvent(type, {
            clientX: x, clientY: y, button: 0,
            bubbles: true, cancelable: true}));
    }
    try {
        out.optics = (v.scene.optics || []).map(function (o) { return o.name; });
        out.editable = !!v.onEdit;
        var m1 = (v.scene.optics || [])[0];
        if (m1) {
            var r = v.svg.getBoundingClientRect();
            var c = m1.center || m1.HRcenter;
            var p = v.sceneToScreen(c[0], c[1]);
            mouse(window, 'mousemove', p[0] + r.left, p[1] + r.top);
            mouse(v.svg, 'mousedown', p[0] + r.left, p[1] + r.top);
            mouse(window, 'mouseup', p[0] + r.left, p[1] + r.top);
            out.selected = v.selectedOptic;
            out.title = document.querySelector('.gt-panel-title span').textContent;
            out.propsShown = v.opticBody.style.display !== 'none';
            out.fields = {};
            for (var k in v.opticFields) {
                var f = v.opticFields[k];
                out.fields[k] = f.editable ? f.el.value : f.el.textContent;
            }
            out.anyEditable = Object.keys(v.opticFields).some(function (k) {
                return v.opticFields[k].editable;
            });
        }
    } catch (e) { out.error = String((e && e.stack) || e); }
    document.getElementById('gt-test-out').textContent = JSON.stringify(out);
})();
</script>
</body>'''

with open(os.path.join(SP, 'stage1_view.html'), encoding='utf-8') as f:
    page = f.read().replace('</body>', driver)
inst = os.path.join(SP, 'stage1_click_optic.html')
with open(inst, 'w', encoding='utf-8') as f:
    f.write(page)

p = run_chrome(inst, ['--dump-dom'])
check('no console error', console_errors(p.stderr) == [],
      '\n        '.join(console_errors(p.stderr)[:3]))
m = re.search(r'<div id="gt-test-out"[^>]*>(.*?)</div>', p.stdout or '', re.S)
if m:
    import json
    payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
               .replace('&lt;', '<').replace('&gt;', '>'))
    res = json.loads(payload)
    check('the driver ran', res['error'] is None, str(res['error'])[:200])
    check('the page knows its optics', res.get('optics') == ['M1', 'M2', 'M3'],
          str(res.get('optics')))
    check('the page is read-only', res.get('editable') is False)
    check('clicking one selects it', res.get('selected') == 'M1',
          str(res.get('selected')))
    check('the panel switches to the properties',
          res.get('propsShown') and res.get('title') == 'Optics properties',
          str(res.get('title')))
    check('and shows real values',
          res.get('fields', {}).get('name') == 'M1'
          and res.get('fields', {}).get('type') == 'Mirror'
          and res.get('fields', {}).get('cx') not in (None, '', '-'),
          str({k: res.get('fields', {}).get(k)
               for k in ['name', 'type', 'cx', 'rocHR']}))
    check('nothing is editable without a Python behind it',
          res.get('anyEditable') is False)
else:
    check('the driver produced output', False)

print('--- screenshot ---')
shot = os.path.join(SP, 'stage1_shot.png')
if os.path.exists(shot):
    os.remove(shot)
p = run_chrome(os.path.join(SP, 'stage1_view.html'), ['--screenshot=' + shot])
size = os.path.getsize(shot) if os.path.exists(shot) else 0
# A blank dark page compresses to a few kB; a drawn scene is much larger.
check('screenshot has content', size > 20000, '(%d bytes)' % size)

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
