'''
The shape editor in a real browser: the side bar it puts up instead
of the layout's, and the messages its rows and buttons send.

The editor is the same viewer, told by a scene channel that what it
is showing is a part rather than a bench. So what wants checking is
that the swap is complete in both directions - the shape panel and
the shape buttons are there, the optics panel and the tracing rules
are not, and a layout scene still gets all of its own - and that the
rows send what Python then does.

Every message is fed to a real ShapeEditor, so the units the panel
works in (millimetres and degrees) are checked against the metres and
radians the model keeps, rather than against the arithmetic the page
used to get there.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

import json
import math
import re
import subprocess

import numpy as np

import gtrace.draw as draw
from gtrace.draw.viewer import viewer_css
from gtrace.mechanics import (Mechanics, models, model_shapes,
                              model_prefix)
from gtrace.draw.viewer.editor import ShapeEditor
from gtrace.unit import *

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

def make_part():
    return Mechanics(shapes=[draw.Rectangle([-0.02, -0.01], 0.04, 0.02),
                             draw.Circle([0.005, 0.0], 0.003)],
                     center=[0.3, 0.1], name='P1',
                     points={'post': [-0.0135, 0.0]})

part = make_part()
editor = ShapeEditor(part)
scene = editor.scene_dict()

with open(os.path.join(SP, 'stage2_widget.mjs'), encoding='utf-8') as f:
    esm = f.read()

def js(obj):
    return json.dumps(obj, ensure_ascii=True).replace('</', '<\\/')

PAGE = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html, body { margin: 0; height: 100%; }
#host { width: 1000px; height: 640px; }
__CSS__
</style></head>
<body>
<div id="host"></div>
<div id="out" style="display:none"></div>
<script>
var ESM_SRC = __ESM__;
var SCENE = __SCENE__;
</script>
<script type="module">
(async function () {
    var out = {error: null, sent: []};
    try {
        var url = URL.createObjectURL(
            new Blob([ESM_SRC], {type: 'text/javascript'}));
        var mod = await import(url);

        var state = {scene: SCENE, height: 640, editable: true, error: ''};
        var handlers = {}, sent = [];
        var model = {
            get: function (k) { return state[k]; },
            set: function (k, v) {
                state[k] = v;
                (handlers['change:' + k] || []).forEach(function (f) { f(); });
            },
            on: function (e, f) { (handlers[e] = handlers[e] || []).push(f); },
            off: function (e, f) {
                handlers[e] = (handlers[e] || []).filter(function (g) {
                    return g !== f;
                });
            },
            send: function (m) { sent.push(m); },
            save_changes: function () {}
        };

        var el = document.getElementById('host');
        mod.default.render({model: model, el: el});
        var v = el.gtraceViewer;

        function buttons(sel) {
            return Array.prototype.map.call(
                el.querySelectorAll(sel), function (b) {
                    return b.textContent;
                });
        }
        function button(text) {
            var found = null;
            Array.prototype.forEach.call(
                el.querySelectorAll('button'), function (b) {
                    if (b.textContent === text) { found = b; }
                });
            return found;
        }
        function panelTitles() {
            return Array.prototype.map.call(
                el.querySelectorAll('.gt-panel-title'),
                function (e) { return e.textContent; });
        }
        function shapeRows() {
            return Array.prototype.map.call(
                el.querySelectorAll('.gt-shaperow'),
                function (b) { return b.textContent; });
        }
        function pointRows() {
            return Array.prototype.map.call(
                el.querySelectorAll('.gt-pointrow'),
                function (b) { return b.textContent; });
        }
        function pointFields() {
            var o = {};
            for (var k in (v.pointFields || {})) {
                var f = v.pointFields[k];
                o[k] = f.editable ? f.el.value : f.el.textContent;
            }
            return o;
        }
        function setPointField(key, text) {
            var f = v.pointFields[key];
            f.el.value = text;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        function fields() {
            var o = {};
            for (var k in (v.shapeFields || {})) {
                var f = v.shapeFields[k];
                o[k] = f.editable ? f.el.value : f.el.textContent;
            }
            return o;
        }
        function setField(key, text) {
            var f = v.shapeFields[key];
            f.el.value = text;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        var before;

        // --- the side bar the editor puts up ---
        // Only the buttons of the row itself: a variant behind an add
        // button lives in a menu, as verify_props_browser has it.
        out.headRows = Array.prototype.map.call(
            el.querySelectorAll('.gt-head .gt-btnrow'), function (row) {
                return Array.prototype.map.call(
                    row.querySelectorAll('button.gt-btn'),
                    function (b) { return b.textContent; });
            });
        out.panels = panelTitles();
        out.panelShown = {
            shape: v.shapeBody.style.display !== 'none',
            beam: v.readoutBody.style.display !== 'none',
            optic: v.opticBody.style.display !== 'none'
        };
        out.rows = shapeRows();
        out.originShown = v.originMark.style.display !== 'none';

        // A layout scene has none of what follows, and asking for it
        // would be a page that fails rather than a check that does.
        if (!SCENE.editor) {
            out.sent = sent;
            document.getElementById('out').textContent = JSON.stringify(out);
            return;
        }

        // --- the list picks a shape, and the rows follow its kind ---
        el.querySelectorAll('.gt-shaperow')[0].click();
        out.pickRect = {selected: v.selectedShape, fields: fields(),
                        marked: v.shapeMark.style.display !== 'none'};
        el.querySelectorAll('.gt-shaperow')[1].click();
        out.pickCircle = {selected: v.selectedShape, fields: fields()};

        // --- editing a number ---
        before = sent.length;
        setField('radius', '4');
        out.setRadius = {msg: sent[before] || null, n: sent.length - before};
        before = sent.length;
        setField('cx', '12.5');
        out.setCx = sent[before] || null;
        // The value the model actually holds decides nothing: the
        // scene has not moved, since this stand-in does not push one
        // back, so 3 mm is still what the circle is.
        before = sent.length;
        setField('radius', '3');
        out.noop = sent.length - before;
        // Nothing usable goes nowhere, and the row goes back to what
        // the model holds rather than to what was typed before it.
        before = sent.length;
        setField('radius', 'wide');
        out.rejected = {n: sent.length - before,
                        back: v.shapeFields.radius.el.value};

        // --- the buttons ---
        // The arrows first, while the selection is still a place the
        // scene in hand has room to move within.
        el.querySelectorAll('.gt-shaperow')[0].click();
        before = sent.length;
        button('↓').click();
        out.down = sent[before] || null;
        el.querySelectorAll('.gt-shaperow')[1].click();
        before = sent.length;
        button('↑').click();
        out.up = sent[before] || null;

        before = sent.length;
        button('+ Circle').click();
        out.addCircle = {msg: sent[before] || null,
                         selected: v.selectedShape};
        before = sent.length;
        el.querySelectorAll('.gt-shaperow')[0].click();
        button('Copy').click();
        out.copy = {msg: sent[before] || null, selected: v.selectedShape};
        before = sent.length;
        button('Remove').click();
        out.remove = sent[before] || null;

        // --- the points a part names for itself ---
        out.pointRows = pointRows();
        out.pointMarks = (v._pointMarkPts || []).length;
        out.pointBefore = pointFields();
        el.querySelectorAll('.gt-pointrow')[0].click();
        out.pickPoint = {selected: v.selectedPoint, fields: pointFields(),
                         marked: v.pointMarkEls[0].ring.classList
                                  .contains('gt-selected'),
                         label: v.pointMarkEls[0].label.textContent};
        before = sent.length;
        setPointField('px', '-14');
        out.setPx = sent[before] || null;
        before = sent.length;
        setPointField('name', 'stud');
        out.rename = sent[before] || null;
        // A name that is not one goes nowhere, and the row goes back
        // to what the part says.
        before = sent.length;
        setPointField('name', '   ');
        out.blankName = {n: sent.length - before,
                         back: v.pointFields.name.el.value};
        before = sent.length;
        setPointField('py', 'over there');
        out.badPy = {n: sent.length - before,
                     back: v.pointFields.py.el.value};
        before = sent.length;
        button('+ Point').click();
        out.addPoint = {msg: sent[before] || null,
                        selected: v.selectedPoint};
        before = sent.length;
        el.querySelectorAll('.gt-pointrow')[0].click();
        button('− Point').click();
        out.removePoint = sent[before] || null;

        // --- the rows are calculators too ---
        // Both kinds of row here hand parseField a unit: a shape row
        // takes it from its own field, a point row is millimetres by
        // the nature of a part. verify_input.js checks the parsing;
        // this checks that the millimetres arrive.
        el.querySelectorAll('.gt-shaperow')[1].click();
        before = sent.length;
        setField('radius', '1[in]');
        out.shapeInch = sent[before] || null;
        el.querySelectorAll('.gt-pointrow')[0].click();
        before = sent.length;
        setPointField('px', '2*25.4');
        out.pointTimes = sent[before] || null;

        // --- the library ---
        v.modelInput.value = 'BROWSER-PART';
        v.modelDesc.value = 'made in the editor';
        v.modelPrefix.value = 'XY';
        before = sent.length;
        button('Save to library').click();
        out.saveModel = sent[before] || null;
        // Left empty, the part says nothing about what its bodies are
        // called rather than saying it is called nothing.
        v.modelPrefix.value = '  ';
        v.modelInput.value = 'BROWSER-PART-2';
        before = sent.length;
        button('Save to library').click();
        out.saveModelNoPrefix = sent[before] || null;
        v.modelInput.value = 'BROWSER-PART';
        v.modelPrefix.value = 'XY';
        v.modelInput.value = '   ';
        before = sent.length;
        button('Save to library').click();
        out.saveBlank = sent.length - before;

        out.sent = sent;
    } catch (e) {
        out.error = String(e && e.stack || e);
    }
    document.getElementById('out').textContent = JSON.stringify(out);
})();
</script>
</body></html>
'''

def run(scene_obj, tag):
    page = PAGE.replace('__CSS__', viewer_css()) \
               .replace('__ESM__', js(esm)) \
               .replace('__SCENE__', js(scene_obj))
    path = os.path.join(SP, 'editor_page_%s.html' % tag)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    p = subprocess.run(
        [CHROME, '--headless=new', '--disable-gpu', '--window-size=1300,800',
         '--virtual-time-budget=6000', '--enable-logging=stderr', '--v=0',
         '--dump-dom', 'file:///' + path.replace('\\', '/')],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=120)
    errs = [l.strip() for l in (p.stderr or '').splitlines()
            if 'CONSOLE' in l and ('Uncaught' in l or 'Error' in l)]
    m = re.search(r'<div id="out"[^>]*>(.*?)</div>', p.stdout or '', re.S)
    payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
               .replace('&lt;', '<').replace('&gt;', '>')) if m else None
    return errs, (json.loads(payload) if payload else None)


print('--- the editor side bar ---')
errs, res = run(scene, 'edit')
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:500])

check('the first row puts shapes down',
      res['headRows'][0] == ['+ Rect', '+ Circle', '+ Line', '+ Poly',
                             '+ Arc', '+ Text'],
      json.dumps(res['headRows'][0]))
check('the second is what a part and a bench share',
      res['headRows'][1] == ['Undo', 'Redo', 'Measure', 'Fit'],
      json.dumps(res['headRows'][1]))
check('the panels are the shapes, the points, the library, the layers '
      'and the help',
      res['panels'] == ['Shapes', 'Named points', 'Model library',
                        'Layers', 'Controls'],
      json.dumps(res['panels']))
check('  so no optics, beams, files or tracing rules',
      not any(p in res['panels'] for p in
              ['Beam width', 'Tracing rules', 'Optical layout (JSON)',
               'Drawing (DXF)']))
check('the shape panel is the one on show',
      res['panelShown']['shape'] and not res['panelShown']['beam']
      and not res['panelShown']['optic'])
check('the list is the part, in the order it is drawn',
      res['rows'] == ['1.  rectangle', '2.  circle'],
      json.dumps(res['rows']))
check('and the origin is marked', res['originShown'])

print('--- the list picks a shape, the rows follow its kind ---')
pr = res['pickRect']
check('picking the rectangle shows its corner and size, in millimetres',
      pr['selected'] == 0
      and abs(float(pr['fields']['x']) + 20) < 1e-9
      and abs(float(pr['fields']['width']) - 40) < 1e-9,
      json.dumps(pr['fields']))
check('  and marks it in the drawing', pr['marked'])
pc = res['pickCircle']
check('picking the circle shows a radius instead',
      pc['selected'] == 1 and 'radius' in pc['fields']
      and 'width' not in pc['fields']
      and abs(float(pc['fields']['radius']) - 3) < 1e-9,
      json.dumps(pc['fields']))

print('--- editing a number ---')
sr = res['setRadius']
check('one message per row committed',
      sr['n'] == 1 and sr['msg'] and sr['msg']['op'] == 'set_shape'
      and sr['msg']['index'] == 1, json.dumps(sr['msg']))
if sr['msg']:
    editor.apply_edit(sr['msg'])
    check('  and 4 in the panel is 4 mm in the model',
          abs(part.shapes[1].radius - 0.004) < 1e-15,
          str(part.shapes[1].radius))
check('a coordinate is sent as the whole point',
      res['setCx'] and 'center' in res['setCx']['attrs']
      and abs(res['setCx']['attrs']['center'][0] - 0.0125) < 1e-12,
      json.dumps(res['setCx']))
if res['setCx']:
    editor.apply_edit(res['setCx'])
    check('  which Python lands where the panel said',
          np.allclose(part.shapes[1].center, [0.0125, 0.0]))
check('the value the model holds sends nothing', res['noop'] == 0)
check('a value that is no number sends nothing, and is put back',
      res['rejected']['n'] == 0
      and abs(float(res['rejected']['back']) - 3) < 1e-9,
      json.dumps(res['rejected']))

print('--- the buttons ---')
ac = res['addCircle']
check('+ Circle asks for one, and selects what it asked for',
      ac['msg'] and ac['msg'] == {'op': 'add_shape', 'type': 'circle'}
      and ac['selected'] == 2, json.dumps(ac))
cp = res['copy']
check('Copy duplicates the picked shape and follows the copy',
      cp['msg'] and cp['msg']['op'] == 'duplicate_shape'
      and cp['msg']['index'] == 0 and cp['selected'] == 1,
      json.dumps(cp))
check('the arrows move it later and earlier',
      res['down'] and res['down']['op'] == 'move_shape'
      and res['down']['to'] == res['down']['index'] + 1
      and res['up'] and res['up']['to'] == res['up']['index'] - 1,
      '%s / %s' % (json.dumps(res['down']), json.dumps(res['up'])))
check('Remove takes the picked shape away',
      res['remove'] and res['remove']['op'] == 'remove_shape',
      json.dumps(res['remove']))

print('--- the points a part names for itself ---')
check('the list is what the part names',
      res['pointRows'] == ['post'], json.dumps(res['pointRows']))
check('  and each is marked in the drawing', res['pointMarks'] == 1)
check('nothing picked shows nothing',
      all(v == '' for v in res['pointBefore'].values()),
      json.dumps(res['pointBefore']))
pp = res['pickPoint']
check('picking one shows its name and its place, in millimetres',
      pp['selected'] == 0 and pp['fields']['name'] == 'post'
      and abs(float(pp['fields']['px']) + 13.5) < 1e-9
      and abs(float(pp['fields']['py'])) < 1e-9,
      json.dumps(pp['fields']))
check('  marks it in the drawing, with its name beside it',
      pp['marked'] and pp['label'] == 'post')

# Every gesture sends the whole list: there is no index that survives
# a rename, so a place and a name arrive the same way.
px = res['setPx']
check('moving one sends the whole list, in metres',
      px and px['op'] == 'set_points' and len(px['points']) == 1
      and abs(px['points'][0]['point'][0] + 0.014) < 1e-12
      and px['points'][0]['name'] == 'post', json.dumps(px))
if px:
    editor.apply_edit(px)
    check('  which Python lands where the panel said',
          np.allclose(part.points['post'], [-0.014, 0.0]))
rn = res['rename']
check('renaming one is the same message',
      rn and rn['op'] == 'set_points'
      and [q['name'] for q in rn['points']] == ['stud'], json.dumps(rn))
if rn:
    editor.apply_edit(rn)
    check('  and the part answers to the new name',
          sorted(part.points) == ['stud'])
check('a blank name sends nothing, and is put back',
      res['blankName']['n'] == 0 and res['blankName']['back'] == 'post',
      json.dumps(res['blankName']))
check('a place that is no number sends nothing either',
      res['badPy']['n'] == 0 and abs(float(res['badPy']['back'])) < 1e-9,
      json.dumps(res['badPy']))

ap = res['addPoint']
check('+ Point names one at the origin, and follows it',
      ap['msg'] and ap['msg']['op'] == 'set_points'
      and len(ap['msg']['points']) == 2
      and ap['msg']['points'][1]['point'] == [0.0, 0.0]
      and ap['selected'] == 1, json.dumps(ap))
check('  under a name that is free to be typed over',
      ap['msg'] and ap['msg']['points'][1]['name'] == 'point')
rp = res['removePoint']
check('\u2212 Point takes the picked one away',
      rp and rp['op'] == 'set_points' and rp['points'] == [],
      json.dumps(rp))
if rp:
    editor.apply_edit(rp)
    check('  and Python is left with none', part.points == {})


print('--- the rows are calculators too ---')
si = res['shapeInch']
check('1[in] in a shape row is 25.4 mm, in metres on the wire',
      si and si['op'] == 'set_shape'
      and abs(si['attrs']['radius'] - 0.0254) < 1e-12, json.dumps(si))
pt = res['pointTimes']
check('2*25.4 in a named point row is 50.8 mm',
      pt and pt['op'] == 'set_points'
      and abs(pt['points'][0]['point'][0] - 0.0508) < 1e-12, json.dumps(pt))

print('--- the library ---')
sm = res['saveModel']
check('Save to library sends the name and the line of description',
      sm and sm['op'] == 'save_model' and sm['name'] == 'BROWSER-PART'
      and sm['description'] == 'made in the editor', json.dumps(sm))
check('  and what the parts built from it are called',
      sm and sm.get('prefix') == 'XY', json.dumps(sm))
sm2 = res['saveModelNoPrefix']
check('a part that says nothing about its name sends no prefix',
      sm2 and 'prefix' not in sm2, json.dumps(sm2))
if sm2:
    editor.apply_edit(sm2)
    check('  and Python leaves it with none',
          model_prefix('BROWSER-PART-2') is None)
if sm:
    editor.apply_edit(sm)
    check('  and Python writes the prefix onto the shelf',
          model_prefix('BROWSER-PART') == 'XY',
          str(model_prefix('BROWSER-PART')))
    check('  and Python puts the part on the shelf',
          'BROWSER-PART' in models()
          and len(model_shapes('BROWSER-PART')) == len(part.shapes))
check('a blank name sends nothing', res['saveBlank'] == 0)

print('--- a layout scene still gets the layout side bar ---')
import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.layout import OpticalLayout, TraceRules, q_from_waist

L = OpticalLayout(
    optics=[opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.pi, name='M1')],
    sources=[GaussianBeam(pos=[0, 0], dirAngle=0.0,
                          q0=q_from_waist(0.2*mm, 0.0, 1064*nm),
                          wl=1064*nm, name='b0')],
    rules=TraceRules(order=2, power_threshold=1e-4))
errs, res2 = run(L.scene_dict(), 'layout')
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res2 is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res2['error'] is None, str(res2['error'])[:300])
check('the add row is the layout\'s again',
      res2['headRows'][0] == ['+ Mirror', '+ Lens', '+ Source',
                              '+ Dump', '+ Mechanics', '+ Assembly',
                              '+ Shape'],
      json.dumps(res2['headRows'][0]))
check('  with Align back on the second row',
      'Align' in res2['headRows'][1], json.dumps(res2['headRows'][1]))
check('the layout panels are all there',
      'Beam readout' in res2['panels'] and 'Tracing rules' in res2['panels']
      and 'Optical layout (JSON)' in res2['panels']
      and 'Model library' not in res2['panels'],
      json.dumps(res2['panels']))
check('no shape list, and no origin mark',
      res2['rows'] == [] and not res2['originShown'])
check('nothing was sent by simply opening it',
      [m for m in res2['sent'] if m.get('op') in ('add_shape',
                                                  'save_model')] == [])

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
