'''
Verification of the keyboard shortcuts, in a real browser.

The viewer lives inside somebody else's page. A notebook binds keys of
its own, and it binds several of the same ones: in JupyterLab 'a'
inserts a cell, 'm' turns the cell into Markdown - which throws the
widget away with it - and Ctrl+Z undoes an edit to the cell. Whoever
gets the key first decides. So this suite is not about what each
shortcut does, which the suite for that tool checks; it is about who
gets the key, and about the key going no further once the viewer has
taken it.

Three rules are checked, and each of them was once broken.

The pointer decides. A key is the viewer's only while the pointer is
over the viewer. The rule used to be relaxed for a page holding a
single viewer, on the reasoning that there was nothing else the key
could be meant for - but a notebook cell is something else, and typing
'f' into one re-fitted the drawing while 'm' armed the measuring tool.

The key is caught on the way down. The page around the viewer is
reached first on the way back up, so a key stopped on the way up has
already been acted on by the notebook. Here a spy sits on the document
in the capture phase, which runs after the window: it stands for that
notebook, and it must not see the keys the viewer takes.

What the viewer does not own, it does not touch. A key with a modifier
the shortcut does not ask for is not the shortcut - Ctrl+F belongs to
the browser's search, not to Fit - and the letters that aim an optics
mean nothing while a shape editor is on show, so there they are left to
the page.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

import json
import re
import subprocess

import numpy as np

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.draw.viewer import viewer_css
from gtrace.layout import OpticalLayout, TraceRules
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

def make_layout():
    b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                           pos=[0, 0], dirAngle=0, name='b0')
    M1 = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=deg2rad(135),
                    diameter=10*cm, thickness=5*cm, name='M1')
    M2 = opt.Mirror(HRcenter=[0.5, 0.4], normAngleHR=deg2rad(-45),
                    diameter=10*cm, thickness=5*cm, name='M2')
    return OpticalLayout(optics=[M1, M2], sources=[b0],
                         rules=TraceRules(order=4, power_threshold=1e-3),
                         name='Keys')

layout = make_layout()
scene = layout.scene_dict()

#: A layout page and an editor page differ in which letters the viewer
#: owns, so both are driven.
editor_scene = {
    'canvas': {'unit': 'm', 'layers': []},
    'beams': [],
    'editor': {'name': 'PART', 'kind': 'mechanics'},
    'shapes': [{'type': 'rect', 'x': 0.0, 'y': 0.0,
                'width': 0.04, 'height': 0.02, 'angle': 0.0}],
    'points': [],
}

with open(os.path.join(SP, 'stage2_widget.mjs'), encoding='utf-8') as f:
    esm = f.read()

def js(obj):
    return json.dumps(obj, ensure_ascii=True).replace('</', '<\\/')

PAGE = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html, body { margin: 0; height: 100%; }
#host { width: 1000px; height: 600px; }
#cell { height: 60px; border: 1px solid #ccc; }
__CSS__
</style></head>
<body>
<div id="host"></div>
<!-- Stands for the editor of the page around the viewer: a notebook
     cell is an element with contenteditable set, and a key aimed at it
     is being typed, not pressed as a shortcut. -->
<div id="cell" contenteditable="true">a cell of the page around it</div>
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

        var state = {scene: SCENE, title: 'Keys', height: 600,
                     editable: true, error: ''};
        var handlers = {}, sent = [];
        var model = {
            get: function (k) { return state[k]; },
            set: function (k, v) {
                state[k] = v;
                (handlers['change:' + k] || []).forEach(function (f) { f(); });
            },
            on: function (e, f) { (handlers[e] = handlers[e] || []).push(f); },
            off: function () {},
            send: function (m) { sent.push(m); },
            save_changes: function () {}
        };

        var el = document.getElementById('host');
        mod.default.render({model: model, el: el});
        var v = el.gtraceViewer;

        // The page around the viewer. It listens on the document in the
        // capture phase, which is where a notebook listens, and that
        // runs after the window: a key the viewer has stopped never
        // arrives here.
        var host = [];
        document.addEventListener('keydown', function (ev) {
            host.push(ev.key);
        }, true);

        function press(k, opts) {
            var ev = new KeyboardEvent('keydown', Object.assign(
                {key: k, bubbles: true, cancelable: true}, opts || {}));
            host.length = 0;
            // Onto whatever has the keyboard, so that the event travels
            // the path a real one does: window, then document, then the
            // element. Dispatching on the window would target the
            // window itself and never reach the spy below it.
            (document.activeElement || document.body).dispatchEvent(ev);
            return {stopped: host.length === 0,
                    prevented: ev.defaultPrevented};
        }

        function state_of() {
            return {scale: Math.round(v.scale),
                    measuring: !!v.measuring,
                    aligning: v.aligning ? v.aligning.want : null,
                    selected: v.selectedOptic};
        }

        // Away from the fitted size, so that a stray fit shows up as a
        // change of scale. There is no call for this: the scale is a
        // property the view applies, and the wheel is what moves it.
        function zoomOut() {
            v.scale = v.scale * 2;
            v._applyTransform();
        }

        v.fit();
        var fitted = Math.round(v.scale);
        zoomOut();
        var zoomed = Math.round(v.scale);
        out.zoomed = {fitted: fitted, zoomed: zoomed};

        // --- with the pointer away from the viewer ---
        v.pointerInside = false;
        out.away = {};
        ['f', 'm', 'a', 'b', '[', ']', 'Escape', 'Delete'].forEach(
            function (k) {
                var r = press(k);
                out.away[k] = {stopped: r.stopped, state: state_of()};
            });
        out.away.undo = {r: press('z', {ctrlKey: true}), state: state_of()};

        // --- with the pointer over the viewer ---
        v.pointerInside = true;

        out.fit = {before: Math.round(v.scale)};
        var rf = press('f');
        out.fit.after = Math.round(v.scale);
        out.fit.stopped = rf.stopped;
        out.fit.prevented = rf.prevented;

        zoomOut();
        var rF = press('F');
        out.fitUpper = {scale: Math.round(v.scale), stopped: rF.stopped};

        // Ctrl+F is the browser's search, not Fit.
        zoomOut();
        var stray = Math.round(v.scale);
        var rc = press('f', {ctrlKey: true});
        out.ctrlF = {scale: Math.round(v.scale), was: stray,
                     stopped: rc.stopped, prevented: rc.prevented};
        v.fit();

        var rm = press('m');
        out.measure = {on: !!v.measuring, stopped: rm.stopped,
                       prevented: rm.prevented};
        press('m');
        out.measureOff = !!v.measuring;

        // Aiming needs something selected; the point here is who gets
        // the key, so the key is claimed either way.
        var rA = press('a');
        out.alignNoSel = {aligning: v.aligning ? v.aligning.want : null,
                          stopped: rA.stopped};
        var first = (v.scene.optics || [])[0];
        if (first) { v._selectOptic(first); }
        out.alignA = {r: press('a'),
                      want: v.aligning ? v.aligning.want : null};
        press('Escape');
        out.alignB = {r: press('b'),
                      want: v.aligning ? v.aligning.want : null};
        press('Escape');

        // Undo and redo, in both spellings.
        if (first) { v._selectOptic(first); }
        out.turn = {r: press(']'), sent: sent.length};
        out.turnBack = {r: press('['), sent: sent.length};
        out.undo = press('z', {ctrlKey: true});
        out.redoShift = press('z', {ctrlKey: true, shiftKey: true});
        out.redoY = press('y', {ctrlKey: true});
        out.undoMeta = press('z', {metaKey: true});
        // Ctrl+Alt+Z is not the shortcut.
        out.undoAlt = press('z', {ctrlKey: true, altKey: true});
        // Nor is a bare z.
        out.bareZ = press('z');

        out.escape = press('Escape');
        out.del = press('Delete');
        // A key the viewer has never claimed goes straight through.
        out.other = press('F2');

        // --- while something is being typed into ---
        // The pointer can rest over the drawing while the caret is in a
        // field or in a cell of the page around it. Then the key is
        // being typed.
        zoomOut();
        var typedScale = Math.round(v.scale);
        var cell = document.getElementById('cell');
        cell.focus();
        var rt = press('f');
        out.typingCell = {scale: Math.round(v.scale), was: typedScale,
                          stopped: rt.stopped};

        var field = v.root.querySelector('input.gt-input');
        out.hasField = !!field;
        if (field) {
            field.focus();
            var rfi = press('f');
            out.typingField = {scale: Math.round(v.scale), was: typedScale,
                               stopped: rfi.stopped};
        }
        document.getElementById('cell').blur();
        if (field) { field.blur(); }

        out.sent = sent.slice();
    } catch (e) {
        out.error = String(e && e.stack || e);
    }
    document.getElementById('out').textContent = JSON.stringify(out);
})();
</script>
</body></html>
'''

EDITOR_PAGE = PAGE.replace("title: 'Keys'", "title: 'Keys editor'")

def run(page_src, scene_obj, name):
    page = (page_src.replace('__CSS__', viewer_css())
                    .replace('__ESM__', js(esm))
                    .replace('__SCENE__', js(scene_obj)))
    path = os.path.join(SP, name)
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

errs, res = run(PAGE, scene, 'keys_page.html')
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:600])

print('--- the pointer decides ---')
away = res['away']
for k in ['f', 'm', 'a', 'b', '[', ']', 'Escape', 'Delete']:
    check('%-6s does nothing with the pointer away' % k,
          not away[k]['stopped'], str(away[k]['state']))
check('nor does Ctrl+Z', not away['undo']['r']['stopped'],
      str(away['undo']['r']))
check('  and the drawing is left where it was',
      away['f']['state']['scale'] == res['zoomed']['zoomed'],
      '%s / %s' % (away['f']['state']['scale'], res['zoomed']['zoomed']))
check('  the measuring tool stays down', not away['m']['state']['measuring'],
      str(away['m']['state']))
check('  and no aim is started', away['a']['state']['aligning'] is None,
      str(away['a']['state']))

print('--- and the page still gets them ---')
# Not stopped means the spy on the document saw the key, which is the
# notebook getting on with what it binds the key to.
check('every one of them reaches the page',
      all(not away[k]['stopped']
          for k in ['f', 'm', 'a', 'b', '[', ']', 'Escape', 'Delete']))

print('--- with the pointer over the viewer ---')
check('f fits the drawing', res['fit']['after'] == res['zoomed']['fitted'],
      '%s -> %s (fitted %s)' % (res['fit']['before'], res['fit']['after'],
                                res['zoomed']['fitted']))
check('  and the key goes no further', res['fit']['stopped'])
check('  and is marked as taken', res['fit']['prevented'])
check('F does the same as f',
      res['fitUpper']['scale'] == res['zoomed']['fitted']
      and res['fitUpper']['stopped'], str(res['fitUpper']))

check('m arms the measuring tool', res['measure']['on'])
check('  and the key goes no further', res['measure']['stopped'])
check('  m again puts it away', not res['measureOff'])

check('a starts a two-point aim', res['alignA']['want'] == 2,
      str(res['alignA']))
check('b starts a three-point aim', res['alignB']['want'] == 3,
      str(res['alignB']))
check('  both keys go no further', res['alignA']['r']['stopped']
      and res['alignB']['r']['stopped'])
# Nothing selected is not a reason to let the page have the key: in a
# notebook 'a' inserts a cell, and the cell the viewer lives in is not
# the viewer's to move.
check('a with nothing selected still takes the key',
      res['alignNoSel']['stopped']
      and res['alignNoSel']['aligning'] is None, str(res['alignNoSel']))

check('] and [ turn the selection', res['turn']['r']['stopped']
      and res['turnBack']['r']['stopped'],
      '%s / %s' % (res['turn']['r'], res['turnBack']['r']))
check('  and reach Python', res['turnBack']['sent'] > res['turn']['sent'] - 1,
      '%d messages' % res['turnBack']['sent'])

print('--- undo and redo ---')
check('Ctrl+Z is taken', res['undo']['stopped'] and res['undo']['prevented'],
      str(res['undo']))
check('Ctrl+Shift+Z is taken', res['redoShift']['stopped'],
      str(res['redoShift']))
check('Ctrl+Y is taken', res['redoY']['stopped'], str(res['redoY']))
check('Command+Z is taken as well, for a Mac', res['undoMeta']['stopped'],
      str(res['undoMeta']))
check('Ctrl+Alt+Z is not the shortcut', not res['undoAlt']['stopped'],
      str(res['undoAlt']))
check('and a bare z is not either', not res['bareZ']['stopped'],
      str(res['bareZ']))

print('--- what the viewer does not own ---')
check('Ctrl+F is left to the browser',
      res['ctrlF']['scale'] == res['ctrlF']['was']
      and not res['ctrlF']['stopped'], str(res['ctrlF']))
check('Escape is taken', res['escape']['stopped'], str(res['escape']))
check('Delete is taken', res['del']['stopped'], str(res['del']))
check('a key it never binds goes straight through',
      not res['other']['stopped'], str(res['other']))

print('--- while something is being typed into ---')
check('a key typed into the page is not a shortcut',
      res['typingCell']['scale'] == res['typingCell']['was']
      and not res['typingCell']['stopped'], str(res['typingCell']))
check('the viewer has a field of its own to try', res['hasField'])
if res.get('typingField'):
    check('nor is one typed into the viewer\'s own field',
          res['typingField']['scale'] == res['typingField']['was']
          and not res['typingField']['stopped'], str(res['typingField']))

print('--- a shape editor ---')
errs2, res2 = run(EDITOR_PAGE, editor_scene, 'keys_editor_page.html')
check('no console error', errs2 == [], '\n        '.join(errs2[:3]))
if res2 is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res2['error'] is None, str(res2['error'])[:600])
# There is no optics to aim in an editor, so the letters mean nothing
# here and are left to the page around it.
check('a is left to the page', not res2['alignA']['r']['stopped'],
      str(res2['alignA']))
check('b is left to the page', not res2['alignB']['r']['stopped'],
      str(res2['alignB']))
check('but ] still turns the shape on show', res2['turn']['r']['stopped'],
      str(res2['turn']))
check('and f still fits', res2['fit']['stopped'], str(res2['fit']))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
