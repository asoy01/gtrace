'''
Aiming an optics from the viewer, in a real browser: the two ways of
naming an angle by places, and the quarter turn.

A drag puts an element approximately anywhere, and Ctrl-drag squares
it onto a beam that already exists. What was missing is the angles a
bench is laid out by before there is a beam to point at - square
across the line between two places, or bisecting the corner at a
place light is to be folded at - which is what this tool is.

Two things carry it and are checked hardest.

The first is that the angle means what it says. Every message the page
sends is fed to Python's apply_edit and the result checked
geometrically rather than against the formula the page used: after a
two-point aim the front face normal is parallel to the line between
those two points, and after a three-point aim it makes the same angle
with both arms - which is the law of reflection, and the whole reason
anyone would bisect a corner.

The second is that a line has two normals and the drawing cannot say
which. The nearer one is taken, so clicking the two points the other
way about is the same aim - which is checked by doing exactly that.

Aiming leaves the element where it stands: which way it faces and
where it sits are two questions, and the second already has answers.
That is checked too, since it is the easiest thing to get wrong.
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

import gtrace.beam as beam
import gtrace.optcomp as opt
from gtrace.draw.viewer import viewer_css
from gtrace.layout import OpticalLayout, TraceRules, q_from_waist
from gtrace.mechanics import breadboard
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
    b0 = beam.GaussianBeam(q0=q_from_waist(0.2*mm, 0.0, 1064*nm), wl=1064*nm,
                           pos=[0.0, 0.0], dirAngle=0.0, name='b0')
    M1 = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=deg2rad(135),
                    diameter=1*inch, thickness=6*mm, name='M1')
    M2 = opt.Mirror(HRcenter=[0.5, 0.4], normAngleHR=deg2rad(-135),
                    diameter=1*inch, thickness=6*mm, name='M2')
    # A board under it all, for its screw holes: the aim is taken by
    # clicking places, and the holes are the places a bench has.
    board = breadboard(0.6, 0.5, center=[0.35, 0.2], name='Board')
    return OpticalLayout(optics=[M1, M2], sources=[b0], mechanics=[board],
                         rules=TraceRules(order=4, power_threshold=1e-4))

layout = make_layout()
scene = layout.scene_dict()

with open(os.path.join(SP, 'stage2_widget.mjs'), encoding='utf-8') as f:
    esm = f.read()

def js(obj):
    return json.dumps(obj, ensure_ascii=True).replace('</', '<\\/')

def hole_near(x, y):
    '''
    The screw hole nearest a place, as both sides pick their points.
    '''
    best, bd = None, float('inf')
    for p in scene['snap']:
        if p['kind'] != 'hole':
            continue
        d = math.hypot(p['point'][0] - x, p['point'][1] - y)
        if d < bd:
            best, bd = p['point'], d
    return best

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
var EDITABLE = __EDITABLE__;
</script>
<script type="module">
(async function () {
    var out = {error: null, sent: []};
    function mouse(target, type, x, y, opts) {
        target.dispatchEvent(new MouseEvent(type, Object.assign({
            clientX: x, clientY: y, button: 0, bubbles: true, cancelable: true
        }, opts || {})));
    }
    function key(k) {
        window.dispatchEvent(new KeyboardEvent('keydown',
            {key: k, bubbles: true}));
    }
    try {
        var url = URL.createObjectURL(
            new Blob([ESM_SRC], {type: 'text/javascript'}));
        var mod = await import(url);

        var state = {scene: SCENE, height: 640, editable: EDITABLE,
                     error: ''};
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

        function rect() { return v.svg.getBoundingClientRect(); }
        function screenOf(p) {
            var r = rect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function clickAt(p) {
            mouse(v.svg, 'mousedown', p[0], p[1]);
            mouse(window, 'mouseup', p[0], p[1]);
        }
        function hover(p) { mouse(window, 'mousemove', p[0], p[1]); }
        function dragFromTo(a, b) {
            mouse(v.svg, 'mousedown', a[0], a[1]);
            mouse(window, 'mousemove', (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
            mouse(window, 'mousemove', b[0], b[1]);
            mouse(window, 'mouseup', b[0], b[1]);
        }
        function holeNear(x, y) {
            var best = null, bd = Infinity;
            (SCENE.snap || []).forEach(function (p) {
                if (p.kind !== 'hole') { return; }
                var d = Math.hypot(p.point[0] - x, p.point[1] - y);
                if (d < bd) { bd = d; best = p.point; }
            });
            return best;
        }
        function menuItems() {
            var found = [];
            (v.addMenus || []).forEach(function (m) {
                if (m.button === v.alignBtn) {
                    found = Array.prototype.map.call(
                        m.menu.querySelectorAll('button'),
                        function (b) { return b.textContent; });
                }
            });
            return found;
        }

        var M1C = SCENE.optics[0].center;
        // Two holes off to one side, and a third making a corner with
        // them, all of them places the snap will take exactly.
        var P1 = holeNear(0.15, 0.05);
        var P2 = holeNear(0.55, 0.35);
        var Q1 = holeNear(0.10, 0.35);
        var Q2 = holeNear(0.30, 0.10);
        var Q3 = holeNear(0.55, 0.05);
        out.points = {P1: P1, P2: P2, Q1: Q1, Q2: Q2, Q3: Q3};
        var before;

        // --- the button and its menu ---
        out.disabledWithNothingSelected = EDITABLE ? v.alignBtn.disabled
                                                   : null;
        out.hasButton = !!v.alignBtn;
        clickAt(screenOf(M1C));
        out.selected = v.selectedOptic;
        out.enabledWithOptic = EDITABLE ? !v.alignBtn.disabled : null;
        out.items = menuItems();

        if (!EDITABLE) {
            out.sent = sent;
            document.getElementById('out').textContent = JSON.stringify(out);
            return;
        }

        // Aiming takes its points from the same marks a measurement
        // does, so the middle of an edge is one of them. Its own mode
        // and its own click handler, though, so ask it here too.
        key('a');
        // Whichever edge middle stands furthest from every other mark.
        // A fixed clearance will not do: a breadboard puts a hole every
        // 25 mm, so on that part nothing is ever far from something.
        var mid = null, midClear = 0;
        v.scene.snap.forEach(function (m) {
            if (m.kind !== 'midpoint') { return; }
            var near = Infinity;
            v.scene.snap.forEach(function (s) {
                if (s === m) { return; }
                near = Math.min(near, Math.hypot(s.point[0] - m.point[0],
                                                 s.point[1] - m.point[1]));
            });
            if (near > midClear) { mid = m; midClear = near; }
        });
        out.midClear = midClear;
        out.alignMid = null;
        if (mid) {
            // Two pixels off, and well inside the clearance measured
            // above, so that the middle is what is nearest.
            var mp = screenOf(mid.point);
            hover([mp[0] + 2, mp[1] - 1]);
            out.alignMid = {hovered: v.snapped && v.snapped.label,
                            scale: v.scale};
            clickAt([mp[0] + 2, mp[1] - 1]);
            out.alignMid.taken = v.aligning
                ? v.aligning.points.slice(-1)[0] : null;
            out.alignMid.want = mid.point.slice();
            out.alignMid.label = mid.label;
        }
        v.cancelAlign();

        // --- two points ---
        key('a');
        out.armed2 = {armed: !!v.aligning, want: v.aligning && v.aligning.want,
                      optic: v.aligning && v.aligning.optic,
                      status: v.statusBar.textContent};
        clickAt(screenOf(P1));
        out.tookFirst = v.aligning ? v.aligning.points.slice() : null;
        // The preview: an arm to the cursor, and the outline turned.
        hover(screenOf(P2));
        out.preview = {
            path: v.alignPath.style.display !== 'none',
            outline: v.outline.style.display !== 'none',
            status: v.statusBar.textContent
        };
        before = sent.length;
        clickAt(screenOf(P2));
        out.align2 = {msg: sent[before] || null, n: sent.length - before,
                      armed: !!v.aligning,
                      path: v.alignPath.style.display !== 'none'};

        // The same two places the other way about: a line has two
        // normals, the nearer is taken, so this is the same aim.
        key('a');
        clickAt(screenOf(P2));
        before = sent.length;
        clickAt(screenOf(P1));
        out.align2rev = {msg: sent[before] || null};

        // --- three points ---
        key('b');
        out.armed3 = {armed: !!v.aligning,
                      want: v.aligning && v.aligning.want};
        clickAt(screenOf(Q1));
        clickAt(screenOf(Q2));
        before = sent.length;
        clickAt(screenOf(Q3));
        out.align3 = {msg: sent[before] || null, n: sent.length - before,
                      armed: !!v.aligning};

        // --- the quarter turn ---
        before = sent.length;
        key(']');
        out.turnPlus = sent[before] || null;
        before = sent.length;
        key('[');
        out.turnMinus = sent[before] || null;

        // --- what the tool refuses ---
        key('a');
        clickAt(screenOf(P1));
        before = sent.length;
        clickAt(screenOf(P1));
        out.samePlace = {points: v.aligning ? v.aligning.points.length : -1,
                         n: sent.length - before};
        before = sent.length;
        key('Escape');
        out.cancelled = {armed: !!v.aligning, n: sent.length - before,
                         stillSelected: v.selectedOptic};

        // Nothing is grabbed while aiming: a drag pans, as it does
        // while measuring.
        key('a');
        var cx0 = v.cx;
        before = sent.length;
        dragFromTo(screenOf(M1C), [screenOf(M1C)[0] + 60, screenOf(M1C)[1]]);
        out.noGrab = {panned: v.cx !== cx0, n: sent.length - before,
                      armed: !!v.aligning};
        key('Escape');

        // Measuring and aiming are both modes, and cannot be on at
        // once.
        key('a');
        v.toggleMeasure(true);
        out.exclusive = {aligning: !!v.aligning, measuring: v.measuring};
        v.toggleMeasure(false);

        // Nothing to aim once the selection is let go of.
        key('Escape');
        clickAt(screenOf([0.05, 0.62]));
        out.disabledAgain = v.alignBtn.disabled;

        out.sent = sent;
    } catch (e) {
        out.error = String(e && e.stack || e);
    }
    document.getElementById('out').textContent = JSON.stringify(out);
})();
</script>
</body></html>
'''

def run(editable):
    page = PAGE.replace('__CSS__', viewer_css()) \
               .replace('__ESM__', js(esm)) \
               .replace('__SCENE__', js(scene)) \
               .replace('__EDITABLE__', 'true' if editable else 'false')
    path = os.path.join(SP, 'align_page_%s.html' % editable)
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

def normal_of(name):
    '''
    The unit normal of an optics' front face, from the model.
    '''
    o = layout.get_optics(name)
    return np.asarray(o.normVectHR, dtype='float64')

def unit(a, b):
    v = np.asarray(b, dtype='float64') - np.asarray(a, dtype='float64')
    return v / np.linalg.norm(v)

def wrapped(a):
    '''
    An angle into (-pi, pi]. Two aims that differ by a turn are the
    same aim, and the page has no reason to prefer one spelling.
    '''
    return (a + np.pi) % (2 * np.pi) - np.pi


print('--- editable viewer ---')
errs, res = run(True)
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:500])

P1, P2 = res['points']['P1'], res['points']['P2']
Q1, Q2, Q3 = (res['points']['Q1'], res['points']['Q2'], res['points']['Q3'])
check('both sides picked the same places',
      P1 == hole_near(0.15, 0.05) and Q2 == hole_near(0.30, 0.10),
      json.dumps(res['points']))

print('--- the button and its menu ---')
check('nothing to aim with nothing selected',
      res['disabledWithNothingSelected'] is True)
check('and something once an optics is', res['selected'] == 'M1'
      and res['enabledWithOptic'] is True)
check('the menu offers both aims and both turns',
      res['items'] == ['Line 2 points', 'Bisect 3 points',
                       'Turn +45°', 'Turn −45°'],
      json.dumps(res['items']))

print('--- aiming by the middle of an edge ---')
am = res.get('alignMid')
check('the scene offers an edge middle to aim by', am is not None,
      '' if am is None else '(%s, %.1f mm clear)'
      % (am['label'], res['midClear']*1000))
if am:
    check('the cursor a couple of pixels off it takes the middle',
          am['hovered'] == am['label'], str(am['hovered']))
    check('  and the click lands on it exactly',
          am['taken'] == am['want'],
          '%s vs %s' % (am['taken'], am['want']))

print('--- square across two points ---')
a2 = res['armed2']
check('a arms the two-point aim for the selection',
      a2['armed'] and a2['want'] == 2 and a2['optic'] == 'M1',
      json.dumps(a2))
check('  and says so', 'Align M1' in a2['status']
      and 'first point' in a2['status'], a2['status'])
check('the first click takes the marked place, not the cursor',
      res['tookFirst'] and np.allclose(res['tookFirst'][0], P1),
      json.dumps(res['tookFirst']))
check('the preview shows the arm and the outline',
      res['preview']['path'] and res['preview']['outline'])
check('  with the angle it would come to',
      '→' in res['preview']['status'], res['preview']['status'])

m2 = res['align2']
check('the second click sends one rotate and puts the tool away',
      m2['n'] == 1 and m2['msg'] and m2['msg']['op'] == 'rotate'
      and m2['msg']['target'] == 'M1' and not m2['armed']
      and not m2['path'], json.dumps(m2))
if m2['msg']:
    before_anchor = np.asarray(layout.get_optics('M1').HRcenter).copy()
    layout.apply_edit(m2['msg'])
    n = normal_of('M1')
    along = unit(P1, P2)
    # The whole claim: the face is square across the line, which is
    # its normal being parallel to it. Checked as a cross product, so
    # either direction along the line passes - which is the point of
    # the next check.
    check('  the face ends up square across the two places',
          abs(n[0] * along[1] - n[1] * along[0]) < 1e-12,
          '(normal %s, line %s)' % (np.round(n, 6), np.round(along, 6)))
    # Which way round is the click order's to say: the face looks
    # from the first place towards the second.
    check('  looking towards the second place, not the first',
          abs(float(np.dot(n, along)) - 1.0) < 1e-12,
          '(dot %.9f)' % np.dot(n, along))
    # Aiming turns and does not move: the point the element is held
    # by stays where it was. Its substrate centre does travel, since
    # turning is about the anchor - that is the model's own rule, not
    # something the aim decides.
    check('  and it turned about its anchor, which stayed put',
          np.allclose(layout.get_optics('M1').HRcenter, before_anchor))
    layout.apply_edit({'op': 'undo'})

rev = res['align2rev']
check('clicking the two places the other way about turns it right round',
      rev['msg'] and m2['msg']
      and abs(abs(wrapped(rev['msg']['normAngleHR']
                          - m2['msg']['normAngleHR'])) - np.pi) < 1e-12,
      json.dumps(rev['msg']))

print('--- bisect three points ---')
a3 = res['armed3']
check('b arms the three-point aim', a3['armed'] and a3['want'] == 3,
      json.dumps(a3))
m3 = res['align3']
check('the third click sends the rotate',
      m3['n'] == 1 and m3['msg'] and m3['msg']['op'] == 'rotate'
      and not m3['armed'], json.dumps(m3))
if m3['msg']:
    layout.apply_edit(m3['msg'])
    n = normal_of('M1')
    arm1 = unit(Q2, Q1)
    arm2 = unit(Q2, Q3)
    # The law of reflection: light from the first place leaves towards
    # the last when the normal makes the same angle with both arms.
    check('  the face bisects the corner',
          abs(float(np.dot(n, arm1)) - float(np.dot(n, arm2))) < 1e-12,
          '(%.9f vs %.9f)' % (np.dot(n, arm1), np.dot(n, arm2)))
    check('  facing into it, not away',
          float(np.dot(n, arm1)) > 0)
    layout.apply_edit({'op': 'undo'})

print('--- the quarter turn ---')
was = float(layout.get_optics('M1').normAngleHR)
tp, tm = res['turnPlus'], res['turnMinus']
check(']  turns a quarter turn counterclockwise',
      tp and tp['op'] == 'rotate'
      and abs(tp['normAngleHR'] - (was + np.pi / 4)) < 1e-12,
      json.dumps(tp))
check('[  turns it back the other way',
      tm and abs(tm['normAngleHR'] - (was - np.pi / 4)) < 1e-12,
      json.dumps(tm))
if tp:
    layout.apply_edit(tp)
    check('  and Python lands it there',
          abs(float(layout.get_optics('M1').normAngleHR)
              - tp['normAngleHR']) < 1e-12)
    layout.apply_edit({'op': 'undo'})

print('--- what the tool refuses ---')
sp = res['samePlace']
check('the same place twice names no direction',
      sp['points'] == 1 and sp['n'] == 0, json.dumps(sp))
ca = res['cancelled']
check('Escape lets the aim go, and nothing is sent',
      not ca['armed'] and ca['n'] == 0, json.dumps(ca))
check('  keeping the selection it was being taken for',
      ca['stillSelected'] == 'M1')
ng = res['noGrab']
check('a drag while aiming pans rather than grabbing',
      res['noGrab']['panned'] and ng['n'] == 0 and ng['armed'],
      json.dumps(ng))
check('measuring and aiming cannot both be on',
      not res['exclusive']['aligning'] and res['exclusive']['measuring'],
      json.dumps(res['exclusive']))
check('nothing to aim once the selection is let go of',
      res['disabledAgain'] is True)

print('--- read-only viewer ---')
errs, res = run(False)
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:500])
check('a page with nowhere to send an aim has no Align button',
      not res['hasButton'])
check('  and clicking an optics still selects it', res['selected'] == 'M1')
check('nothing was ever sent', res['sent'] == [], str(res['sent'][:2]))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
