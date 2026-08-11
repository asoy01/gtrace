'''
Drawing a part with the mouse: picking a shape out of the drawing,
carrying it, and taking hold of it by its grips.

The panel already had every number a shape is made of, and typing
them is how a part is drawn exactly. This is the other half - the
rough half - and what wants checking is that the two agree: a corner
dragged to a place lands on that place, the opposite corner does not
drift, and the message that says so is the same set_shape a typed row
sends.

Every message the page produces is applied to a real ShapeEditor,
built fresh from the same part, and the shape that comes out is
checked geometrically. Nothing here trusts the arithmetic the page
used to get there - the page is asked what it would do, and Python is
asked what that would mean.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

import copy
import json
import math
import re
import subprocess

import numpy as np

import gtrace.draw as draw
from gtrace.draw.viewer import viewer_css
from gtrace.mechanics import Mechanics
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

#{{{ The part, and a fresh editor for every message

# A part with one of everything that can be taken hold of, laid out so
# that no two of them are hit by the same click except where that is
# the thing being checked: the circle sits inside the rectangle, which
# is how the pick order is measured.
def make_part():
    return Mechanics(
        shapes=[draw.Rectangle([-0.02, -0.01], 0.04, 0.02),
                draw.Circle([0.005, 0.0], 0.003),
                draw.PolyLine([0.03, 0.05, 0.04], [0.0, 0.0, 0.02]),
                draw.Line([-0.05, 0.03], [-0.02, 0.03]),
                draw.Arc([0.0, -0.04], 0.01, 0.0, np.pi)],
        center=[0.3, 0.1], name='P1')

RECT, CIRCLE, POLY, LINE, ARC = range(5)

part = make_part()
scene = ShapeEditor(part).scene_dict()

def applied(msg):
    '''
    A part with one message applied to it, from the state the page
    was handed. The page never gets a scene back, so every gesture it
    makes is against the original - and so is every check.
    '''
    p = make_part()
    ShapeEditor(p).apply_edit(copy.deepcopy(msg))
    return p

#}}}

#{{{ The page

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
    function mouse(target, type, x, y, opts) {
        target.dispatchEvent(new MouseEvent(type, Object.assign({
            clientX: x, clientY: y, button: 0, bubbles: true, cancelable: true
        }, opts || {})));
    }
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

        // Re-measured every time: the status bar changes length as the
        // cursor moves, and that reflows the page.
        function rect() { return v.svg.getBoundingClientRect(); }
        function screenOf(p) {
            var r = rect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function clientOfHandle(k) {
            var r = rect();
            var h = v._shapeHandlePts[k];
            return [h.px + r.left, h.py + r.top];
        }
        function clickAt(p, opts) {
            mouse(v.svg, 'mousedown', p[0], p[1], opts);
            mouse(window, 'mouseup', p[0], p[1], opts);
        }
        function dragFromTo(a, b, opts) {
            mouse(v.svg, 'mousedown', a[0], a[1], opts);
            mouse(window, 'mousemove',
                  (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, opts);
            mouse(window, 'mousemove', b[0], b[1], opts);
            mouse(window, 'mouseup', b[0], b[1], opts);
        }
        // Alt is "the cursor means what it says": without it a drag
        // settles on the marked points, which is a separate check.
        var FREE = {altKey: true};
        function last(n) { return sent[sent.length - 1] || null; }
        function fields() {
            var o = {};
            for (var k in (v.shapeFields || {})) {
                var f = v.shapeFields[k];
                o[k] = f.editable ? f.el.value : f.el.textContent;
            }
            return o;
        }
        function handlePoints() {
            return (v._shapeHandlePts || []).map(function (h) {
                return [h.px, h.py, h.role, h.i];
            });
        }
        var before;

        // --- picking a shape out of the drawing ---
        clickAt(screenOf([-0.018, 0.008]));      // inside the rectangle
        out.pickRect = v.selectedShape;
        clickAt(screenOf([0.005, 0.0]));         // the circle inside it
        out.pickCircle = v.selectedShape;
        clickAt(screenOf([-0.035, 0.03]));       // along the line
        out.pickLine = v.selectedShape;
        clickAt(screenOf([0.0, 0.05]));          // nothing there
        out.pickNothing = v.selectedShape;
        // The same place twice steps down through what overlaps.
        var over = screenOf([0.006, 0.001]);
        clickAt(over);
        out.cycle0 = v.selectedShape;
        clickAt(over);
        out.cycle1 = v.selectedShape;
        clickAt(over);
        out.cycle2 = v.selectedShape;

        // --- carrying a shape ---
        clickAt(screenOf([-0.018, 0.008]));
        before = sent.length;
        dragFromTo(screenOf([-0.018, 0.008]), screenOf([-0.008, 0.013]), FREE);
        out.moveRect = {msg: last(), n: sent.length - before};

        // An unselected shape is not grabbed: the press pans, exactly
        // as it does on a breadboard.
        clickAt(screenOf([0.0, 0.05]));
        before = sent.length;
        var cx0 = v.cx;
        dragFromTo(screenOf([-0.018, 0.008]), screenOf([-0.008, 0.013]), FREE);
        out.panNotMove = {n: sent.length - before, panned: v.cx !== cx0};

        // --- the grips ---
        clickAt(screenOf([-0.018, 0.008]));
        out.rectHandles = handlePoints();
        out.rectHandleWant = [[-0.02, -0.01], [0.02, -0.01],
                              [0.02, 0.01], [-0.02, 0.01]].map(function (p) {
            return v.sceneToScreen(p[0], p[1]);
        });
        before = sent.length;
        dragFromTo(clientOfHandle(0), screenOf([-0.03, -0.02]), FREE);
        out.dragCorner = {msg: last(), n: sent.length - before};

        clickAt(screenOf([0.005, 0.0]));
        out.circleHandles = handlePoints();
        before = sent.length;
        dragFromTo(clientOfHandle(0), screenOf([0.013, 0.0]), FREE);
        out.dragRadius = {msg: last(), n: sent.length - before};

        clickAt(screenOf([-0.035, 0.03]));
        out.lineHandles = handlePoints();
        before = sent.length;
        dragFromTo(clientOfHandle(1), screenOf([-0.025, 0.045]), FREE);
        out.dragEnd = {msg: last(), n: sent.length - before};

        clickAt(screenOf([0.01, -0.04]));        // the right end of the arc
        out.arcSelected = v.selectedShape;
        out.arcHandles = handlePoints();
        before = sent.length;
        dragFromTo(clientOfHandle(0), screenOf([0.006, -0.032]), FREE);
        out.dragArcStart = {msg: last(), n: sent.length - before};

        // A grip that never moved decides nothing.
        before = sent.length;
        clickAt(clientOfHandle(2));
        out.gripClickSends = sent.length - before;

        // --- a polyline, vertex by vertex ---
        clickAt(screenOf([0.04, 0.0]));          // along its bottom edge
        out.pickPoly = v.selectedShape;
        out.polyFields = fields();
        out.polyHandles = handlePoints();
        // Taking hold of a vertex is how one is picked out.
        clickAt(clientOfHandle(1));
        out.vertexPicked = {selected: v.selectedVertex,
                            fields: fields()};
        before = sent.length;
        dragFromTo(clientOfHandle(1), screenOf([0.055, 0.008]), FREE);
        out.dragVertex = {msg: last(), n: sent.length - before};

        function button(text) {
            var found = null;
            Array.prototype.forEach.call(
                el.querySelectorAll('button'), function (b) {
                    if (b.textContent === text) { found = b; }
                });
            return found;
        }
        out.vertexRowShown = v.vertexFoot.style.display !== 'none';
        before = sent.length;
        button('+ Vertex').click();
        out.addVertex = {msg: last(), n: sent.length - before,
                         selected: v.selectedVertex};
        before = sent.length;
        button('\\u2212 Vertex').click();
        out.removeVertex = {msg: last(), n: sent.length - before};
        // Typing a vertex row sends the whole list, with one point
        // changed - which is what a polyline is made of.
        var f = v.shapeFields.vx;
        f.el.value = '60';
        f.el.dispatchEvent(new Event('change', {bubbles: true}));
        out.typedVertex = last();

        // The row of vertex buttons belongs to a polyline alone.
        clickAt(screenOf([0.005, 0.0]));
        out.vertexRowHidden = v.vertexFoot.style.display === 'none';

        // --- settling on the marked points ---
        // The centre of the circle carried to within a few pixels of
        // the origin, which is the one point every part is drawn from.
        clickAt(screenOf([0.0, 0.05]));
        clickAt(screenOf([0.005, 0.0]));
        before = sent.length;
        var near = screenOf([0.0005, 0.0004]);
        var from = screenOf([0.005, 0.0]);
        mouse(v.svg, 'mousedown', from[0], from[1]);
        mouse(window, 'mousemove', near[0], near[1]);
        out.snapMarked = v.snapMark.style.display !== 'none';
        mouse(window, 'mouseup', near[0], near[1]);
        out.snapped = {msg: last(), n: sent.length - before,
                       marked: out.snapMarked};
        before = sent.length;
        dragFromTo(screenOf([0.005, 0.0]), near, FREE);
        out.notSnapped = last();

        // The middle of an edge catches as a corner does: the circle
        // carried to near the middle of the line, which is the only
        // marked point anywhere about there.
        clickAt(screenOf([0.0, 0.05]));
        clickAt(screenOf([0.005, 0.0]));
        before = sent.length;
        dragFromTo(screenOf([0.005, 0.0]), screenOf([-0.0345, 0.0295]));
        out.snappedMid = last();

        // --- turning ---
        // Shift takes hold of a shape to turn it, as it does out on
        // the bench. Taken hold of by an edge rather than a vertex:
        // a grip does its own job whatever else is held down.
        clickAt(screenOf([0.0, 0.05]));
        clickAt(screenOf([0.04, 0.0]));
        out.turnPicked = v.selectedShape;
        before = sent.length;
        var tf = screenOf([0.04, 0.0]), tt = screenOf([0.05, 0.01]);
        mouse(v.svg, 'mousedown', tf[0], tf[1], {shiftKey: true});
        mouse(window, 'mousemove', tt[0], tt[1], {shiftKey: true});
        out.turnStatus = v.statusBar.textContent;
        mouse(window, 'mouseup', tt[0], tt[1], {shiftKey: true});
        out.turnDrag = {msg: last(), n: sent.length - before};

        // The quarter turn, on the keys that turn an optics out on
        // the bench - and on a rectangle, which cannot carry one.
        clickAt(screenOf([0.0, 0.05]));
        clickAt(screenOf([-0.018, 0.008]));
        before = sent.length;
        window.dispatchEvent(new KeyboardEvent('keydown',
                                               {key: ']', bubbles: true}));
        out.quarterTurn = {msg: last(), n: sent.length - before};
        window.dispatchEvent(new KeyboardEvent('keydown',
                                               {key: '[', bubbles: true}));
        out.quarterBack = last();

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
    path = os.path.join(SP, 'editor_drag_%s.html' % tag)
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

#}}}

errs, res = run(scene, 'part')
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:600])

print('--- picking a shape out of the drawing ---')
check('a click inside a plate picks it', res['pickRect'] == RECT,
      str(res['pickRect']))
check('a smaller shape on top of it wins', res['pickCircle'] == CIRCLE,
      str(res['pickCircle']))
check('a line is picked by its outline, having no inside',
      res['pickLine'] == LINE, str(res['pickLine']))
check('a click on nothing lets go', res['pickNothing'] is None,
      str(res['pickNothing']))
check('the same place again steps down through what overlaps',
      [res['cycle0'], res['cycle1'], res['cycle2']] == [CIRCLE, RECT, CIRCLE],
      json.dumps([res['cycle0'], res['cycle1'], res['cycle2']]))

print('--- carrying a shape ---')
mv = res['moveRect']
check('a drag sends one message, and it is a set_shape',
      mv['n'] == 1 and mv['msg'] and mv['msg']['op'] == 'set_shape'
      and mv['msg']['index'] == RECT, json.dumps(mv['msg']))
if mv['msg']:
    p = applied(mv['msg'])
    r0, r1 = part.shapes[RECT], p.shapes[RECT]
    check('  the plate is carried, and keeps its size',
          abs(r1.width - r0.width) < 1e-15
          and abs(r1.height - r0.height) < 1e-15
          and not np.allclose(r1.point, r0.point),
          '%s -> %s' % (np.round(r0.point, 5), np.round(r1.point, 5)))
    check('  by as far as the cursor went',
          abs((r1.point[0] - r0.point[0]) - 0.010) < 2e-4
          and abs((r1.point[1] - r0.point[1]) - 0.005) < 2e-4,
          str(np.round(np.subtract(r1.point, r0.point), 6)))
    check('  and everything else is left alone',
          np.allclose(p.shapes[CIRCLE].center, part.shapes[CIRCLE].center))
pn = res['panNotMove']
check('an unselected shape is not grabbed; the view pans instead',
      pn['n'] == 0 and pn['panned'], json.dumps(pn))

print('--- the grips ---')
check('a rectangle stands on four, at its corners',
      len(res['rectHandles']) == 4
      and all(h[2] == 'corner' for h in res['rectHandles'])
      and all(abs(res['rectHandles'][i][0] - res['rectHandleWant'][i][0]) < 0.5
              and abs(res['rectHandles'][i][1] - res['rectHandleWant'][i][1]) < 0.5
              for i in range(4)),
      json.dumps(res['rectHandles']))
dc = res['dragCorner']
check('dragging one sends a set_shape', dc['n'] == 1 and dc['msg']
      and dc['msg']['index'] == RECT, json.dumps(dc['msg']))
if dc['msg']:
    r0, r1 = part.shapes[RECT], applied(dc['msg']).shapes[RECT]
    far0 = np.add(r0.point, [r0.width, r0.height])
    far1 = np.add(r1.point, [r1.width, r1.height])
    check('  the opposite corner stays exactly where it was',
          np.allclose(far0, far1, atol=1e-15), str(np.round(far1, 6)))
    check('  and the dragged one lands under the cursor',
          abs(r1.point[0] + 0.03) < 2e-4 and abs(r1.point[1] + 0.02) < 2e-4,
          str(np.round(r1.point, 6)))
check('a circle stands on four, on its rim',
      len(res['circleHandles']) == 4
      and all(h[2] == 'radius' for h in res['circleHandles']),
      json.dumps([h[2] for h in res['circleHandles']]))
dr = res['dragRadius']
if dr['msg']:
    c0, c1 = part.shapes[CIRCLE], applied(dr['msg']).shapes[CIRCLE]
    check('dragging one sets the radius and leaves the centre',
          np.allclose(c1.center, c0.center, atol=1e-15)
          and abs(c1.radius - 0.008) < 2e-4,
          '%s r=%s' % (np.round(c1.center, 5), round(c1.radius, 6)))
check('a line stands on its two ends',
      len(res['lineHandles']) == 2
      and all(h[2] == 'end' for h in res['lineHandles']),
      json.dumps([h[2] for h in res['lineHandles']]))
de = res['dragEnd']
if de['msg']:
    l0, l1 = part.shapes[LINE], applied(de['msg']).shapes[LINE]
    check('dragging the far end moves that end alone',
          np.allclose(l1.start, l0.start, atol=1e-15)
          and abs(l1.stop[0] + 0.025) < 2e-4
          and abs(l1.stop[1] - 0.045) < 2e-4,
          '%s -> %s' % (np.round(l0.stop, 5), np.round(l1.stop, 5)))
check('an arc stands on three: where it starts, stops, and how far out',
      [h[2] for h in res['arcHandles']]
      == ['startangle', 'stopangle', 'radius'],
      json.dumps([h[2] for h in res['arcHandles']]))
da = res['dragArcStart']
if da['msg']:
    a0, a1 = part.shapes[ARC], applied(da['msg']).shapes[ARC]
    check('dragging the start turns that end, keeping the radius',
          abs(a1.radius - a0.radius) < 1e-15
          and abs(a1.stopangle - a0.stopangle) < 1e-15
          and abs(a1.startangle - math.atan2(0.008, 0.006)) < 0.02,
          'start %.4f -> %.4f' % (a0.startangle, a1.startangle))
check('a grip pressed and let go decides nothing',
      res['gripClickSends'] == 0, str(res['gripClickSends']))

print('--- a polyline, vertex by vertex ---')
check('the outline is picked by its edge', res['pickPoly'] == POLY,
      str(res['pickPoly']))
check('its rows are about one vertex, and say which of how many',
      res['polyFields'].get('vertex') == '1 of 3'
      and abs(float(res['polyFields']['vx']) - 30) < 1e-9
      and abs(float(res['polyFields']['vy'])) < 1e-9,
      json.dumps(res['polyFields']))
check('it stands on one grip per vertex',
      len(res['polyHandles']) == 3
      and all(h[2] == 'vertex' for h in res['polyHandles']),
      json.dumps([h[2] for h in res['polyHandles']]))
vp = res['vertexPicked']
check('taking hold of one is how it is picked out',
      vp['selected'] == 1 and vp['fields'].get('vertex') == '2 of 3',
      json.dumps(vp))
dv = res['dragVertex']
check('dragging it sends the whole list',
      dv['n'] == 1 and dv['msg'] and 'x' in dv['msg']['attrs']
      and 'y' in dv['msg']['attrs'], json.dumps(dv['msg']))
if dv['msg']:
    q0, q1 = part.shapes[POLY], applied(dv['msg']).shapes[POLY]
    check('  that vertex moves and the others do not',
          len(q1.x) == 3
          and abs(q1.x[1] - 0.055) < 2e-4 and abs(q1.y[1] - 0.008) < 2e-4
          and abs(q1.x[0] - q0.x[0]) < 1e-15
          and abs(q1.x[2] - q0.x[2]) < 1e-15,
          '%s %s' % (np.round(q1.x, 5), np.round(q1.y, 5)))
check('the vertex buttons are up for a polyline', res['vertexRowShown'])
check('  and away for anything else', res['vertexRowHidden'])
av = res['addVertex']
check('+ Vertex puts one in after the one in hand, and takes hold of it',
      av['n'] == 1 and av['msg'] and av['selected'] == 2, json.dumps(av))
if av['msg']:
    q0, q1 = part.shapes[POLY], applied(av['msg']).shapes[POLY]
    check('  halfway along to the next one',
          len(q1.x) == 4
          and abs(q1.x[2] - (q0.x[1] + q0.x[2]) / 2) < 1e-15
          and abs(q1.y[2] - (q0.y[1] + q0.y[2]) / 2) < 1e-15,
          '%s %s' % (np.round(q1.x, 5), np.round(q1.y, 5)))
rv = res['removeVertex']
check('- Vertex takes the one in hand out', rv['n'] == 1 and rv['msg'],
      json.dumps(rv['msg']))
if rv['msg']:
    q1 = applied(rv['msg']).shapes[POLY]
    check('  leaving the rest of the outline as it was',
          len(q1.x) == 2 and abs(q1.x[0] - 0.03) < 1e-15
          and abs(q1.x[1] - 0.05) < 1e-15,
          '%s %s' % (np.round(q1.x, 5), np.round(q1.y, 5)))
tv = res['typedVertex']
check('a typed vertex row is the same message as a dragged grip',
      tv and tv['op'] == 'set_shape' and 'x' in tv['attrs'],
      json.dumps(tv))
if tv:
    q1 = applied(tv).shapes[POLY]
    check('  and 60 in the panel is 60 mm in the model',
          abs(q1.x[1] - 0.060) < 1e-15 and len(q1.x) == 3,
          str(np.round(q1.x, 5)))

print('--- settling on the marked points ---')
sn = res['snapped']
check('a drag near the origin settles exactly on it',
      sn['msg'] and np.allclose(sn['msg']['attrs']['center'], [0.0, 0.0],
                                atol=1e-15),
      json.dumps(sn['msg']))
check('  and the mark says where it caught', sn['marked'])
ns = res['notSnapped']
check('with Alt the cursor means what it says',
      ns and not np.allclose(ns['attrs']['center'], [0.0, 0.0], atol=1e-9)
      and abs(ns['attrs']['center'][0] - 0.0005) < 2e-4,
      json.dumps(ns))

sm = res['snappedMid']
check('the middle of a line catches a drag as its ends do',
      sm and np.allclose(sm['attrs']['center'], [-0.035, 0.03], atol=1e-15),
      json.dumps(sm))

print('--- turning ---')
check('the outline is the shape in hand', res['turnPicked'] == POLY,
      str(res['turnPicked']))
td = res['turnDrag']
check('Shift + drag sends a turn, about the middle of the shape',
      td['n'] == 1 and td['msg'] and td['msg']['op'] == 'rotate_shape'
      and td['msg']['index'] == POLY
      and np.allclose(td['msg']['pivot'], [0.04, 0.01], atol=1e-9),
      json.dumps(td['msg']))
# The two arms are named in scene coordinates and clicked in whole
# pixels, so a degree either way is the screen rather than the turn.
# What the angle exactly is, is settled below against the vertices.
check('  by the angle the cursor swung through',
      td['msg'] and abs(td['msg']['angle'] - np.pi / 2) < 0.05,
      str(round(td['msg']['angle'], 4)) if td['msg'] else '')
check('  and says so while it is being made',
      'turned by' in (res['turnStatus'] or ''), res['turnStatus'])
if td['msg']:
    q0 = part.shapes[POLY]
    q1 = applied(td['msg']).shapes[POLY]
    a = td['msg']['angle']
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    want = (np.column_stack([q0.x, q0.y]) - [0.04, 0.01]) @ R.T + [0.04, 0.01]
    check('  which is where Python puts the vertices',
          np.allclose(np.column_stack([q1.x, q1.y]), want, atol=1e-15),
          str(np.round(np.column_stack([q1.x, q1.y]), 5).tolist()))
qt = res['quarterTurn']
check('] turns the shape in hand by 45 degrees',
      qt['n'] == 1 and qt['msg'] and qt['msg']['op'] == 'rotate_shape'
      and qt['msg']['index'] == RECT
      and abs(qt['msg']['angle'] - np.pi / 4) < 1e-12, json.dumps(qt['msg']))
check('  and [ turns it back the other way',
      res['quarterBack'] and abs(res['quarterBack']['angle'] + np.pi / 4)
      < 1e-12, json.dumps(res['quarterBack']))
if qt['msg']:
    r1 = applied(qt['msg']).shapes[RECT]
    check('  a turned rectangle reaching Python is a closed outline',
          isinstance(r1, draw.PolyLine) and len(r1.x) == 5
          and r1.x[0] == r1.x[-1] and r1.y[0] == r1.y[-1],
          type(r1).__name__)

print('--- what the whole session sent ---')
ops = set(m.get('op') for m in res['sent'])
check('nothing but set_shape and rotate_shape ever left the page',
      ops == {'set_shape', 'rotate_shape'}, json.dumps(sorted(ops)))
check('every message names a shape the part has',
      all(0 <= m['index'] < len(part.shapes) for m in res['sent']),
      str(len(res['sent'])) + ' messages')

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
