'''
Editing the one shape of a hand-drawn body, from the layout itself.

A body of one shape is a drawing rather than a part - a tank wall, an
aperture, the edge of a table, a note - and it is what ``+ Shape``
puts down. Until now the layout could move it and turn it and nothing
else: its radius, its width, the ends of its line were reachable only
through ``Mechanics.edit()``, which is a cell in a notebook and a
different viewer. A shape that can be put down and not resized is not
worth putting down.

So the body carries its shape in the ``mechanics`` channel, the panel
shows that shape's own rows under the pose rows, and its grips stand
on the drawing where the body carries them. Three things are checked
hardest here.

The first is the frame. A shape is written in the body's own
coordinates and the body has a pose, so a grip stands at the shape's
point turned by that pose and carried - and a drag has to come back
the other way before it can say anything about the shape. A body that
is turned is therefore the case worth driving, not the easy one.

The second is that the arithmetic is the shape editor's. The rows are
``SHAPE_FIELDS``, the drag is ``shapeHandleAttrs``: the same functions
answering the same question in the other viewer, so a corner dragged
here lands where a corner dragged there would.

The third is what has no shape to edit. A part off the library shelf
is cut to size by its corner handles instead, one of several shapes is
edited where there is a list to pick from, and an attached body is not
edited here at all - each of them offers no rows and no grips rather
than offering something that would be refused.

Every message the page sends is fed to a real ``OpticalLayout`` and
the result checked against what the gesture meant.
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
import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.layout import OpticalLayout, TraceRules, EditError, q_from_waist
from gtrace.mechanics import Mechanics, breadboard
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
    '''
    Four bodies, each answering a different question about what has a
    shape to edit: a circle drawn by hand, a rectangle drawn by hand
    and turned, a breadboard off the shelf, and a mount of two shapes.
    They stand well apart so that a click on one is a click on it.
    '''
    b0 = GaussianBeam(q0=q_from_waist(1*mm, 0.0, 1064*nm), wl=1064*nm,
                      pos=[0, 0], dirAngle=0, name='b0')
    M1 = opt.Mirror(HRcenter=[0.4, 0.0], normAngleHR=np.pi, diameter=5*cm,
                    thickness=2*cm, name='M1')
    ring = Mechanics(shapes=[draw.Circle([0.0, 0.0], 0.05)],
                     center=[0.0, 0.4], name='Ring')
    # Turned, and its shape written off its own origin: the frame is
    # the thing being checked, so neither the pose nor the shape may
    # be the identity.
    plate = Mechanics(shapes=[draw.Rectangle([0.0, -0.02], 0.16, 0.04)],
                      center=[0.6, 0.4], rotationAngle=0.6, name='Plate')
    line = Mechanics(shapes=[draw.Line([-0.06, 0.0], [0.06, 0.0])],
                     center=[0.0, -0.4], name='Line1')
    board = breadboard(0.3, 0.2, center=[0.6, -0.4], name='Board')
    mount = Mechanics(shapes=[draw.Circle([0.0, 0.0], 0.012),
                              draw.Rectangle([-0.015, -0.015], 0.03, 0.03)],
                      center=[1.1, 0.25], name='Mount')
    clamp = Mechanics(shapes=[draw.Circle([0.0, 0.0], 0.02)],
                      name='Clamp', attached_to=M1, offset=[0.0, -0.12])
    return OpticalLayout(optics=[M1], sources=[b0],
                         mechanics=[ring, plate, line, board, mount, clamp],
                         rules=TraceRules(order=1, power_threshold=1e-3))

layout = make_layout()
scene = layout.scene_dict()

def js(obj):
    return json.dumps(obj).replace('</', '<\\/')

with open(os.path.join(REPO, 'gtrace', 'draw', 'viewer', 'viewer.js'),
          encoding='utf-8') as f:
    VIEWER_JS = f.read()
with open(os.path.join(REPO, 'gtrace', 'draw', 'viewer', 'viewer.css'),
          encoding='utf-8') as f:
    VIEWER_CSS = f.read()

PAGE = '''<!doctype html><html><head><meta charset="utf-8">
<style>__CSS__</style></head><body>
<div id="host" style="width:1200px;height:700px"></div>
<div id="out" style="display:none"></div>
<script>__JS__</script>
<script>
(function () {
    var out = {error: null, sent: []};
    try {
        var scene = __SCENE__;
        var v = GTraceViewer.mount(document.getElementById('host'), scene, {
            onEdit: function (msg) { out.sent.push(msg); }
        });
        var sent = out.sent;

        function rect() { return v.svg.getBoundingClientRect(); }
        function screenOf(p) {
            var r = rect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function mouse(target, type, x, y) {
            target.dispatchEvent(new MouseEvent(type, {
                clientX: x, clientY: y, button: 0,
                bubbles: true, cancelable: true}));
        }
        function clickAt(p) {
            mouse(v.svg, 'mousedown', p[0], p[1]);
            mouse(window, 'mouseup', p[0], p[1]);
        }
        function dragFromTo(a, b) {
            mouse(v.svg, 'mousedown', a[0], a[1]);
            mouse(window, 'mousemove', (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
            mouse(window, 'mousemove', b[0], b[1]);
            mouse(window, 'mouseup', b[0], b[1]);
        }
        function body(name) {
            var found = null;
            (v.scene.mechanics || []).forEach(function (b) {
                if (b.name === name) { found = b; }
            });
            return found;
        }
        function shapeRows() {
            var o = {};
            Object.keys(v.mechShapeFields || {}).forEach(function (k) {
                var f = v.mechShapeFields[k];
                o[k] = {shown: f.row.style.display !== 'none',
                        value: f.editable ? f.el.value : f.el.textContent,
                        editable: !!f.editable};
            });
            return o;
        }
        function typeRow(key, text) {
            var f = (v.mechShapeFields || {})[key];
            if (!f || !f.editable) { return null; }
            var n = sent.length;
            f.el.value = text;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
            return {msg: sent[n] || null, n: sent.length - n};
        }
        function gripScreen(i) {
            var r = rect();
            return [v._mechShapePts[i][0] + r.left,
                    v._mechShapePts[i][1] + r.top];
        }

        // --- a circle drawn by hand ---
        var ring = body('Ring');
        clickAt(screenOf(ring.center));
        out.ring = {picked: v.selectedMech, rows: shapeRows(),
                    grips: (v._mechShapePts || []).length};
        // Where the grips stand: on the rim, as the body carries it.
        out.ring.gripOff = null;
        if ((v._mechShapePts || []).length) {
            var want = v.sceneToScreen(ring.center[0] + ring.shape.radius,
                                       ring.center[1]);
            out.ring.gripOff = [Math.abs(v._mechShapePts[0][0] - want[0]),
                                Math.abs(v._mechShapePts[0][1] - want[1])];
        }
        out.ring.typed = typeRow('radius', '80');
        // And a drag on a grip, out to a place we can name.
        if ((v._mechShapePts || []).length) {
            var n0 = sent.length;
            var to = screenOf([ring.center[0] + 0.12, ring.center[1]]);
            dragFromTo(gripScreen(0), to);
            out.ring.dragged = sent[n0] || null;
            out.ring.dragN = sent.length - n0;
        }

        // --- a rectangle drawn by hand, on a body that is turned ---
        var plate = body('Plate');
        // The middle of the plate, not the body's origin: the shape is
        // written off that origin, so the origin is on its edge.
        var pa = plate.rotationAngle || 0;
        var pmid = [plate.center[0] + 0.08 * Math.cos(pa),
                    plate.center[1] + 0.08 * Math.sin(pa)];
        clickAt(screenOf(pmid));
        out.plate = {picked: v.selectedMech, rows: shapeRows(),
                     grips: (v._mechShapePts || []).length,
                     angle: plate.rotationAngle};
        // A corner grip, dragged to a place on the bench. The frame is
        // what this is about: the shape is written off the body's
        // origin and the body is turned, so nothing lines up by
        // accident.
        if ((v._mechShapePts || []).length === 4) {
            var m0 = sent.length;
            var target = [plate.center[0] + 0.25, plate.center[1] + 0.18];
            dragFromTo(gripScreen(2), screenOf(target));
            out.plate.dragged = sent[m0] || null;
            out.plate.target = target;
            out.plate.dragN = sent.length - m0;
        }
        out.plate.typedWidth = typeRow('width', '250');

        // --- a line: its ends ---
        var line = body('Line1');
        clickAt(screenOf(line.center));
        out.line = {picked: v.selectedMech, rows: shapeRows(),
                    grips: (v._mechShapePts || []).length};
        if ((v._mechShapePts || []).length === 2) {
            var k0 = sent.length;
            var end = [line.center[0] + 0.2, line.center[1] + 0.1];
            dragFromTo(gripScreen(1), screenOf(end));
            out.line.dragged = sent[k0] || null;
            out.line.end = end;
        }

        // --- what has no shape to edit here ---
        clickAt(screenOf([0.6, -0.4]));
        out.board = {picked: v.selectedMech,
                     rows: Object.keys(v.mechShapeFields || {}).length,
                     grips: (v._mechShapePts || []).length,
                     resizeHandles: (v._handlePts || []).length};
        // Off the beam: a beam within twelve pixels of a click wins
        // it, and a body under one is reached by clicking where the
        // beam is not.
        clickAt(screenOf([1.1, 0.25]));
        out.mount = {picked: v.selectedMech,
                     rows: Object.keys(v.mechShapeFields || {}).length,
                     grips: (v._mechShapePts || []).length};
        var clamp = body('Clamp');
        clickAt(screenOf(clamp.center));
        out.clamp = {picked: v.selectedMech,
                     grips: (v._mechShapePts || []).length};
    } catch (e) { out.error = String((e && e.stack) || e); }
    document.getElementById('out').textContent = JSON.stringify(out);
})();
</script>
</body></html>
'''

page = (PAGE.replace('__CSS__', VIEWER_CSS)
            .replace('__JS__', VIEWER_JS)
            .replace('__SCENE__', js(scene)))
path = os.path.join(SP, 'mech_shape_page.html')
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
check('no console error', errs == [], '\n        '.join(errs[:3]))
m = re.search(r'<div id="out"[^>]*>(.*?)</div>', p.stdout or '', re.S)
if m is None:
    print('  FAIL  no output')
    print()
    print('%d passed, %d failed' % (npass, nfail + 1))
    sys.exit(1)
payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
           .replace('&lt;', '<').replace('&gt;', '>'))
res = json.loads(payload)
check('ran without exception', res['error'] is None, str(res['error'])[:500])

def applied(msg):
    '''
    The message fed to a layout built fresh from the same bodies, so
    each check starts from what the page was looking at.
    '''
    L = make_layout()
    L.apply_edit(msg)
    return L

print()
print('--- the scene ---')
mechs = dict((b['name'], b) for b in scene['mechanics'])
check('a body of one hand-drawn shape carries that shape',
      mechs['Ring']['shape'] == {'type': 'circle', 'center': [0.0, 0.0],
                                 'radius': 0.05, 'thickness': 0.0},
      json.dumps(mechs['Ring']['shape']))
check('  in the frame it is written in, not on the bench',
      mechs['Ring']['shape']['center'] == [0.0, 0.0]
      and mechs['Ring']['center'] == [0.0, 0.4])
check('a body off the library shelf carries none',
      mechs['Board']['shape'] is None and mechs['Board']['resizable'] == 'box')
check('nor does one of several shapes', mechs['Mount']['shape'] is None)
check('an attached body carries its shape all the same',
      mechs['Clamp']['shape'] is not None,
      'the panel decides what to offer, not the channel')

print()
print('--- a circle ---')
r = res['ring']
check('clicking it picks it', r['picked'] == 'Ring')
check('the panel shows the circle\'s own rows',
      set(r['rows']) == {'cx', 'cy', 'radius'}
      and all(v['shown'] and v['editable'] for v in r['rows'].values()),
      json.dumps(r['rows']))
check('  in millimetres, of the shape and not of the body',
      abs(float(r['rows']['radius']['value']) - 50) < 1e-9
      and abs(float(r['rows']['cx']['value'])) < 1e-9,
      json.dumps({k: v['value'] for k, v in r['rows'].items()}))
check('four grips stand on the circle', r['grips'] == 4)
check('  on its rim, where the body carries it',
      r['gripOff'] and max(r['gripOff']) < 0.5,
      json.dumps(r['gripOff']))

t = r['typed']
check('typing a radius sends one message, the shape whole',
      t and t['n'] == 1 and t['msg']['op'] == 'set'
      and t['msg']['target'] == 'Ring'
      and abs(t['msg']['attrs']['shape']['radius'] - 0.08) < 1e-12,
      json.dumps(t['msg']))
if t and t['msg']:
    L = applied(t['msg'])
    check('  and Python gives the body that radius',
          abs(L.get_mechanics('Ring').shapes[0].radius - 0.08) < 1e-12)
    check('  and moves nothing',
          np.allclose(L.get_mechanics('Ring').center, [0.0, 0.4]))

d = r['dragged']
check('dragging a grip sends one message on release',
      d and r['dragN'] == 1 and d['op'] == 'set' and d['target'] == 'Ring',
      json.dumps(d))
if d:
    L = applied(d)
    check('  and the radius is where it was let go',
          abs(L.get_mechanics('Ring').shapes[0].radius - 0.12) < 2e-3,
          '%.4f' % L.get_mechanics('Ring').shapes[0].radius)

print()
print('--- a rectangle, on a body that is turned ---')
pl = res['plate']
check('clicking it picks it', pl['picked'] == 'Plate')
check('the panel shows the rectangle\'s rows',
      set(pl['rows']) == {'x', 'y', 'width', 'height', 'angle', 'px', 'py'},
      json.dumps(sorted(pl['rows'])))
check('  the width in millimetres of the shape itself',
      abs(float(pl['rows']['width']['value']) - 160) < 1e-9,
      pl['rows']['width']['value'])
check('four grips stand on it', pl['grips'] == 4)

pd = pl['dragged']
check('dragging a corner sends one message',
      pd and pl['dragN'] == 1 and pd['op'] == 'set'
      and pd['target'] == 'Plate' and 'shape' in pd['attrs'],
      json.dumps(pd))
if pd:
    L = applied(pd)
    body = L.get_mechanics('Plate')
    world = [body.to_world(c) for c in body.shapes[0].corners()]
    want = np.asarray(pl['target'], dtype='float64')
    near = min(np.hypot(*(np.asarray(c) - want)) for c in world)
    check('  and a corner of the rectangle lands where the drag ended',
          near < 5e-3, '%.4f m away' % near)
    check('  the body has not moved',
          np.allclose(body.center, [0.6, 0.4])
          and abs(body.rotationAngle - 0.6) < 1e-12)
    check('  and it is still a rectangle',
          isinstance(body.shapes[0], draw.Rectangle))

tw = pl['typedWidth']
check('typing a width sends the shape whole',
      tw and tw['n'] == 1
      and abs(tw['msg']['attrs']['shape']['width'] - 0.25) < 1e-12,
      json.dumps(tw['msg']) if tw else '')
if tw and tw['msg']:
    check('  and Python cuts the shape, not the body',
          abs(applied(tw['msg']).get_mechanics('Plate').shapes[0].width
              - 0.25) < 1e-12)

print()
print('--- a line ---')
ln = res['line']
check('the panel shows both ends',
      set(ln['rows']) == {'x1', 'y1', 'x2', 'y2'}, json.dumps(sorted(ln['rows'])))
check('a grip on each end', ln['grips'] == 2)
ld = ln['dragged']
check('dragging an end sends one set', ld and ld['op'] == 'set'
      and ld['target'] == 'Line1', json.dumps(ld))
if ld:
    L = applied(ld)
    stop = L.get_mechanics('Line1').to_world(L.get_mechanics('Line1').shapes[0].stop)
    check('  and that end is where it was let go',
          np.allclose(stop, ln['end'], atol=5e-3),
          str(np.round(stop, 4).tolist()))

print()
print('--- what has no shape to edit here ---')
b = res['board']
check('a breadboard offers no shape rows',
      b['picked'] == 'Board' and b['rows'] == 0, json.dumps(b))
check('  and no shape grips - it is cut to size instead',
      b['grips'] == 0 and b['resizeHandles'] == 4, json.dumps(b))
mo = res['mount']
check('a body of several shapes offers none either',
      mo['picked'] == 'Mount' and mo['rows'] == 0 and mo['grips'] == 0,
      json.dumps(mo))
cl = res['clamp']
check('an attached body offers no grips - its pose is its host\'s',
      cl['picked'] == 'Clamp' and cl['grips'] == 0, json.dumps(cl))

print()
print('--- what Python refuses ---')
for attrs, why in [
        ({'shape': {'type': 'circle', 'center': [0, 0], 'radius': 0}},
         'a radius of nothing'),
        ({'shape': {'type': 'circle', 'center': [0, 0],
                    'radius': float('inf')}}, 'an infinity'),
        ({'shape': {'type': 'rectangle', 'point': [0, 0], 'width': 1,
                    'height': 1}}, 'a shape of another kind'),
        ({'shape': 'a circle'}, 'something that is not a shape')]:
    L = make_layout()
    try:
        L.apply_edit({'op': 'set', 'target': 'Ring', 'attrs': attrs})
        ok = False
    except EditError:
        ok = True
    check('%s is refused' % why, ok)
    check('  and the body is untouched',
          abs(L.get_mechanics('Ring').shapes[0].radius - 0.05) < 1e-12)

for target, why in [('Board', 'a body cut to size'),
                    ('Mount', 'a body of several shapes')]:
    L = make_layout()
    try:
        L.apply_edit({'op': 'set', 'target': target,
                      'attrs': {'shape': {'type': 'circle',
                                          'center': [0, 0], 'radius': 0.1}}})
        ok = False
    except EditError:
        ok = True
    check('setting the shape of %s is refused' % why, ok)

L = make_layout()
L.apply_edit({'op': 'set', 'target': 'Ring',
              'attrs': {'shape': {'type': 'circle', 'center': [0, 0],
                                  'radius': 0.09, 'thickness': 0.0}}})
L.apply_edit({'op': 'undo'})
check('undo puts the shape back',
      abs(L.get_mechanics('Ring').shapes[0].radius - 0.05) < 1e-12)

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
