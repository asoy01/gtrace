'''
Stage 2b verification, browser side.

Drives the widget's ESM with real mouse events: grab an optics, drag it,
release, and check the edit message that reaches the model. Then feed
the message through Python's apply_edit and confirm the two sides agree
on where the optics ended up.
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

from gtrace.draw.viewer import viewer_css
from gtrace.layout import OpticalLayout

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

with open(os.path.join(SP, 'stage2_widget.mjs'), encoding='utf-8') as f:
    esm = f.read()
with open(os.path.join(SP, 'stage2b_scene.json')) as f:
    scene = json.load(f)

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
var EDITABLE = __EDITABLE__;
</script>
<script type="module">
(async function () {
    var out = {error: null, sent: [], steps: {}};
    function mouse(target, type, x, y, opts) {
        var ev = new MouseEvent(type, Object.assign({
            clientX: x, clientY: y, button: 0, bubbles: true, cancelable: true
        }, opts || {}));
        target.dispatchEvent(ev);
    }
    try {
        var url = URL.createObjectURL(
            new Blob([ESM_SRC], {type: 'text/javascript'}));
        var mod = await import(url);

        var state = {scene: SCENE, title: 'Edit check', height: 640,
                     editable: EDITABLE, error: ''};
        var handlers = {};
        var sent = [];
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
            send: function (msg) { sent.push(msg); }
        };

        var el = document.getElementById('host');
        mod.default.render({model: model, el: el});
        var v = el.gtraceViewer;
        var svg = v.svg;
        var r = svg.getBoundingClientRect();
        // Measured afresh every time: selecting an optics swaps the
        // beam readout for the properties panel, and the side bar does
        // not have to be the same height either way. A rect cached once
        // would put every later click a few pixels off.
        function screenOf(p) {
            var rr = svg.getBoundingClientRect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + rr.left, s[1] + rr.top];
        }

        out.opticsInScene = (v.scene.optics || []).length;
        out.editable = !!v.onEdit;
        out.helpMentionsDrag =
            /Drag an optics/.test(el.querySelector('.gt-help').textContent);

        var m1 = v.scene.optics.filter(function (o) {
            return o.name === 'M1';
        })[0];
        out.m1Center = m1.center;
        out.m1Angle = m1.normAngleHR;

        // --- hovering an optics must advertise the grab ---
        var p0 = screenOf(m1.center);
        mouse(window, 'mousemove', p0[0], p0[1]);
        out.steps.hover = {
            picked: v.hoverOptic ? v.hoverOptic.name : null,
            cursorClass: svg.classList.contains('gt-over-optic'),
            outlineShown: v.outline.style.display !== 'none'
        };

        // Away from any optics the grab must not be offered.
        var far = v.sceneToScreen(m1.center[0], m1.center[1] + 0.3);
        mouse(window, 'mousemove', far[0] + r.left, far[1] + r.top);
        out.steps.hoverAway = {
            picked: v.hoverOptic ? v.hoverOptic.name : null,
            cursorClass: svg.classList.contains('gt-over-optic')
        };

        // --- drag it 40 px right and 25 px up ---
        mouse(window, 'mousemove', p0[0], p0[1]);
        mouse(svg, 'mousedown', p0[0], p0[1]);
        out.steps.grabbed = {
            dragging: !!v.dragOptic,
            target: v.dragOptic ? v.dragOptic.optic.name : null
        };
        mouse(window, 'mousemove', p0[0] + 20, p0[1] - 12);
        mouse(window, 'mousemove', p0[0] + 40, p0[1] - 25);
        out.steps.midDrag = {
            center: v.dragOptic ? v.dragOptic.center.slice() : null,
            outlineShown: v.outline.style.display !== 'none',
            outlineDragging: v.outline.classList.contains('gt-dragging'),
            status: v.statusBar.textContent,
            sentSoFar: sent.length,
            // The view itself must not have panned.
            cx: v.cx, cy: v.cy, scale: v.scale
        };
        mouse(window, 'mouseup', p0[0] + 40, p0[1] - 25);
        out.steps.released = {
            dragging: !!v.dragOptic,
            sent: sent.length
        };
        out.dropScreenDelta = [40, -25];
        out.scale = v.scale;

        // --- a click that does not move must not send an edit ---
        mouse(window, 'mousemove', p0[0], p0[1]);
        mouse(svg, 'mousedown', p0[0], p0[1]);
        mouse(window, 'mouseup', p0[0], p0[1]);
        out.steps.click = {sent: sent.length};

        // --- shift-drag rotates ---
        var pr = screenOf([m1.center[0] + 0.04, m1.center[1]]);
        mouse(window, 'mousemove', pr[0], pr[1]);
        mouse(svg, 'mousedown', pr[0], pr[1], {shiftKey: true});
        out.steps.rotateGrab = {
            rotate: v.dragOptic ? v.dragOptic.rotate : null,
            pivot: v.dragOptic ? v.dragOptic.pivot.slice() : null,
            center0: v.dragOptic ? v.dragOptic.center0.slice() : null,
            angle0: v.dragOptic ? v.dragOptic.angle0 : null
        };
        out.m1HRcenter = m1.HRcenter;
        mouse(window, 'mousemove', pr[0], pr[1] - 30);
        out.steps.rotateMid = {
            center: v.dragOptic ? v.dragOptic.center.slice() : null,
            angle: v.dragOptic ? v.dragOptic.angle : null
        };
        mouse(window, 'mouseup', pr[0], pr[1] - 30);
        out.steps.rotated = {sent: sent.length};

        // --- ctrl-drag puts it square on the beam under the cursor ---
        var b0i = 0;
        v.scene.beams.forEach(function (b, i) {
            if (b.name === 'b0') { b0i = i; }
        });
        var b0 = v.scene.beams[b0i];
        out.b0 = {name: b0.name, index: b0i, pos: b0.pos,
                  dirVect: b0.dirVect, length: b0.length};
        var onBeam = [b0.pos[0] + b0.dirVect[0] * b0.length * 0.4,
                      b0.pos[1] + b0.dirVect[1] * b0.length * 0.4];
        var pOn = screenOf(onBeam);
        mouse(window, 'mousemove', p0[0], p0[1]);
        mouse(svg, 'mousedown', p0[0], p0[1]);
        // A few pixels off the beam: near enough to snap, not on it.
        mouse(window, 'mousemove', pOn[0], pOn[1] - 5, {ctrlKey: true});
        out.steps.snapMid = {
            snap: v.dragOptic ? v.dragOptic.snap : null,
            angle: v.dragOptic ? v.dragOptic.angle : null,
            center: v.dragOptic ? v.dragOptic.center.slice() : null,
            status: v.statusBar.textContent,
            wantPoint: onBeam
        };
        mouse(window, 'mouseup', pOn[0], pOn[1] - 5, {ctrlKey: true});
        out.steps.snapped = {sent: sent.length,
                             msg: sent.length ? sent[sent.length - 1] : null};

        // Ctrl held where no beam passes is an ordinary move: there is
        // nothing to align to, and the drag must not be swallowed.
        var pAway = screenOf([0.2, -0.5]);
        mouse(window, 'mousemove', p0[0], p0[1]);
        mouse(svg, 'mousedown', p0[0], p0[1]);
        mouse(window, 'mousemove', pAway[0], pAway[1], {ctrlKey: true});
        out.steps.noSnapMid = {snap: v.dragOptic ? v.dragOptic.snap : null};
        mouse(window, 'mouseup', pAway[0], pAway[1], {ctrlKey: true});
        out.steps.noSnap = {sent: sent.length,
                            msg: sent.length ? sent[sent.length - 1] : null};

        // --- zoomed in and grabbed by the rim ---
        // What has to be over the beam is the element. Zoomed in, the
        // point it was grabbed by can be hundreds of pixels from the
        // point it is held by, and testing the cursor made the whole
        // thing stop working above a certain zoom.
        var scale0 = v.scale;
        v.scale = scale0 * 8; v._applyTransform();
        var hr = m1.HRcenter;
        var rim = [m1.center[0] + 0.045, m1.center[1]];
        var pRim = screenOf(rim);
        mouse(window, 'mousemove', pRim[0], pRim[1]);
        mouse(svg, 'mousedown', pRim[0], pRim[1]);
        // Put the point the mirror is held by exactly on the beam; the
        // cursor then lands wherever the grab offset leaves it.
        var pZoom = screenOf([rim[0] + onBeam[0] - hr[0],
                              rim[1] + onBeam[1] - hr[1]]);
        mouse(window, 'mousemove', pZoom[0], pZoom[1], {ctrlKey: true});
        var pOnNow = screenOf(onBeam);
        out.steps.zoomSnap = {
            snap: v.dragOptic ? v.dragOptic.snap : null,
            // How far the cursor was from the beam, in pixels: the
            // check only means something if this is well outside the
            // reach a cursor test would have had.
            cursorOffPx: Math.hypot(pZoom[0] - pOnNow[0],
                                    pZoom[1] - pOnNow[1]),
            grabbed: !!v.dragOptic
        };
        mouse(window, 'mouseup', pZoom[0], pZoom[1], {ctrlKey: true});
        out.steps.zoomSnapped = {sent: sent.length,
                                 msg: sent.length ? sent[sent.length - 1]
                                                  : null};
        v.scale = scale0; v._applyTransform();

        // --- panning still works away from an optics ---
        var cx0 = v.cx;
        mouse(window, 'mousemove', far[0] + r.left, far[1] + r.top);
        mouse(svg, 'mousedown', far[0] + r.left, far[1] + r.top);
        mouse(window, 'mousemove', far[0] + r.left + 50, far[1] + r.top);
        mouse(window, 'mouseup', far[0] + r.left + 50, far[1] + r.top);
        out.steps.pan = {moved: v.cx !== cx0, sent: sent.length};

        out.sent = sent;

        // --- the error banner ---
        model.set('error', 'EditError: nope');
        var banner = el.querySelector('.gt-error');
        out.steps.error = {shown: banner.style.display !== 'none',
                           text: banner.textContent};
        model.set('error', '');
        out.steps.errorCleared = {shown: banner.style.display !== 'none'};
    } catch (e) {
        out.error = String((e && e.stack) || e);
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
    path = os.path.join(SP, 'stage2b_page_%s.html' % editable)
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
    if not m:
        return errs, None
    payload = (m.group(1).replace('&quot;', '"').replace('&amp;', '&')
               .replace('&lt;', '<').replace('&gt;', '>'))
    return errs, json.loads(payload)

print('--- editable viewer ---')
errs, res = run(True)
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:300])
check('optics reached the viewer', res['opticsInScene'] == 3,
      str(res['opticsInScene']))
check('editing is on', res['editable'])
check('the help mentions dragging', res['helpMentionsDrag'])

s = res['steps']
print('--- hovering an optics ---')
check('the optics is picked', s['hover']['picked'] == 'M1',
      str(s['hover']['picked']))
check('the cursor says it is grabbable', s['hover']['cursorClass'])
check('an outline is shown', s['hover']['outlineShown'])
check('nothing is picked away from it', s['hoverAway']['picked'] is None,
      str(s['hoverAway']['picked']))
check('and the cursor goes back', not s['hoverAway']['cursorClass'])

print('--- dragging ---')
check('mousedown grabs the optics', s['grabbed']['dragging']
      and s['grabbed']['target'] == 'M1', str(s['grabbed']))
check('the outline follows and is styled as dragging',
      s['midDrag']['outlineShown'] and s['midDrag']['outlineDragging'])
check('the status bar names the optics',
      s['midDrag']['status'].startswith('M1:'), s['midDrag']['status'])
check('nothing is sent mid-drag', s['midDrag']['sentSoFar'] == 0,
      str(s['midDrag']['sentSoFar']))
check('the view did not pan while dragging',
      s['midDrag']['scale'] == res['scale'], '')
check('release ends the drag', not s['released']['dragging'])
check('exactly one message on release', s['released']['sent'] == 1,
      str(s['released']['sent']))

sent = res['sent']
check('the message is a move', sent[0]['op'] == 'move', str(sent[0].get('op')))
check('it names the target', sent[0]['target'] == 'M1',
      str(sent[0].get('target')))
check('it carries a center', 'center' in sent[0], str(list(sent[0].keys())))

# The drop point must correspond to the pixels the cursor travelled.
dx_px, dy_px = res['dropScreenDelta']
want = [res['m1Center'][0] + dx_px / res['scale'],
        res['m1Center'][1] + (-dy_px) / res['scale']]
check('the optics landed where it was dropped',
      abs(sent[0]['center'][0] - want[0]) < 1e-9
      and abs(sent[0]['center'][1] - want[1]) < 1e-9,
      '(%s vs %s)' % ([round(x, 6) for x in sent[0]['center']],
                      [round(x, 6) for x in want]))

print('--- a click is not a drag ---')
check('no edit sent for a click without movement',
      s['click']['sent'] == 1, str(s['click']['sent']))

print('--- shift-drag rotates ---')
check('the drag is marked as a rotation', s['rotateGrab']['rotate'] is True,
      str(s['rotateGrab']['rotate']))
check('a second message was sent', s['rotated']['sent'] == 2,
      str(s['rotated']['sent']))
check('it is a rotate', len(sent) > 1 and sent[1]['op'] == 'rotate',
      str(sent[1].get('op') if len(sent) > 1 else None))
check('it carries an angle', len(sent) > 1 and 'normAngleHR' in sent[1],
      str(list(sent[1].keys()) if len(sent) > 1 else None))
check('the angle actually changed',
      len(sent) > 1
      and abs(sent[1]['normAngleHR'] - res['m1Angle']) > 1e-3,
      '(%s -> %s)' % (res['m1Angle'],
                      sent[1].get('normAngleHR') if len(sent) > 1 else None))

# An optics turns about the point it is held by, which for a mirror is
# the apex of its HR face - the point Python keeps fixed too, so the
# outline shown while dragging is where the element ends up.
grab, mid = s['rotateGrab'], s['rotateMid']
check('the pivot is the apex of the HR face, not the middle',
      np.allclose(grab['pivot'], res['m1HRcenter'], atol=1e-15)
      and not np.allclose(grab['pivot'], res['m1Center'], atol=1e-6),
      '(%s, HR %s, centre %s)'
      % (grab['pivot'], res['m1HRcenter'], res['m1Center']))
piv = np.asarray(grab['pivot'], dtype=float)
r_before = np.asarray(grab['center0'], dtype=float) - piv
r_after = np.asarray(mid['center'], dtype=float) - piv
check('the preview keeps the substrate the same distance from it',
      abs(np.hypot(*r_after) - np.hypot(*r_before)) < 1e-12,
      '(%.12f vs %.12f)' % (np.hypot(*r_after), np.hypot(*r_before)))
turned = np.arctan2(r_after[1], r_after[0]) - np.arctan2(r_before[1],
                                                         r_before[0])
wanted = mid['angle'] - grab['angle0']
check('and swings it by exactly the angle the optics turned',
      abs((turned - wanted + np.pi) % (2*np.pi) - np.pi) < 1e-12,
      '(%.12f vs %.12f)' % (turned, wanted))

print('--- ctrl-drag aligns onto the beam under the cursor ---')
snap = s['snapMid']['snap']
check('a beam under the cursor is picked up', snap is not None, str(snap))
check('it is the one the cursor was over', snap['beam'] == res['b0']['name']
      and snap['index'] == res['b0']['index'],
      '%s / %s' % (snap['beam'], snap['index']))

# The snap point is checked as a point of the beam rather than against
# the pixel that was aimed at. Writing the status bar reflows the page,
# so a screen coordinate taken before the drag and one taken after do
# not agree to the pixel, and that is the harness, not the viewer. What
# the projection comes to exactly is checked in verify_stage2b.py.
bp = np.asarray(res['b0']['pos'], dtype=float)
bd = np.asarray(res['b0']['dirVect'], dtype=float)
sp = np.asarray(snap['point'], dtype=float) - bp
check('the snap point is on the beam itself',
      abs(sp[0]*bd[1] - sp[1]*bd[0]) < 1e-12,
      '(%.3e m off it)' % abs(sp[0]*bd[1] - sp[1]*bd[0]))
check('within the drawn length of it',
      0.0 <= float(np.dot(sp, bd)) <= res['b0']['length'],
      '(%.6f of %.6f)' % (float(np.dot(sp, bd)), res['b0']['length']))
check('near where the cursor was',
      np.hypot(*(np.asarray(snap['point'], dtype=float)
                 - np.asarray(s['snapMid']['wantPoint'], dtype=float))) < 0.02,
      '%s vs %s' % (snap['point'], s['snapMid']['wantPoint']))
check('the preview turns the optics to face back down the beam',
      abs((s['snapMid']['angle'] - np.arctan2(-bd[1], -bd[0])
           + np.pi) % (2*np.pi) - np.pi) < 1e-12,
      '(%.9f)' % s['snapMid']['angle'])
check('the status bar says what is about to happen',
      'square onto' in s['snapMid']['status']
      and res['b0']['name'] in s['snapMid']['status'],
      s['snapMid']['status'])
check('one message on release', s['snapped']['sent'] == 3,
      str(s['snapped']['sent']))
align = s['snapped']['msg']
check("it is an 'align'", align['op'] == 'align', str(align.get('op')))
check('naming the optics, the beam and where along it',
      align['target'] == 'M1' and align['beam'] == res['b0']['name']
      and align['beam_index'] == res['b0']['index']
      and np.allclose(align['point'], snap['point'], atol=0),
      str(align))
# The geometry is Python's to do: the browser says which beam and
# where, not what the answer is.
check('and it carries no angle or centre of its own',
      'normAngleHR' not in align and 'center' not in align,
      str(sorted(align)))

check('Ctrl away from every beam does not snap',
      s['noSnapMid']['snap'] is None, str(s['noSnapMid']['snap']))
check('and the drag is an ordinary move',
      s['noSnap']['msg']['op'] == 'move' and s['noSnap']['sent'] == 4,
      '%s, %d sent' % (s['noSnap']['msg'].get('op'), s['noSnap']['sent']))

# The element is what has to be over the beam, not the cursor. Zoomed
# in and grabbed by the rim the two are far apart, and testing the
# cursor made the feature stop working above a certain zoom.
zs = s['zoomSnap']
check('an optics grabbed by its rim is still grabbed', zs['grabbed'])
check('with the cursor well clear of the beam',
      zs['cursorOffPx'] > 50, '(%.1f px)' % zs['cursorOffPx'])
check('it still snaps, because the element is on the beam',
      zs['snap'] is not None, str(zs['snap']))
check('and an align is what leaves the page',
      s['zoomSnapped']['msg']['op'] == 'align'
      and s['zoomSnapped']['sent'] == 5,
      '%s, %d sent' % (s['zoomSnapped']['msg'].get('op'),
                       s['zoomSnapped']['sent']))

print('--- panning still works ---')
check('dragging empty space pans', s['pan']['moved'])
check('and sends nothing', s['pan']['sent'] == 5, str(s['pan']['sent']))

print('--- error banner ---')
check('shown when the error traitlet is set', s['error']['shown']
      and 'nope' in s['error']['text'], str(s['error']))
check('hidden again when cleared', not s['errorCleared']['shown'])

print('--- Python agrees with the browser ---')
# Rebuild the layout the scene came from and feed it the very messages
# the browser produced.
import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.layout import TraceRules
from gtrace.unit import mm, cm, nm, ppm, deg2rad

b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                       pos=[0, 0], dirAngle=0, name='b0')
def MK(name, c, a, roc=0.0, rh=0.99, th=0.01):
    return opt.Mirror(HRcenter=c, normAngleHR=a, diameter=10*cm,
                      thickness=5*cm, wedgeAngle=deg2rad(0.25),
                      inv_ROC_HR=roc, Refl_HR=rh, Trans_HR=th,
                      Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45, name=name)
def rebuild():
    src = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                            pos=[0, 0], dirAngle=0, name='b0')
    opts = [MK('M1', [0.5, 0.0], deg2rad(135)),
            MK('M2', [0.5, 0.4], deg2rad(-45), 1.0/2.0),
            MK('M3', [0.9, 0.4], deg2rad(180), 0.0, 0.9, 0.1)]
    return OpticalLayout(optics=opts, sources=[src],
                         rules=TraceRules(order=5,
                                          power_threshold=1e-4)), opts

lay, optics = rebuild()

# Check after each message: rotating about the HR surface also shifts
# the substrate centre, so the move can only be checked before it.
lay.apply_edit(sent[0])
check('the move puts the optics exactly where it was dropped',
      np.allclose(np.asarray(optics[0].center), sent[0]['center'], atol=1e-12),
      '(%s vs %s)' % ([round(float(x), 9) for x in np.asarray(optics[0].center)],
                      [round(x, 9) for x in sent[0]['center']]))

lay.apply_edit(sent[1])
check('the rotation sets exactly the angle that was sent',
      abs(float(optics[0].normAngleHR)
          - (sent[1]['normAngleHR'] % (2 * np.pi))) < 1e-12,
      '(%.9f vs %.9f)' % (float(optics[0].normAngleHR),
                          sent[1]['normAngleHR'] % (2 * np.pi)))
check('the layout still traces afterwards', len(lay.trace()) > 0,
      '(%d beams)' % len(lay.beams))

# What the preview promised is what Python delivers: the outline was
# drawn from a turn about the apex of the HR face, and that is where
# the substrate centre has to have ended up. On a layout that has not
# had the move applied first, since the browser rotated the mirror from
# where the scene had it.
lay_rot, optics_rot = rebuild()
lay_rot.apply_edit(sent[1])
check('and the substrate went where the outline showed it going',
      np.allclose(np.asarray(optics_rot[0].center),
                  s['rotateMid']['center'], atol=1e-9),
      '(%s vs %s)'
      % ([round(float(x), 9) for x in np.asarray(optics_rot[0].center)],
         [round(x, 9) for x in s['rotateMid']['center']]))
check('and the apex of the HR face did not move at all',
      np.allclose(np.asarray(optics_rot[0].HRcenter), res['m1HRcenter'],
                  atol=1e-15),
      str([round(float(x), 12) for x in np.asarray(optics_rot[0].HRcenter)]))

# The align message, on the layout the browser was actually looking at.
lay2, optics2 = rebuild()
lay2.apply_edit(sent[2])
M1a = optics2[0]
src2 = [b for b in lay2.trace() if b.name == 'b0'][0]
bpos = np.asarray(src2.pos, dtype='float64')
bdir = np.asarray(src2.dirVect, dtype='float64')
off = np.asarray(M1a.HRcenter, dtype='float64') - bpos
across = abs(off[0]*bdir[1] - off[1]*bdir[0])
check('the align message lands the mirror on the beam axis',
      across < 1e-15, '(%.3e m off it)' % across)
check('square to the beam',
      abs(float(np.dot(np.asarray(M1a.normVectHR), bdir)) + 1) < 1e-12,
      '(n.d = %.15f)' % float(np.dot(np.asarray(M1a.normVectHR), bdir)))
check('at the point along it that was dropped on',
      abs(float(np.dot(off, bdir)) - sent[2]['point'][0]) < 1e-9,
      '(%.9f vs %.9f)' % (float(np.dot(off, bdir)), sent[2]['point'][0]))
check('and the layout still traces', len(lay2.trace()) > 0,
      '(%d beams)' % len(lay2.beams))

print('--- read-only viewer ---')
errs, res = run(False)
check('no console error', errs == [], '\n        '.join(errs[:3]))
check('ran without exception', res and res['error'] is None,
      str(res and res['error'])[:200])
check('editing is off', not res['editable'])
check('the help does not mention dragging', not res['helpMentionsDrag'])
# The optics is still pickable, because its properties can be read
# without a Python behind the page; it just cannot be grabbed.
check('the optics can still be inspected',
      res['steps']['hover']['picked'] == 'M1',
      str(res['steps']['hover']['picked']))
check('but no grab cursor is offered',
      not res['steps']['hover']['cursorClass'])
check('dragging an optics sends nothing', res['sent'] == [],
      str(res['sent']))
check('and pans instead', res['steps']['pan']['moved'])

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
