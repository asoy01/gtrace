'''
The laser drawn for a source, in a real browser: that it is there, that
it is where the light comes from, that it stays the same size on screen,
that clicking it opens the source panel, and that dragging it sends what
Python then does.

Why a laser is drawn at all: nothing else in the picture says which of
the beams the user put there. A source is traced from a copy of itself,
so its own beam sits in the scene looking exactly like the ones the
trace made from it. The box is both the answer to that and the handle
the source is edited by.

Why it is sized in screen pixels: a layout runs from a bench to a
kilometre, so a body sized in metres would be a dot on one and would
cover the other. The check that matters is the one at the bottom of the
first section - zoom in forty times and the box is the same size and
still clickable. The Ctrl-drag snap was written the other way round
once and stopped working above a certain zoom.

Every edit that leaves the page is fed to Python's apply_edit, so the
two sides are compared on the same numbers rather than on the page's own
account of what it did.
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
from gtrace.layout import (OpticalLayout, TraceRules, q_from_waist,
                           source_waist)
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
                           pos=[0, 0], dirAngle=0, name='b0')
    # A second one, aimed across the first, so that the checks see more
    # than a single box lying along +x.
    b1 = beam.GaussianBeam(q0=q_from_waist(0.5*mm, 0.1, 532*nm), wl=532*nm,
                           P=0.25, pos=[0.5, -0.4], dirAngle=np.pi/2,
                           name='b1')

    def M(name, c, a):
        return opt.Mirror(HRcenter=c, normAngleHR=a, diameter=10*cm,
                          thickness=5*cm, wedgeAngle=deg2rad(0.25),
                          Refl_HR=0.99, Trans_HR=0.01, n=1.45, name=name)

    optics = [M('M1', [0.5, 0.0], deg2rad(135)),
              M('M2', [0.5, 0.4], deg2rad(-45)),
              # Just behind the first laser, facing the way it fires, so
              # that its footprint covers the box: the check that the
              # box wins the pick needs something to win against. Not on
              # the origin itself - a source standing on a surface makes
              # a beam of no length, which draw() divides by.
              M('M3', [-0.04, 0.0], 0.0)]
    return OpticalLayout(optics=optics, sources=[b0, b1],
                         rules=TraceRules(order=4, power_threshold=1e-4))

layout = make_layout()
scene = layout.scene_dict()

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

        // Re-measured every time it is used: the status bar changes
        // length as the cursor moves, and that reflows the page by a
        // pixel or two.
        function rect() { return v.svg.getBoundingClientRect(); }
        function screenOf(p) {
            var r = rect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function polys() {
            return Array.prototype.map.call(
                el.querySelectorAll('.gt-source'), function (e) {
                    return {points: e.getAttribute('points'),
                            cls: e.getAttribute('class'),
                            shown: e.style.display !== 'none'};
                });
        }
        function pointsOf(i) {
            return polys()[i].points.split(' ').map(function (p) {
                return p.split(',').map(Number);
            });
        }
        function extent(pts) {
            var xs = pts.map(function (p) { return p[0]; });
            var ys = pts.map(function (p) { return p[1]; });
            return [Math.max.apply(null, xs) - Math.min.apply(null, xs),
                    Math.max.apply(null, ys) - Math.min.apply(null, ys)];
        }
        function panel() {
            return {
                title: el.querySelector('.gt-panel-title span').textContent,
                beamShown: v.readoutBody.style.display !== 'none',
                opticShown: v.opticBody.style.display !== 'none',
                sourceShown: v.sourceBody.style.display !== 'none'
            };
        }
        function fields() {
            var o = {};
            for (var k in v.sourceFields) {
                var f = v.sourceFields[k];
                o[k] = f.editable ? f.el.value : f.el.textContent;
            }
            return o;
        }
        function setField(key, text) {
            var f = v.sourceFields[key];
            f.el.value = text;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        function button(text) {
            var found = null;
            Array.prototype.forEach.call(
                el.querySelectorAll('button'), function (b) {
                    if (b.textContent === text) { found = b; }
                });
            return found;
        }
        function clickAt(p) {
            mouse(v.svg, 'mousedown', p[0], p[1]);
            mouse(window, 'mouseup', p[0], p[1]);
        }

        // --- the lasers are drawn ---
        out.count = polys().length;
        out.start = panel();
        var s0 = SCENE.sources[0], s1 = SCENE.sources[1];

        // The nose of the box is at the origin of the beam: that is
        // where the light comes from, and the body runs back from it so
        // that it does not sit on top of what the beam is pointing at.
        var o0 = v.sceneToScreen(s0.pos[0], s0.pos[1]);
        var pts0 = pointsOf(0);
        out.nose = {
            origin: o0,
            // The two points of the shape that sit at u = 0.
            atOrigin: pts0.filter(function (p) {
                return Math.hypot(p[0] - o0[0], p[1] - o0[1]) < 5;
            }).length,
            // and none of it is in front
            ahead: pts0.filter(function (p) {
                return (p[0] - o0[0]) * s0.dirVect[0]
                     + (p[1] - o0[1]) * (-s0.dirVect[1]) > 0.5;
            }).length
        };
        // The second one fires along +y, so its box runs the other way
        // on screen. Same shape, turned.
        out.aimed = {
            first: extent(pointsOf(0)),
            second: extent(pointsOf(1))
        };

        // --- the size, against the beam it emits ---
        //
        // The box is in screen pixels so that it stays legible across a
        // layout that may be a bench or a kilometre. That holds only
        // while the beam is a line: zoomed in far enough the drawn
        // envelope is wider than the aperture it comes out of, which is
        // a picture of something that cannot happen. Past that point it
        // grows with the view instead.
        //
        // Both halves are measured off the drawing rather than from the
        // constants, so the checks say what is seen.
        function noseHalfPx(i) {
            var s = SCENE.sources[i];
            var o = v.sceneToScreen(s.pos[0], s.pos[1]);
            // The two corners of the nose are the points of the shape
            // at the origin's own distance across it, and so the two
            // nearest the origin whichever way the laser is aimed.
            var d = pointsOf(i).map(function (p) {
                return Math.hypot(p[0] - o[0], p[1] - o[1]);
            }).sort(function (a, b) { return a - b; });
            return (d[0] + d[1]) / 2;
        }
        var SIGMA = SCENE.display.sigma_main;
        function beamHalfPx(i) {
            return SIGMA * SCENE.sources[i].width[0] * v.scale;
        }
        var scale0 = v.scale;

        // Far enough out that the beam is nothing: the nominal size.
        v.scale = 1e-3;
        v._applyTransform();
        var nominal = noseHalfPx(0);
        var nominalBox = extent(pointsOf(0));

        // The zoom at which the beam is exactly as wide as the nose.
        var threshold = nominal / (SIGMA * SCENE.sources[0].width[0]);

        // Below it: pixel-fixed, whatever the zoom.
        out.zoom = {nominal: nominal};
        v.scale = threshold / 8;
        v._applyTransform();
        out.zoom.belowBox = extent(pointsOf(0));
        out.zoom.belowNose = noseHalfPx(0);
        out.zoom.belowBeam = beamHalfPx(0);

        // Well past it: the nose keeps up with the beam.
        v.scale = threshold * 25;
        v._applyTransform();
        out.zoom.aboveBox = extent(pointsOf(0));
        out.zoom.aboveNose = noseHalfPx(0);
        out.zoom.aboveBeam = beamHalfPx(0);
        // And the shape is the same shape, only bigger.
        out.zoom.aspectBelow = out.zoom.belowBox[0] / out.zoom.belowBox[1];
        out.zoom.aspectAbove = out.zoom.aboveBox[0] / out.zoom.aboveBox[1];

        // Right at the crossing, nothing jumps.
        v.scale = threshold * 0.999;
        v._applyTransform();
        var justUnder = noseHalfPx(0);
        v.scale = threshold * 1.001;
        v._applyTransform();
        out.zoom.acrossJump = Math.abs(noseHalfPx(0) - justUnder);
        out.zoom.acrossNose = noseHalfPx(0);

        // Still clickable on both sides of it, which is the point of
        // the whole choice: a box sized in metres would be off the
        // screen zoomed in, and a dot zoomed out.
        function pickedAt(scale) {
            v.scale = scale;
            v._applyTransform();
            var p = screenOf(SCENE.sources[0].pos);
            var r = rect();
            // A little back along the beam, in the body, scaled the
            // way the box is.
            var k = Math.max(1, noseHalfPx(0) / nominal);
            var hit = v._pickSource(p[0] - r.left - 14 * k,
                                    p[1] - r.top);
            return hit ? hit.name : null;
        }
        out.zoom.pickedBelow = pickedAt(threshold / 8);
        out.zoom.pickedAbove = pickedAt(threshold * 25);
        out.zoom.pickedFarOut = pickedAt(1e-3);

        v.scale = scale0;
        v._applyTransform();
        out.zoom.nominalBox = nominalBox;

        // --- clicking one opens the source panel ---
        var at0 = screenOf(s0.pos);
        // A few pixels back along the beam, so the click lands on the
        // body rather than exactly on the nose.
        var body0 = [at0[0] - 14 * s0.dirVect[0],
                     at0[1] + 14 * s0.dirVect[1]];
        clickAt(body0);
        out.clicked = panel();
        out.clickedName = v.selectedSource;
        out.clickedFields = fields();
        out.selectedClass = polys()[0].cls;
        // M3 covers the same spot. The laser is picked first, as a
        // dimension is: it is a small mark that an element would
        // otherwise shadow for good. Both pickers are asked as well as
        // read off the selection, so that the check cannot pass by the
        // element simply not being there.
        var rb = rect();
        var scenePt = v.screenToScene(body0[0] - rb.left, body0[1] - rb.top);
        out.overOptic = {
            sourceThere: !!v._pickSource(body0[0] - rb.left,
                                         body0[1] - rb.top),
            opticThere: !!v._pickOptic(scenePt[0], scenePt[1]),
            source: v.selectedSource, optic: v.selectedOptic
        };

        // Clicking the other one moves the panel over to it.
        var at1 = screenOf(s1.pos);
        clickAt([at1[0] - 14 * s1.dirVect[0], at1[1] + 14 * s1.dirVect[1]]);
        out.second = {name: v.selectedSource, fields: fields()};

        // Clicking empty space lets it go.
        clickAt([rect().left + 12, rect().top + 12]);
        out.cleared = {panel: panel(), name: v.selectedSource};

        // --- editing the fields ---
        clickAt(body0);
        if (EDITABLE) {
            setField('w0x', '0.35');
            setField('dx', '0.12');
            setField('wl', '532');
            setField('P', '0.4');
            setField('angle', '12.5');
            setField('px', '0.02');
            setField('length', '2.5');

            // Nothing a source can be is infinite, and JSON could not
            // carry it anyway: the field goes back to what the model
            // holds rather than sending it. Checked before the rename,
            // which leaves the page addressing a name the scene it was
            // handed does not have yet.
            var n0 = sent.length;
            setField('w0x', 'inf');
            setField('dx', 'nonsense');
            out.refusedCount = sent.length - n0;
            out.afterRefusal = fields().w0x;

            setField('name', 'laser');
            out.edits = sent.slice();
            // The optimistic rename: the page goes on addressing the
            // source it just asked to have renamed.
            out.renamedTo = v.selectedSource;
        }

        // --- dragging it ---
        if (EDITABLE) {
            sent.length = 0;
            var from = screenOf(s0.pos);
            var grab = [from[0] - 14, from[1]];
            var to = [grab[0] + 90, grab[1] - 60];
            mouse(v.svg, 'mousedown', grab[0], grab[1]);
            mouse(window, 'mousemove', to[0], to[1]);
            out.dragMid = {
                dragging: !!v.dragSource,
                target: v.dragSource ? v.dragSource.source.name : null,
                pos: v.dragSource ? v.dragSource.pos.slice() : null,
                cls: polys()[0].cls
            };
            // Where the preview says the nose is, so that what is shown
            // and what is sent can be compared.
            out.dragMid.previewNose = v.sceneToScreen(
                v.dragSource.pos[0], v.dragSource.pos[1]);
            mouse(window, 'mouseup', to[0], to[1]);
            out.dragged = {sent: sent.slice(), released: !v.dragSource,
                           selected: v.selectedSource};
            // The scene has not moved: Python owns the model, and it
            // has not answered yet.
            out.dragged.sceneUntouched = SCENE.sources[0].pos.slice();

            // Shift turns it, about the point the light leaves from.
            sent.length = 0;
            var p = screenOf(s0.pos);
            var hold = [p[0] - 20, p[1]];
            mouse(v.svg, 'mousedown', hold[0], hold[1], {shiftKey: true});
            mouse(window, 'mousemove', p[0] - 20, p[1] - 20,
                  {shiftKey: true});
            out.turnMid = {
                rotate: v.dragSource ? v.dragSource.rotate : null,
                angle: v.dragSource ? v.dragSource.angle : null,
                // The origin is untouched by a turn.
                pos: v.dragSource ? v.dragSource.pos.slice() : null,
                // And the drawn nose is still at the origin.
                nose: pointsOf(0).filter(function (q) {
                    return Math.hypot(q[0] - v.sceneToScreen(
                        SCENE.sources[0].pos[0],
                        SCENE.sources[0].pos[1])[0],
                        q[1] - v.sceneToScreen(
                            SCENE.sources[0].pos[0],
                            SCENE.sources[0].pos[1])[1]) < 5;
                }).length
            };
            mouse(window, 'mouseup', p[0] - 20, p[1] - 20, {shiftKey: true});
            out.turned = sent.slice();

            // A grab that goes nowhere is a click, not a move.
            sent.length = 0;
            v.selectedSource = null;
            v._showPanel('beam');
            mouse(v.svg, 'mousedown', body0[0], body0[1]);
            mouse(window, 'mouseup', body0[0], body0[1]);
            out.tap = {sent: sent.slice(), selected: v.selectedSource,
                       panel: panel()};
        }

        // --- adding and removing ---
        if (EDITABLE) {
            sent.length = 0;
            // The button arms a place; the click that follows says
            // where the laser stands. The place is read back from the
            // viewer, since a click carries whole pixels.
            button('+ Source').click();
            out.arm = {armed: v.placing && v.placing.kind,
                       type: v.placing && v.placing.type,
                       lit: button('+ Source').classList.contains('gt-btn-on'),
                       sent: sent.length};
            var sp = screenOf([0.42, 0.31]);
            mouse(window, 'mousemove', sp[0], sp[1]);
            var atSource = v.placePreview.slice();
            clickAt(sp);
            out.added = {sent: sent.slice(), selected: v.selectedSource,
                         at: atSource, stillArmed: !!v.placing,
                         optic: v.selectedOptic};

            sent.length = 0;
            clickAt(body0);
            var rm = null;
            Array.prototype.forEach.call(
                v.sourceBody.querySelectorAll('button'), function (b) {
                    if (b.textContent === 'Remove') { rm = b; }
                });
            out.hasRemove = !!rm;
            if (rm) { rm.click(); }
            out.removed = sent.slice();
        }

        // --- the tracing rules ---
        out.rules = {present: !!v.ruleFields};
        if (v.ruleFields) {
            out.rules.shown = {};
            for (var rk in v.ruleFields) {
                out.rules.shown[rk] = v.ruleFields[rk].el.value;
            }
            sent.length = 0;
            var of = v.ruleFields.order;
            of.el.value = '9';
            of.el.dispatchEvent(new Event('change', {bubbles: true}));
            var pf = v.ruleFields.power_threshold;
            pf.el.value = '1e-9';
            pf.el.dispatchEvent(new Event('change', {bubbles: true}));
            // Not a number: put back what the layout holds, send nothing.
            var nbefore = sent.length;
            of.el.value = 'deep';
            of.el.dispatchEvent(new Event('change', {bubbles: true}));
            out.rules.sent = sent.slice();
            out.rules.refused = sent.length - nbefore === 0;
            out.rules.restored = of.el.value;
        }

        // --- the laser goes with the layer its beam is drawn on ---
        var layer = SCENE.sources[0].layer;
        v.setLayerVisible(layer, false);
        out.hidden = polys().map(function (p) { return p.shown; });
        v.setLayerVisible(layer, true);
        out.shownAgain = polys().map(function (p) { return p.shown; });

        // --- a scene from Python replaces what the page had ---
        var next = JSON.parse(JSON.stringify(SCENE));
        next.sources[0].pos = [0.2, 0.3];
        clickAt(body0);
        model.set('scene', next);
        out.afterPush = {
            selected: v.selectedSource,
            panel: panel(),
            px: v.sourceFields.px.editable ? v.sourceFields.px.el.value
                                           : v.sourceFields.px.el.textContent,
            drawnAt: pointsOf(0).filter(function (q) {
                var o = v.sceneToScreen(0.2, 0.3);
                return Math.hypot(q[0] - o[0], q[1] - o[1]) < 5;
            }).length
        };

        // A scene that no longer has the source drops the panel.
        var gone = JSON.parse(JSON.stringify(next));
        gone.sources = [gone.sources[1]];
        model.set('scene', gone);
        out.afterGone = {selected: v.selectedSource, panel: panel(),
                         count: polys().length};
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
    path = os.path.join(SP, 'source_page_%s.html' % editable)
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


print('--- editable viewer ---')
errs, res = run(True)
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:500])

s0 = scene['sources'][0]
s1 = scene['sources'][1]

print('--- the lasers are drawn ---')
check('one laser per registered source', res['count'] == 2, str(res['count']))
check('the panel starts on the beam readout', res['start']['beamShown'])
# The nose is at the origin of the beam, and the body runs back from it.
# A laser drawn forward would cover the very beam it emits.
check('two corners of the box sit at the origin of the beam',
      res['nose']['atOrigin'] == 2, str(res['nose']['atOrigin']))
check('and nothing is drawn ahead of it', res['nose']['ahead'] == 0,
      str(res['nose']['ahead']))
# The second source fires across the first, so the same shape lies the
# other way on screen. Comparing the extents is enough to say it turned.
w0, h0 = res['aimed']['first']
w1, h1 = res['aimed']['second']
check('a laser aimed along +x is wider than it is tall', w0 > h0,
      '%.1f x %.1f' % (w0, h0))
check('one aimed along +y is taller than it is wide', h1 > w1,
      '%.1f x %.1f' % (w1, h1))
check('and the two are the same box, turned',
      abs(w0 - h1) < 0.6 and abs(h0 - w1) < 0.6,
      '%.1f x %.1f  vs  %.1f x %.1f' % (w0, h0, w1, h1))

print('--- the size, against the beam it emits ---')
z = res['zoom']
# While the beam is thinner than the aperture, the box is in screen
# pixels: a layout runs from a bench to a kilometre, and a body sized in
# metres would be a dot on one and would cover the other.
check('below the crossing it keeps its nominal size',
      abs(z['belowNose'] - z['nominal']) < 0.01,
      '%.3f vs %.3f' % (z['belowNose'], z['nominal']))
check('  and the box with it',
      abs(z['belowBox'][0] - z['nominalBox'][0]) < 0.5
      and abs(z['belowBox'][1] - z['nominalBox'][1]) < 0.5,
      '%s vs %s' % (z['belowBox'], z['nominalBox']))
check('  which is the case where the beam is the thinner of the two',
      z['belowBeam'] < z['belowNose'],
      '%.3f vs %.3f' % (z['belowBeam'], z['belowNose']))

# Past it the picture would otherwise show a beam wider than the
# aperture it comes out of, so the box grows with the view instead.
check('past it the box grows with the zoom',
      z['aboveBox'][1] > z['belowBox'][1] * 5,
      '%.1f -> %.1f' % (z['belowBox'][1], z['aboveBox'][1]))
check('  and the aperture tracks the beam exactly',
      abs(z['aboveNose'] - z['aboveBeam']) < 0.01 * z['aboveBeam'],
      '%.2f vs %.2f' % (z['aboveNose'], z['aboveBeam']))
check('  never letting the beam out of a narrower opening',
      z['aboveNose'] >= z['aboveBeam'] - 1e-6,
      '%.3f vs %.3f' % (z['aboveNose'], z['aboveBeam']))
check('  keeping the same shape, only bigger',
      abs(z['aspectAbove'] - z['aspectBelow']) < 1e-6,
      '%.6f vs %.6f' % (z['aspectAbove'], z['aspectBelow']))
# The two rules meet at the crossing rather than stepping, so zooming
# through it does not make the laser jump.
check('and the two meet at the crossing without a step',
      z['acrossJump'] < 0.01 * z['acrossNose'],
      '%.5f px' % z['acrossJump'])

check('it is clickable below the crossing', z['pickedBelow'] == 'b0',
      str(z['pickedBelow']))
check('  above it', z['pickedAbove'] == 'b0', str(z['pickedAbove']))
check('  and zoomed right out, where the layout is a dot',
      z['pickedFarOut'] == 'b0', str(z['pickedFarOut']))

print('--- clicking one shows the source ---')
check('the panel turns to the source', res['clicked']['sourceShown']
      and not res['clicked']['beamShown'] and not res['clicked']['opticShown'],
      str(res['clicked']))
check('titled as such', res['clicked']['title'] == 'Source properties',
      res['clicked']['title'])
check('and it is the one that was clicked', res['clickedName'] == 'b0',
      str(res['clickedName']))
check('the drawn laser says it is selected',
      'gt-selected' in (res['selectedClass'] or ''),
      str(res['selectedClass']))
# M3 stands at the same point. A laser is a small mark and an element is
# an area, so the laser is picked first - the element is still there to
# be clicked anywhere off it.
check('the spot really has both a laser and an optics on it',
      res['overOptic']['sourceThere'] and res['overOptic']['opticThere'],
      str(res['overOptic']))
check('  and the laser is the one picked',
      res['overOptic']['source'] == 'b0'
      and res['overOptic']['optic'] is None, str(res['overOptic']))

f = res['clickedFields']
check('the type is named', f['type'] == 'Source', f['type'])
check('the name is the source name', f['name'] == 'b0', f['name'])
# What the panel shows is what Python reports, in the units a laser is
# spoken of in: a waist in millimetres, a wavelength in nanometres, a
# direction in degrees.
w = source_waist(layout.get_source('b0'))
check('the waist size is shown in millimetres',
      abs(float(f['w0x']) - w['waist_size'][0] / mm) < 1e-12,
      '%s vs %g' % (f['w0x'], w['waist_size'][0] / mm))
check('  in both directions',
      abs(float(f['w0y']) - w['waist_size'][1] / mm) < 1e-12, f['w0y'])
check('the waist position is shown in metres',
      abs(float(f['dx']) - w['waist_pos'][0]) < 1e-15, f['dx'])
check('the wavelength is shown in nanometres',
      abs(float(f['wl']) - 1064.0) < 1e-9, f['wl'])
check('the power as it is', abs(float(f['P']) - 1.0) < 1e-15, f['P'])
check('the index as it is', abs(float(f['n']) - 1.0) < 1e-15, f['n'])
check('the position in metres',
      abs(float(f['px']) - s0['pos'][0]) < 1e-15
      and abs(float(f['py']) - s0['pos'][1]) < 1e-15,
      '%s, %s' % (f['px'], f['py']))
check('the direction in degrees',
      abs(float(f['angle']) - math.degrees(s0['dirAngle'])) < 1e-9, f['angle'])

sec = res['second']
check('clicking the other laser moves the panel to it',
      sec['name'] == 'b1', str(sec['name']))
check('  and the values change with it',
      abs(float(sec['fields']['wl']) - 532.0) < 1e-9
      and abs(float(sec['fields']['P']) - 0.25) < 1e-15,
      '%s / %s' % (sec['fields']['wl'], sec['fields']['P']))
check('  including the waist it was given',
      abs(float(sec['fields']['w0y']) - 0.5) < 1e-12,
      sec['fields']['w0y'])
check('clicking empty space lets the source go',
      res['cleared']['name'] is None and res['cleared']['panel']['beamShown'],
      str(res['cleared']))

print('--- editing a field sends what Python then does ---')
edits = res['edits']
by_key = {}
for m in edits:
    if m['op'] == 'set':
        for k in m['attrs']:
            by_key[k] = m
    else:
        by_key[m['op']] = m

check('a waist size goes out in metres',
      abs(by_key['waist_size_x']['attrs']['waist_size_x'] - 0.35*mm) < 1e-18,
      str(by_key.get('waist_size_x')))
check('a waist position goes out as it stands',
      by_key['waist_pos_x']['attrs']['waist_pos_x'] == 0.12,
      str(by_key.get('waist_pos_x')))
check('a wavelength goes out in metres',
      abs(by_key['wl']['attrs']['wl'] - 532*nm) < 1e-18, str(by_key.get('wl')))
check('a power goes out as it stands',
      by_key['P']['attrs']['P'] == 0.4, str(by_key.get('P')))
check('a free length goes out in metres',
      by_key['length']['attrs']['length'] == 2.5, str(by_key.get('length')))
check('a position is a move, not a set',
      by_key['move']['op'] == 'move' and by_key['move']['pos'][0] == 0.02,
      str(by_key.get('move')))
check('a direction is a rotate, in radians',
      abs(by_key['rotate']['dirAngle'] - math.radians(12.5)) < 1e-12,
      str(by_key.get('rotate')))
check('the name is its own operation',
      by_key['rename']['op'] == 'rename'
      and by_key['rename']['name'] == 'laser', str(by_key.get('rename')))
check('and the page addresses the new name at once',
      res['renamedTo'] == 'laser', str(res['renamedTo']))
check('a value that cannot travel is not sent',
      res['refusedCount'] == 0, str(res['refusedCount']))
check('  and the field goes back to what the model holds',
      abs(float(res['afterRefusal']) - 0.2) < 1e-12, res['afterRefusal'])

# Now the other half: hand every one of those messages to Python and
# check the source comes to what the panel said it would.
L = make_layout()
for m in edits:
    L.apply_edit(m)
src = L.get_source('laser')
w = source_waist(src)
check('Python applies the whole run of them',
      abs(w['waist_size'][0] - 0.35*mm) < 1e-18
      and abs(w['waist_pos'][0] - 0.12) < 1e-15
      and abs(src.wl - 532*nm) < 1e-18 and src.P == 0.4
      and abs(src.length - 2.5) < 1e-15, str(w))
check('  and the pose is what was typed',
      abs(src.pos[0] - 0.02) < 1e-15
      and abs(src.dirAngle - math.radians(12.5)) < 1e-12,
      '%s %s' % (src.pos, src.dirAngle))
# The wavelength was set in the same run as the waist. It decides what a
# q comes to as a waist, so if it were applied afterwards the waist
# would come out measured against the old light.
check('  and the waist means what it said, not what the old light made of it',
      abs(w['waist_size'][0] - 0.35*mm) < 1e-18, str(w['waist_size'][0]))

print('--- dragging ---')
dm = res['dragMid']
check('pressing on a laser takes hold of it', dm['dragging'], str(dm))
check('  of the one that was pressed', dm['target'] == 'b0',
      str(dm['target']))
check('  and it shows that it is being dragged',
      'gt-dragging' in (dm['cls'] or ''), str(dm['cls']))
dr = res['dragged']
check('releasing lets go', dr['released'])
check('  and sends one message', len(dr['sent']) == 1, str(dr['sent']))
msg = dr['sent'][0] if dr['sent'] else {}
check('  a move naming the source', msg.get('op') == 'move'
      and msg.get('target') == 'b0', str(msg))
check('  at the position the preview was showing',
      msg.get('pos') and abs(msg['pos'][0] - dm['pos'][0]) < 1e-12
      and abs(msg['pos'][1] - dm['pos'][1]) < 1e-12,
      '%s vs %s' % (msg.get('pos'), dm['pos']))
# Python owns the model. The page previews and asks; it does not decide.
check('  and the scene it was handed is untouched',
      dr['sceneUntouched'] == s0['pos'], str(dr['sceneUntouched']))
check('  the source it moved is the one on show',
      dr['selected'] == 'b0', str(dr['selected']))

# What the drag asked for is what Python does with it.
L = make_layout()
L.apply_edit(msg)
check('Python puts it where the drag said',
      abs(L.get_source('b0').pos[0] - msg['pos'][0]) < 1e-15
      and abs(L.get_source('b0').pos[1] - msg['pos'][1]) < 1e-15,
      str(L.get_source('b0').pos))

tm = res['turnMid']
check('Shift turns it instead', tm['rotate'] is True, str(tm['rotate']))
check('  about the point the light leaves from',
      tm['pos'] == s0['pos'], str(tm['pos']))
check('  which stays under the nose of the box while it turns',
      tm['nose'] == 2, str(tm['nose']))
check('  and one rotate is sent', len(res['turned']) == 1
      and res['turned'][0]['op'] == 'rotate', str(res['turned']))
turn = res['turned'][0] if res['turned'] else {}
check('  carrying the angle the preview showed',
      abs(turn.get('dirAngle', 0) - tm['angle']) < 1e-12,
      '%s vs %s' % (turn.get('dirAngle'), tm['angle']))
# The preview and the answer agree: Python turns a source about its
# origin, which is what the box was drawn turning about.
L = make_layout()
L.apply_edit(turn)
check('Python turns it to the same angle and leaves it standing',
      abs(math.fmod(L.get_source('b0').dirAngle - turn['dirAngle'],
                    2 * math.pi)) < 1e-12
      and np.allclose(L.get_source('b0').pos, s0['pos']),
      str(L.get_source('b0').dirAngle))

tap = res['tap']
check('a grab that goes nowhere is a click', tap['sent'] == [],
      str(tap['sent']))
check('  and selects the laser', tap['selected'] == 'b0',
      str(tap['selected']))
check('  showing its panel', tap['panel']['sourceShown'], str(tap['panel']))

print('--- adding and removing ---')
arm = res['arm']
check('+ Source arms a place rather than adding one',
      arm['sent'] == 0 and arm['armed'] == 'optics'
      and arm['type'] == 'Source', json.dumps(arm))
check('  and lights while it is armed', arm['lit'])
add = res['added']
check('+ Source sends one message', len(add['sent']) == 1, str(add['sent']))
amsg = add['sent'][0] if add['sent'] else {}
check('  an add of a Source', amsg.get('op') == 'add'
      and amsg.get('type') == 'Source', str(amsg))
# A source is not an optics and does not take an optics' pose: it stands
# at a point and is aimed.
check('  carrying a position and a direction, not a face',
      'pos' in (amsg.get('params') or {})
      and 'dirAngle' in (amsg.get('params') or {})
      and 'HRcenter' not in (amsg.get('params') or {}),
      str(amsg.get('params')))
check('  where it was clicked',
      amsg['params']['pos'] == add['at'],
      '%s vs %s' % (amsg['params']['pos'], add['at']))
check('  and the mode is put away once it is placed',
      not add['stillArmed'])
check('  with a name nothing in the layout has taken',
      amsg.get('name') not in [o['name'] for o in scene['optics']]
      + [s['name'] for s in scene['sources']], str(amsg.get('name')))
check('  and the page selects it as a source, not as an optics',
      add['selected'] == amsg.get('name') and add['optic'] is None,
      '%s / %s' % (add['selected'], add['optic']))
L = make_layout()
L.apply_edit(amsg)
check('Python makes the source it asked for',
      amsg['name'] in [s.name for s in L.sources],
      str([s.name for s in L.sources]))

check('the source panel offers a Remove', res['hasRemove'])
rmsg = res['removed'][-1] if res['removed'] else {}
check('  which sends a remove naming the source',
      rmsg.get('op') == 'remove' and rmsg.get('target') == 'b0', str(rmsg))

print('--- the tracing rules ---')
check('the panel is there', res['rules']['present'])
shown = res['rules']['shown']
check('it shows the order the layout is tracing at',
      abs(float(shown['order']) - 4) < 1e-12, shown['order'])
check('  and the threshold', abs(float(shown['power_threshold']) - 1e-4) < 1e-18,
      shown['power_threshold'])
check('  and the open beam length',
      abs(float(shown['open_beam_length']) - 1.0) < 1e-15,
      shown['open_beam_length'])
rsent = res['rules']['sent']
check('changing one sends a rules message', len(rsent) == 2, str(rsent))
check('  naming only what changed',
      rsent and rsent[0] == {'op': 'rules', 'rules': {'order': 9}},
      str(rsent[0] if rsent else None))
check('  and the threshold likewise',
      len(rsent) > 1 and rsent[1] == {'op': 'rules',
                                      'rules': {'power_threshold': 1e-9}},
      str(rsent[1] if len(rsent) > 1 else None))
check('something that is not a number is not sent', res['rules']['refused'])
check('  and the field goes back to the layout value',
      abs(float(res['rules']['restored']) - 4) < 1e-12,
      res['rules']['restored'])
L = make_layout()
for m in rsent:
    L.apply_edit(m)
check('Python takes both', L.rules.order == 9
      and L.rules.power_threshold == 1e-9, str(L.rules.to_dict()))

print('--- the laser goes with its beam ---')
# Hiding a layer hides the beams on it. A laser left drawn there would
# be pointing at nothing.
check('hiding the layer hides the lasers drawn on it',
      res['hidden'] == [False, False], str(res['hidden']))
check('showing it again brings them back',
      res['shownAgain'] == [True, True], str(res['shownAgain']))

print('--- a scene from Python ---')
ap = res['afterPush']
check('the selection survives a new scene', ap['selected'] == 'b0',
      str(ap['selected']))
check('  and the panel stays on it', ap['panel']['sourceShown'], str(ap['panel']))
check('  showing the value Python came back with',
      abs(float(ap['px']) - 0.2) < 1e-15, ap['px'])
check('  and the laser is drawn where Python put it',
      ap['drawnAt'] == 2, str(ap['drawnAt']))
ag = res['afterGone']
check('a scene without the source drops the panel',
      ag['selected'] is None and ag['panel']['beamShown'], str(ag))
check('  and stops drawing it', ag['count'] == 1, str(ag['count']))


print()
print('--- read-only viewer ---')
errs, ro = run(False)
check('no console error', errs == [], '\n        '.join(errs[:3]))
if ro is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', ro['error'] is None, str(ro['error'])[:500])

# Reading is always allowed: a written page shows what the layout is,
# and a laser is part of that.
check('the lasers are drawn there too', ro['count'] == 2, str(ro['count']))
check('and clicking one still shows the source',
      ro['clicked']['sourceShown'] and ro['clickedName'] == 'b0',
      str(ro['clicked']))
check('  with the same values, read-only',
      abs(float(ro['clickedFields']['w0x']) - 0.2) < 1e-12
      and ro['clickedFields']['wl'].strip().startswith('1064'),
      str(ro['clickedFields']))
check('there is no Remove on a page with no Python behind it',
      'hasRemove' not in ro or not ro.get('hasRemove'))
check('and no tracing rules to change either', not ro['rules']['present'])
check('nothing was sent', ro['sent'] == [], str(ro['sent']))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
