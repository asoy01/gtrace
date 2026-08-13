'''
The mechanics in a real browser: that a breadboard can be picked by its
area and only by its area, that everything else wins the click over it,
and that its pose is edited the way the others are - panel, drag,
Shift-drag - with every message fed back to Python and compared.

Two decisions carry the interaction and are checked hardest.

The first is the pick order. A mechanics is the largest thing in the
picture, and it is picked last: a beam crossing a breadboard, an optics
standing on it, a mount lying on it - all of them win, or they could
never be pointed at again. Among mechanics themselves the smallest
wins, for the same reason.

The second is that a mechanics is grabbed only while it is selected.
A breadboard can cover most of the bench, and a press on it usually
means "pan the view"; so the first click selects, and only then does a
drag move the body. The check drags an unselected board and
requires the view to move and the board to stay.
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
import gtrace.draw as draw
import gtrace.optcomp as opt
from gtrace.draw.viewer import viewer_css
from gtrace.layout import OpticalLayout, TraceRules, q_from_waist
from gtrace.mechanics import (Mechanics, breadboard, round_breadboard,
                              pedestal, clamping_fork)
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
    M1 = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.pi, diameter=10*cm,
                    thickness=5*cm, Refl_HR=0.99, Trans_HR=0.01, n=1.45,
                    name='M1')
    # The board runs under the whole beam path and under M1, so that
    # every "X wins over the board" check has a real overlap to decide.
    # A real breadboard rather than a bare rectangle: its holes are
    # what the drag snap and the resize checks work against.
    board = breadboard(0.6, 0.4, center=[0.3, 0.0], name='Board')
    # A mount on the board: the smallest-wins rule needs two mechanics
    # on top of each other. It carries a model so the model row has
    # something to show; the board has none so the row can hide.
    mount = Mechanics(shapes=[draw.Circle([0.0, 0.0], 0.012),
                              draw.Rectangle([-0.015, -0.015], 0.03, 0.03)],
                      center=[0.42, -0.12], name='Mount', model='POLARIS-K1')
    # A clamp standing on M1, whose pose is the mirror's doing: the
    # panel has to say so, its pose rows have to refuse the keyboard,
    # and a drag on it has to pan rather than move. M1's substrate
    # centre is [0.525, 0] facing pi, so an offset of [0, -0.09] in
    # its frame stands the clamp at [0.525, 0.09] - on the board, off
    # the beam, and clear of the mirror's own pick circle.
    clamp = Mechanics(shapes=[draw.Rectangle([-0.01, -0.01], 0.02, 0.02)],
                      name='Clamp', attached_to=M1, offset=[0.0, -0.09])
    # A pedestal standing clear of everything, to be dropped onto a
    # screw hole, and a fork standing on the pedestal with its turn
    # free, to be swung about it.
    # A round board off on its own: a tank is round, and so is what
    # goes in the bottom of it. Clear of the beam and of everything
    # else, so that a click on it is a click on it.
    tank = round_breadboard(0.30, name='Tank', center=[0.0, -0.5])
    post = pedestal(name='Post', center=[0.15, 0.15])
    fork = clamping_fork(name='Fork', attached_to=post,
                         fix_rotation=False)
    return OpticalLayout(optics=[M1], sources=[b0],
                         mechanics=[board, mount, clamp, post, fork,
                                    tank],
                         rules=TraceRules(order=2, power_threshold=1e-4))

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
        // length as the cursor moves, and that reflows the page.
        function rect() { return v.svg.getBoundingClientRect(); }
        function screenOf(p) {
            var r = rect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function panel() {
            return {
                title: el.querySelector('.gt-panel-title span').textContent,
                beamShown: v.readoutBody.style.display !== 'none',
                opticShown: v.opticBody.style.display !== 'none',
                mechShown: v.mechBody.style.display !== 'none',
                selectedMech: v.selectedMech,
                selectedOptic: v.selectedOptic
            };
        }
        function fields() {
            var o = {};
            for (var k in v.mechFields) {
                var f = v.mechFields[k];
                o[k] = f.editable ? f.el.value : f.el.textContent;
                o[k + '_shown'] = f.row.style.display !== 'none';
            }
            return o;
        }
        function setField(key, text) {
            var f = v.mechFields[key];
            f.el.value = text;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        function outlinePts() {
            if (v.mechOutline.style.display === 'none') { return null; }
            return v.mechOutline.getAttribute('points').split(' ')
                .map(function (p) { return p.split(',').map(Number); });
        }
        function button(text) {
            var found = null;
            Array.prototype.forEach.call(
                el.querySelectorAll('button'), function (b) {
                    if (b.textContent === text) { found = b; }
                });
            return found;
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

        // Scene points chosen against the layout: on the board but off
        // the beam, the optics, the mount and the laser box.
        var BOARD_PT = [0.2, 0.15];
        var MOUNT_PT = [0.42, -0.12];
        var BEAM_PT = [0.25, 0.0];
        var OFF_PT = [0.3, 0.45];

        out.mechCount = (SCENE.mechanics || []).length;

        // --- the pick order ---
        clickAt(screenOf(BOARD_PT));
        out.clickBoard = panel();
        out.boardFields = fields();
        out.boardOutline = outlinePts();
        out.boardOutlineWant = SCENE.mechanics[0].outline.map(function (p) {
            return v.sceneToScreen(p[0], p[1]);
        });

        clickAt(screenOf(BEAM_PT));
        out.clickBeam = panel();

        clickAt(screenOf(MOUNT_PT));
        out.clickMount = panel();
        out.mountFields = fields();

        var m1c = [0.525, 0.0];    // substrate centre of M1, on the board
        clickAt(screenOf(m1c));
        out.clickOptic = panel();

        clickAt(screenOf(OFF_PT));
        out.clickOff = panel();

        // --- a hidden layer is unpickable ---
        v.setLayerVisible('mechanics', false);
        clickAt(screenOf(BOARD_PT));
        out.clickHidden = panel();
        v.setLayerVisible('mechanics', true);

        // --- panning is not moving ---
        // The board is not selected: a drag across it pans the view and
        // sends nothing.
        var cx0 = v.cx;
        var a = screenOf(BOARD_PT);
        dragFromTo(a, [a[0] + 60, a[1]]);
        out.panned = {moved: v.cx !== cx0, sent: sent.length,
                      selected: v.selectedMech};

        // --- dragging the selection ---
        clickAt(screenOf(BOARD_PT));
        var before = sent.length;
        a = screenOf(BOARD_PT);
        // Alt: a plain drag settles on whatever marked point it comes
        // near, which is the check below rather than this one.
        dragFromTo(a, [a[0] + 50, a[1] + 30], {altKey: true});
        out.dragMove = {msg: sent[before] || null,
                        n: sent.length - before,
                        scale: v.scale};

        // --- Shift-drag turns about the centre ---
        // Grab a point of the board, swing it a quarter turn about the
        // centre, and require the angle in the message. The grab has
        // to land on bare board - a grab on M1 would take the mirror,
        // which is the pick order doing its job - so it starts above
        // the centre and swings to the left.
        clickAt(screenOf(BOARD_PT));
        before = sent.length;
        var c = SCENE.mechanics[0].center;
        var g0 = screenOf([c[0], c[1] + 0.15]);
        var g1 = screenOf([c[0] - 0.15, c[1]]);
        dragFromTo(g0, g1, {shiftKey: true});
        out.dragRotate = {msg: sent[before] || null, n: sent.length - before};

        // --- the panel edits ---
        clickAt(screenOf(BOARD_PT));
        before = sent.length;
        setField('cx', '0.31');
        out.editCx = sent[before] || null;
        before = sent.length;
        setField('angle', '15');
        out.editAngle = sent[before] || null;
        before = sent.length;
        setField('name', 'Bench');
        out.rename = {msg: sent[before] || null, selected: v.selectedMech};
        v.selectedMech = 'Board';   // the model was not really renamed

        // --- remove ---
        clickAt(screenOf(MOUNT_PT));
        before = sent.length;
        var foot = v.mechBody.querySelector('.gt-btn-danger');
        out.removeShown = !!foot;
        if (foot) { foot.click(); }
        out.remove = {msg: sent[before] || null, panel: panel()};

        // --- placing a body on a marked point ---
        // The pedestal dropped near a screw hole of the board: a
        // plain drag settles on it exactly, and Alt takes the cursor
        // at its word instead.
        var POST_PT = [0.15, 0.15];
        // Whichever hole of the board lies nearest the pedestal: the
        // grid is the board's business, not this check's.
        var hole = null, holeD = 1e9;
        (SCENE.snap || []).forEach(function (s) {
            if (s.kind !== 'hole' || s.optic !== 'Board') { return; }
            var dd = Math.hypot(s.point[0] - 0.16, s.point[1] - 0.16);
            if (dd < holeD) { hole = s.point; holeD = dd; }
        });
        out.hole = hole || null;
        if (hole) {
            clickAt(screenOf(POST_PT));
            before = sent.length;
            // A few pixels short of the hole, so that only the snap
            // can land it there.
            var near = v.screenToScene(0, 0);
            var to = screenOf([hole[0] - 0.004, hole[1] - 0.003]);
            dragFromTo(screenOf(POST_PT), to);
            out.postSnap = {msg: sent[before] || null,
                            n: sent.length - before};
            before = sent.length;
            dragFromTo(screenOf(POST_PT), to, {altKey: true});
            out.postFree = sent[before] || null;
        }

        // --- swinging a body whose turn is free ---
        var FORK_PT = [0.15 - 0.03, 0.15];
        clickAt(screenOf(FORK_PT));
        out.clickFork = {panel: panel(), fields: fields()};
        before = sent.length;
        var f0 = screenOf(FORK_PT);
        dragFromTo(f0, [f0[0], f0[1] - 60], {shiftKey: true});
        out.forkSwing = {msg: sent[before] || null, n: sent.length - before};
        // With its turn fixed, the same gesture pans instead.
        before = sent.length;
        var cy1 = v.cy;
        v.scene.mechanics.forEach(function (m) {
            if (m.name === 'Fork') { m.fix_rotation = true; }
        });
        dragFromTo(f0, [f0[0], f0[1] - 60], {shiftKey: true});
        out.forkFixed = {n: sent.length - before, panned: v.cy !== cy1};
        v.scene.mechanics.forEach(function (m) {
            if (m.name === 'Fork') { m.fix_rotation = false; }
        });

        // --- the Fix rotation checkbox ---
        // A row that could be clicked and did nothing: it was read for
        // a number, like every other row, and a checkbox has none.
        clickAt(screenOf(FORK_PT));
        var fr = v.mechFields.fix_rotation;
        out.forkFixRow = {
            shown: !!(fr && fr.row.style.display !== 'none'),
            editable: !!(fr && fr.editable),
            checked: (fr && fr.editable) ? fr.el.checked : null
        };
        before = sent.length;
        if (fr && fr.editable) {
            fr.el.checked = true;
            fr.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        out.forkFixToggled = {msg: sent[before] || null,
                              n: sent.length - before};
        // And ticking it where it already stands says nothing.
        before = sent.length;
        if (fr && fr.editable) {
            fr.el.checked = false;
            fr.el.dispatchEvent(new Event('change', {bubbles: true}));
            fr.el.checked = false;
            fr.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        out.forkFixAgain = sent.length - before;

        // --- Escape lets go ---
        clickAt(screenOf(BOARD_PT));
        window.dispatchEvent(new KeyboardEvent('keydown',
            {key: 'Escape', bubbles: true}));
        out.escape = panel();

        // --- an attached body ---
        var CLAMP_PT = [0.525, 0.09];
        clickAt(screenOf(CLAMP_PT));
        out.clickClamp = panel();
        out.clampFields = fields();
        out.clampDisabled = EDITABLE ? {
            cx: v.mechFields.cx.el.disabled,
            cy: v.mechFields.cy.el.disabled,
            angle: v.mechFields.angle.el.disabled,
            boardCx: null
        } : null;
        before = sent.length;
        var cp = screenOf(CLAMP_PT);
        var cvx = v.cx;
        dragFromTo(cp, [cp[0] + 40, cp[1] + 20]);
        out.clampDrag = {sent: sent.length - before, panned: v.cx !== cvx,
                         selected: v.selectedMech};
        // And the pose rows come back to life on a free body.
        if (EDITABLE) {
            clickAt(screenOf(BOARD_PT));
            out.clampDisabled.boardCx = v.mechFields.cx.el.disabled;
        }

        // --- the attachment is edited from the panel ---
        if (EDITABLE) {
            clickAt(screenOf(MOUNT_PT));
            out.attachOptions = Array.prototype.map.call(
                v.mechFields.attached.el.options,
                function (o) { return o.value; });
            before = sent.length;
            setField('attached', 'M1');
            out.attach = {msg: sent[before] || null};
            clickAt(screenOf(CLAMP_PT));
            before = sent.length;
            setField('attached', '');
            out.detach = {msg: sent[before] || null};
            // Choosing what is already chosen decides nothing.
            clickAt(screenOf(MOUNT_PT));
            before = sent.length;
            setField('attached', '');
            out.attachNoop = sent.length - before;

            // The offset rows: the adjustment an attached body still
            // owns.
            clickAt(screenOf(CLAMP_PT));
            before = sent.length;
            setField('oy', '-80');
            out.editOffset = sent[before] || null;
        }

        // --- a body under an element, reached by cycling ---
        // The board runs under M1, and M1's pick circle wins the
        // click; a mount with no offset is in exactly this position,
        // covered completely by its own mirror. Clicking the same
        // spot again walks element -> beams -> body, so the board
        // has to turn up within one lap.
        clickAt(screenOf(OFF_PT));
        var reached = null;
        for (var ci = 0; ci < 12 && !reached; ci++) {
            clickAt(screenOf(m1c));
            if (v.selectedMech) {
                reached = {clicks: ci + 1, mech: v.selectedMech};
            }
        }
        out.cycleToMech = reached;
        // And one more lap of the same spot comes back to the mirror.
        clickAt(screenOf(m1c));
        out.cycleWraps = panel();

        // --- the model library menu ---
        var hm = v.mechMenu;
        out.hwMenu = {
            shown: !!hm && hm.wrap.style.display !== 'none',
            items: hm ? Array.prototype.map.call(
                hm.menu.querySelectorAll('button'),
                function (b) { return b.textContent; }) : []
        };
        before = sent.length;
        var bbItem = null;
        if (hm) {
            Array.prototype.forEach.call(hm.menu.querySelectorAll('button'),
                function (b) { if (b.textContent === 'BB3030') { bbItem = b; } });
        }
        if (bbItem) { bbItem.click(); }
        out.hwAdd = {msg: sent[before] || null, selected: v.selectedMech};
        v.selectedMech = null;   // nothing was really added

        // What each model calls the bodies built from it. A part is
        // known by what it is - a mount is MT1 - and the shelf is
        // where that is written down.
        out.hwNames = {};
        if (hm) {
            Array.prototype.forEach.call(hm.menu.querySelectorAll('button'),
                function (b) {
                    var was = sent.length;
                    b.click();
                    var msg = sent[was] || null;
                    out.hwNames[b.textContent] = msg ? msg.name : null;
                    v.selectedMech = null;
                });
        }

        // --- every add button opens its own menu ---
        // Each of these is built from a wrap, a button and a menu held
        // in local variables, and `var` is function-scoped: two blocks
        // that name theirs the same share one, and the click handlers
        // then close over whichever ran last. That is not visible in
        // what the menus contain - only in what a click on the button
        // does - so it is checked here by clicking the buttons.
        out.menuOwn = (v.addMenus || []).map(function (entry, i) {
            v.closeAddMenus();
            entry.button.click();
            var shown = (v.addMenus || []).map(function (m) {
                return m.menu.style.display !== 'none';
            });
            return {label: entry.button.textContent,
                    open: shown.indexOf(true),
                    count: shown.filter(Boolean).length,
                    i: i};
        });
        v.closeAddMenus();

        // --- the assembly menu ---
        // An element and the parts that hold it, in one message. The
        // menu is filled from the scene, like the model shelf.
        var am = v.assemblyMenu;
        out.asmMenu = {
            shown: !!am && am.wrap.style.display !== 'none',
            items: am ? Array.prototype.map.call(
                am.menu.querySelectorAll('button'),
                function (b) { return b.textContent; }) : []
        };
        out.asmAdd = {};
        if (am) {
            Array.prototype.forEach.call(am.menu.querySelectorAll('button'),
                function (b) {
                    var was = sent.length;
                    b.click();
                    out.asmAdd[b.textContent] = {
                        msg: sent[was] || null,
                        n: sent.length - was,
                        selectedOptic: v.selectedOptic,
                        selectedMech: v.selectedMech
                    };
                    v.selectedOptic = null;   // nothing was really added
                });
        }

        // --- the shape menu ---
        // Where the view is looking, read here: what + Shape puts down
        // lands at the centre, and the driver moves the view later on.
        out.cx = v.cx;
        out.cy = v.cy;
        var sm = v.shapeMenu;
        out.shMenu = {
            shown: !!sm && sm.wrap.style.display !== 'none',
            items: sm ? Array.prototype.map.call(
                sm.menu.querySelectorAll('button'),
                function (b) { return b.textContent; }) : []
        };
        out.shAdd = {};
        if (sm) {
            Array.prototype.forEach.call(sm.menu.querySelectorAll('button'),
                function (b, i) {
                    var was = sent.length;
                    b.click();
                    out.shAdd[b.textContent] = {
                        msg: sent[was] || null,
                        selected: v.selectedMech,
                        scale: v.shapeScale()
                    };
                    v.selectedMech = null;   // nothing was really added
                });
        }
        // The same button at two zooms: what it puts down is sized to
        // what is on screen, so the second is twice the first.
        out.shZoom = null;
        if (sm) {
            var circleBtn = null;
            Array.prototype.forEach.call(sm.menu.querySelectorAll('button'),
                function (b) { if (b.textContent === 'Circle') { circleBtn = b; } });
            if (circleBtn) {
                var keep = v.scale;
                var n0 = sent.length;
                circleBtn.click();
                var first = sent[n0] || null;
                v.scale = keep / 2;          // twice as much of the bench
                var n1 = sent.length;
                circleBtn.click();
                var second = sent[n1] || null;
                v.scale = keep;
                v.selectedMech = null;
                out.shZoom = {first: first, second: second};
            }
        }

        // --- the size rows, and the corner handles ---
        clickAt(screenOf(BOARD_PT));
        out.sizeRows = {board: fields()};
        var hcorners = [[0.0, -0.2], [0.6, -0.2], [0.6, 0.2], [0.0, 0.2]];
        out.handles = v.mechHandles.map(function (elh, i) {
            var s = v.sceneToScreen(hcorners[i][0], hcorners[i][1]);
            return {shown: elh.style.display !== 'none',
                    dx: Math.abs(Number(elh.getAttribute('x')) + 3.5 - s[0]),
                    dy: Math.abs(Number(elh.getAttribute('y')) + 3.5 - s[1])};
        });
        clickAt(screenOf(MOUNT_PT));
        out.sizeRows.mount = fields();
        out.mountHandles = v.mechHandles.some(function (elh) {
            return elh.style.display !== 'none';
        });
        // A round board is cut to one number, so it offers one row.
        clickAt(screenOf([0.0, -0.5]));
        out.sizeRows.tank = fields();
        out.tankPicked = v.selectedMech;

        // --- dragging a corner cuts the board to size ---
        clickAt(screenOf(BOARD_PT));
        before = sent.length;
        dragFromTo(screenOf([0.6, 0.2]), screenOf([0.75, 0.25]));
        out.resize = {msg: sent[before] || null, n: sent.length - before};

        // The same gesture on a round one: the centre stays where it
        // is - a disc has no opposite corner to hold - and the drag
        // sets one size rather than two.
        clickAt(screenOf([0.0, -0.5]));
        before = sent.length;
        var rh = v.mechHandles[2], rr = rect();
        var rfrom = [Number(rh.getAttribute('x')) + 3.5 + rr.left,
                     Number(rh.getAttribute('y')) + 3.5 + rr.top];
        var rto = screenOf([0.20, -0.30]);
        mouse(v.svg, 'mousedown', rfrom[0], rfrom[1]);
        mouse(window, 'mousemove', rto[0], rto[1]);
        out.roundStatus = v.statusBar.textContent;
        mouse(window, 'mouseup', rto[0], rto[1]);
        out.roundResize = {msg: sent[before] || null,
                           n: sent.length - before};
        // The rows follow the same rule: one number, sent as the size
        // it is.
        clickAt(screenOf([0.0, -0.5]));
        before = sent.length;
        v.mechFields.diameter.el.value = '400';
        v.mechFields.diameter.el.dispatchEvent(
            new Event('change', {bubbles: true}));
        out.roundRow = sent[before] || null;

        // --- the screw holes catch a dragged anchor ---
        // Zoomed in first: the hole reach is capped in metres, and the
        // page reflows by a pixel or two as the status bar changes, so
        // the test wants those pixels to be small on the bench.
        v.scale = 1200;
        v.cx = 0.45; v.cy = 0.05;
        v._applyTransform();
        var holes = (SCENE.snap || []).filter(function (p) {
            return p.kind === 'hole';
        });
        out.holeCount = holes.length;
        var anchor = SCENE.optics[0].HRcenter;
        var hole = null, hbest = Infinity;
        holes.forEach(function (p) {
            var dd = Math.hypot(p.point[0] - 0.4, p.point[1] - 0.1);
            if (dd < hbest) { hbest = dd; hole = p; }
        });
        var hdelta = [hole.point[0] + 0.001 - anchor[0],
                      hole.point[1] + 0.0005 - anchor[1]];
        before = sent.length;
        dragFromTo(screenOf(m1c),
                   screenOf([m1c[0] + hdelta[0], m1c[1] + hdelta[1]]));
        out.holeSnap = {msg: sent[before] || null, hole: hole.point};
        // The same drag with Alt held rides free.
        before = sent.length;
        dragFromTo(screenOf(m1c),
                   screenOf([m1c[0] + hdelta[0], m1c[1] + hdelta[1]]),
                   {altKey: true});
        out.holeFree = {msg: sent[before] || null};

        // A laser is bolted down like anything else, so the point its
        // light leaves from catches on the holes too. Held by the
        // box behind that point rather than by the point itself, so
        // that what lands on the hole is the model's own place and
        // not the cursor.
        var lp = SCENE.sources[0].pos;
        var lhole = null, lbest = Infinity;
        holes.forEach(function (p) {
            var dd = Math.hypot(p.point[0] - lp[0], p.point[1] - lp[1]);
            if (dd < lbest) { lbest = dd; lhole = p; }
        });
        out.srcHole = lhole.point;
        var ldelta = [lhole.point[0] + 0.0012 - lp[0],
                      lhole.point[1] - 0.0007 - lp[1]];
        var lgrab = screenOf(lp);
        lgrab = [lgrab[0] - 12, lgrab[1]];
        var lto = screenOf([lp[0] + ldelta[0], lp[1] + ldelta[1]]);
        lto = [lto[0] - 12, lto[1]];
        before = sent.length;
        mouse(v.svg, 'mousedown', lgrab[0], lgrab[1]);
        mouse(window, 'mousemove', lto[0], lto[1]);
        out.srcSnapMarked = v.snapMark.style.display !== 'none';
        out.srcStatus = v.statusBar.textContent;
        mouse(window, 'mouseup', lto[0], lto[1]);
        out.srcSnap = {msg: sent[before] || null, n: sent.length - before};
        before = sent.length;
        mouse(v.svg, 'mousedown', lgrab[0], lgrab[1], {altKey: true});
        mouse(window, 'mousemove', lto[0], lto[1], {altKey: true});
        mouse(window, 'mouseup', lto[0], lto[1], {altKey: true});
        out.srcFree = {msg: sent[before] || null};

        // --- Copy takes the element and what stands on it ---
        // The panel has to be showing an element: the button belongs
        // to it, and a body or a laser has no stack to bring along.
        clickAt(screenOf(m1c));
        out.copyPanel = v.panelKind;
        out.copyButtons = Array.prototype.map.call(
            v.opticBody.querySelectorAll('.gt-props-foot button'),
            function (b) { return b.textContent; });
        if (EDITABLE) {
            before = sent.length;
            button('Copy').click();
            out.copy = {msg: sent[before] || null, n: sent.length - before,
                        selected: v.selectedOptic};
            // A body selected instead: the optics panel is not on
            // show, so the button is not the one that answers.
            clickAt(screenOf(BOARD_PT));
            before = sent.length;
            out.copyOnBody = {kind: v.panelKind,
                              msg: v.copySelected(),
                              n: sent.length - before};
        }

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
    path = os.path.join(SP, 'mech_page_%s.html' % editable)
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

board = scene['mechanics'][0]

print('--- the pick order ---')
check('the scene carries every body: a board, a mount, a clamp, '
      'a post, a fork and a round board',
      res['mechCount'] == 6, str(res['mechCount']))
p = res['clickBoard']
check('a click on the empty board selects it',
      p['mechShown'] and p['selectedMech'] == 'Board'
      and p['title'] == 'Mechanics properties', json.dumps(p))
check('its outline is drawn', res['boardOutline'] is not None)
if res['boardOutline']:
    want = res['boardOutlineWant']
    got = res['boardOutline']
    check('  where the scene says it stands',
          all(math.hypot(g[0] - w[0], g[1] - w[1]) < 1e-6
              for g, w in zip(got, want)))
check('a beam crossing the board wins the click',
      res['clickBeam']['beamShown']
      and not res['clickBeam']['selectedMech'], json.dumps(res['clickBeam']))
check('the mount wins over the board it lies on',
      res['clickMount']['selectedMech'] == 'Mount',
      str(res['clickMount']['selectedMech']))
check('an optics standing on the board wins over it',
      res['clickOptic']['opticShown']
      and res['clickOptic']['selectedOptic'] == 'M1'
      and not res['clickOptic']['selectedMech'],
      json.dumps(res['clickOptic']))
check('a click off everything lets go of the board',
      res['clickOff']['beamShown'] and not res['clickOff']['selectedMech'])
check('a hidden mechanics layer is unpickable',
      not res['clickHidden']['selectedMech']
      and res['clickHidden']['beamShown'], json.dumps(res['clickHidden']))

print('--- the panel ---')
f = res['boardFields']
check('name, type and layer', f['name'] == 'Board'
      and f['type'] == 'Mechanics' and f['layer'] == 'mechanics',
      json.dumps({k: f[k] for k in ['name', 'type', 'layer']}))
check('the pose in metres and degrees',
      abs(float(f['cx']) - board['center'][0]) < 1e-12
      and abs(float(f['cy']) - board['center'][1]) < 1e-12
      and abs(float(f['angle'])) < 1e-12,
      json.dumps({k: f[k] for k in ['cx', 'cy', 'angle']}))
check('no model row for a body with no model', not f['model_shown'])
check('the mount shows its model',
      res['mountFields']['model_shown']
      and res['mountFields']['model'] == 'POLARIS-K1',
      str(res['mountFields']['model']))

print('--- panning is not moving ---')
check('dragging an unselected board pans the view',
      res['panned']['moved'] and res['panned']['sent'] == 0
      and not res['panned']['selected'], json.dumps(res['panned']))

print('--- dragging the selection ---')
d = res['dragMove']
check('one move message per drag', d['n'] == 1
      and d['msg'] and d['msg']['op'] == 'move'
      and d['msg']['target'] == 'Board', json.dumps(d['msg']))
if d['msg']:
    dx = d['msg']['center'][0] - board['center'][0]
    dy = d['msg']['center'][1] - board['center'][1]
    # 50 px right and 30 px down on screen, in a y-up scene.
    check('  by what the cursor moved',
          abs(dx - 50 / d['scale']) < 1e-9
          and abs(dy + 30 / d['scale']) < 1e-9,
          '(%.6f, %.6f at scale %.1f)' % (dx, dy, d['scale']))
    # What Python then does is what the preview showed.
    layout.apply_edit(d['msg'])
    check('  and Python lands it there',
          np.allclose(layout.get_mechanics('Board').center,
                      d['msg']['center']))
    layout.apply_edit({'op': 'undo'})

r = res['dragRotate']
check('one rotate message per Shift-drag', r['n'] == 1
      and r['msg'] and r['msg']['op'] == 'rotate'
      and r['msg']['target'] == 'Board', json.dumps(r['msg']))
if r['msg']:
    # Within half a degree, not exactly: the status bar changes as the
    # cursor moves, which reflows the page and shifts the drawing by a
    # pixel or two between the press and the release - the same
    # re-measurement trap verify_stage2b_browser documents. The exact
    # contract is the next check: Python lands the angle the preview
    # sent, whatever the pixels came to.
    check('  a quarter turn about the centre',
          abs(r['msg']['rotationAngle'] - np.pi / 2) < np.deg2rad(0.5),
          '(%.6f rad)' % r['msg']['rotationAngle'])
    layout.apply_edit(r['msg'])
    check('  and Python lands it there',
          abs(layout.get_mechanics('Board').rotationAngle
              - r['msg']['rotationAngle']) < 1e-12)
    layout.apply_edit({'op': 'undo'})

print('--- the panel edits ---')
check('Center x edits as a move',
      res['editCx'] and res['editCx']['op'] == 'move'
      and abs(res['editCx']['center'][0] - 0.31) < 1e-12
      and abs(res['editCx']['center'][1] - board['center'][1]) < 1e-12,
      json.dumps(res['editCx']))
check('Angle edits as a rotate, in radians',
      res['editAngle'] and res['editAngle']['op'] == 'rotate'
      and abs(res['editAngle']['rotationAngle'] - np.deg2rad(15)) < 1e-12,
      json.dumps(res['editAngle']))
check('the name edits as a rename, carried optimistically',
      res['rename']['msg'] and res['rename']['msg']['op'] == 'rename'
      and res['rename']['msg']['name'] == 'Bench'
      and res['rename']['selected'] == 'Bench', json.dumps(res['rename']))

print('--- remove and escape ---')
check('the Remove button is offered', res['removeShown'])
check('and sends the removal',
      res['remove']['msg'] and res['remove']['msg']['op'] == 'remove'
      and res['remove']['msg']['target'] == 'Mount',
      json.dumps(res['remove']['msg']))
check('  and lets go of the selection',
      not res['remove']['panel']['selectedMech']
      and res['remove']['panel']['beamShown'])
check('Escape lets go too', not res['escape']['selectedMech']
      and res['escape']['beamShown'])

print('--- an attached body ---')
check('clicking the clamp selects it',
      res['clickClamp']['selectedMech'] == 'Clamp',
      json.dumps(res['clickClamp']))
cf = res['clampFields']
check('the panel names its host',
      cf['attached'] == 'M1' and cf['attached_shown'], str(cf['attached']))
check('a free body offers the row too, standing free',
      res['boardFields']['attached_shown']
      and res['boardFields']['attached'] == '',
      repr(res['boardFields']['attached']))
# Against the pose Python derived and put in the scene, not against
# the nominal numbers of the layout: the substrate centre of M1 sits a
# hair off 0.525 by way of its default wedge.
clamp = [m for m in scene['mechanics'] if m['name'] == 'Clamp'][0]
check('its pose in the panel is the derived one',
      abs(float(cf['cx']) - clamp['center'][0]) < 1e-12
      and abs(float(cf['cy']) - clamp['center'][1]) < 1e-12,
      json.dumps({'cx': cf['cx'], 'cy': cf['cy']}))
check('its pose rows refuse the keyboard',
      res['clampDisabled']['cx'] and res['clampDisabled']['cy']
      and res['clampDisabled']['angle'], json.dumps(res['clampDisabled']))
check('  and come back to life on a free body',
      res['clampDisabled']['boardCx'] is False)
check('a drag on it pans and sends nothing',
      res['clampDrag']['sent'] == 0 and res['clampDrag']['panned']
      and res['clampDrag']['selected'] == 'Clamp',
      json.dumps(res['clampDrag']))

print('--- placing a body on a marked point ---')
check('the board offers a screw hole to aim at',
      res['hole'] is not None
      and any(np.allclose(sp['point'], res['hole'])
              for sp in layout.snap_points() if sp['kind'] == 'hole'),
      json.dumps(res['hole']))
ps = res.get('postSnap') or {}
check('a dragged body settles on it exactly',
      ps.get('n') == 1 and ps.get('msg')
      and np.allclose(ps['msg']['center'], res['hole'], atol=1e-12),
      json.dumps(ps.get('msg')))
if ps.get('msg'):
    layout.apply_edit(ps['msg'])
    check('  and Python stands it there',
          np.allclose(layout.get_mechanics('Post').center, res['hole'],
                      atol=1e-12))
    check('  which is what the pedestal names its axis',
          np.allclose(layout.get_mechanics('Post').world_points()['axis'],
                      res['hole'], atol=1e-12))
pf = res.get('postFree')
check('Alt rides free of the marks',
      pf and not np.allclose(pf['center'], res['hole'], atol=1e-6),
      json.dumps(pf))

print('--- swinging a body whose turn is free ---')
ck = res['clickFork']
check('the fork panel says what it stands on and that it may turn',
      ck['panel']['selectedMech'] == 'Fork'
      and ck['fields'].get('attached') == 'Post', json.dumps(ck['fields']))
fs = res['forkSwing']
check('Shift + drag on it sends one rotate',
      fs['n'] == 1 and fs['msg'] and fs['msg']['op'] == 'rotate'
      and fs['msg']['target'] == 'Fork', json.dumps(fs['msg']))
if fs['msg']:
    post0 = np.array(layout.get_mechanics('Post').center)
    layout.apply_edit(fs['msg'])
    fork = layout.get_mechanics('Fork')
    check('  which Python turns about the post it is held by',
          np.allclose(fork.world_points()['bore'], post0, atol=1e-12)
          and abs(fork.rotationAngle - fs['msg']['rotationAngle']) < 1e-12)
ff = res['forkFixed']
check('with its turn fixed the same gesture pans instead',
      ff['n'] == 0 and ff['panned'], json.dumps(ff))

print('--- the attachment is edited from the panel ---')
check('the choices are free, every optics and every other body',
      res['attachOptions'] == ['', 'M1', 'Board', 'Clamp', 'Post', 'Fork',
                               'Tank'],
      json.dumps(res['attachOptions']))
at = res['attach']
check('picking an optics sends the attachment',
      at['msg'] and at['msg']['op'] == 'set'
      and at['msg']['target'] == 'Mount'
      and at['msg']['attrs'] == {'attached_to': 'M1'}, json.dumps(at))
if at['msg']:
    mount_m = layout.get_mechanics('Mount')
    pos0 = mount_m.center.copy()
    layout.apply_edit(at['msg'])
    # Where a mount belongs on its host is the model's to say, so
    # attaching seats it at the designed position - the host's
    # substrate centre - rather than leaving it where it was dropped.
    check('  and it seats at its designed position on M1',
          mount_m.attached_to is layout.get_optics('M1')
          and np.allclose(mount_m.center, layout.get_optics('M1').center)
          and not np.allclose(mount_m.center, pos0))
    layout.apply_edit({'op': 'undo'})
dt = res['detach']
check('picking free sends the detachment',
      dt['msg'] and dt['msg']['op'] == 'set'
      and dt['msg']['target'] == 'Clamp'
      and dt['msg']['attrs'] == {'attached_to': None}, json.dumps(dt))
if dt['msg']:
    clamp_m = layout.get_mechanics('Clamp')
    pos0 = clamp_m.center.copy()
    layout.apply_edit(dt['msg'])
    check('  and the clamp is freed in place',
          clamp_m.attached_to is None and np.allclose(clamp_m.center, pos0))
    layout.apply_edit({'op': 'undo'})
check('choosing what is already chosen decides nothing',
      res['attachNoop'] == 0, str(res['attachNoop']))

cf = res['clampFields']
check('an attached body offers its offset, in millimetres',
      cf['ox_shown'] and cf['oy_shown'] and cf['oangle_shown']
      and abs(float(cf['ox'])) < 1e-9 and abs(float(cf['oy']) + 90) < 1e-9
      and abs(float(cf['oangle'])) < 1e-9,
      json.dumps({'ox': cf['ox'], 'oy': cf['oy'], 'oa': cf['oangle']}))
check('a free body has no offset rows',
      not res['boardFields']['ox_shown']
      and not res['boardFields']['oangle_shown'])
eo = res['editOffset']
check('editing one sends the whole offset',
      eo and eo['op'] == 'set' and eo['target'] == 'Clamp'
      and np.allclose(eo['attrs']['offset'], [0.0, -0.08]),
      json.dumps(eo))
if eo:
    clamp_m = layout.get_mechanics('Clamp')
    layout.apply_edit(eo)
    # The offset lives in the host frame: M1 faces pi, so a -0.08
    # across in its frame lands +0.08 across on the bench.
    check('  and the body moves off its designed spot accordingly',
          np.allclose(clamp_m.center,
                      np.asarray(layout.get_optics('M1').center)
                      + [0.0, 0.08]),
          str(list(clamp_m.center)))
    layout.apply_edit({'op': 'undo'})

print('--- a body under an element, reached by cycling ---')
check('clicking the same spot again reaches the body under M1',
      res['cycleToMech'] and res['cycleToMech']['mech'] == 'Board',
      json.dumps(res['cycleToMech']))
check('and the next click wraps back to the mirror',
      res['cycleWraps']['selectedOptic'] == 'M1'
      and not res['cycleWraps']['selectedMech'],
      json.dumps(res['cycleWraps']))

print('--- the Fix rotation checkbox ---')
fr = res['forkFixRow']
check('an attached body whose turn is free offers the row',
      fr['shown'] and fr['editable'], json.dumps(fr))
check('  showing the turn as free', fr['checked'] is False, json.dumps(fr))
ft = res['forkFixToggled']
check('ticking it sends one message, and says what a checkbox says',
      ft['n'] == 1 and ft['msg'] and ft['msg']['op'] == 'set'
      and ft['msg']['target'] == 'Fork'
      and ft['msg']['attrs'] == {'fix_rotation': True},
      json.dumps(ft))
if ft['msg']:
    layout.apply_edit(ft['msg'])
    check('  and Python freezes the turn',
          layout.get_mechanics('Fork').fix_rotation is True)
    layout.apply_edit({'op': 'undo'})
    check('  which undo lets go again',
          layout.get_mechanics('Fork').fix_rotation is False)
# The scene is what the model says, and nothing here is applied to it,
# so putting the box back where the body stands is not an edit.
check('setting it to what the body already is says nothing',
      res['forkFixAgain'] == 0, str(res['forkFixAgain']))

print('--- the model library menu ---')
check('+ Mechanics lists exactly the library shelf',
      res['hwMenu']['shown']
      and set(res['hwMenu']['items'])
          == set(e['name'] for e in scene['mechlib']),
      json.dumps(res['hwMenu']['items']))
ha = res['hwAdd']
check('choosing a model sends the add, carried optimistically',
      ha['msg'] and ha['msg']['op'] == 'add'
      and ha['msg']['type'] == 'Mechanics'
      and ha['msg']['params']['model'] == 'BB3030'
      and ha['selected'] == ha['msg']['name'], json.dumps(ha))
layout.apply_edit(ha['msg'])
check('  and Python builds it from the shelf',
      layout.get_mechanics(ha['msg']['name']).resizable)
layout.apply_edit({'op': 'undo'})

# What a body off the shelf is called. The model says it, so a mount is
# MT1 whatever catalogue the footprint came from - and 'PD' is nobody's
# prefix here, since a photodetector is what that reads as.
WANT_PREFIX = {'BB3030': 'BB', 'BB4530': 'BB', 'BB6045': 'BB',
               'BBR30': 'BB', 'BBR45': 'BB',
               'MOUNT-25': 'MT', 'MOUNT-50': 'MT',
               'HOLDER-25': 'HLD', 'HOLDER-50': 'HLD',
               'PEDESTAL-25': 'P', 'FORK-125': 'FK'}
names = res['hwNames']
check('every model on the shelf was asked for', set(names) == set(WANT_PREFIX),
      json.dumps(sorted(names)))
for model, prefix in sorted(WANT_PREFIX.items()):
    got = names.get(model)
    check('%s is added as %s1, not H1' % (model, prefix),
          got == prefix + '1', str(got))
check('and no two of them collide with what is already there',
      all(v not in [m.name for m in layout.mechanics]
          for v in names.values()), json.dumps(names))

def scaled(shape, k):
    '''
    A shape dict with every length multiplied and every angle left
    alone - written out here rather than taken from the page, since
    what is being checked is that the page did it right.
    '''
    out = dict(shape)
    for key in ('point', 'center', 'start', 'stop', 'pivot'):
        if out.get(key) is not None:
            out[key] = [v * k for v in out[key]]
    for key in ('x', 'y'):
        if out.get(key) is not None:
            out[key] = [v * k for v in out[key]]
    for key in ('width', 'height', 'radius'):
        if out.get(key) is not None:
            out[key] = out[key] * k
    return out

def same_shape(a, b, tol=1e-9):
    if set(a) != set(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, list) != isinstance(y, list):
            return False
        if isinstance(x, list):
            if len(x) != len(y) or any(abs(p - q) > tol for p, q in zip(x, y)):
                return False
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) > tol:
                return False
        elif x != y:
            return False
    return True

print('--- every add button opens its own menu ---')
own = res['menuOwn']
check('there is a menu for every add button that has one',
      len(own) >= 4, json.dumps([m['label'] for m in own]))
for m in own:
    check('%s opens its own menu, and only that one' % m['label'],
          m['open'] == m['i'] and m['count'] == 1, json.dumps(m))

print('--- the assembly menu ---')
KINDS = [a['label'] for a in scene['assemblies']]
check('+ Assembly lists the kinds the scene carries',
      res['asmMenu']['shown'] and res['asmMenu']['items'] == KINDS,
      json.dumps(res['asmMenu']['items']))
for entry in scene['assemblies']:
    aa = res['asmAdd'].get(entry['label']) or {}
    msg = aa.get('msg')
    check('choosing %s sends one add of that kind' % entry['label'],
          aa.get('n') == 1 and msg and msg['op'] == 'add'
          and msg['type'] == 'Assembly' and msg['kind'] == entry['kind'],
          json.dumps(msg))
    if not msg:
        continue
    check('  named for the element, and selected as one',
          msg['name'].startswith(entry['prefix'])
          and aa['selectedOptic'] == msg['name']
          and not aa['selectedMech'], json.dumps(msg['name']))
    check('  at the centre of the view',
          np.allclose(msg['params']['center'], [res['cx'], res['cy']]),
          json.dumps(msg['params']['center']))
    n_optics = len(layout.optics)
    n_mech = len(layout.mechanics)
    layout.apply_edit(msg)
    check('  and Python builds the element and three parts from it',
          len(layout.optics) == n_optics + 1
          and len(layout.mechanics) == n_mech + 3,
          '%d optics, %d bodies' % (len(layout.optics), len(layout.mechanics)))
    check('  the parts standing on the element',
          layout.mechanics[-3].attached_to is layout.optics[-1],
          layout.mechanics[-3].name)
    layout.apply_edit({'op': 'undo'})
    check('  and one undo takes all four away',
          len(layout.optics) == n_optics and len(layout.mechanics) == n_mech)

print('--- the shape menu ---')
KINDS = ['Rect', 'Circle', 'Line', 'Poly', 'Arc', 'Text']
check('+ Shape offers every kind the editor draws with',
      res['shMenu']['shown'] and res['shMenu']['items'] == KINDS,
      json.dumps(res['shMenu']['items']))
TYPES = {'Rect': 'rectangle', 'Circle': 'circle', 'Line': 'line',
         'Poly': 'polyline', 'Arc': 'arc', 'Text': 'text'}
for label, kind in TYPES.items():
    sa = res['shAdd'].get(label) or {}
    msg = sa.get('msg')
    check('choosing %s sends a body of one %s' % (label, kind),
          msg and msg['op'] == 'add' and msg['type'] == 'Mechanics'
          and len(msg['params']['shapes']) == 1
          and msg['params']['shapes'][0]['type'] == kind
          and 'model' not in msg['params'],
          json.dumps(msg))
    if not msg:
        continue
    check('  carried optimistically, like a model off the shelf',
          sa['selected'] == msg['name'])
    check('  and named for the shape it is',
          msg['name'] == {'Rect': 'RECT1', 'Circle': 'CIRC1',
                          'Line': 'LINE1', 'Poly': 'POLY1',
                          'Arc': 'ARC1', 'Text': 'TEXT1'}[label],
          msg['name'])
    check('  at the centre of the view',
          np.allclose(msg['params']['center'], [res['cx'], res['cy']]),
          json.dumps(msg['params']['center']))
    # The shape is Python's own answer to what a new one looks like,
    # scaled: the page holds no second opinion about what a circle is.
    want = scene['newshapes'][kind]
    got = msg['params']['shapes'][0]
    k = sa['scale']
    check('  which is the scene\'s own new %s, scaled by the view' % kind,
          same_shape(got, scaled(want, k)),
          json.dumps({'got': got, 'want': scaled(want, k)}))
    layout.apply_edit(msg)
    body = layout.get_mechanics(msg['name'])
    check('  and Python builds it where the page said',
          len(body.shapes) == 1
          and np.allclose(body.center, msg['params']['center']))
    layout.apply_edit({'op': 'undo'})

sz = res['shZoom']
check('the same button puts down twice as much at half the zoom',
      sz and sz['first'] and sz['second']
      and abs(sz['second']['params']['shapes'][0]['radius']
              - 2 * sz['first']['params']['shapes'][0]['radius']) < 1e-12,
      json.dumps([sz['first']['params']['shapes'][0]['radius'],
                  sz['second']['params']['shapes'][0]['radius']])
      if sz and sz['first'] and sz['second'] else '')

print('--- the size rows and the corner handles ---')
bf = res['sizeRows']['board']
mf = res['sizeRows']['mount']
check('the board shows its size in millimetres',
      bf['width_shown'] and bf['height_shown']
      and abs(float(bf['width']) - 600) < 1e-9
      and abs(float(bf['height']) - 400) < 1e-9,
      json.dumps({'w': bf['width'], 'h': bf['height']}))
check('a hand-drawn body has no size rows',
      not mf['width_shown'] and not mf['height_shown']
      and not mf['diameter_shown'])
tf = res['sizeRows']['tank']
check('a round board offers one size row instead of two',
      res['tankPicked'] == 'Tank'
      and tf['diameter_shown']
      and not tf['width_shown'] and not tf['height_shown']
      and abs(float(tf['diameter']) - 300) < 1e-9,
      json.dumps({'d': tf['diameter'], 'w': tf['width_shown']}))
check('  and the rectangular one offers the two and not the one',
      not bf['diameter_shown'])
check('four handles stand on the corners',
      all(h['shown'] and h['dx'] < 1e-6 and h['dy'] < 1e-6
          for h in res['handles']), json.dumps(res['handles']))
check('and none on a hand-drawn body', not res['mountHandles'])

print('--- dragging a corner cuts the board to size ---')
rz = res['resize']
check('one set message per corner drag',
      rz['n'] == 1 and rz['msg'] and rz['msg']['op'] == 'set',
      json.dumps(rz['msg']))
if rz['msg']:
    at = rz['msg']['attrs']
    check('  roughly the rectangle that was dragged',
          abs(at['width'] - 0.75) < 0.02 and abs(at['height'] - 0.45) < 0.02,
          '(%.4f x %.4f)' % (at['width'], at['height']))
    # The dragged corner was the top right, so the bottom left stayed.
    check('  with the opposite corner standing still',
          abs(at['center'][0] - at['width'] / 2 - 0.0) < 0.02
          and abs(at['center'][1] - at['height'] / 2 + 0.2) < 0.02)
    n0 = sum(1 for s in layout.get_mechanics('Board').shapes
             if isinstance(s, draw.Circle))
    layout.apply_edit(rz['msg'])
    n1 = sum(1 for s in layout.get_mechanics('Board').shapes
             if isinstance(s, draw.Circle))
    check('  and Python re-drills the grid rather than scaling it',
          n1 > n0 and abs(layout.get_mechanics('Board').params['width']
                          - at['width']) < 1e-12,
          '(%d -> %d holes)' % (n0, n1))
    layout.apply_edit({'op': 'undo'})

print('--- and a round one is cut to a diameter ---')
rr = res['roundResize']
check('one set message per corner drag on it too',
      rr['n'] == 1 and rr['msg'] and rr['msg']['op'] == 'set'
      and rr['msg']['target'] == 'Tank', json.dumps(rr['msg']))
if rr['msg']:
    at = rr['msg']['attrs']
    check('  carrying one size, the same on both names',
          at['width'] == at['height'], json.dumps(at))
    check('  and no centre: a disc is cut about where it stands',
          'center' not in at, json.dumps(sorted(at)))
    check('  roughly the disc that was dragged',
          abs(at['width'] - 0.40) < 0.02, str(at['width']))
    layout.apply_edit(rr['msg'])
    check('  which Python re-drills at that diameter',
          abs(layout.get_mechanics('Tank').params['width'] - at['width'])
          < 1e-12
          and np.allclose(layout.get_mechanics('Tank').center, [0.0, -0.5]))
    layout.apply_edit({'op': 'undo'})
check('the status bar says the one size, not two',
      '⌀' in res['roundStatus'] and '×' not in res['roundStatus'],
      res['roundStatus'])
rw = res['roundRow']
check('the Diameter row sends the size on both names',
      rw and rw['op'] == 'set' and abs(rw['attrs']['width'] - 0.40) < 1e-12
      and rw['attrs']['width'] == rw['attrs']['height'], json.dumps(rw))

print('--- the screw holes catch a dragged anchor ---')
check('the scene has a grid to snap to', res['holeCount'] >= 384,
      str(res['holeCount']))
hs = res['holeSnap']
check('the drag sends a move', hs['msg'] and hs['msg']['op'] == 'move'
      and hs['msg']['target'] == 'M1', json.dumps(hs['msg']))
if hs['msg']:
    layout.apply_edit(hs['msg'])
    check('  landing the anchor exactly on the hole',
          np.allclose(layout.get_optics('M1').HRcenter, hs['hole'],
                      atol=1e-9),
          '(%s vs %s)' % (list(layout.get_optics('M1').HRcenter),
                          hs['hole']))
    layout.apply_edit({'op': 'undo'})
hf = res['holeFree']
check('Alt rides free of the grid',
      hf['msg'] and hs['msg']
      and math.hypot(hf['msg']['center'][0] - hs['msg']['center'][0],
                     hf['msg']['center'][1] - hs['msg']['center'][1]) > 1e-6,
      json.dumps(hf['msg']))

ss = res['srcSnap']
check('a dragged laser sends one move of where its light leaves from',
      ss['n'] == 1 and ss['msg'] and ss['msg']['op'] == 'move'
      and ss['msg']['target'] == 'b0', json.dumps(ss['msg']))
if ss['msg']:
    check('  landing that point exactly on the hole',
          np.allclose(ss['msg']['pos'], res['srcHole'], atol=1e-9),
          '(%s vs %s)' % (ss['msg']['pos'], res['srcHole']))
    layout.apply_edit(ss['msg'])
    check('  which is where Python puts the laser',
          np.allclose(layout.get_source('b0').pos, res['srcHole'],
                      atol=1e-9))
    layout.apply_edit({'op': 'undo'})
check('  and the mark and the bar say so while the drag is on',
      res['srcSnapMarked'] and 'Board hole' in res['srcStatus'],
      res['srcStatus'])
sf = res['srcFree']
check('Alt rides the laser free of the grid too',
      sf['msg'] and ss['msg']
      and math.hypot(sf['msg']['pos'][0] - ss['msg']['pos'][0],
                     sf['msg']['pos'][1] - ss['msg']['pos'][1]) > 1e-6,
      json.dumps(sf['msg']))

print('--- copying an element ---')
check('the optics panel offers Copy beside Remove',
      res['copyPanel'] == 'optic'
      and res['copyButtons'] == ['Copy', 'Remove'],
      json.dumps(res['copyButtons']))
cp = res['copy']
check('Copy asks for one, under the next free name',
      cp['n'] == 1 and cp['msg'] and cp['msg']['op'] == 'copy'
      and cp['msg']['target'] == 'M1' and cp['msg']['name'] == 'M2',
      json.dumps(cp['msg']))
check('  and the selection follows the copy', cp['selected'] == 'M2')
if cp['msg']:
    was = [m.name for m in layout.mechanics]
    layout.apply_edit(cp['msg'])
    now = [m.name for m in layout.mechanics]
    # Only the clamp stands on M1 here - the mount of this fixture is
    # free on purpose, so that the attach gesture has something to
    # attach - and only what stands on it is copied.
    check('  which Python builds with what stands on it',
          layout._is_optics('M2')
          and [n for n in now if n not in was] == ['Clamp1'],
          json.dumps(now))
    check('  pinned to the copy rather than to the original',
          layout.get_mechanics('Clamp1').attached_to
          is layout.get_optics('M2')
          and np.allclose(layout.get_mechanics('Clamp1').offset,
                          layout.get_mechanics('Clamp').offset))
    layout.apply_edit({'op': 'undo'})
    check('  and one undo takes the copy and its bodies back',
          not layout._is_optics('M2')
          and [m.name for m in layout.mechanics] == was)
check('Copy belongs to the optics panel alone',
      res['copyOnBody']['kind'] == 'mech'
      and res['copyOnBody']['msg'] is None
      and res['copyOnBody']['n'] == 0, json.dumps(res['copyOnBody']))

print('--- read-only viewer ---')
errs, res = run(False)
check('no console error', errs == [], '\n        '.join(errs[:3]))
if res is None:
    print('  FAIL  no output')
    sys.exit(1)
check('ran without exception', res['error'] is None, str(res['error'])[:500])
check('a click still shows the body',
      res['clickBoard']['mechShown']
      and res['clickBoard']['selectedMech'] == 'Board')
f = res['boardFields']
check('the pose reads as static text',
      abs(float(f['cx']) - board['center'][0]) < 1e-12, str(f['cx']))
check('the clamp still names its host',
      res['clampFields']['attached'] == 'M1'
      and res['clampFields']['attached_shown'])
check('no Remove on offer', not res['removeShown'])
check('  and no Copy either', res['copyButtons'] == [],
      json.dumps(res['copyButtons']))
check('no model menu either', not res['hwMenu']['shown'])
check('and no resize handles',
      all(not h['shown'] for h in res['handles']))
check('nothing was ever sent', res['sent'] == [], str(res['sent'][:2]))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
