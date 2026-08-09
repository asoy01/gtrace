'''
Verification of the optics properties panel, in a real browser.

Clicks an optics, reads back every field, edits several of them and
checks the edit messages that leave the page. The messages are then fed
to Python's apply_edit so that both sides are compared on the same
numbers.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, require_chrome

import json
import math
import os
import re
import subprocess
import sys

import numpy as np

import gtrace.beam as beam
import gtrace.draw.serialize as ser
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

    def M(name, c, a, roc=0.0, rh=0.99, th=0.01):
        return opt.Mirror(HRcenter=c, normAngleHR=a, diameter=10*cm,
                          thickness=5*cm, wedgeAngle=deg2rad(0.25),
                          inv_ROC_HR=roc, Refl_HR=rh, Trans_HR=th,
                          Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45,
                          name=name)

    optics = [M('M1', [0.5, 0.0], deg2rad(135)),
              M('M2', [0.5, 0.4], deg2rad(-45), 1.0/2.0),
              M('M3', [0.9, 0.4], deg2rad(180), 0.0, 0.9, 0.1)]
    return OpticalLayout(optics=optics, sources=[b0],
                         rules=TraceRules(order=5, power_threshold=1e-4)), optics

layout, (M1, M2, M3) = make_layout()
scene = layout.scene_dict()

# A real lens, serialized the way a scene carries one, handed to the
# page separately. It is pushed into the scene from the browser rather
# than registered in the layout, so that adding it does not disturb the
# trace the checks above it are written against - the same trick the
# CyMirror row uses. What Python actually puts in a scene is checked in
# verify_stage2b.py; this is the panel's side of the same dict.
LENS_OPTIC = ser.optic_to_dict(
    opt.Lens(f=0.3, center=[0.7, 0.15], normAngleHR=np.pi, name='L1'))

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
var LENS = __LENS__;
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

        var state = {scene: SCENE, title: 'Props', height: 640,
                     editable: EDITABLE, error: ''};
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
            // The real model has one; a stand-in that lacks it
            // would let a missing call pass unnoticed.
            save_changes: function () {}
        };

        var el = document.getElementById('host');
        mod.default.render({model: model, el: el});
        var v = el.gtraceViewer;
        var r = v.svg.getBoundingClientRect();
        function screenOf(p) {
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function panel() {
            return {
                title: el.querySelector('.gt-panel-title span').textContent,
                beamShown: v.readoutBody.style.display !== 'none',
                propsShown: v.opticBody.style.display !== 'none'
            };
        }
        function fields() {
            var o = {};
            for (var k in v.opticFields) {
                var f = v.opticFields[k];
                if (f.kind === 'bool') { o[k] = f.editable ? f.el.checked
                                                           : f.el.textContent; }
                else { o[k] = f.editable ? f.el.value : f.el.textContent; }
            }
            return o;
        }
        function rowShown(key) {
            return v.opticFields[key].row.style.display !== 'none';
        }
        function setField(key, text) {
            var f = v.opticFields[key];
            f.el.value = text;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
        }
        function toggleField(key, on) {
            var f = v.opticFields[key];
            f.el.checked = on;
            f.el.dispatchEvent(new Event('change', {bubbles: true}));
        }

        out.start = panel();
        function button(text) {
            var found = null;
            Array.prototype.forEach.call(
                el.querySelectorAll('button'), function (b) {
                    if (b.textContent === text) { found = b; }
                });
            return found;
        }
        // --- making room ---
        // A notebook cell is a letterbox and a bench drawing is not, so
        // the two controls that give the drawing more of it: fold the
        // side panel away, and drag the bottom edge down.
        var stageW = function () {
            return Math.round(v.svg.getBoundingClientRect().width);
        };
        out.room = {wideBefore: stageW()};
        el.querySelector('.gt-sidetoggle').click();
        out.room.folded = {
            sideShown: v.side.style.display !== 'none',
            wide: stageW(),
            // The drawing has to notice: the viewport is what the
            // scene is mapped onto, and a stale one distorts it.
            viewBox: v.svg.getAttribute('viewBox'),
            label: el.querySelector('.gt-sidetoggle').textContent
        };
        el.querySelector('.gt-sidetoggle').click();
        out.room.unfolded = {
            sideShown: v.side.style.display !== 'none',
            wide: stageW(),
            label: el.querySelector('.gt-sidetoggle').textContent
        };

        var grip = el.querySelector('.gt-resize');
        out.room.hasGrip = !!grip;
        if (grip) {
            // The element the viewer was mounted into, which is the one
            // whose height the grip changes - not the div this page
            // pinned at a fixed size around it.
            var box = v.container;
            var h0 = Math.round(box.getBoundingClientRect().height);
            var scale0 = v.scale;
            mouse(grip, 'mousedown', 0, 300);
            mouse(window, 'mousemove', 0, 500);
            mouse(window, 'mouseup', 0, 500);
            out.room.resized = {
                from: h0,
                to: Math.round(box.getBoundingClientRect().height),
                height: state.height,
                svgHeight: Math.round(v.svg.getBoundingClientRect().height),
                // Dragging taller must not reframe: the view is already
                // where the user put it.
                scaleKept: v.scale === scale0
            };
            // And it will not be dragged away to nothing.
            mouse(grip, 'mousedown', 0, 500);
            mouse(window, 'mousemove', 0, -4000);
            mouse(window, 'mouseup', 0, -4000);
            out.room.floor = Math.round(box.getBoundingClientRect().height);
            box.style.height = h0 + 'px';
            v._resize();
        }

        // Which row each button is on. The two kinds are kept apart on
        // purpose, so that which row a button lands on does not depend
        // on how wide the panel happens to be.
        out.headRows = Array.prototype.map.call(
            el.querySelectorAll('.gt-head .gt-btnrow'),
            function (r) {
                return Array.prototype.map.call(
                    r.querySelectorAll('button'),
                    function (b) { return b.textContent; });
            });
        // The head is buttons and nothing else: the layout is labelled
        // by the cell that made it, or by the browser tab.
        out.headHasTitle = !!el.querySelector('.gt-title');
        out.headChildren = Array.prototype.map.call(
            el.querySelector('.gt-head').children,
            function (c) { return c.className; });

        out.addButton = !!button('+ Mirror');
        out.addCyButton = !!button('+ CyMirror');
        out.addLensButton = !!button('+ Lens');
        out.addCyLensButton = !!button('+ CyLens');
        // Scoped to the optics panel. The dimension panel builds one of
        // its own whether or not there is anywhere to send edits, since
        // a viewer with no Python behind it can still take back a
        // measurement it drew itself.
        out.removeButton = !!v.opticBody.querySelector('.gt-btn-danger');
        out.undoButton = button('Undo') ? {
            // The scene handed over has no history behind it, so the
            // button has to start out of reach.
            disabled: button('Undo').disabled,
            sceneSays: v.scene.can_undo
        } : null;
        out.redoButton = button('Redo') ? {
            disabled: button('Redo').disabled,
            sceneSays: v.scene.can_redo
        } : null;
        out.saveButton = !!button('Save');
        out.loadButton = !!button('Load');
        out.dxfButton = !!button('Export');
        out.panelTitles = Array.prototype.map.call(
            el.querySelectorAll('.gt-panel-title'),
            function (t) { return t.textContent; });
        out.pathShown = v.pathInput ? v.pathInput.value : null;

        // --- click M1 ---
        var m1 = v.scene.optics.filter(function (o) {
            return o.name === 'M1';
        })[0];
        var p = screenOf(m1.center);
        mouse(window, 'mousemove', p[0], p[1]);
        mouse(v.svg, 'mousedown', p[0], p[1]);
        mouse(window, 'mouseup', p[0], p[1]);
        out.afterClick = panel();
        out.selected = v.selectedOptic;
        out.fields = fields();
        out.editableFields = Object.keys(v.opticFields).filter(function (k) {
            return v.opticFields[k].editable;
        });
        out.outlineSelected =
            v.outline.classList.contains('gt-selected')
            || v.outline.style.display !== 'none';
        out.m1 = m1;
        // Scoped to the optics panel: the dimension panel is styled the
        // same way and has a heading of its own.
        out.groupHeadings = Array.prototype.map.call(
            v.opticBody.querySelectorAll('.gt-group'),
            function (g) { return g.textContent; });
        out.curveRowShown = rowShown('curve_direction');
        // Recorded before any edit, so that the starting state of the
        // controls is compared with the scene as it was handed over.
        out.display = {present: !!v.displayControls};
        if (v.displayControls) {
            out.display.sigmaOptions = Array.prototype.map.call(
                v.displayControls.sigma.options, function (o) {
                    return o.value;
                });
            out.display.modeOptions = Array.prototype.map.call(
                v.displayControls.width_mode.options, function (o) {
                    return o.value;
                });
            out.display.sigmaShown = v.displayControls.sigma.value;
            out.display.modeShown = v.displayControls.width_mode.value;
            out.display.sceneSays = v.scene.display;
        }

        if (EDITABLE) {
            // --- edit several fields ---
            setField('cx', '0.55');
            setField('angle', '120');
            setField('rocHR', '3.5');
            setField('rocAR', '-2.5');
            setField('diameter', '0.2');
            setField('wedgeAngle', '0.5');
            setField('Refl_HR', '0.95');
            out.sentAfterEdits = sent.length;

            // Rubbish must not be sent, and the field must snap back.
            setField('diameter', 'not a number');
            out.afterBadInput = {sent: sent.length,
                                 shown: v.opticFields.diameter.el.value};

            // Setting a field to the value it already has sends nothing.
            setField('n', v.opticFields.n.el.value);
            out.afterNoop = {sent: sent.length};


            // M2 is curved, so flattening it is a real change: the only
            // way to check that 'inf' maps to a zero inverse ROC.
            var m2 = v.scene.optics.filter(function (o) {
                return o.name === 'M2';
            })[0];
            var p2 = screenOf(m2.center);
            mouse(window, 'mousemove', p2[0], p2[1]);
            mouse(v.svg, 'mousedown', p2[0], p2[1]);
            mouse(window, 'mouseup', p2[0], p2[1]);
            out.m2 = {selected: v.selectedOptic,
                      rocShown: v.opticFields.rocHR.el.value,
                      wantRoc: 1 / m2.inv_ROC_HR};
            setField('rocHR', 'inf');
            out.sentAfterFlatten = sent.length;

            // A field that may be unset. Clearing it only counts as a
            // change once Python has echoed the value back, so push the
            // scene in between - exactly what the real loop does.
            var nNull = sent.length;
            setField('max_stray_order', '2');
            var capped = JSON.parse(JSON.stringify(v.scene));
            capped.optics.filter(function (o) {
                return o.name === 'M2';
            })[0].max_stray_order = 2;
            model.set('scene', capped);
            out.nullableShown = v.opticFields.max_stray_order.el.value;
            setField('max_stray_order', 'auto');
            setField('max_stray_order', 'not a number');
            out.nullable = {sent: sent.length - nNull,
                            msgs: sent.slice(nNull),
                            shown: v.opticFields.max_stray_order.el.value};

            // --- the tracing flags are checkboxes ---
            var nFlag = sent.length;
            toggleField('HRtransmissive', true);
            toggleField('term_on_HR', true);
            setField('term_on_HR_order', '2');
            // Echo one back for whichever optics is selected, then set
            // it to what it already is.
            var flagged = JSON.parse(JSON.stringify(v.scene));
            flagged.optics.filter(function (o) {
                return o.name === v.selectedOptic;
            })[0].HRtransmissive = true;
            model.set('scene', flagged);
            out.flagEchoed = v.opticFields.HRtransmissive.el.checked;
            toggleField('HRtransmissive', true);
            out.flags = {sent: sent.length - nFlag,
                         msgs: sent.slice(nFlag)};

            // --- a row for something this class lacks stays hidden ---
            var withCurve = JSON.parse(JSON.stringify(v.scene));
            var cy = JSON.parse(JSON.stringify(withCurve.optics[1]));
            cy.name = 'Cy';
            cy.type = 'CyMirror';
            cy.curve_direction = 'v';
            withCurve.optics.push(cy);
            model.set('scene', withCurve);
            v._selectOptic(cy);
            out.curve = {shown: rowShown('curve_direction'),
                         value: v.opticFields.curve_direction.el.value,
                         kind: v.opticFields.curve_direction.kind,
                         options: Array.prototype.map.call(
                             v.opticFields.curve_direction.el.options,
                             function (o) { return o.value; })};
            setField('curve_direction', 'h');
            out.curve.msg = sent[sent.length - 1];
            // Echo it back, then pick the same value again: no message.
            var turned = JSON.parse(JSON.stringify(v.scene));
            turned.optics.filter(function (o) {
                return o.name === v.selectedOptic;
            })[0].curve_direction = 'h';
            model.set('scene', turned);
            var nCurve = sent.length;
            setField('curve_direction', 'h');
            out.curve.sentAfterSame = sent.length - nCurve;
            out.curve.shownAfterEcho = v.opticFields.curve_direction.el.value;

            // --- a lens: the focal length row and the anchor ---
            var withLens = JSON.parse(JSON.stringify(v.scene));
            withLens.optics.push(JSON.parse(JSON.stringify(LENS)));
            model.set('scene', withLens);
            var lensOptic = withLens.optics[withLens.optics.length - 1];
            v._selectOptic(lensOptic);
            out.lens = {
                type: fields().type,
                rowShown: rowShown('f'),
                shown: v.opticFields.f.el.value,
                // The panel is in millimetres; the scene carries the
                // power, in reciprocal metres.
                wantF: 1 / (LENS.inv_f * 0.001),
                label: v.opticFields.f.row.querySelector('.gt-key')
                        .textContent,
                anchorRowShown: rowShown('anchor_point'),
                anchorShown: v.opticFields.anchor_point.el.value,
                anchorKind: v.opticFields.anchor_point.kind,
                anchorOptions: Array.prototype.map.call(
                    v.opticFields.anchor_point.el.options,
                    function (o) { return o.value; })
            };

            var nLens = sent.length;
            setField('f', '200');
            out.lens.msg = sent[sent.length - 1];
            out.lens.sent = sent.length - nLens;

            // An infinity would not survive JSON: what reaches Python is
            // a null it cannot use. The field must refuse it outright
            // and put back what the model holds.
            setField('f', 'inf');
            out.lens.afterInf = {sent: sent.length - nLens,
                                 shown: v.opticFields.f.el.value};

            // The anchor is a choice, like the curve direction.
            var nAnchor = sent.length;
            setField('anchor_point', 'HRcenter');
            out.lens.anchorMsg = sent[sent.length - 1];
            out.lens.anchorSent = sent.length - nAnchor;

            // A mirror has no focal length, so that row is not for it.
            v._selectOptic(v.scene.optics[0]);
            out.lens.rowOnMirror = rowShown('f');
            out.lens.anchorRowOnMirror = rowShown('anchor_point');
            out.lens.anchorOnMirror = v.opticFields.anchor_point.el.value;

            // --- moving along a beam by a typed distance ---
            // M1 has the source beam landing on it, so the rows are
            // offered; the third mirror has nothing through it.
            var slideBeams = v._beamsThrough(v.scene.optics[0]);
            out.slide = {
                rowShown: rowShown('slide_beam'),
                byRowShown: rowShown('slide_by'),
                kind: v.opticFields.slide_beam.kind,
                options: Array.prototype.map.call(
                    v.opticFields.slide_beam.el.options,
                    function (o) { return o.textContent; }),
                chosen: v.slideBeam,
                nearest: slideBeams.length ? slideBeams[0].beam.name : null,
                byShown: v.opticFields.slide_by.el.value,
                label: v.opticFields.slide_by.row.querySelector('.gt-key')
                        .textContent
            };
            var nSlide = sent.length;
            setField('slide_by', '50');
            out.slide.msg = sent[sent.length - 1];
            out.slide.sent = sent.length - nSlide;
            // A distance is an instruction, not a value to hold: the
            // field has to come back to zero without waiting for a
            // scene, or leaning on Enter walks the optics down the
            // bench.
            out.slide.byAfter = v.opticFields.slide_by.el.value;
            setField('slide_by', '0');
            setField('slide_by', 'sideways');
            out.slide.afterQuiet = {sent: sent.length - nSlide,
                                    shown: v.opticFields.slide_by.el.value};

            // --- Ctrl + click a beam names it in the row ---
            // Pick a beam through M1 other than the one already chosen,
            // and click it with Ctrl held.
            // Somewhere along a different beam and clear of every
            // optics: over an optics the click grabs the optics, which
            // is what a click there is for.
            var other = null, pickPt = null;
            slideBeams.forEach(function (h) {
                if (other || h.beam.name === v.slideBeam.name) { return; }
                var bb = h.beam, q = null;
                for (var t = 0.1; t < 0.95 && !q; t += 0.05) {
                    var p = [bb.pos[0] + bb.dirVect[0] * bb.length * t,
                             bb.pos[1] + bb.dirVect[1] * bb.length * t];
                    if (!v._pickOptic(p[0], p[1])) { q = p; }
                }
                if (q) { other = h; pickPt = q; }
            });
            out.slide.otherBeam = other ? other.beam.name : null;
            out.slide.pickPoint = pickPt;
            if (other && pickPt) {
                var pc = screenOf(pickPt);
                var nCtrl = sent.length;
                mouse(window, 'mousemove', pc[0], pc[1]);
                mouse(v.svg, 'mousedown', pc[0], pc[1], {ctrlKey: true});
                mouse(window, 'mouseup', pc[0], pc[1], {ctrlKey: true});
                out.slide.byClick = {
                    chosen: v.slideBeam,
                    // Several beams routinely lie on top of each other,
                    // so what was clicked is not one beam but a bundle.
                    // What the choice has to be is one of that bundle
                    // and one that passes through the optics.
                    under: v._pickAll(pickPt[0], pickPt[1],
                                      12 / v.scale).map(function (h) {
                        return h.beam.name;
                    }),
                    through: v._beamsThrough(v.scene.optics[0])
                              .map(function (h) { return h.beam.name; }),
                    shownInPicker: v.opticFields.slide_beam.el.value,
                    // The selection must survive: choosing what to move
                    // along is part of working on that element.
                    stillSelected: v.selectedOptic,
                    panel: panel(),
                    markShown: v.slideMark.style.display !== 'none',
                    sent: sent.length - nCtrl
                };

                // Clicking the same place again steps through the beams
                // that lie on top of each other there.
                var through = {};
                v._beamsThrough(v.scene.optics[0]).forEach(function (h) {
                    through[h.index] = true;
                });
                var bundle = v._pickAll(pickPt[0], pickPt[1], 12 / v.scale)
                              .filter(function (h) { return through[h.index]; });
                out.slide.cycle = {
                    bundle: bundle.map(function (h) { return h.index; }),
                    dirs: bundle.map(function (h) {
                        return h.beam.dirVect;
                    }),
                    picks: [],
                    // The mark and its arrow have to follow the choice:
                    // two beams on one line commonly run opposite ways,
                    // and which way is the sign of Move by.
                    arrows: [],
                    marks: []
                };
                for (var k = 0; k <= bundle.length; k++) {
                    mouse(window, 'mousemove', pc[0], pc[1]);
                    mouse(v.svg, 'mousedown', pc[0], pc[1], {ctrlKey: true});
                    mouse(window, 'mouseup', pc[0], pc[1], {ctrlKey: true});
                    out.slide.cycle.picks.push(v.slideBeam.index);
                    out.slide.cycle.arrows.push(
                        v.slideArrow.getAttribute('d'));
                    out.slide.cycle.marks.push((function () {
                        var b = v.scene.beams[v.slideBeam.index];
                        var a0 = v.sceneToScreen(b.pos[0], b.pos[1]);
                        var a1 = v.sceneToScreen(b.end[0], b.end[1]);
                        var m = ['x1', 'y1', 'x2', 'y2'].map(function (k) {
                            return parseFloat(v.slideMark.getAttribute(k));
                        });
                        return Math.max(Math.abs(m[0] - a0[0]),
                                        Math.abs(m[1] - a0[1]),
                                        Math.abs(m[2] - a1[0]),
                                        Math.abs(m[3] - a1[1])) < 1e-6;
                    })());
                }
                // A Ctrl-click somewhere else starts the cycle over
                // rather than carrying on from where it was.
                var pFar = screenOf([pickPt[0], pickPt[1] + 0.25]);
                mouse(window, 'mousemove', pFar[0], pFar[1]);
                mouse(v.svg, 'mousedown', pFar[0], pFar[1], {ctrlKey: true});
                mouse(window, 'mouseup', pFar[0], pFar[1], {ctrlKey: true});
                mouse(window, 'mousemove', pc[0], pc[1]);
                mouse(v.svg, 'mousedown', pc[0], pc[1], {ctrlKey: true});
                mouse(window, 'mouseup', pc[0], pc[1], {ctrlKey: true});
                out.slide.cycle.afterElsewhere = v.slideBeam.index;

                // Without Ctrl the same click is the readout, as before.
                mouse(window, 'mousemove', pc[0], pc[1]);
                mouse(v.svg, 'mousedown', pc[0], pc[1]);
                mouse(window, 'mouseup', pc[0], pc[1]);
                out.slide.plainClick = {
                    panel: panel(),
                    selected: v.selectedOptic,
                    pinned: v.pinned ? v.pinned.beam.name : null,
                    markShown: v.slideMark.style.display !== 'none'
                };
                v._selectOptic(v.scene.optics[0]);
            }

            // An optics with no beam through it has nothing to slide
            // along, and the pair of rows goes away.
            var lonely = JSON.parse(JSON.stringify(v.scene.optics[0]));
            lonely.name = 'Lonely';
            lonely.center = [9.0, 9.0];
            lonely.HRcenter = [9.0, 9.0];
            var withLonely = JSON.parse(JSON.stringify(v.scene));
            withLonely.optics.push(lonely);
            model.set('scene', withLonely);
            v._selectOptic(lonely);
            out.slide.rowsWhenNoBeam = [rowShown('slide_beam'),
                                        rowShown('slide_by')];
            v._selectOptic(v.scene.optics[0]);

            // Back to M1 for the checks that follow.
            mouse(window, 'mousemove', p[0], p[1]);
            mouse(v.svg, 'mousedown', p[0], p[1]);
            mouse(window, 'mouseup', p[0], p[1]);
        }
        out.sent = sent;

        // --- a scene coming back keeps the selection ---
        var moved = JSON.parse(JSON.stringify(SCENE));
        moved.optics[0].center = [0.77, 0.06];
        model.set('scene', moved);
        out.afterPush = {panel: panel(), selected: v.selectedOptic,
                         cx: v.opticFields.cx.editable
                             ? v.opticFields.cx.el.value
                             : v.opticFields.cx.el.textContent};

        // --- renaming ---
        if (EDITABLE) {
            var nBefore = sent.length;
            setField('name', 'PRM');
            out.rename = {msg: sent[sent.length - 1],
                          sent: sent.length - nBefore,
                          selected: v.selectedOptic,
                          fallback: v.selectionFallback};

            // Python refuses it: the viewer must go back to the old name.
            model.set('error', 'EditError: nope');
            out.renameRefused = {selected: v.selectedOptic,
                                 shown: v.opticFields.name.el.value,
                                 panel: panel()};
            model.set('error', '');

            // Now one that succeeds: the scene comes back renamed.
            setField('name', 'PRM');
            var renamed = JSON.parse(JSON.stringify(v.scene));
            renamed.optics[0].name = 'PRM';
            model.set('scene', renamed);
            out.renameAccepted = {selected: v.selectedOptic,
                                  shown: v.opticFields.name.el.value,
                                  fallback: v.selectionFallback,
                                  panel: panel()};

            // Put the original name back for the checks that follow.
            setField('name', 'M1');
            var restored = JSON.parse(JSON.stringify(v.scene));
            restored.optics[0].name = 'M1';
            model.set('scene', restored);

            // Blank and unchanged names must not be sent.
            var nQuiet = sent.length;
            setField('name', '   ');
            setField('name', 'M1');
            out.renameQuiet = {sent: sent.length - nQuiet,
                               shown: v.opticFields.name.el.value};
        }

        // --- adding and removing a mirror ---
        if (EDITABLE) {
            var nSent = sent.length;
            v.cx = 0.31; v.cy = 0.22; v._applyTransform();
            button('+ Mirror').click();
            out.add = {msg: sent[sent.length - 1],
                       sent: sent.length - nSent,
                       selected: v.selectedOptic,
                       viewCentre: [v.cx, v.cy]};

            // Python answers with a scene that has the new mirror in it.
            var grown = JSON.parse(JSON.stringify(v.scene));
            var proto = grown.optics[0];
            var added = JSON.parse(JSON.stringify(proto));
            added.name = out.add.msg.name;
            added.center = out.add.msg.params.HRcenter.slice();
            added.HRcenter = out.add.msg.params.HRcenter.slice();
            added.normAngleHR = out.add.msg.params.normAngleHR;
            added.inv_ROC_HR = 0;
            grown.optics.push(added);
            model.set('scene', grown);
            out.afterAdd = {panel: panel(), selected: v.selectedOptic,
                            name: v.opticFields.name.editable
                                ? v.opticFields.name.el.value
                                : v.opticFields.name.el.textContent};

            // A second one must not reuse the name.
            button('+ Mirror').click();
            out.secondName = sent[sent.length - 1].name;

            // The cylindrical kind, from its own button.
            var nCy = sent.length;
            button('+ CyMirror').click();
            out.addCy = {msg: sent[sent.length - 1],
                         sent: sent.length - nCy,
                         selected: v.selectedOptic};

            // A lens carries no parameters of its own from the view:
            // Python builds it from catalogue defaults, since a lens is
            // not cut to match the mirrors around it.
            var nLensBtn = sent.length;
            button('+ Lens').click();
            out.addLens = {msg: sent[sent.length - 1],
                           sent: sent.length - nLensBtn,
                           selected: v.selectedOptic};

            // The cylindrical lens: catalogue defaults like a Lens,
            // plus the curve direction like a CyMirror.
            var nCyLensBtn = sent.length;
            button('+ CyLens').click();
            out.addCyLens = {msg: sent[sent.length - 1],
                             sent: sent.length - nCyLensBtn,
                             selected: v.selectedOptic};

            // Put the first one back in the panel, then remove it.
            v._selectOptic(added);
            button('Remove').click();
            out.remove = {msg: sent[sent.length - 1],
                          selected: v.selectedOptic,
                          panel: panel()};

            // --- undo ---
            // Python answers an edit with a scene that says whether
            // there is now something to go back to.
            var nUndo = sent.length;
            button('Undo').click();
            out.undo = {sentWhileEmpty: sent.length - nUndo};
            var withHistory = JSON.parse(JSON.stringify(v.scene));
            withHistory.can_undo = true;
            model.set('scene', withHistory);
            out.undo.enabled = !button('Undo').disabled;
            button('Undo').click();
            out.undo.msg = sent[sent.length - 1];
            out.undo.sent = sent.length - nUndo;
            // Ctrl+Z does the same, but only with the pointer over this
            // viewer: a notebook page has its own undo.
            v.pointerInside = true;
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true}));
            out.undo.byKey = sent.length - nUndo;
            v.pointerInside = false;
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'z', ctrlKey: true, bubbles: true}));
            out.undo.byKeyOutside = sent.length - nUndo;
            var noHistory = JSON.parse(JSON.stringify(v.scene));
            noHistory.can_undo = false;
            model.set('scene', noHistory);
            out.undo.disabledAgain = button('Undo').disabled;

            // --- redo ---
            // The same shape as undo, driven by its own flag: the two
            // are independent, and a scene can offer either, both or
            // neither.
            var nRedo = sent.length;
            button('Redo').click();
            out.redo = {sentWhileEmpty: sent.length - nRedo};
            var withFuture = JSON.parse(JSON.stringify(v.scene));
            withFuture.can_redo = true;
            model.set('scene', withFuture);
            out.redo.enabled = !button('Redo').disabled;
            out.redo.undoStillDisabled = button('Undo').disabled;
            button('Redo').click();
            out.redo.msg = sent[sent.length - 1];
            out.redo.sent = sent.length - nRedo;
            // Ctrl+Shift+Z and Ctrl+Y both mean redo; Ctrl+Z alone must
            // not, or the shifted spelling would step the wrong way.
            v.pointerInside = true;
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Z', ctrlKey: true, shiftKey: true, bubbles: true}));
            out.redo.byShiftKey = sent[sent.length - 1];
            out.redo.afterShiftKey = sent.length - nRedo;
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'y', ctrlKey: true, bubbles: true}));
            out.redo.byCtrlY = sent[sent.length - 1];
            out.redo.afterCtrlY = sent.length - nRedo;
            v.pointerInside = false;
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'y', ctrlKey: true, bubbles: true}));
            out.redo.byKeyOutside = sent.length - nRedo;
            var noFuture = JSON.parse(JSON.stringify(v.scene));
            noFuture.can_redo = false;
            model.set('scene', noFuture);
            out.redo.disabledAgain = button('Redo').disabled;
        }

        // --- clicking a beam goes back to the readout ---
        var b = v.scene.beams[0];
        var pb = screenOf([b.pos[0] + b.dirVect[0] * b.length * 0.4,
                           b.pos[1] + b.dirVect[1] * b.length * 0.4]);
        mouse(window, 'mousemove', pb[0], pb[1]);
        mouse(v.svg, 'mousedown', pb[0], pb[1]);
        mouse(window, 'mouseup', pb[0], pb[1]);
        out.afterBeamClick = {panel: panel(), selected: v.selectedOptic,
                              beam: v.pinned ? v.pinned.beam.name : null};

        // --- repeated clicks cycle: the element, then the beams under
        // it, then the element again. Tried where a beam ends on an
        // element, since that is exactly the spot the element's grab
        // circle used to shadow for good.
        var cbeam = null, copt = null;
        for (var ci = 0; ci < v.scene.beams.length && !cbeam; ci++) {
            var cb = v.scene.beams[ci];
            var cg = v.layerGroups[cb.layer];
            var co = v._pickOptic(cb.end[0], cb.end[1]);
            if (co && (!cg || cg.visible)) { cbeam = cb; copt = co; }
        }
        out.cycleClicks = null;
        if (cbeam) {
            var pc = screenOf(cbeam.end);
            var clickAt = function () {
                mouse(window, 'mousemove', pc[0], pc[1]);
                mouse(v.svg, 'mousedown', pc[0], pc[1]);
                mouse(window, 'mouseup', pc[0], pc[1]);
            };
            // The cycle belongs to the place clicked; make sure the
            // clicks before this section do not count as that place.
            v.lastClick = null;
            var under = v._pickAll(cbeam.end[0], cbeam.end[1],
                                   12 / v.scale).length;
            clickAt();
            out.cycleClicks = {
                optic: copt.name, under: under,
                first: {panel: panel(), selected: v.selectedOptic}
            };
            clickAt();
            out.cycleClicks.second = {panel: panel(),
                                      selected: v.selectedOptic,
                                      beam: v.pinned ? v.pinned.beam.name
                                                     : null};
            // Keep clicking until the cycle comes back to the element.
            // The exact bundle size under the clicked pixel need not
            // match the probe above (the click coordinate is rounded),
            // so the count is bounded rather than assumed.
            var more = 0;
            while (more < under + 3 && v.selectedOptic === null) {
                clickAt();
                more += 1;
            }
            out.cycleClicks.wrapped = {panel: panel(),
                                       selected: v.selectedOptic,
                                       pinned: !!v.pinned,
                                       steps: more};
        }

        // --- the layout file buttons ---
        if (v.pathInput) {
            var nFile = sent.length;
            v.pathInput.value = 'chosen.json';
            button('Save').click();
            out.save = {msg: sent[sent.length - 1],
                        sent: sent.length - nFile};

            // Loading throws the selection away and reframes the view,
            // since the file may describe something else entirely.
            v._selectOptic(v.scene.optics[0]);
            var scaleBefore = v.scale;
            v.scale *= 4; v._applyTransform();
            button('Load').click();
            out.load = {msg: sent[sent.length - 1],
                        sent: sent.length - nFile - 1,
                        selectedAfterClick: v.selectedOptic,
                        fitPending: v.fitOnNextScene};
            // Python answers with a scene; the view must be refitted.
            model.set('scene', JSON.parse(JSON.stringify(v.scene)));
            out.load.fitPendingAfter = v.fitOnNextScene;
            out.load.refitted = v.scale !== scaleBefore * 4;

            // The drawing has a panel and a file name of its own: it
            // is not the layout, and Load must never be pressed on it.
            out.dxfStart = v.dxfInput.value;
            var nDxf = sent.length;
            v.dxfInput.value = 'drawing.dxf';
            button('Export').click();
            out.dxf = {msg: sent[sent.length - 1], sent: sent.length - nDxf};
            // Changing the layout's name does not touch the drawing's.
            v.pathInput.value = 'other.json';
            button('Export').click();
            out.dxfIndependent = sent[sent.length - 1].path;
            v.pathInput.value = 'chosen.json';

            // An extension the user typed is left alone; one they did
            // not type is filled in.
            out.dxfNames = ['plan', 'plan.dxf', 'plan.DXF', 'a.b/c',
                            'dir/'].map(function (p) {
                v.dxfInput.value = p;
                button('Export').click();
                return sent[sent.length - 1].path;
            });
            v.dxfInput.value = 'drawing.dxf';

            // A blank file name asks for nothing, in either panel.
            v.pathInput.value = '   ';
            v.dxfInput.value = '   ';
            var nBlank = sent.length;
            button('Save').click();
            button('Load').click();
            button('Export').click();
            out.blankPath = sent.length - nBlank;
            v.pathInput.value = 'chosen.json';
            v.dxfInput.value = 'drawing.dxf';
        }

        // --- the beam width controls, last so as not to shift the
        //     indices the field-edit checks above rely on ---
        if (v.displayControls) {
            var nDisp = sent.length;
            v.displayControls.sigma.value = '1';
            v.displayControls.sigma.dispatchEvent(
                new Event('change', {bubbles: true}));
            v.displayControls.width_mode.value = 'y';
            v.displayControls.width_mode.dispatchEvent(
                new Event('change', {bubbles: true}));
            out.display.sent = sent.length - nDisp;
            out.display.msgs = sent.slice(nDisp);

            // A scene drawn the new way puts the controls where it says.
            var redrawn = JSON.parse(JSON.stringify(v.scene));
            redrawn.display = {sigma_main: 3, sigma_stray: 3,
                               width_mode: 'avg'};
            model.set('scene', redrawn);
            out.display.afterPush = {
                sigma: v.displayControls.sigma.value,
                mode: v.displayControls.width_mode.value
            };
        }
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
               .replace('__LENS__', js(LENS_OPTIC)) \
               .replace('__EDITABLE__', 'true' if editable else 'false')
    path = os.path.join(SP, 'props_page_%s.html' % editable)
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
check('ran without exception', res['error'] is None, str(res['error'])[:400])

print('--- making room ---')
rm = res['room']
check('folding the side panel away hides it',
      not rm['folded']['sideShown'])
check('and the drawing takes the width it gave up',
      rm['folded']['wide'] > rm['wideBefore'] + 300,
      '%d -> %d' % (rm['wideBefore'], rm['folded']['wide']))
# The viewport is what the scene is mapped onto; a stale one would
# stretch the drawing across the new width.
check('the viewport follows',
      rm['folded']['viewBox'].split()[2] == str(rm['folded']['wide'])
      or abs(float(rm['folded']['viewBox'].split()[2])
             - rm['folded']['wide']) < 1.5,
      rm['folded']['viewBox'])
check('the button turns round to bring it back',
      rm['folded']['label'] != rm['unfolded']['label'],
      '%s / %s' % (rm['folded']['label'], rm['unfolded']['label']))
check('and it comes back to the width it had',
      rm['unfolded']['sideShown']
      and abs(rm['unfolded']['wide'] - rm['wideBefore']) < 1.5,
      '%d vs %d' % (rm['unfolded']['wide'], rm['wideBefore']))

check('there is a grip to drag it taller by', rm['hasGrip'])
rz = rm['resized']
check('dragging it down makes it taller',
      abs(rz['to'] - rz['from'] - 200) < 2,
      '%d -> %d' % (rz['from'], rz['to']))
# The height belongs to whatever mounted the viewer, so the new one is
# written back to the traitlet that set it: that is what makes it
# survive a re-render and lets Python read what it was dragged to.
check('and tells the model what it became',
      abs(rz['height'] - rz['to']) < 2,
      '%s vs %d' % (rz['height'], rz['to']))
check('the drawing gets the height, less the grip',
      0 < rz['to'] - rz['svgHeight'] < 40,
      '%d vs %d' % (rz['svgHeight'], rz['to']))
check('and the view is left where the user put it', rz['scaleKept'])
check('it cannot be dragged away to nothing', rm['floor'] >= 240,
      str(rm['floor']))

print('--- the head ---')
check('the buttons are on two rows', len(res['headRows']) == 2,
      str(res['headRows']))
check('what adds to the layout on the first',
      res['headRows'][0] == ['+ Mirror', '+ CyMirror', '+ Lens', '+ CyLens'],
      str(res['headRows'][0]))
check('what acts on it or on the view on the second',
      res['headRows'][1] == ['Undo', 'Redo', 'Measure', 'Fit'],
      str(res['headRows'][1]))
# No heading. In a notebook the layout is already labelled by the cell
# that made it, and a written page carries its name in the browser tab;
# a line of the side bar spent repeating it is a line not spent on the
# readout.
check('and there is no heading above them', not res['headHasTitle'],
      str(res['headChildren']))
check('nothing but the rows is in the head',
      res['headChildren'] == ['gt-btnrow', 'gt-btnrow'],
      str(res['headChildren']))

print('--- panel switching ---')
check('starts on the beam readout',
      res['start']['title'] == 'Beam readout' and res['start']['beamShown']
      and not res['start']['propsShown'], str(res['start']))
check('clicking an optics shows its properties',
      res['afterClick']['title'] == 'Optics properties'
      and res['afterClick']['propsShown']
      and not res['afterClick']['beamShown'], str(res['afterClick']))
check('the optics is recorded as selected', res['selected'] == 'M1',
      str(res['selected']))
check('and outlined in the drawing', res['outlineSelected'])

print('--- field values ---')
f = res['fields']
m1 = res['m1']
check('name', f['name'] == 'M1', f['name'])
check('type', f['type'] == 'Mirror', f['type'])
check('center x', abs(float(f['cx']) - m1['center'][0]) < 1e-15, f['cx'])
check('center y', abs(float(f['cy']) - m1['center'][1]) < 1e-15, f['cy'])
check('angle is shown in degrees',
      abs(float(f['angle']) - math.degrees(m1['normAngleHR'])) < 1e-9,
      '%s vs %.6f' % (f['angle'], math.degrees(m1['normAngleHR'])))
check('wedge is shown in degrees',
      abs(float(f['wedgeAngle']) - math.degrees(m1['wedgeAngle'])) < 1e-9,
      f['wedgeAngle'])
check('diameter', abs(float(f['diameter']) - m1['diameter']) < 1e-15,
      f['diameter'])
check('thickness', abs(float(f['thickness']) - m1['thickness']) < 1e-15,
      f['thickness'])
check('a flat surface reads as an infinite ROC',
      m1['inv_ROC_HR'] == 0 and f['rocHR'] == 'inf', f['rocHR'])
check('index n', abs(float(f['n']) - m1['n']) < 1e-15, f['n'])
check('reflectivity', abs(float(f['Refl_HR']) - m1['Refl_HR']) < 1e-15,
      f['Refl_HR'])
check('everything but the type is editable',
      set(res['editableFields']) == {'name', 'cx', 'cy', 'angle', 'diameter',
                                     'thickness', 'wedgeAngle', 'rocHR',
                                     'rocAR', 'n', 'Refl_HR', 'Trans_HR',
                                     'Refl_AR', 'Trans_AR', 'max_stray_order',
                                     'HRtransmissive', 'HRreflective',
                                     'term_on_HR',
                                     'term_on_HR_order', 'curve_direction',
                                     # Built for every optics; the rows hide
                                     # themselves where they do not apply -
                                     # a focal length on anything but a lens,
                                     # a beam to slide along where none
                                     # passes through.
                                     'f', 'anchor_point',
                                     'slide_beam', 'slide_by'},
      str(sorted(res['editableFields'])))
check('the type is not editable', 'type' not in res['editableFields'])
check('an unset max stray order reads as auto',
      f['max_stray_order'] == 'auto', f['max_stray_order'])

print('--- edits leaving the panel ---')
sent = res['sent']
check('one message per edited field', res['sentAfterEdits'] == 7,
      str(res['sentAfterEdits']))
by_kind = {}
for msg in sent:
    key = msg['op'] + ':' + ','.join(sorted(msg.get('attrs', {}).keys()))
    by_kind[key] = msg

check('center x becomes a move',
      sent[0]['op'] == 'move' and abs(sent[0]['center'][0] - 0.55) < 1e-15
      and abs(sent[0]['center'][1] - m1['center'][1]) < 1e-15,
      str(sent[0]))
check('the angle becomes a rotate in radians',
      sent[1]['op'] == 'rotate'
      and abs(sent[1]['normAngleHR'] - math.radians(120)) < 1e-15,
      str(sent[1]))
check('ROC becomes an inverse ROC',
      sent[2]['op'] == 'set'
      and abs(sent[2]['attrs']['inv_ROC_HR'] - 1/3.5) < 1e-15,
      str(sent[2]))
check('a negative ROC keeps its sign',
      abs(sent[3]['attrs']['inv_ROC_AR'] - (-1/2.5)) < 1e-15, str(sent[3]))
check('diameter goes through as it is',
      sent[4]['attrs']['diameter'] == 0.2, str(sent[4]))
check('the wedge is converted to radians',
      abs(sent[5]['attrs']['wedgeAngle'] - math.radians(0.5)) < 1e-15,
      str(sent[5]))
check('reflectivity goes through as it is',
      sent[6]['attrs']['Refl_HR'] == 0.95, str(sent[6]))

print('--- flattening a curved mirror ---')
check('selecting another optics switches the panel to it',
      res['m2']['selected'] == 'M2', str(res['m2']['selected']))
check('its ROC is shown as a radius',
      abs(float(res['m2']['rocShown']) - res['m2']['wantRoc']) < 1e-12,
      '%s vs %s' % (res['m2']['rocShown'], res['m2']['wantRoc']))
check("'inf' becomes a zero inverse ROC",
      res['sentAfterFlatten'] == 8 and sent[7]['target'] == 'M2'
      and sent[7]['attrs']['inv_ROC_HR'] == 0, str(sent[7]))

print('--- bad and no-op input ---')
check('rubbish is not sent', res['afterBadInput']['sent'] == 7,
      str(res['afterBadInput']['sent']))
check('and the field snaps back to the model value',
      abs(float(res['afterBadInput']['shown']) - m1['diameter']) < 1e-15,
      res['afterBadInput']['shown'])
check('setting a field to its current value sends nothing',
      res['afterNoop']['sent'] == 7, str(res['afterNoop']['sent']))

print('--- a field that may be unset ---')
nul = res['nullable']
check('setting a number and clearing it are two messages',
      nul['sent'] == 2, str(nul['sent']))
check('the number goes through',
      nul['msgs'][0]['attrs']['max_stray_order'] == 2, str(nul['msgs'][0]))
check('the echoed value is displayed', res['nullableShown'] == '2',
      str(res['nullableShown']))
check("'auto' clears it back to null",
      nul['msgs'][1]['attrs']['max_stray_order'] is None, str(nul['msgs'][1]))
check('rubbish in a nullable field is still refused',
      nul['shown'] == '2', nul['shown'])

print('--- a scene pushed back from Python ---')
check('the selection survives', res['afterPush']['selected'] == 'M1',
      str(res['afterPush']['selected']))
check('the panel stays on the properties',
      res['afterPush']['panel']['propsShown'])
check('and shows the value Python came back with',
      abs(float(res['afterPush']['cx']) - 0.77) < 1e-15,
      res['afterPush']['cx'])

print('--- the layout file panel ---')
check('the buttons are there',
      res['saveButton'] and res['loadButton'] and res['dxfButton'])
check('the file name starts on the default',
      res['pathShown'] == 'layout.json', str(res['pathShown']))
sv = res['save']
check('Save sends one message', sv['sent'] == 1, str(sv['sent']))
check("it is a 'save' naming the file typed in",
      sv['msg']['op'] == 'save' and sv['msg']['path'] == 'chosen.json',
      str(sv['msg']))
ld = res['load']
check('Load sends one message', ld['sent'] == 1, str(ld['sent']))
check("it is a 'load' with the same file",
      ld['msg']['op'] == 'load' and ld['msg']['path'] == 'chosen.json',
      str(ld['msg']))
check('the selection is dropped before the answer arrives',
      ld['selectedAfterClick'] is None, str(ld['selectedAfterClick']))
check('and the next scene is refitted, not left where the old one was',
      ld['fitPending'] and not ld['fitPendingAfter'] and ld['refitted'],
      str(ld))
check('a blank file name sends nothing', res['blankPath'] == 0,
      str(res['blankPath']))

print('--- the drawing panel ---')
# The layout is the model, saved and loaded as the same system. The DXF
# is a drawing of it, going out to something that will never send it
# back. They get separate panels and separate names so that Load is
# never pressed on a drawing.
check('the two panels are named for what they hold',
      'Optical layout (JSON)' in res['panelTitles']
      and 'Drawing (DXF)' in res['panelTitles'], str(res['panelTitles']))
check('the drawing has a file name of its own',
      res['dxfStart'] == 'layout.dxf', str(res['dxfStart']))
dx = res['dxf']
check('Export sends one message', dx['sent'] == 1, str(dx['sent']))
check("it is an 'export' of a dxf",
      dx['msg']['op'] == 'export' and dx['msg']['format'] == 'dxf',
      str(dx['msg']))
check('naming the file typed in its own panel',
      dx['msg']['path'] == 'drawing.dxf', str(dx['msg']['path']))
check('which the layout file name does not touch',
      res['dxfIndependent'] == 'drawing.dxf', str(res['dxfIndependent']))
check('an extension the user typed is left alone, one they omitted is added',
      res['dxfNames'] == ['plan.dxf', 'plan.dxf', 'plan.DXF', 'a.b/c.dxf',
                          'dir/.dxf'],
      str(res['dxfNames']))

print('--- the beam width controls ---')
d = res['display']
check('the panel is there', d['present'])
check('the envelope widths offered', d['sigmaOptions'] == ['1', '2.7', '3'],
      str(d['sigmaOptions']))
check('the directions offered', d['modeOptions'] == ['x', 'y', 'avg'],
      str(d['modeOptions']))
check('the scene states what is in force',
      d['sceneSays'].get('width_mode') == scene['display']['width_mode']
      and d['sceneSays'].get('sigma_main') == scene['display']['sigma_main'],
      str(d['sceneSays']))
check('the controls start on it',
      d['sigmaShown'] == str(scene['display']['sigma_main'])
      and d['modeShown'] == scene['display']['width_mode'],
      '%s / %s' % (d['sigmaShown'], d['modeShown']))
check('each change is one message', d['sent'] == 2, str(d['sent']))
check('the envelope width goes to both beam kinds',
      d['msgs'][0]['op'] == 'draw'
      and d['msgs'][0]['params'] == {'sigma_main': 1, 'sigma_stray': 1},
      str(d['msgs'][0]))
check('the direction goes through as it is',
      d['msgs'][1]['params'] == {'width_mode': 'y'}, str(d['msgs'][1]))
check('a redrawn scene moves the controls',
      d['afterPush']['sigma'] == '3' and d['afterPush']['mode'] == 'avg',
      str(d['afterPush']))

print('--- the tracing group ---')
check('the panel has a Tracing heading',
      res['groupHeadings'] == ['Tracing'], str(res['groupHeadings']))
check('the flags show as checkboxes',
      f['HRtransmissive'] is False and f['term_on_HR'] is False,
      '%r / %r' % (f['HRtransmissive'], f['term_on_HR']))
check('term_on_HR_order shows its value', f['term_on_HR_order'] == '0',
      str(f['term_on_HR_order']))
check('a plain Mirror hides the curve direction row',
      not res['curveRowShown'])

flags = res['flags']
check('each real change sends one message, a redundant one none',
      flags['sent'] == 3, str(flags['sent']))
check('an echoed flag ticks the box', res['flagEchoed'] is True,
      str(res['flagEchoed']))
check('HR transmissive goes through as a boolean',
      flags['msgs'][0]['attrs']['HRtransmissive'] is True,
      str(flags['msgs'][0]))
check('terminate on HR too',
      flags['msgs'][1]['attrs']['term_on_HR'] is True, str(flags['msgs'][1]))
check('and the order as a number',
      flags['msgs'][2]['attrs']['term_on_HR_order'] == 2,
      str(flags['msgs'][2]))

print('--- a row only some classes have ---')
cur = res['curve']
check('a CyMirror shows the curve direction', cur['shown'])
check('as a choice, not a text box', cur['kind'] == 'choice',
      str(cur['kind']))
check('offering exactly the values gtrace allows',
      cur['options'] == ['h', 'v'], str(cur['options']))
check('with its value selected', cur['value'] == 'v', str(cur['value']))
check('changing it sends a set',
      cur['msg']['op'] == 'set'
      and cur['msg']['attrs']['curve_direction'] == 'h', str(cur['msg']))
check('the echoed value is selected', cur['shownAfterEcho'] == 'h',
      str(cur['shownAfterEcho']))
check('picking the value it already has sends nothing',
      cur['sentAfterSame'] == 0, str(cur['sentAfterSame']))

print('--- the focal length of a lens ---')
ln = res['lens']
check('the panel knows it is a lens', ln['type'] == 'Lens', str(ln['type']))
check('a lens shows a focal length row', ln['rowShown'])
check('but a mirror does not', not ln['rowOnMirror'])
check('the focal length shown is the one the scene carries',
      abs(float(ln['shown']) - ln['wantF']) < 1e-6,
      '%s vs %.9f' % (ln['shown'], ln['wantF']))
# A lens is listed in a catalogue in millimetres and spoken of that
# way, unlike everything else on a bench of this size.
check('the row is labelled in millimetres', ln['label'] == 'Focal length [mm]',
      str(ln['label']))
check('and shows the focal length asked for, in those units',
      abs(float(ln['shown']) - 300.0) < 1e-6, ln['shown'])
check('editing it sends one message', ln['sent'] == 1, str(ln['sent']))
check('which is a set of f itself, not of a curvature',
      ln['msg']['op'] == 'set' and set(ln['msg']['attrs']) == {'f'},
      str(ln['msg']))
check('carrying metres, whatever the panel shows',
      abs(ln['msg']['attrs']['f'] - 0.2) < 1e-15,
      '200 mm -> %r m' % ln['msg']['attrs']['f'])

# JSON has no infinity, so an infinite focal length would arrive as a
# null. A lens with no power is a flat window: a different element,
# not somewhere to get to by typing.
check('an infinite focal length is not sent',
      ln['afterInf']['sent'] == 1, str(ln['afterInf']['sent']))
check('and the field goes back to what the model holds',
      abs(float(ln['afterInf']['shown']) - ln['wantF']) < 1e-6,
      str(ln['afterInf']['shown']))

check('the anchor row is shown for a lens', ln['anchorRowShown'])
check('and for a mirror too, since both have one', ln['anchorRowOnMirror'])
check('as a choice, not a text box', ln['anchorKind'] == 'choice',
      str(ln['anchorKind']))
check('offering exactly the values gtrace allows',
      ln['anchorOptions'] == ['HRcenter', 'center'],
      str(ln['anchorOptions']))
check('a lens is anchored at the middle of its substrate',
      ln['anchorShown'] == 'center', str(ln['anchorShown']))
check('a mirror at the apex of its HR face',
      ln['anchorOnMirror'] == 'HRcenter', str(ln['anchorOnMirror']))
check('changing it sends a set', ln['anchorSent'] == 1
      and ln['anchorMsg']['op'] == 'set'
      and ln['anchorMsg']['attrs']['anchor_point'] == 'HRcenter',
      str(ln['anchorMsg']))

print('--- moving along a beam by a typed distance ---')
sl = res['slide']
check('an optics with a beam through it offers the pair of rows',
      sl['rowShown'] and sl['byRowShown'],
      '%s / %s' % (sl['rowShown'], sl['byRowShown']))
check('and neither is offered when no beam passes through it',
      sl['rowsWhenNoBeam'] == [False, False], str(sl['rowsWhenNoBeam']))
check('the beam is picked from a list, not typed', sl['kind'] == 'choice',
      str(sl['kind']))
check('offering the beams that actually reach the optics',
      len(sl['options']) > 0 and sl['nearest'] in sl['options'],
      '%s (nearest %s)' % (sl['options'], sl['nearest']))
check('with the nearest one chosen to start from',
      sl['chosen'] and sl['chosen']['name'] == sl['nearest'],
      str(sl['chosen']))
# The one row on the panel in millimetres besides the focal length: an
# adjustment on a bench is spoken of in mm, and 0.05 invites the slip.
check('the distance row is labelled in millimetres',
      sl['label'] == 'Move by [mm]', str(sl['label']))
check('and starts at zero, since it is a distance to move, not a place',
      sl['byShown'] == '0', str(sl['byShown']))
check('typing one sends a single message', sl['sent'] == 1, str(sl['sent']))
check("it is a 'slide' naming the optics and the beam",
      sl['msg']['op'] == 'slide' and sl['msg']['target'] == 'M1'
      and sl['msg']['beam'] == sl['chosen']['name']
      and sl['msg']['beam_index'] == sl['chosen']['index'],
      str(sl['msg']))
check('carrying metres, whatever the panel shows',
      abs(sl['msg']['distance'] - 0.05) < 1e-15,
      '50 mm -> %r m' % sl['msg']['distance'])
check('the field goes back to zero at once',
      sl['byAfter'] == '0', str(sl['byAfter']))
check('zero and rubbish send nothing and leave it at zero',
      sl['afterQuiet']['sent'] == 1 and sl['afterQuiet']['shown'] == '0',
      str(sl['afterQuiet']))

# A name like 'b0:M1t1' says nothing about which line in the picture it
# is, so the beam is chosen by clicking it. Ctrl, because that already
# means "this optics, against that beam" in a drag.
check('there is a second beam through the optics to pick',
      sl['otherBeam'] is not None and sl['otherBeam'] != sl['chosen']['name'],
      str(sl['otherBeam']))
check('and a point on it clear of every optics to click',
      sl['pickPoint'] is not None, str(sl['pickPoint']))
bc = sl['byClick']
check('Ctrl + clicking a beam changes the row to one under the cursor',
      bc['chosen']['name'] != sl['chosen']['name']
      and bc['chosen']['name'] in bc['under'],
      '%s (was %s, under the cursor: %s)'
      % (bc['chosen']['name'], sl['chosen']['name'], bc['under']))
# Only ever a beam of the element's own: sliding along one that misses
# it is not something to mean, and such a click stays a readout.
check('and never one that misses the optics',
      bc['chosen']['name'] in bc['through'],
      '%s not in %s' % (bc['chosen']['name'], bc['through']))
check('and the picker shows it',
      bc['shownInPicker'] == str(bc['chosen']['index']),
      '%s vs %s' % (bc['shownInPicker'], bc['chosen']['index']))
# Choosing what to move along is part of working on that element, so it
# must not throw the element away the way a plain click does.
check('the optics stays selected', bc['stillSelected'] == 'M1',
      str(bc['stillSelected']))
check('and its properties stay on screen', bc['panel']['propsShown'],
      str(bc['panel']))
check('the chosen beam is marked in the drawing', bc['markShown'])
check('nothing is sent: the choice lives in the viewer', bc['sent'] == 0,
      str(bc['sent']))

# Beams routinely lie on top of one another, so pointing cannot tell
# them apart. Clicking again steps through the bundle, as it does for
# the readout.
cy = sl['cycle']
n = len(cy['bundle'])
check('there is a bundle of beams at that point to step through',
      n >= 2, str(cy['bundle']))
# The order within a bundle of near-coincident beams is settled by
# distances that differ in the last digits, and a click coordinate
# makes a round trip through the screen to get here, so which of them
# comes first is not something to pin down. That every one of them is
# reachable, once each, is.
check('each Ctrl + click takes a different beam of it',
      len(set(cy['picks'][:n])) == n, str(cy['picks'][:n]))
check('between them they cover the whole bundle',
      set(cy['picks'][:n]) == set(cy['bundle']),
      '%s vs %s' % (sorted(set(cy['picks'][:n])), sorted(cy['bundle'])))
check('and it comes back round to where it started',
      cy['picks'][n] == cy['picks'][0],
      '%s after %d' % (cy['picks'], n))

# Stepping through a bundle has to be visible, and moving a faint mark
# from one line onto the same line is not visible at all: what tells
# two beams on one axis apart is which way they run, and that is also
# the sign of Move by.
dir_of = dict(zip(cy['bundle'], [tuple(d) for d in cy['dirs']]))
check('the bundle holds beams running opposite ways',
      len(set(dir_of.values())) > 1, str(sorted(set(dir_of.values()))))
check('the arrow is drawn for the chosen beam throughout',
      all(a for a in cy['arrows']), str(cy['arrows'][:2]))
# Between two picks that run different ways the arrow must differ; the
# mark alone need not, since they lie on the same line.
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
         if dir_of[cy['picks'][i]] != dir_of[cy['picks'][j]]]
check('there are two picks running different ways to compare',
      len(pairs) > 0, str(len(pairs)))
check('and the arrow turns round between them',
      all(cy['arrows'][i] != cy['arrows'][j] for i, j in pairs),
      str([(cy['picks'][i], cy['picks'][j]) for i, j in pairs][:3]))
check('and the mark spans the chosen beam every time',
      all(cy['marks']), str(cy['marks']))
# The cycle belongs to the place that was clicked, not to the viewer:
# clicking elsewhere and coming back starts at the nearest again, which
# is what the first click here gave.
check('a Ctrl + click somewhere else starts the cycle again',
      cy['afterElsewhere'] == bc['chosen']['index'],
      '%s (the first click here gave %s)'
      % (cy['afterElsewhere'], bc['chosen']['index']))

pc = sl['plainClick']
check('the same click without Ctrl is still the readout',
      pc['panel']['beamShown'] and pc['selected'] is None,
      str(pc['panel']))
check('pinning one of the beams it was over',
      pc['pinned'] in bc['under'],
      '%s (under the cursor: %s)' % (pc['pinned'], bc['under']))
check('and the mark goes with the panel', not pc['markShown'])

print('--- renaming ---')
ren = res['rename']
check('one message', ren['sent'] == 1, str(ren['sent']))
check("it is a 'rename'", ren['msg']['op'] == 'rename', str(ren['msg']))
check('naming the old and the new', ren['msg']['target'] == 'M1'
      and ren['msg']['name'] == 'PRM', str(ren['msg']))
check('the viewer follows the new name straight away',
      ren['selected'] == 'PRM', str(ren['selected']))
check('while remembering the old one', ren['fallback'] == 'M1',
      str(ren['fallback']))

refused = res['renameRefused']
check('a refused rename puts the selection back',
      refused['selected'] == 'M1', str(refused['selected']))
check('and the field shows the name that is real again',
      refused['shown'] == 'M1', str(refused['shown']))
check('the properties stay on screen', refused['panel']['propsShown'])

ok = res['renameAccepted']
check('an accepted rename sticks', ok['selected'] == 'PRM',
      str(ok['selected']))
check('the field shows it', ok['shown'] == 'PRM', str(ok['shown']))
check('and nothing is left to revert to', ok['fallback'] is None,
      str(ok['fallback']))
check('the panel still shows the optics', ok['panel']['propsShown'])

check('a blank or unchanged name is not sent',
      res['renameQuiet']['sent'] == 0, str(res['renameQuiet']['sent']))
check('and the field snaps back',
      res['renameQuiet']['shown'] == 'M1', str(res['renameQuiet']['shown']))

print('--- adding a mirror ---')
check('the add button is there', res['addButton'])
check('the remove button is there', res['removeButton'])
add = res['add']
check('one message per click', add['sent'] == 1, str(add['sent']))
check("it is an 'add'", add['msg']['op'] == 'add', str(add['msg'].get('op')))
check('of a Mirror', add['msg']['type'] == 'Mirror', str(add['msg'].get('type')))
check('with a name the layout does not use yet',
      add['msg']['name'] not in [o['name'] for o in scene['optics']],
      str(add['msg'].get('name')))
check('placed at the centre of the view',
      abs(add['msg']['params']['HRcenter'][0] - add['viewCentre'][0]) < 1e-12
      and abs(add['msg']['params']['HRcenter'][1] - add['viewCentre'][1]) < 1e-12,
      str(add['msg']['params']['HRcenter']))
check('the new optics is selected right away',
      add['selected'] == add['msg']['name'], str(add['selected']))
check('and its properties show once the scene comes back',
      res['afterAdd']['panel']['propsShown']
      and res['afterAdd']['name'] == add['msg']['name'],
      str(res['afterAdd']))
check('a second mirror gets a different name',
      res['secondName'] != add['msg']['name'],
      '%s vs %s' % (add['msg']['name'], res['secondName']))

print('--- adding a cylindrical mirror ---')
check('it has its own button', res['addCyButton'])
cy = res['addCy']
check('one message per click', cy['sent'] == 1, str(cy['sent']))
check('of the cylindrical type', cy['msg']['type'] == 'CyMirror',
      str(cy['msg'].get('type')))
check('with a curve direction to start from',
      cy['msg']['params'].get('curve_direction') == 'h',
      str(cy['msg']['params']))
check('named apart from the plain mirrors',
      cy['msg']['name'].startswith('CY')
      and cy['msg']['name'] != add['msg']['name'],
      str(cy['msg']['name']))
check('placed at the centre of the view like the others',
      cy['msg']['params']['HRcenter'] == add['msg']['params']['HRcenter'],
      str(cy['msg']['params']['HRcenter']))
check('and selected right away', cy['selected'] == cy['msg']['name'],
      str(cy['selected']))

print('--- adding a lens ---')
check('it has its own button', res['addLensButton'])
ln2 = res['addLens']
check('one message per click', ln2['sent'] == 1, str(ln2['sent']))
check('of the lens type', ln2['msg']['type'] == 'Lens',
      str(ln2['msg'].get('type')))
# The view has nothing to say about what kind of lens: Python owns the
# catalogue defaults, and there is one place they live.
check('carrying only where to put it',
      set(ln2['msg']['params']) == {'HRcenter', 'normAngleHR'},
      str(sorted(ln2['msg']['params'])))
check('named apart from the mirrors',
      ln2['msg']['name'].startswith('L')
      and ln2['msg']['name'] not in (add['msg']['name'], cy['msg']['name']),
      str(ln2['msg']['name']))
check('placed at the centre of the view like the others',
      ln2['msg']['params']['HRcenter'] == add['msg']['params']['HRcenter'],
      str(ln2['msg']['params']['HRcenter']))
check('and selected right away', ln2['selected'] == ln2['msg']['name'],
      str(ln2['selected']))

print('--- adding a cylindrical lens ---')
check('it has its own button', res['addCyLensButton'])
cl = res['addCyLens']
check('one message per click', cl['sent'] == 1, str(cl['sent']))
check('of the cylindrical lens type', cl['msg']['type'] == 'CyLens',
      str(cl['msg'].get('type')))
# Catalogue defaults like a Lens, plus the direction like a CyMirror.
check('carrying where to put it and which way it curves',
      set(cl['msg']['params']) == {'HRcenter', 'normAngleHR',
                                   'curve_direction'},
      str(sorted(cl['msg']['params'])))
check('with a curve direction to start from',
      cl['msg']['params'].get('curve_direction') == 'h',
      str(cl['msg']['params']))
check('named apart from the lenses and the mirrors',
      cl['msg']['name'].startswith('CL')
      and cl['msg']['name'] not in (add['msg']['name'], cy['msg']['name'],
                                    ln2['msg']['name']),
      str(cl['msg']['name']))
check('placed at the centre of the view like the others',
      cl['msg']['params']['HRcenter'] == add['msg']['params']['HRcenter'],
      str(cl['msg']['params']['HRcenter']))
check('and selected right away', cl['selected'] == cl['msg']['name'],
      str(cl['selected']))

print('--- undo ---')
ub = res['undoButton']
check('there is an Undo button', ub is not None)
check('out of reach until there is something to go back to',
      ub['disabled'] and ub['sceneSays'] is False, str(ub))
un = res['undo']
check('and pressing it then sends nothing',
      un['sentWhileEmpty'] == 0, str(un['sentWhileEmpty']))
check('a scene with a history behind it enables it', un['enabled'])
check('pressing it sends one undo',
      un['sent'] == 1 and un['msg'] == {'op': 'undo'}, str(un['msg']))
# The layout is Python's, so the button asks rather than guesses: an
# undo carries nothing but the word.
check('carrying nothing else', list(un['msg'].keys()) == ['op'],
      str(list(un['msg'].keys())))
check('Ctrl+Z over the viewer does the same', un['byKey'] == 2,
      str(un['byKey']))
# A notebook page has its own undo, and taking the key from everywhere
# would undo edits meant for that.
check('but not with the pointer elsewhere', un['byKeyOutside'] == 2,
      str(un['byKeyOutside']))
check('a scene with nothing behind it puts it out of reach again',
      un['disabledAgain'])

print('--- redo ---')
rb = res['redoButton']
check('there is a Redo button', rb is not None)
check('out of reach until there is something to come back to',
      rb['disabled'] and rb['sceneSays'] is False, str(rb))
rd = res['redo']
check('and pressing it then sends nothing',
      rd['sentWhileEmpty'] == 0, str(rd['sentWhileEmpty']))
check('a scene with an undo behind it enables it', rd['enabled'])
# The two flags are independent: one is filled by editing, the other by
# undoing, so a scene can offer either without the other.
check('without enabling Undo along with it', rd['undoStillDisabled'])
check('pressing it sends one redo',
      rd['sent'] == 1 and rd['msg'] == {'op': 'redo'}, str(rd['msg']))
check('carrying nothing else', list(rd['msg'].keys()) == ['op'],
      str(list(rd['msg'].keys())))
check('Ctrl+Shift+Z over the viewer does the same',
      rd['afterShiftKey'] == 2 and rd['byShiftKey'] == {'op': 'redo'},
      str(rd['byShiftKey']))
check('and so does Ctrl+Y',
      rd['afterCtrlY'] == 3 and rd['byCtrlY'] == {'op': 'redo'},
      str(rd['byCtrlY']))
check('but not with the pointer elsewhere', rd['byKeyOutside'] == 3,
      str(rd['byKeyOutside']))
check('a scene with nothing ahead of it puts it out of reach again',
      rd['disabledAgain'])

print('--- removing an optics ---')
rem = res['remove']
check("it is a 'remove'", rem['msg']['op'] == 'remove', str(rem['msg']))
check('naming the selected optics', rem['msg']['target'] == add['msg']['name'],
      str(rem['msg'].get('target')))
check('the selection is dropped', rem['selected'] is None,
      str(rem['selected']))
check('and the panel goes back to the beam readout',
      rem['panel']['beamShown'], str(rem['panel']))

print('--- Python accepts what the buttons sent ---')
lay2, _ = make_layout()
lay2.apply_edit(res['add']['msg'])
check('the add message is accepted',
      res['add']['msg']['name'] in [o.name for o in lay2.optics],
      str([o.name for o in lay2.optics]))
lay2.apply_edit(res['addCy']['msg'])
made = lay2.get_optics(res['addCy']['msg']['name'])
check('so is the cylindrical one',
      type(made).__name__ == 'CyMirror'
      and made.curve_direction == 'h',
      '%s / %s' % (type(made).__name__, made.curve_direction))
check('and it appears in the scene as a CyMirror',
      [o for o in lay2.scene_dict()['optics']
       if o['name'] == made.name][0]['type'] == 'CyMirror')

lay2.apply_edit(res['addLens']['msg'])
lens2 = lay2.get_optics(res['addLens']['msg']['name'])
check('the lens button builds a lens',
      type(lens2).__name__ == 'Lens', type(lens2).__name__)
check('with a focal length Python chose',
      np.isfinite(float(lens2.f)) and float(lens2.f) > 0,
      'f=%.4f' % float(lens2.f))
# The two messages the panel sent for a lens have to be ones Python
# takes: this is the only place both halves meet on the same values.
lay2.apply_edit({'op': 'set', 'target': lens2.name,
                 'attrs': dict(res['lens']['msg']['attrs'])})
check('and the focal length the panel sent is applied',
      abs(float(lens2.f) - res['lens']['msg']['attrs']['f']) < 1e-9,
      'f=%.9f' % float(lens2.f))
lay2.apply_edit({'op': 'set', 'target': lens2.name,
                 'attrs': dict(res['lens']['anchorMsg']['attrs'])})
check('so is the anchor', lens2.anchor_point == 'HRcenter',
      str(lens2.anchor_point))
check('and it appears in the scene as a Lens carrying its power',
      [o for o in lay2.scene_dict()['optics']
       if o['name'] == lens2.name][0]['type'] == 'Lens'
      and 'inv_f' in [o for o in lay2.scene_dict()['optics']
                      if o['name'] == lens2.name][0])

# The slide, on its own: it moves M1, which the replay above is written
# around.
lay2.trace()
m1s = lay2.get_optics('M1')
slide_msg = res['slide']['msg']
slide_along = [b for b in lay2.beams if b.name == slide_msg['beam']][0]
bdir = np.asarray(slide_along.dirVect, dtype='float64')
before = np.asarray(m1s.center, dtype='float64').copy()
angle_before = float(m1s.normAngleHR)
lay2.apply_edit(slide_msg)
step = np.asarray(m1s.center, dtype='float64') - before
check('Python moves it 50 mm along the beam the panel named',
      np.allclose(step, bdir*0.05, atol=1e-15),
      '%s (want %s)' % (list(step.round(15)), list((bdir*0.05).round(15))))
check('without turning it', float(m1s.normAngleHR) == angle_before)
new = lay2.get_optics(res['add']['msg']['name'])
check('the mirror is where the view was',
      np.allclose(np.asarray(new.HRcenter),
                  res['add']['msg']['params']['HRcenter'], atol=1e-12),
      str(list(np.asarray(new.HRcenter))))
check('it inherited the size of the others',
      abs(float(new.diameter) - 10*cm) < 1e-15, str(float(new.diameter)))
check('the layout traces with it', len(lay2.trace()) > 0,
      '(%d beams)' % len(lay2.beams))
lay2.apply_edit(res['remove']['msg'])
check('the remove message is accepted',
      res['add']['msg']['name'] not in [o.name for o in lay2.optics],
      str([o.name for o in lay2.optics]))

print('--- back to the beam readout ---')
check('clicking a beam switches the panel',
      res['afterBeamClick']['panel']['title'] == 'Beam readout'
      and res['afterBeamClick']['panel']['beamShown'],
      str(res['afterBeamClick']['panel']))
check('the optics is deselected',
      res['afterBeamClick']['selected'] is None,
      str(res['afterBeamClick']['selected']))
check('and a beam got pinned',
      res['afterBeamClick']['beam'] is not None,
      str(res['afterBeamClick']['beam']))

print('--- repeated clicks cycle through the element and its beams ---')
cyc = res['cycleClicks']
check('a beam ending on an element exists to try', cyc is not None)
if cyc is not None:
    check('the first click selects the element',
          cyc['first']['panel']['propsShown']
          and cyc['first']['selected'] == cyc['optic'], str(cyc['first']))
    check('the next click reaches a beam under it',
          cyc['second']['panel']['beamShown']
          and cyc['second']['selected'] is None
          and cyc['second']['beam'] is not None, str(cyc['second']))
    check('and the cycle comes back around to the element',
          cyc['wrapped']['panel']['propsShown']
          and cyc['wrapped']['selected'] == cyc['optic']
          and not cyc['wrapped']['pinned'],
          '%s under=%s' % (cyc['wrapped'], cyc['under']))

print('--- Python applies what the panel sent ---')
lay, optics = make_layout()
o = optics[0]

# The move first, on its own: rotating turns the substrate about the HR
# surface, and changing the diameter or the wedge changes the sagitta,
# so both shift the substrate centre afterwards. That is the model being
# self-consistent, not the edit being lost.
lay.apply_edit(sent[0])
check('center x lands exactly where it was typed',
      abs(float(np.asarray(o.center)[0]) - 0.55) < 1e-12,
      str(float(np.asarray(o.center)[0])))

hr_before = np.asarray(o.HRcenter).copy()
center_before = np.asarray(o.center).copy()
# Five kinds of message are left out. The renames deliberately include
# one Python refuses, so the sequence is not a coherent script; 'Cy' and
# 'L1' exist only in the synthetic scene the browser was handed, so this
# layout has never heard of them; save/load would write files into the
# repository, which a check has no business doing; the slide moves M1
# off the place the checks below expect it, so it is applied on its own
# further down instead; and the redos outnumber the undos, because the
# browser was driven with hand-made scenes saying there was more to come
# back to than this layout ever stepped out of. Redo is applied on its
# own at the end of this section.
SYNTHETIC = ('Cy', 'L1')
for msg in sent[1:]:
    # 'export' joins save/load for the same reason and one more: the
    # paths in those messages are probes of how the extension is
    # swapped, not places to write - 'a/b/layout.dxf' names no
    # directory, and the ones that do would land in the repository.
    if (msg['op'] not in ('rename', 'save', 'load', 'slide', 'redo',
                          'export')
            and msg.get('target') not in SYNTHETIC):
        lay.apply_edit(msg)
check('angle', abs(float(o.normAngleHR) - math.radians(120)) < 1e-12,
      str(float(o.normAngleHR)))
check('the HR surface stayed put while the substrate turned',
      np.allclose(np.asarray(o.HRcenter), hr_before, atol=1e-15)
      and not np.allclose(np.asarray(o.center), center_before, atol=1e-9),
      'HR %s, centre %s -> %s'
      % (list(hr_before.round(6)), list(center_before.round(6)),
         list(np.asarray(o.center).round(6))))
check('ROC HR', abs(float(o.inv_ROC_HR) - 1/3.5) < 1e-15,
      str(float(o.inv_ROC_HR)))
check('ROC AR keeps its sign', abs(float(o.inv_ROC_AR) + 1/2.5) < 1e-15,
      str(float(o.inv_ROC_AR)))
check('M2 was flattened', float(optics[1].inv_ROC_HR) == 0.0,
      str(float(optics[1].inv_ROC_HR)))
check('diameter', abs(float(o.diameter) - 0.2) < 1e-15, str(float(o.diameter)))
check('wedge', abs(float(o.wedgeAngle) - math.radians(0.5)) < 1e-15,
      str(float(o.wedgeAngle)))
check('reflectivity', abs(float(o.Refl_HR) - 0.95) < 1e-15,
      str(float(o.Refl_HR)))

# The tracing flags the checkboxes sent, on whichever optics they were
# aimed at.
flag_msgs = [m for m in sent if m['op'] == 'set'
             and set(m['attrs']) & {'HRtransmissive', 'term_on_HR',
                                    'term_on_HR_order'}]
target = lay.get_optics(flag_msgs[0]['target'])
check('the tracing flags were applied',
      bool(target.HRtransmissive) and bool(target.term_on_HR)
      and target.term_on_HR_order == 2,
      '%s / %s / %s' % (target.HRtransmissive, target.term_on_HR,
                        target.term_on_HR_order))

# 'Cy' has to be created here first; the browser only ever saw it in the
# scene it was handed.
lay.apply_edit({'op': 'add', 'name': 'Cy', 'type': 'CyMirror',
                'params': {'HRcenter': [0.2, 0.6], 'inv_ROC_HR': 0.5,
                           'curve_direction': 'v'}})
cy_msgs = [m for m in sent if m.get('target') == 'Cy']
for m in cy_msgs:
    lay.apply_edit(m)
check('the curve direction message is accepted',
      lay.get_optics('Cy').curve_direction == 'h',
      str(lay.get_optics('Cy').curve_direction))
lay.apply_edit({'op': 'remove', 'target': 'Cy'})
check('the layout still traces', len(lay.trace()) > 0,
      '(%d beams)' % len(lay.beams))

# What the panel would show after the round trip must match what was typed.
back = lay.scene_dict()['optics'][0]
check('the ROC reads back as it was typed',
      abs(1/back['inv_ROC_HR'] - 3.5) < 1e-12, str(1/back['inv_ROC_HR']))
check('the angle reads back in degrees as it was typed',
      abs(math.degrees(back['normAngleHR']) - 120) < 1e-12,
      str(math.degrees(back['normAngleHR'])))

# The redo the browser sent, on a step this layout really did take: an
# undo of its own to step out of, then the message as the button spelt
# it. Undo and redo together must leave the layout where it was, or the
# read-back above would no longer describe it.
redo_msg = [m for m in sent if m['op'] == 'redo'][0]
lay.apply_edit({'op': 'set', 'target': o.name, 'attrs': {'diameter': 0.3}})
lay.apply_edit({'op': 'undo'})
lay.apply_edit(redo_msg)
check('the redo the panel sent is accepted', abs(float(o.diameter) - 0.3) < 1e-15,
      str(float(o.diameter)))

# The export the page sent, on a path of this suite's choosing rather
# than the name-mangling probes above.
export_msg = [m for m in sent if m['op'] == 'export'][0]
export_msg = dict(export_msg, path=os.path.join(SP, 'props_export.dxf'))
check('the export the panel sent is accepted',
      lay.apply_edit(export_msg) is lay
      and os.path.getsize(export_msg['path']) > 1000,
      '(%d bytes)' % os.path.getsize(export_msg['path']))
lay.apply_edit({'op': 'undo'})
check('and a round trip leaves the layout as it was',
      abs(float(o.diameter) - 0.2) < 1e-15
      and lay.scene_dict()['optics'][0] == back, str(float(o.diameter)))

print('--- read-only viewer ---')
errs, res = run(False)
check('no console error', errs == [], '\n        '.join(errs[:3]))
check('ran without exception', res and res['error'] is None,
      str(res and res['error'])[:300])
check('an optics can still be selected', res['selected'] == 'M1',
      str(res['selected']))
check('its properties are shown',
      res['afterClick']['title'] == 'Optics properties', str(res['afterClick']))
check('but nothing is editable', res['editableFields'] == [],
      str(res['editableFields']))
check('there are no add buttons',
      not res['addButton'] and not res['addCyButton']
      and not res['addLensButton'] and not res['addCyLensButton'])
# Nothing to add and nothing to undo, so the head is down to one row.
# Measure stays: it needs no Python, and a written page you can measure
# on is most of the reason to have one.
check('and the head is down to a single row',
      res['headRows'] == [['Measure', 'Fit']], str(res['headRows']))
check('there is no remove button on the optics panel',
      not res['removeButton'])
check('and no beam width controls, since redrawing needs Python',
      not res['display']['present'])
check('and no layout file panel, since writing files needs Python',
      not res['saveButton'] and not res['loadButton']
      and res['pathShown'] is None)
check('the values are still readable',
      res['fields']['cx'] not in ('', '-')
      and abs(float(res['fields']['cx']) - res['m1']['center'][0]) < 1e-15,
      str(res['fields']['cx']))
check('and nothing is sent', res['sent'] == [], str(res['sent']))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
