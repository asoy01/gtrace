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
            send: function (m) { sent.push(m); }
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
        out.addButton = !!button('+ Mirror');
        out.addCyButton = !!button('+ CyMirror');
        out.removeButton = !!button('Remove');
        out.saveButton = !!button('Save');
        out.loadButton = !!button('Load');
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
        out.groupHeadings = Array.prototype.map.call(
            el.querySelectorAll('.gt-props .gt-group'),
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

            // Put the first one back in the panel, then remove it.
            v._selectOptic(added);
            button('Remove').click();
            out.remove = {msg: sent[sent.length - 1],
                          selected: v.selectedOptic,
                          panel: panel()};
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

            // A blank file name asks for nothing.
            v.pathInput.value = '   ';
            var nBlank = sent.length;
            button('Save').click();
            button('Load').click();
            out.blankPath = sent.length - nBlank;
            v.pathInput.value = 'chosen.json';
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
                                     'HRtransmissive', 'term_on_HR',
                                     'term_on_HR_order', 'curve_direction'},
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
check('the buttons are there', res['saveButton'] and res['loadButton'])
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
# Three kinds of message are left out. The renames deliberately include
# one Python refuses, so the sequence is not a coherent script; 'Cy'
# exists only in the synthetic scene the browser was handed, so this
# layout has never heard of it; and save/load would write files into the
# repository, which a check has no business doing.
for msg in sent[1:]:
    if (msg['op'] not in ('rename', 'save', 'load')
            and msg.get('target') != 'Cy'):
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
      not res['addButton'] and not res['addCyButton'])
check('there is no remove button', not res['removeButton'])
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
