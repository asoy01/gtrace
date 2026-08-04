'''
Verification of the measuring tool, in a real browser.

Arms the tool, moves the cursor, clicks two points and checks the edit
message that leaves the page - then hands that message to Python's
apply_edit, so that what the browser asked for and what the model makes
of it are compared on the same numbers.

Two things here are worth stating.

The tool is a mode, and while it is on the picture must stop answering
to what is under the cursor: no optics is grabbed, no beam is pinned. A
click means "measure here" and nothing else. That is checked directly,
because the failure it guards against - dragging an element out from
under the cursor halfway through measuring it - would move the very
thing being measured.

The snapping is checked in scene coordinates rather than in pixels. The
same mistake was made once before, when a Ctrl-drag judged its snap from
the cursor position and quietly stopped working as soon as the drawing
was enlarged. The reach is stated in screen pixels on purpose - it is
about what the eye can aim at - so the check zooms in and asks again.
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
from gtrace.layout import OpticalLayout, TraceRules, EditError, Dimension
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
                    diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                    Refl_HR=0.99, Trans_HR=0.01, Refl_AR=500*ppm,
                    Trans_AR=1-500*ppm, n=1.45, name='M1')
    M2 = opt.Mirror(HRcenter=[0.5, 0.4], normAngleHR=deg2rad(-45),
                    diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                    Refl_HR=0.99, Trans_HR=0.01, Refl_AR=500*ppm,
                    Trans_AR=1-500*ppm, n=1.45, name='M2')
    L1 = opt.Lens(f=500*mm, HRcenter=[0.25, 0.0], normAngleHR=np.pi,
                  name='L1')
    lay = OpticalLayout(optics=[M1, M2, L1], sources=[b0],
                        rules=TraceRules(order=4, power_threshold=1e-3),
                        name='Measure')
    return lay, M1, M2, L1

layout, M1, M2, L1 = make_layout()
# One dimension already on the layout, so that the checks about
# selecting, editing and removing one have something to work with
# without having to place it through the page first.
#: How far aside D1's line is drawn. The span it measures is inside the
#: lens, which is the case the offset exists for: a line drawn straight
#: between those two points lies in the glass, over the beams going
#: through it, where it can be neither read nor taken hold of.
D1_OFFSET = 20*mm
layout.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                   'params': {'p1': [float(x) for x in np.asarray(L1.HRcenter)],
                              'p2': [float(x) for x in np.asarray(L1.ARcenter)],
                              'offset': D1_OFFSET}})
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
    var out = {error: null};
    function mouse(target, type, x, y, opts) {
        target.dispatchEvent(new MouseEvent(type, Object.assign({
            clientX: x, clientY: y, button: 0, bubbles: true, cancelable: true
        }, opts || {})));
    }
    try {
        var url = URL.createObjectURL(
            new Blob([ESM_SRC], {type: 'text/javascript'}));
        var mod = await import(url);

        var state = {scene: SCENE, title: 'Measure', height: 640,
                     editable: EDITABLE, error: ''};
        var handlers = {}, sent = [];
        var model = {
            get: function (k) { return state[k]; },
            set: function (k, v) {
                state[k] = v;
                (handlers['change:' + k] || []).forEach(function (f) { f(); });
            },
            on: function (e, f) { (handlers[e] = handlers[e] || []).push(f); },
            off: function () {},
            send: function (m) { sent.push(m); }
        };

        var el = document.getElementById('host');
        mod.default.render({model: model, el: el});
        var v = el.gtraceViewer;
        // Measured afresh every time: the status bar changes length as
        // the tool reports what it is doing, and the page relaying out
        // moves the drawing by a pixel or two.
        function screenOf(p) {
            var r = v.svg.getBoundingClientRect();
            var s = v.sceneToScreen(p[0], p[1]);
            return [s[0] + r.left, s[1] + r.top];
        }
        function button(text) {
            var found = null;
            Array.prototype.forEach.call(
                el.querySelectorAll('button'), function (b) {
                    if (b.textContent === text) { found = b; }
                });
            return found;
        }
        function panel() {
            return {
                title: el.querySelector('.gt-panel-title span').textContent,
                beamShown: v.readoutBody.style.display !== 'none',
                propsShown: v.opticBody.style.display !== 'none',
                dimShown: v.dimBody.style.display !== 'none'
            };
        }
        function dimFields() {
            var o = {};
            for (var k in v.dimFields) {
                var f = v.dimFields[k];
                o[k] = f.editable ? f.el.value : f.el.textContent;
            }
            return o;
        }
        function rowShown(key) {
            return v.dimFields[key].row.style.display !== 'none';
        }
        function clickAt(p) {
            mouse(window, 'mousemove', p[0], p[1]);
            mouse(v.svg, 'mousedown', p[0], p[1]);
            mouse(window, 'mouseup', p[0], p[1]);
        }

        out.measureButton = !!button('Measure');
        out.startArmed = v.measuring;

        // --- the dimension already on the layout, drawn ---
        out.drawn = {
            groups: el.querySelectorAll('.gt-dims .gt-dim').length,
            labels: Array.prototype.map.call(
                el.querySelectorAll('.gt-dims .gt-dim-label'),
                function (t) { return {text: t.textContent,
                                       hidden: t.style.display === 'none'}; }),
            ticks: el.querySelectorAll('.gt-dims .gt-dim-tick').length,
            exts: Array.prototype.map.call(
                el.querySelectorAll('.gt-dims .gt-dim-ext'),
                function (e) { return e.style.display !== 'none'; })
        };

        // --- selecting it ---
        var d1 = v.scene.dimensions[0];
        var span = [(d1.p1[0] + d1.p2[0]) / 2, (d1.p1[1] + d1.p2[1]) / 2];
        var mid = [(d1.line[0][0] + d1.line[1][0]) / 2,
                   (d1.line[0][1] + d1.line[1][1]) / 2];
        // Zoom in first: the lens is 6 mm thick and the whole bench is
        // most of a metre, so at the fitted scale the dimension is a
        // couple of pixels long and nothing could be aimed at it.
        v.scale *= 60; v.cx = mid[0]; v.cy = mid[1]; v._applyTransform();

        // Aiming at the span rather than at the line does not take the
        // dimension: the span is inside the lens, and that is the very
        // place the line was carried away from.
        clickAt(screenOf(span));
        out.atSpan = {selected: v.selectedDim, optic: v.selectedOptic,
                      panel: panel()};

        clickAt(screenOf(mid));
        out.afterDimClick = {panel: panel(), selected: v.selectedDim,
                             fields: dimFields(),
                             insideShown: rowShown('inside'),
                             opticalShown: rowShown('optical'),
                             marked: el.querySelector('.gt-dims .gt-dim')
                                 .classList.contains('gt-selected')};

        // Editing an end sends a set naming that end only.
        var nEdit = sent.length;
        v.dimFields.p2x.el.value = '0.9';
        v.dimFields.p2x.el.dispatchEvent(new Event('change', {bubbles: true}));
        out.editEnd = {msg: sent[sent.length - 1], sent: sent.length - nEdit};

        // Renaming goes through the same operation an optics uses.
        var nRen = sent.length;
        v.dimFields.name.el.value = 'thickness';
        v.dimFields.name.el.dispatchEvent(new Event('change', {bubbles: true}));
        out.rename = {msg: sent[sent.length - 1], sent: sent.length - nRen,
                      selected: v.selectedDim};
        v.selectedDim = 'D1';
        v._refreshDimPanel();

        // Remove, from the same button the optics panel has. A read-only
        // viewer has neither the button nor anything to send.
        if (button('Remove')) {
            var nRem = sent.length;
            button('Remove').click();
            out.remove = {msg: sent[sent.length - 1], sent: sent.length - nRem,
                          selected: v.selectedDim, panel: panel()};
        }

        v.fit();

        // A viewer with nowhere to send edits measures all the same,
        // keeping the result to itself. That path is its own, since
        // nothing leaves the page, so it is checked on its own and the
        // rest below - which is all about what Python is told - is
        // skipped.
        if (!EDITABLE) {
            // The layout's own dimension is not the reader's to remove:
            // a read-only viewer must not appear to change what it was
            // handed.
            v._selectDim(v.scene.dimensions[0]);
            out.removeOfferedForTheirs =
                v.dimFoot.style.display !== 'none';

            var nLocal = sent.length;
            var nDim = v.scene.dimensions.length;
            button('Measure').click();
            out.localArmed = v.measuring;

            var hrL = v.scene.optics.filter(function (o) {
                return o.name === 'M1';
            })[0].HRcenter;
            var m2L = v.scene.optics.filter(function (o) {
                return o.name === 'M2';
            })[0].HRcenter;
            clickAt(screenOf(hrL));
            clickAt(screenOf(m2L));
            // The span runs up the page, so a point to one side of it is
            // one displaced in x.
            clickAt(screenOf([(hrL[0] + m2L[0]) / 2 - 0.06,
                              (hrL[1] + m2L[1]) / 2]));

            var made = v.scene.dimensions[v.scene.dimensions.length - 1];
            out.local = {
                sent: sent.length - nLocal,
                added: v.scene.dimensions.length - nDim,
                name: made.name,
                p1: made.p1.slice(), p2: made.p2.slice(),
                wantP1: hrL.slice(), wantP2: m2L.slice(),
                length: made.length,
                offset: made.offset,
                optical: made.optical,
                inside: made.inside,
                isLocal: !!made.local,
                drawn: el.querySelectorAll('.gt-dims .gt-dim').length,
                selected: v.selectedDim,
                measuring: v.measuring,
                panel: panel(),
                removeOffered: v.dimFoot.style.display !== 'none'
            };

            button('Remove').click();
            out.localRemoved = {
                left: v.scene.dimensions.length,
                drawn: el.querySelectorAll('.gt-dims .gt-dim').length,
                sent: sent.length - nLocal,
                panel: panel()
            };

            out.sent = sent;
            document.getElementById('out').textContent = JSON.stringify(out);
            return;
        }

        // --- arming the tool ---
        button('Measure').click();
        out.armed = {measuring: v.measuring,
                     lit: button('Measure').classList.contains('gt-btn-on'),
                     cursor: v.svg.classList.contains('gt-measuring')};

        // --- snapping ---
        // Aim a few pixels off the HR apex of M1: the tool should take
        // the apex, not the pixel under the cursor.
        var hr = v.scene.optics.filter(function (o) {
            return o.name === 'M1';
        })[0].HRcenter;
        var near = screenOf(hr);
        mouse(window, 'mousemove', near[0] + 4, near[1] - 3);
        out.snapNear = {label: v.snapped && v.snapped.label,
                        point: v.snapped && v.snapped.point.slice(),
                        preview: v.measurePreview.slice(),
                        markShown: v.snapMark.style.display !== 'none'};

        // Well away from anything, the cursor is the answer.
        var away = screenOf([hr[0] + 0.25, hr[1] + 0.25]);
        mouse(window, 'mousemove', away[0], away[1]);
        out.snapFar = {snapped: v.snapped,
                       markShown: v.snapMark.style.display !== 'none'};

        // The reach is in screen pixels, so it survives a zoom: the
        // same offset in pixels still snaps when the drawing is 40x
        // larger, which a reach in scene units would not.
        var scale0 = v.scale;
        v.scale *= 40; v.cx = hr[0]; v.cy = hr[1]; v._applyTransform();
        var zoomed = screenOf(hr);
        mouse(window, 'mousemove', zoomed[0] + 4, zoomed[1] - 3);
        out.snapZoomed = {label: v.snapped && v.snapped.label};
        v.fit();

        // A beam end is on offer too. Not any beam end: most of them sit
        // on an optics, where a point of the element is nearer. Take one
        // that ends in the open - a beam that hit nothing - so that the
        // beam's own end is the only thing to snap to.
        var loose = null;
        v.scene.beams.forEach(function (b) {
            var clear = v.scene.snap.every(function (s) {
                return Math.hypot(s.point[0] - b.end[0],
                                  s.point[1] - b.end[1]) > 0.05;
            });
            if (clear && !loose) { loose = b; }
        });
        out.hasLooseBeam = !!loose;
        if (loose) {
            var bend = screenOf(loose.end);
            mouse(window, 'mousemove', bend[0] + 2, bend[1] + 2);
            out.snapBeam = {label: v.snapped && v.snapped.label,
                            point: v.snapped && v.snapped.point.slice(),
                            want: loose.end.slice()};
        }

        // --- while measuring, nothing else answers the mouse ---
        var opticPt = screenOf(v.scene.optics[0].center);
        mouse(window, 'mousemove', opticPt[0], opticPt[1]);
        out.overOptic = {hoverOptic: v.hoverOptic, hover: v.hover};
        mouse(v.svg, 'mousedown', opticPt[0], opticPt[1]);
        out.pressOnOptic = {dragOptic: v.dragOptic};
        mouse(window, 'mouseup', opticPt[0], opticPt[1]);
        // That release was a click, and while measuring a click fixes an
        // end. Start the tool over so the pair below is a clean one.
        out.pressWasAMeasurePoint = !!v.measureFrom;
        v.toggleMeasure(false);
        button('Measure').click();

        // --- the two clicks ---
        var nMeas = sent.length;
        // Deliberately a few pixels off, to land on the apex by snapping
        // rather than by aim.
        var p1 = screenOf(hr);
        clickAt([p1[0] + 3, p1[1] + 3]);
        out.firstClick = {from: v.measureFrom && v.measureFrom.slice(),
                          sent: sent.length - nMeas,
                          measuring: v.measuring,
                          rubberShown: v.rubber.style.display !== 'none'};

        // The line follows the cursor between the clicks.
        var m2hr = v.scene.optics.filter(function (o) {
            return o.name === 'M2';
        })[0].HRcenter;
        var p2 = screenOf(m2hr);
        mouse(window, 'mousemove', p2[0] - 3, p2[1] + 2);
        out.betweenClicks = {
            rubberShown: v.rubber.style.display !== 'none',
            x2: Number(v.rubber.getAttribute('x2')),
            wantX2: screenOf(m2hr)[0] - v.svg.getBoundingClientRect().left,
            status: el.querySelector('.gt-status').textContent
        };

        clickAt([p2[0] - 3, p2[1] + 2]);
        out.secondClick = {sent: sent.length - nMeas,
                           to: v.measureTo && v.measureTo.slice(),
                           measuring: v.measuring,
                           pendingShown: !!v.pendingEls
                               && v.pendingEls.group.style.display !== 'none',
                           rubberShown: v.rubber.style.display !== 'none',
                           wantP1: hr.slice(), wantP2: m2hr.slice()};

        // --- the third click places the line ---
        // Off to one side of the span, by a distance we can predict: the
        // offset is measured square to the way the two points run.
        var span = [m2hr[0] - hr[0], m2hr[1] - hr[1]];
        var spanLen = Math.hypot(span[0], span[1]);
        var wantOff = 0.06;
        var side = [hr[0] + span[0]/2 - span[1]/spanLen*wantOff,
                    hr[1] + span[1]/2 + span[0]/spanLen*wantOff];
        var pOff = screenOf(side);
        mouse(window, 'mousemove', pOff[0], pOff[1]);
        out.placing = {
            offset: v.measureOffset,
            wantOffset: wantOff,
            scale: v.scale,
            snapped: v.snapped,
            pendingShown: v.pendingEls.group.style.display !== 'none',
            extShown: v.pendingEls.e1.style.display !== 'none',
            status: el.querySelector('.gt-status').textContent
        };

        // Near the span itself the offset is zero, so a line drawn
        // straight between the two points can be had without aiming at
        // it exactly.
        var mid = screenOf([hr[0] + span[0]/2, hr[1] + span[1]/2]);
        mouse(window, 'mousemove', mid[0] + 2, mid[1] + 2);
        out.deadzone = {offset: v.measureOffset,
                        extShown: v.pendingEls.e1.style.display !== 'none'};

        mouse(window, 'mousemove', pOff[0], pOff[1]);
        var previewOffset = v.measureOffset;
        clickAt(pOff);
        out.thirdClick = {msg: sent[sent.length - 1],
                          previewOffset: previewOffset,
                          sent: sent.length - nMeas,
                          measuring: v.measuring,
                          lit: button('Measure').classList.contains('gt-btn-on'),
                          selected: v.selectedDim,
                          pendingShown: v.pendingEls.group.style.display
                              !== 'none'};

        // --- cancelling ---
        button('Measure').click();
        clickAt(screenOf(hr));
        var nCancel = sent.length;
        document.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Escape', bubbles: true}));
        out.cancelled = {measuring: v.measuring, from: v.measureFrom,
                         sent: sent.length - nCancel,
                         rubberShown: v.rubber.style.display !== 'none'};

        // Escape after the second click drops the whole thing, not just
        // the point being placed.
        button('Measure').click();
        clickAt(screenOf(hr));
        clickAt(screenOf(m2hr));
        var nCancel2 = sent.length;
        document.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Escape', bubbles: true}));
        out.cancelledLate = {measuring: v.measuring, from: v.measureFrom,
                             to: v.measureTo,
                             sent: sent.length - nCancel2,
                             pendingShown: v.pendingEls.group.style.display
                                 !== 'none'};

        // 'm' arms it from the keyboard, but only with the pointer over
        // this viewer.
        v.pointerInside = true;
        document.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'm', bubbles: true}));
        out.byKey = v.measuring;
        document.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'm', bubbles: true}));
        out.byKeyAgain = v.measuring;

        // Both ends in one place is not a measurement: nothing is sent,
        // the second point is not taken, and the tool stays up.
        button('Measure').click();
        var nSame = sent.length;
        clickAt(screenOf(hr));
        clickAt(screenOf(hr));
        out.samePoint = {sent: sent.length - nSame, measuring: v.measuring,
                         to: v.measureTo};
        v.toggleMeasure(false);

        // --- a scene with no dimensions in it ---
        var bare = JSON.parse(JSON.stringify(v.scene));
        bare.dimensions = [];
        model.set('scene', bare);
        out.afterEmpty = {groups: el.querySelectorAll('.gt-dims .gt-dim').length,
                          panel: panel(), selected: v.selectedDim};

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
    page = (PAGE.replace('__CSS__', viewer_css())
                .replace('__ESM__', js(esm))
                .replace('__SCENE__', js(scene))
                .replace('__EDITABLE__', 'true' if editable else 'false'))
    path = os.path.join(SP, 'measure_page.html')
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
check('ran without exception', res['error'] is None, str(res['error'])[:600])
check('there is a Measure button', res['measureButton'])
check('and the tool starts disarmed', res['startArmed'] is False)

print('--- a dimension in the drawing ---')
dr = res['drawn']
check('one group per dimension', dr['groups'] == 1, str(dr['groups']))
check('with a tick at each end', dr['ticks'] == 2, str(dr['ticks']))
check('the physical distance is written on it',
      dr['labels'][0]['text'].endswith('mm')
      and not dr['labels'][0]['hidden'], str(dr['labels'][0]))
check('and the optical distance under it',
      dr['labels'][1]['text'].endswith('optical')
      and not dr['labels'][1]['hidden'], str(dr['labels'][1]))
# The lens is 6 mm of glass with n = 1.45, so the two numbers must
# differ by that factor - the point of writing both.
phys = float(dr['labels'][0]['text'].split()[0])
optl = float(dr['labels'][1]['text'].split()[0])
check('and they differ by the index',
      abs(optl / phys - float(L1.n)) < 1e-4, '%.6f' % (optl / phys))
check('with an extension line at each end, since the line is carried aside',
      dr['exts'] == [True, True], str(dr['exts']))

print('--- picking it ---')
# The complaint the offset answers: a line drawn straight between the
# two points lies over what it measures, and neither can be picked.
sp0 = res['atSpan']
check('aiming at the span does not take the dimension',
      sp0['selected'] is None, str(sp0['selected']))
check('what is there is taken instead', sp0['optic'] == 'L1',
      str(sp0['optic']))

print('--- selecting one ---')
a = res['afterDimClick']
check('clicking it shows the dimension panel',
      a['panel']['title'] == 'Dimension' and a['panel']['dimShown']
      and not a['panel']['beamShown'] and not a['panel']['propsShown'],
      str(a['panel']))
check('it is recorded as selected', a['selected'] == 'D1', str(a['selected']))
check('and marked in the drawing', a['marked'])
check('the panel names it', a['fields']['name'] == 'D1',
      str(a['fields']['name']))
check('shows both ends',
      abs(float(a['fields']['p1x']) - float(L1.HRcenter[0])) < 1e-12
      and abs(float(a['fields']['p2x']) - float(L1.ARcenter[0])) < 1e-12,
      '%s / %s' % (a['fields']['p1x'], a['fields']['p2x']))
check('where the line is drawn, in mm',
      abs(float(a['fields']['offset']) - D1_OFFSET/mm) < 1e-9,
      str(a['fields']['offset']))
check('the distance', a['fields']['length'].endswith('mm'),
      a['fields']['length'])
check('the direction', a['fields']['angle'].endswith('deg')
      or '°' in a['fields']['angle'], a['fields']['angle'])
check('which optics the span is inside', a['fields']['inside'] == 'L1',
      a['fields']['inside'])
check('and the optical distance', a['opticalShown'] and a['insideShown'],
      '%s / %s' % (a['insideShown'], a['opticalShown']))

print('--- editing one ---')
e = res['editEnd']
check('typing an end sends one set', e['sent'] == 1, str(e['sent']))
check("naming that end only",
      e['msg']['op'] == 'set' and e['msg']['target'] == 'D1'
      and list(e['msg']['attrs']) == ['p2'], str(e['msg']))
check('with the typed value in it',
      abs(e['msg']['attrs']['p2'][0] - 0.9) < 1e-15, str(e['msg']['attrs']))
check('and the other coordinate untouched',
      abs(e['msg']['attrs']['p2'][1] - float(L1.ARcenter[1])) < 1e-15,
      str(e['msg']['attrs']['p2'][1]))

r = res['rename']
check('renaming sends a rename', r['sent'] == 1
      and r['msg'] == {'op': 'rename', 'target': 'D1', 'name': 'thickness'},
      str(r['msg']))
check('and the viewer follows it', r['selected'] == 'thickness',
      str(r['selected']))

rm = res['remove']
check('Remove sends one remove naming the dimension',
      rm['sent'] == 1 and rm['msg'] == {'op': 'remove', 'target': 'D1'},
      str(rm['msg']))
check('the selection is dropped', rm['selected'] is None, str(rm['selected']))
check('and the panel goes back to the readout', rm['panel']['beamShown'],
      str(rm['panel']))

print('--- arming the tool ---')
ar = res['armed']
check('the button arms it', ar['measuring'])
check('and lights up', ar['lit'])
check('the cursor says so', ar['cursor'])

print('--- snapping ---')
sn = res['snapNear']
check('a point near a marked one takes it', sn['label'] == 'M1 HR',
      str(sn['label']))
check('exactly, not approximately',
      abs(sn['point'][0] - float(M1.HRcenter[0])) < 1e-15
      and abs(sn['point'][1] - float(M1.HRcenter[1])) < 1e-15,
      str(sn['point']))
check('the preview is the marked point, not the cursor',
      sn['preview'] == sn['point'], str(sn['preview']))
check('and it is shown', sn['markShown'])
check('a point away from everything takes the cursor',
      res['snapFar']['snapped'] is None
      and not res['snapFar']['markShown'], str(res['snapFar']))
# The reach is in screen pixels, so zooming in does not lose it. The
# same mistake was made once with the Ctrl-drag snap.
check('the reach survives a 40x zoom', res['snapZoomed']['label'] == 'M1 HR',
      str(res['snapZoomed']['label']))
check('the layout has a beam ending in the open', res['hasLooseBeam'])
sb = res['snapBeam']
check('a beam end is on offer', sb['label'] and sb['label'].endswith(' end'),
      str(sb['label']))
# Taken from the scene as it stands, not worked out again in the page.
check('and it is the end the scene carries',
      sb['point'] == sb['want'], str(sb['point']))

print('--- nothing else answers the mouse while measuring ---')
check('no optics is hovered', res['overOptic']['hoverOptic'] is None,
      str(res['overOptic']['hoverOptic']))
check('no beam is picked', res['overOptic']['hover'] is None,
      str(res['overOptic']['hover']))
check('and pressing on an optics does not grab it',
      res['pressOnOptic']['dragOptic'] is None,
      str(res['pressOnOptic']['dragOptic']))
# What it does instead is what a click always does while measuring: it
# fixes an end. That is the point of the mode.
check('the release counts as a point of the measurement',
      res['pressWasAMeasurePoint'])

print('--- the three clicks ---')
fc = res['firstClick']
check('the first click fixes an end, sending nothing', fc['sent'] == 0,
      str(fc['sent']))
check('at the marked point it snapped to',
      abs(fc['from'][0] - float(M1.HRcenter[0])) < 1e-15
      and abs(fc['from'][1] - float(M1.HRcenter[1])) < 1e-15, str(fc['from']))
check('the tool stays up', fc['measuring'])
check('and the line appears', fc['rubberShown'])

bc = res['betweenClicks']
check('the line follows the cursor', bc['rubberShown'])
check('to where the next click would land',
      abs(bc['x2'] - bc['wantX2']) < 1.5,
      '%.2f vs %.2f' % (bc['x2'], bc['wantX2']))
check('and the status bar reports the distance as it stands',
      'Measure:' in bc['status'] and 'mm' in bc['status'], bc['status'])

sc = res['secondClick']
# The second click no longer commits anything: the two points worth
# measuring between are usually the two the drawing is busiest around,
# so where the line goes is a third question.
check('the second click sends nothing either', sc['sent'] == 0,
      str(sc['sent']))
check('it fixes the other end, snapped', sc['to'] is not None
      and abs(sc['to'][0] - sc['wantP2'][0]) < 1e-15
      and abs(sc['to'][1] - sc['wantP2'][1]) < 1e-15, str(sc['to']))
check('the tool is still up', sc['measuring'])
check('the bare line to the cursor goes away', sc['rubberShown'] is False)
check('and the dimension itself is previewed instead', sc['pendingShown'])

print('--- placing the line ---')
pl = res['placing']
# Judged in pixels, not in metres: this is a cursor position, and the
# status bar changing length as the tool reports what it is doing moves
# the drawing by a pixel or two. What has to be exact is that the
# message carries the offset the preview showed, which is checked below.
check('the cursor sets how far aside the line goes',
      abs(pl['offset'] - pl['wantOffset']) * pl['scale'] < 3,
      '%.6f vs %.6f (%.1f px)'
      % (pl['offset'], pl['wantOffset'],
         abs(pl['offset'] - pl['wantOffset']) * pl['scale']))
check('to the side the cursor is on', pl['offset'] > 0, str(pl['offset']))
# Where the line is drawn is a matter of where there is room; nothing in
# the model has an opinion, so nothing is snapped to.
check('nothing is snapped to while placing it', pl['snapped'] is None,
      str(pl['snapped']))
check('the preview is up', pl['pendingShown'])
check('with extension lines carrying the ends out to it', pl['extShown'])
check('and the status bar says the distance is settled',
      'place the line' in pl['status'] and 'offset' in pl['status'],
      pl['status'])
dz = res['deadzone']
check('near the span itself the offset is zero', dz['offset'] == 0,
      str(dz['offset']))
check('and the extension lines go away with it',
      dz['extShown'] is False)

tc = res['thirdClick']
check('the third click sends one add', tc['sent'] == 1, str(tc['sent']))
check('of a Dimension',
      tc['msg']['op'] == 'add' and tc['msg']['type'] == 'Dimension',
      str(tc['msg'])[:120])
check('carrying both ends and where the line goes',
      set(tc['msg']['params']) == {'p1', 'p2', 'offset'},
      str(sorted(tc['msg']['params'])))
check('the first end snapped exactly',
      abs(tc['msg']['params']['p1'][0] - sc['wantP1'][0]) < 1e-15
      and abs(tc['msg']['params']['p1'][1] - sc['wantP1'][1]) < 1e-15,
      str(tc['msg']['params']['p1']))
check('and so did the second',
      abs(tc['msg']['params']['p2'][0] - sc['wantP2'][0]) < 1e-15
      and abs(tc['msg']['params']['p2'][1] - sc['wantP2'][1]) < 1e-15,
      str(tc['msg']['params']['p2']))
check('with exactly the offset the preview was showing',
      tc['msg']['params']['offset'] == tc['previewOffset'],
      '%r vs %r' % (tc['msg']['params']['offset'], tc['previewOffset']))
check('and a name nothing else in the scene uses',
      tc['msg']['name'] not in ['D1', 'M1', 'M2', 'L1'], str(tc['msg']['name']))
# A mode that stays on until it is switched off is a mode that gets
# left on, and the button is right there.
check('the tool puts itself away', tc['measuring'] is False)
check('and the button goes out', tc['lit'] is False)
check('the preview goes with it', tc['pendingShown'] is False)
check('the new dimension is selected ahead of the scene coming back',
      tc['selected'] == tc['msg']['name'], str(tc['selected']))

print('--- cancelling ---')
cn = res['cancelled']
check('Escape disarms the tool', cn['measuring'] is False)
check('drops the end already fixed', cn['from'] is None, str(cn['from']))
check('sends nothing', cn['sent'] == 0, str(cn['sent']))
check('and takes the line away', cn['rubberShown'] is False)
cl = res['cancelledLate']
check('Escape after the second click drops the whole measurement',
      cl['measuring'] is False and cl['from'] is None and cl['to'] is None,
      str(cl))
check('sending nothing', cl['sent'] == 0, str(cl['sent']))
check('and taking the preview away', cl['pendingShown'] is False)
check("'m' arms it", res['byKey'] is True)
check('and puts it away again', res['byKeyAgain'] is False)
sp = res['samePoint']
check('two clicks in one place send nothing', sp['sent'] == 0, str(sp['sent']))
check('the second point is not taken', sp['to'] is None, str(sp['to']))
check('and the tool is left up to try again', sp['measuring'])

print('--- a scene with no dimensions ---')
ae = res['afterEmpty']
check('nothing is drawn', ae['groups'] == 0, str(ae['groups']))
check('and the panel falls back to the readout',
      ae['panel']['beamShown'] and ae['selected'] is None, str(ae['panel']))

print('--- Python makes the same of it ---')
lay, M1b, M2b, L1b = make_layout()
for msg in res['sent']:
    if msg['op'] == 'remove' or msg['op'] == 'rename':
        continue          # the sequence is not a coherent script
    if msg['op'] == 'set':
        continue          # aimed at the dimension the removes took out
    lay.apply_edit(msg)
check('the add the page sent is accepted', len(lay.dimensions) == 1,
      str([d.name for d in lay.dimensions]))
made = lay.dimensions[0]
check('with the ends it named',
      abs(float(made.p1[0]) - float(M1b.HRcenter[0])) < 1e-15
      and abs(float(made.p2[0]) - float(M2b.HRcenter[0])) < 1e-15,
      '%s / %s' % (list(made.p1), list(made.p2)))
m = made.measure(lay.optics)
check('and it measures the distance between the two mirrors',
      abs(m['length'] - float(np.linalg.norm(np.asarray(M2b.HRcenter)
                                             - np.asarray(M1b.HRcenter))))
      < 1e-15, str(m['length']))
check('a span across the bench is not inside anything',
      m['optical'] is None, str(m['optical']))
check('the line was carried aside by exactly what the page asked for',
      made.offset == res['thirdClick']['msg']['params']['offset'],
      str(made.offset))
# Where the line is drawn cannot change what was measured.
check('which leaves the distance alone',
      abs(m['length'] - Dimension(made.p1, made.p2).length) < 1e-15)
a, b = made.line_ends()
check('and puts the line square off the span by that much',
      abs(float(np.linalg.norm(a - made.p1)) - abs(made.offset)) < 1e-15
      and abs(float(np.dot(a - made.p1, made.p2 - made.p1))) < 1e-15,
      str(list(a)))

print('--- read-only viewer ---')
errs, res = run(False)
check('no console error', errs == [], '\n        '.join(errs[:3]))
check('ran without exception', res and res['error'] is None,
      str(res and res['error'])[:400])
check('the dimensions are still drawn', res['drawn']['groups'] == 1,
      str(res['drawn']['groups']))
check('and can still be read',
      res['afterDimClick']['panel']['dimShown']
      and res['afterDimClick']['selected'] == 'D1',
      str(res['afterDimClick']['panel']))
check('with nothing editable',
      res['afterDimClick']['fields']['name'] == 'D1',
      str(res['afterDimClick']['fields']['name']))

print('--- measuring without Python ---')
# The points to snap to are in the scene and the distance between two of
# them is arithmetic, so a written page can be measured on. That is the
# whole reason a file you can mail to a collaborator is worth having.
check('there is a Measure button all the same', res['measureButton'])
check('and it arms the tool', res['localArmed'])
lc = res['local']
check('the measurement is kept to the page', lc['sent'] == 0,
      str(lc['sent']))
check('one dimension is added', lc['added'] == 1, str(lc['added']))
check('and drawn', lc['drawn'] == 2, str(lc['drawn']))
check('with the ends it snapped to',
      abs(lc['p1'][0] - lc['wantP1'][0]) < 1e-15
      and abs(lc['p1'][1] - lc['wantP1'][1]) < 1e-15
      and abs(lc['p2'][0] - lc['wantP2'][0]) < 1e-15
      and abs(lc['p2'][1] - lc['wantP2'][1]) < 1e-15,
      '%s / %s' % (lc['p1'], lc['p2']))
check('the distance between them',
      abs(lc['length'] - float(np.linalg.norm(np.asarray(M2.HRcenter)
                                              - np.asarray(M1.HRcenter))))
      < 1e-12, str(lc['length']))
check('and the line carried aside as the third click asked',
      lc['offset'] != 0, str(lc['offset']))
# Whether a span runs inside a substrate is a question about the
# surfaces, and those live in the model rather than in the drawing.
check('but no optical distance, which would need the model',
      lc['optical'] is None and lc['inside'] is None,
      '%s / %s' % (lc['optical'], lc['inside']))
check('the tool puts itself away as usual', lc['measuring'] is False)
check('and the new dimension is selected',
      lc['selected'] == lc['name'] and lc['panel']['dimShown'],
      str(lc['selected']))

check('Remove is offered for what the reader drew', lc['removeOffered'])
# But not for the layout's own: a read-only viewer must not appear to
# change what it was handed.
check('and not for the layout\'s own dimension',
      not res['removeOfferedForTheirs'])
lr = res['localRemoved']
check('Remove takes back the reader\'s measurement', lr['left'] == 1
      and lr['drawn'] == 1, '%s / %s' % (lr['left'], lr['drawn']))
check('still sending nothing', lr['sent'] == 0, str(lr['sent']))
check('and the panel goes back to the readout', lr['panel']['beamShown'],
      str(lr['panel']))
check('nothing left the page at any point', res['sent'] == [],
      str(res['sent'])[:200])

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
