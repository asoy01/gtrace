'''
Regenerate the viewer screenshots used by the tutorial.

    pixi run python docs/make_viewer_figures.py

The figures are photographs of the real front end rather than mock-ups:
the widget's ESM is loaded into headless Chrome, driven with a stand-in
model of the kind anywidget provides, and screenshotted. Run this again
whenever the viewer's appearance changes, so that the tutorial does not
end up describing a version of the interface that no longer exists.

The layout is the one the tutorial builds in its "Optical layouts"
section, so the pictures and the text agree.
'''

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# The browser search already exists for the verification suites, and
# having two lists of where Chrome might live is one too many.
sys.path.insert(0, os.path.join(REPO, 'tests', 'gui'))
from _harness import find_chrome                                 # noqa: E402

import numpy as np                                               # noqa: E402

import gtrace.beam as beam                                       # noqa: E402
import gtrace.optcomp as opt                                     # noqa: E402
import gtrace.optics.gaussian as gauss                           # noqa: E402
from gtrace.draw.viewer import viewer_css                        # noqa: E402
from gtrace.draw.viewer.widget import widget_esm                 # noqa: E402
from gtrace.layout import OpticalLayout, TraceRules              # noqa: E402
from gtrace.unit import *                                        # noqa: E402

OUT = os.path.join(REPO, 'docs', 'source', 'tutorial', 'figures')

#: Size of the viewer in CSS pixels, and the scale the browser renders
#: it at. The figures are shown at half width in the notebook, so the
#: doubled scale is what keeps the text sharp. The height is set by the
#: side bar rather than by the drawing: it is tall enough that every
#: panel is visible at once, since a panel scrolled out of frame is a
#: panel the tutorial describes and the picture appears to contradict.
WIDTH, HEIGHT, SCALE = 1180, 940, 2


def make_layout():
    '''
    The system the tutorial traces, as an OpticalLayout.

    Same mirrors and same source beam as the sections above it, so that
    the reader recognises the picture.
    '''
    q0 = gauss.Rw2q(ROC=np.inf, w=0.3 * mm)
    b0 = beam.GaussianBeam(q0=q0, wl=1064 * nm, length=30 * cm, P=1.0,
                           pos=[0.0, 0.0], dirAngle=deg2rad(10), name='b0')

    M1 = opt.Mirror(HRcenter=[50 * cm, 10 * cm], normAngleHR=np.pi,
                    diameter=25 * cm, thickness=10 * cm,
                    wedgeAngle=deg2rad(0.25), inv_ROC_HR=1. / (120 * cm),
                    inv_ROC_AR=0, Refl_HR=0.9, Trans_HR=1 - 0.9,
                    Refl_AR=500 * ppm, Trans_AR=1 - 500 * ppm,
                    n=1.45, name='M1')
    M2 = opt.Mirror(HRcenter=[0 * cm, 18 * cm], normAngleHR=deg2rad(5.0),
                    diameter=15 * cm, thickness=5 * cm,
                    wedgeAngle=deg2rad(0.25), inv_ROC_HR=-1. / (350 * cm),
                    inv_ROC_AR=0, Refl_HR=0.9, Trans_HR=1 - 0.9,
                    Refl_AR=500 * ppm, Trans_AR=1 - 500 * ppm,
                    n=1.45, name='M2')
    M3 = opt.Mirror(HRcenter=[30 * cm, 30 * cm], normAngleHR=deg2rad(21.3),
                    diameter=15 * cm, thickness=5 * cm,
                    wedgeAngle=deg2rad(1), inv_ROC_HR=1. / (350 * cm),
                    inv_ROC_AR=0, Refl_HR=0.9, Trans_HR=1 - 0.9,
                    Refl_AR=500 * ppm, Trans_AR=1 - 500 * ppm,
                    n=1.45, name='M3')

    return OpticalLayout(optics=[M1, M2, M3], sources=[b0],
                         rules=TraceRules(order=30, power_threshold=1e-6),
                         name='Tutorial')


PAGE = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html, body { margin: 0; background: #fff; }
#host { width: __W__px; height: __H__px; }
__CSS__
</style></head>
<body>
<div id="host"></div>
<script>
var ESM_SRC = __ESM__;
var SCENE = __SCENE__;
</script>
<script type="module">
(async function () {
    var url = URL.createObjectURL(
        new Blob([ESM_SRC], {type: 'text/javascript'}));
    var mod = await import(url);

    /* The slice of the anywidget model the front end actually uses.
       Nothing is sent anywhere: these pictures are of the interface,
       not of a round trip. */
    var state = {scene: SCENE, title: 'Tutorial', height: __H__,
                 editable: true, error: '', notice: '',
                 layout_path: 'layout.json'};
    var handlers = {};
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
        send: function () {}
    };

    var el = document.getElementById('host');
    mod.default.render({model: model, el: el});
    var v = el.gtraceViewer;

    function screenOf(x, y) { return v.sceneToScreen(x, y); }
    function clickOptic(name) {
        var o = (v.scene.optics || []).filter(function (p) {
            return p.name === name;
        })[0];
        if (!o) { return; }
        var c = o.center || o.HRcenter;
        var r = v.svg.getBoundingClientRect();
        var p = screenOf(c[0], c[1]);
        ['mousemove', 'mousedown', 'mouseup'].forEach(function (t) {
            (t === 'mousedown' ? v.svg : window).dispatchEvent(
                new MouseEvent(t, {clientX: p[0] + r.left,
                                   clientY: p[1] + r.top,
                                   button: 0, bubbles: true,
                                   cancelable: true}));
        });
    }

__ACTIONS__

    document.title = 'ready';
})();
</script>
</body></html>
'''


def build_page(scene, actions):
    css = viewer_css()
    page = PAGE.replace('__CSS__', css)
    page = page.replace('__ESM__', json.dumps(widget_esm()))
    page = page.replace('__SCENE__', json.dumps(scene).replace('</', '<\\/'))
    page = page.replace('__ACTIONS__', actions)
    page = page.replace('__W__', str(WIDTH)).replace('__H__', str(HEIGHT))
    return page


def shoot(chrome, page, name):
    html = os.path.join(OUT, '_shot.html')
    with open(html, 'w', encoding='utf-8') as f:
        f.write(page)
    png = os.path.join(OUT, name)
    if os.path.exists(png):
        os.remove(png)

    cmd = [chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars',
           '--window-size=%d,%d' % (WIDTH, HEIGHT),
           '--force-device-scale-factor=%d' % SCALE,
           '--virtual-time-budget=6000',
           '--screenshot=' + png,
           'file:///' + html.replace('\\', '/')]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    os.remove(html)

    if not os.path.exists(png):
        raise SystemExit('no screenshot written for %s' % name)
    print('  %-28s %7d bytes' % (name, os.path.getsize(png)))


#: What each figure shows, as a fragment of the driver above.
#:
#: There is deliberately no picture of the untouched viewer: it differs
#: from the readout one only in that every row of the panel reads '-',
#: and two nearly identical screenshots cost the reader more attention
#: than the second one returns.
FIGURES = [
    ('viewer_readout.png', '''
    /* Pin the readout partway along the longest main beam, which is
       what a reader sees after clicking one. */
    var main = v.scene.beams.filter(function (b) {
        return b.layer === 'main_beam';
    });
    main.sort(function (a, b) { return b.length - a.length; });
    var t = main[0];
    var p = screenOf(t.pos[0] + t.dirVect[0] * t.length * 0.45,
                     t.pos[1] + t.dirVect[1] * t.length * 0.45);
    v._onHover(p[0], p[1]);
    v._onClick(p[0], p[1]);
'''),
    ('viewer_properties.png', '''
    clickOptic('M1');
'''),
    ('viewer_measure.png', '''
    /* A measurement taken across the substrate of M1, from the apex of
       its HR face to the apex of its AR face - the span that runs
       inside the glass, and so the one that carries an optical distance
       as well as a physical one. The dimension line is carried clear of
       the element, with extension lines back to the two points, which
       is the whole point of the third click: drawn straight between
       them it would lie in the glass, on top of the beams going
       through it.

       Pushed in as a scene, since that is how it would arrive from
       Python once the last click had been made, and then selected so
       that the panel is showing.

       Zoomed onto the element, because a 10 cm substrate on a metre of
       bench is otherwise a few pixels of green. */
    var m1 = v.scene.optics.filter(function (o) {
        return o.name === 'M1';
    })[0];
    var mid = [(m1.HRcenter[0] + m1.ARcenter[0]) / 2,
               (m1.HRcenter[1] + m1.ARcenter[1]) / 2];
    var withDim = JSON.parse(JSON.stringify(v.scene));
    var n = m1.n;
    var vx = m1.ARcenter[0] - m1.HRcenter[0];
    var vy = m1.ARcenter[1] - m1.HRcenter[1];
    var len = Math.hypot(vx, vy);
    var off = 0.17;                       /* clear of the 25 cm aperture */
    var nx = -vy / len * off, ny = vx / len * off;
    withDim.dimensions = [{type: 'Dimension', name: 'D1',
                           p1: m1.HRcenter, p2: m1.ARcenter, offset: off,
                           line: [[m1.HRcenter[0] + nx, m1.HRcenter[1] + ny],
                                  [m1.ARcenter[0] + nx, m1.ARcenter[1] + ny]],
                           length: len, optical: n * len,
                           inside: 'M1', n: n}];
    model.set('scene', withDim);
    v.scale *= 3; v.cx = mid[0] + nx / 2; v.cy = mid[1] + ny / 2;
    v._applyTransform();
    var r = v.svg.getBoundingClientRect();
    var p = v.sceneToScreen(mid[0] + nx, mid[1] + ny);
    ['mousemove', 'mousedown', 'mouseup'].forEach(function (t) {
        (t === 'mousedown' ? v.svg : window).dispatchEvent(
            new MouseEvent(t, {clientX: p[0] + r.left, clientY: p[1] + r.top,
                               button: 0, bubbles: true, cancelable: true}));
    });
'''),
]


def main():
    chrome = find_chrome()
    if chrome is None:
        raise SystemExit('No Chrome-like browser found. Set GTRACE_CHROME '
                         'to the executable to regenerate the figures.')

    layout = make_layout()
    scene = layout.scene_dict()
    print('layout: %d optics, %d beams' % (len(layout.optics),
                                           len(scene['beams'])))
    print('browser: %s' % chrome)

    os.makedirs(OUT, exist_ok=True)
    for name, actions in FIGURES:
        shoot(chrome, build_page(scene, actions), name)


if __name__ == '__main__':
    main()
