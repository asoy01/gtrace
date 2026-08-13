/*
 * What the page makes of a turned rectangle, against what gtrace makes
 * of the same one.
 *
 * A rectangle now carries an angle and the point it is taken about,
 * and both sides work its corners out for themselves: Python to draw,
 * bound and export it, the page to draw, pick and drag it. Two answers
 * to one question, so they are asked here side by side - the numbers
 * come from _work/rect_cases.json, which verify_rect.py writes from
 * the classes themselves.
 *
 * The drag is checked differently, because Python has no drag: what a
 * corner grip sends is held to what dragging a corner means. The
 * corner under the pointer lands on the pointer, the opposite corner
 * does not move, and the turn is not disturbed - on a rectangle at an
 * angle just as on one square to the axes.
 *
 * Run by run_all.py, after verify_rect.py, and needs nothing but Node.
 */

const fs = require('fs');
const path = require('path');

const REPO = process.argv[2];
const WORK = path.join(REPO, 'tests', 'gui', '_work');
const V = require(path.join(REPO, 'gtrace', 'draw', 'viewer', 'viewer.js'));

const CASES = path.join(WORK, 'rect_cases.json');
if (!fs.existsSync(CASES)) {
    console.log('rect_cases.json is not there - run verify_rect.py first');
    process.exit(77);
}
const data = JSON.parse(fs.readFileSync(CASES, 'utf8'));

let npass = 0, nfail = 0;

function check(name, cond, detail) {
    if (cond) { npass++; }
    else { nfail++; console.log('  FAIL  ' + name + '  ' + (detail || '')); }
}

function near(a, b, tol) {
    return Math.abs(a - b) <= (tol === undefined ? 1e-12 : tol);
}

function nearPt(a, b, tol) {
    return near(a[0], b[0], tol) && near(a[1], b[1], tol);
}

function nearPts(a, b, tol) {
    return a.length === b.length
        && a.every(function (p, i) { return nearPt(p, b[i], tol); });
}

function label(c) {
    return ('rect angle ' + c.shape.angle.toFixed(3)
            + ' about ' + (c.shape.pivot ? c.shape.pivot : 'itself'));
}

// --- the geometry, against gtrace ------------------------------------
data.cases.forEach(function (c) {
    const s = c.shape;
    check(label(c) + ': the pivot', nearPt(V.rectanglePivot(s), c.pivot));
    check(label(c) + ': the corners', nearPts(V.rectangleCorners(s), c.corners),
          JSON.stringify(V.rectangleCorners(s)));

    // The bounding box the panel marks, and the one Fit frames.
    const b = V.shapeBounds(s);
    const xs = b.map(function (p) { return p[0]; });
    const ys = b.map(function (p) { return p[1]; });
    check(label(c) + ': the bounding box',
          near(Math.min.apply(null, xs), c.bbox[0][0])
          && near(Math.min.apply(null, ys), c.bbox[0][1])
          && near(Math.max.apply(null, xs), c.bbox[1][0])
          && near(Math.max.apply(null, ys), c.bbox[1][1]));

    // The middle is among the points a drag settles on, and it is the
    // middle of the corners rather than of the box it was written from.
    const snaps = V.shapeSnapPoints(s);
    check(label(c) + ': the middle is a place to settle on',
          snaps.some(function (p) { return nearPt(p, c.centre, 1e-12); }));
    check(label(c) + ': every corner is too',
          c.corners.every(function (q) {
              return snaps.some(function (p) { return nearPt(p, q, 1e-12); });
          }));
    check(label(c) + ': and the middle of every side',
          c.corners.every(function (q, i) {
              const r = c.corners[(i + 1) % 4];
              const m = [(q[0] + r[0]) / 2, (q[1] + r[1]) / 2];
              return snaps.some(function (p) { return nearPt(p, m, 1e-12); });
          }));

    // What a click lands on. A probe that gtrace marked null landed on
    // the outline itself, where the answer turns on the last bit of
    // the arithmetic; the two are not held to agreeing there.
    c.probes.forEach(function (p, i) {
        if (c.encloses[i] === null) { return; }
        check(label(c) + ': encloses ' + JSON.stringify(p),
              V.shapeEncloses(s, p[0], p[1]) === c.encloses[i],
              'page says ' + V.shapeEncloses(s, p[0], p[1]));
    });

    // What a turn of the whole part previews, against turned_shape.
    const t = V.turnedShape(s, data.turn.angle, data.turn.pivot);
    const got = (t.type === 'polyline'
                 ? t.x.map(function (x, i) { return [x, t.y[i]]; })
                 : V.rectangleCorners(t));
    check(label(c) + ': turned by the part', nearPts(got.slice(0, 4), c.turned));
    if (s.angle !== 0) {
        check(label(c) + ': a turn keeps its own angle in the drawing',
              t.type === 'polyline' || near(t.angle, s.angle));
    }

    // Carrying it: the pivot goes along, or stays unsaid.
    const moved = V.shapeMoveAttrs(s, data.move[0], data.move[1]);
    const after = Object.assign({}, s, moved);
    check(label(c) + ': carried', nearPts(V.rectangleCorners(after), c.moved));
    check(label(c) + ': carried keeps its pivot under it',
          (s.pivot === null && !after.pivot)
          || nearPt(V.rectanglePivot(after),
                    [c.pivot[0] + data.move[0], c.pivot[1] + data.move[1]]));
});

// --- the panel rows --------------------------------------------------
data.cases.forEach(function (c) {
    const s = c.shape;
    const deg = V.shapeFieldValue(s, 'angle');
    check(label(c) + ': the angle row is degrees',
          near(deg * Math.PI / 180,
               Math.atan2(Math.sin(s.angle), Math.cos(s.angle)), 1e-12),
          'row says ' + deg);
    check(label(c) + ': the pivot rows are millimetres of the real pivot',
          near(V.shapeFieldValue(s, 'px'), c.pivot[0] * 1000, 1e-9)
          && near(V.shapeFieldValue(s, 'py'), c.pivot[1] * 1000, 1e-9));

    // Typing into a row sends what that row sets and nothing else.
    const a1 = V.shapeFieldAttrs(s, 'angle', 45);
    check(label(c) + ': typing an angle sends radians',
          near(a1.angle, Math.PI / 4) && Object.keys(a1).length === 1);
    const a2 = V.shapeFieldAttrs(s, 'px', 12);
    check(label(c) + ': typing a pivot sends the pair, in metres',
          nearPt(a2.pivot, [0.012, c.pivot[1]], 1e-12)
          && Object.keys(a2).length === 1);
});

// --- the drag --------------------------------------------------------
// A corner dragged to a place ends up on that place, the corner
// opposite it does not move, and neither the angle nor the pivot is
// disturbed. The same three things whether or not the rectangle is
// turned - which is the whole point, since the width and the height
// are lengths along its own axes and the drag has to work there.
//
// Which numbered corner the pointer holds afterwards is not among
// them: dragged past the corner it is measured from, the lower left
// becomes the upper right, exactly as it always did on a rectangle
// square to the axes. What has to be true is that the place is a
// corner and the fixed one is still where it was.
const TARGETS = [[0.4, 0.3], [-0.2, 0.35], [0.0, 0.0], [0.31, -0.22]];
data.cases.forEach(function (c) {
    const s = c.shape;
    const handles = V.shapeHandles(s);
    check(label(c) + ': four corner grips',
          handles.length === 4
          && handles.every(function (h) { return h.role === 'corner'; }));
    check(label(c) + ': the grips stand on the corners',
          nearPts(handles.map(function (h) { return h.p; }), c.corners));

    handles.forEach(function (h, i) {
        TARGETS.forEach(function (pt) {
            const after = Object.assign({}, s, V.shapeHandleAttrs(s, h, pt));
            const cs = V.rectangleCorners(after);
            const on = function (q) {
                return cs.some(function (p) { return nearPt(p, q, 1e-9); });
            };
            check(label(c) + ': corner ' + i + ' dragged to '
                  + JSON.stringify(pt) + ' ends up there',
                  on(pt), 'corners ' + JSON.stringify(cs));
            check(label(c) + ': and the corner across from it does not move',
                  on(c.corners[(i + 2) % 4]));
            check(label(c) + ': and the turn is not disturbed',
                  (after.angle || 0) === (s.angle || 0)
                  && ((s.pivot === null && !after.pivot)
                      || nearPt(after.pivot, s.pivot)));
            check(label(c) + ': and it is still a rectangle',
                  near(Math.hypot(cs[1][0] - cs[0][0], cs[1][1] - cs[0][1]),
                       after.width, 1e-9)
                  && Math.abs((cs[1][0] - cs[0][0]) * (cs[2][0] - cs[1][0])
                              + (cs[1][1] - cs[0][1]) * (cs[2][1] - cs[1][1]))
                     < 1e-15);
        });
    });
});

// A drag that would take a rectangle down to nothing is held at the
// smallest size a grip can be found again at - on a turned one too,
// where the pointer is not on either axis.
data.cases.forEach(function (c) {
    const s = c.shape;
    const h = V.shapeHandles(s)[0];
    const opposite = c.corners[2];
    const after = Object.assign({}, s, V.shapeHandleAttrs(s, h, opposite));
    check(label(c) + ': a corner dragged onto its opposite keeps a size',
          after.width > 0 && after.height > 0);
});

console.log('');
console.log(npass + ' passed, ' + nfail + ' failed');
process.exit(nfail ? 1 : 0);
