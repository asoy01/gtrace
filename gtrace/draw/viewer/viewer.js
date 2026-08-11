/*
 * gtrace viewer core
 *
 * A dependency-free SVG viewer for gtrace scenes. The scene is the
 * JSON structure produced by gtrace.draw.serialize.scene_to_dict():
 *
 *   {canvas: {unit, layers: [{name, color, shapes: [...]}, ...]},
 *    beams:  [{name, layer, pos, end, dirVect, length, wl, n, P,
 *              qx, qy, wx, wy, Gouyx, Gouyy, optDist, stray_order}, ...]}
 *
 * Geometry is drawn in scene coordinates (meters, y up) inside a single
 * transformed group, so zoom and pan are just an update of that transform.
 * Text labels are drawn in screen coordinates so that they keep a constant
 * readable size at any zoom level.
 *
 * The same core is reused by the anywidget front end (Stage 2) and the
 * live server (Stage 3), so it must not depend on anything but the DOM.
 *
 * Copyright (c) 2011-2026, Yoichi Aso. BSD license.
 */

(function (global) {
'use strict';

var SVGNS = 'http://www.w3.org/2000/svg';

//{{{ Small DOM helpers

function svgEl(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) { for (var k in attrs) { e.setAttribute(k, attrs[k]); } }
    return e;
}

function htmlEl(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined) { e.textContent = text; }
    return e;
}

//}}}

//{{{ Formatting

/*
 * Format a length in meters with an automatically chosen SI prefix.
 */
function fmtLen(v) {
    if (v === null || v === undefined || isNaN(v)) { return '-'; }
    if (!isFinite(v)) { return v > 0 ? '∞' : '-∞'; }
    var a = Math.abs(v);
    if (a === 0) { return '0'; }
    if (a >= 1e3) { return (v / 1e3).toPrecision(5) + ' km'; }
    if (a >= 1) { return v.toPrecision(5) + ' m'; }
    if (a >= 1e-3) { return (v * 1e3).toPrecision(5) + ' mm'; }
    if (a >= 1e-6) { return (v * 1e6).toPrecision(5) + ' µm'; }
    if (a >= 1e-9) { return (v * 1e9).toPrecision(5) + ' nm'; }
    return v.toExponential(4) + ' m';
}

function fmtNum(v, digits) {
    if (v === null || v === undefined || isNaN(v)) { return '-'; }
    if (!isFinite(v)) { return v > 0 ? '∞' : '-∞'; }
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
        return v.toExponential(digits === undefined ? 4 : digits);
    }
    return v.toPrecision(digits === undefined ? 5 : digits);
}

/*
 * Format a q parameter, held as a [real, imag] pair. The unit (meters)
 * is carried by the row label, so that the value fits on one line.
 */
function fmtQ(q) {
    if (!q) { return '-'; }
    var im = q[1];
    return fmtNum(q[0], 4) + (im < 0 ? ' - ' : ' + ') +
           fmtNum(Math.abs(im), 4) + 'i';
}

function fmtDeg(rad) {
    if (rad === null || rad === undefined || isNaN(rad)) { return '-'; }
    return (rad * 180 / Math.PI).toFixed(2) + '°';
}

/*
 * Wrap an angle into (-pi, pi]. Beams accumulate their direction angle
 * reflection after reflection, so it can grow well beyond one turn.
 */
function normAngle(rad) {
    var a = rad % (2 * Math.PI);
    if (a > Math.PI) { a -= 2 * Math.PI; }
    if (a <= -Math.PI) { a += 2 * Math.PI; }
    return a;
}

//}}}

//{{{ Beam physics

/*
 * Beam parameters at a distance d from the origin of the beam.
 *
 * This is the JavaScript counterpart of GaussianBeam.width() / R() /
 * propagate(): propagating a Gaussian beam by d in a medium of index n
 * is simply q -> q + d, and then
 *
 *   w = sqrt(-2 / (k * Im(1/q))),   R = 1 / Re(1/q),   k = 2*pi*n/wl
 *
 * The Rayleigh range Im(q) is invariant along the propagation, the waist
 * sits at a distance -Re(q) ahead of the evaluation point, and the Gouy
 * phase accumulates as atan(Re(q)/Im(q)).
 */
function beamParamsAt(beam, d) {
    var k = 2 * Math.PI * beam.n / beam.wl;
    var out = {d: d, optDist: beam.optDist + beam.n * d};
    ['x', 'y'].forEach(function (ax) {
        var q0 = beam['q' + ax];
        var q = [q0[0] + d, q0[1]];
        var den = q[0] * q[0] + q[1] * q[1];
        var invRe = q[0] / den;
        var invIm = -q[1] / den;
        out['q' + ax] = q;
        out['w' + ax] = Math.sqrt(-2.0 / (k * invIm));
        out['R' + ax] = 1.0 / invRe;
        out['zR' + ax] = q[1];
        out['w0' + ax] = Math.sqrt(2.0 * q[1] / k);
        out['waist' + ax] = -q[0];
        out['Gouy' + ax] = beam['Gouy' + ax] +
            Math.atan(q[0] / q[1]) - Math.atan(q0[0] / q0[1]);
    });
    return out;
}

/*
 * The four corners of the substrate of an optics, in scene coordinates.
 * Used to outline an optics while it is being dragged; the exact shape
 * (wedge, curvature) belongs to the canvas, this is only a handle.
 */
function opticOutline(o, center, angle) {
    var c = center || o.center || o.HRcenter || [0, 0];
    var a = angle === undefined ? (o.normAngleHR || 0) : angle;
    var ux = Math.cos(a), uy = Math.sin(a);          // along the normal
    var vx = -uy, vy = ux;                           // across the face
    var h = (o.thickness || 0) / 2;
    var w = (o.diameter || 0) / 2;
    return [[c[0] + ux * h + vx * w, c[1] + uy * h + vy * w],
            [c[0] + ux * h - vx * w, c[1] + uy * h - vy * w],
            [c[0] - ux * h - vx * w, c[1] - uy * h - vy * w],
            [c[0] - ux * h + vx * w, c[1] - uy * h + vy * w]];
}

/*
 * Radius of a circle enclosing the substrate, for hit testing.
 */
function opticRadius(o) {
    var w = (o.diameter || 0) / 2;
    var h = (o.thickness || 0) / 2;
    return Math.hypot(w, h);
}

/*
 * Whether a point lies inside a polygon, by ray casting. The same
 * algorithm as Mechanics.contains on the Python side: a mechanics is
 * picked by its area, because the enclosing circle the optics use
 * would let a breadboard cover the whole bench around itself.
 */
function pointInPolygon(x, y, poly) {
    var inside = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        var xi = poly[i][0], yi = poly[i][1];
        var xj = poly[j][0], yj = poly[j][1];
        if ((yi > y) !== (yj > y) &&
                x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
    }
    return inside;
}

/*
 * Area of a polygon, by the shoelace formula. Used to pick the
 * smallest of several overlapping mechanics: the mount standing on
 * the breadboard must win over the breadboard, or it could never be
 * pointed at.
 */
function polygonArea(poly) {
    var a = 0;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        a += poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
    }
    return Math.abs(a) / 2;
}

/*
 * Project a scene point onto a beam segment.
 * Returns {d, foot, dist} where d is the distance from the beam origin
 * (clamped to the segment), foot the projected point and dist the
 * distance from the given point to the segment.
 */
function projectOnBeam(beam, px, py) {
    var vx = beam.dirVect[0], vy = beam.dirVect[1];
    var d = (px - beam.pos[0]) * vx + (py - beam.pos[1]) * vy;
    if (d < 0) { d = 0; }
    if (d > beam.length) { d = beam.length; }
    var fx = beam.pos[0] + vx * d;
    var fy = beam.pos[1] + vy * d;
    return {d: d, foot: [fx, fy], dist: Math.hypot(px - fx, py - fy)};
}

//}}}

//{{{ Colors

function srgbToLinear(c) {
    c = c / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function linearToSrgb(c) {
    c = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.round(Math.max(0, Math.min(1, c)) * 255);
}

/*
 * Relative luminance above which a color is too pale to be seen on the
 * white background. 0.30 is the value that gives a 3:1 contrast ratio.
 */
var MAX_LUMINANCE = 0.30;

/*
 * Layer colors come from the DXF palette, which assumes a black
 * background: pure green and pure cyan are perfectly readable there but
 * nearly invisible on white. Darken any color that is too light, keeping
 * its hue by scaling in linear light. Colors that are already dark
 * enough (red, magenta, black) pass through untouched.
 */
function layerColor(rgb) {
    if (!rgb) { return '#000000'; }
    var r = srgbToLinear(rgb[0]);
    var g = srgbToLinear(rgb[1]);
    var b = srgbToLinear(rgb[2]);
    var lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if (lum > MAX_LUMINANCE) {
        var k = MAX_LUMINANCE / lum;
        r *= k; g *= k; b *= k;
    }
    return 'rgb(' + linearToSrgb(r) + ',' + linearToSrgb(g) + ',' +
           linearToSrgb(b) + ')';
}

//}}}

//{{{ Viewer

/*
 * Every live viewer on the page. Used only to scope the keyboard
 * shortcuts: with one viewer the keys work wherever the pointer is,
 * with several they act on the one being pointed at.
 */
var VIEWERS = [];

/*
 * What the front end can put into a layout. The name prefix is only a
 * starting point: the layout has the last word on whether it is free,
 * and it can be renamed afterwards like any other.
 */
var ADDABLE_TYPES = [
    {type: 'Mirror', label: 'Mirror', prefix: 'M',
     title: 'Add a mirror'},
    {type: 'CyMirror', label: 'CyMirror', prefix: 'CY',
     title: 'Add a cylindrical mirror',
     params: {curve_direction: 'h'}},
    // No params: a lens is not cut to match the mirrors around it, so
    // Python builds it from its own catalogue defaults rather than from
    // anything the view can say. Its focal length is edited afterwards
    // in the panel, like any other property.
    {type: 'Lens', label: 'Lens', prefix: 'L',
     title: 'Add a lens'},
    // A cylindrical lens carries its direction the way a CyMirror
    // does; everything else it takes from the catalogue, like a Lens.
    {type: 'CyLens', label: 'CyLens', prefix: 'CL',
     title: 'Add a cylindrical lens',
     params: {curve_direction: 'h'}},
    // A source is not an optics, and does not take an optics' pose: it
    // starts at a point and is aimed, rather than standing with a face
    // turned. 'source' says to send the other pair of parameters.
    {type: 'Source', label: 'Source', prefix: 'S',
     title: 'Add a laser source', source: true}
];

/*
 * How those are offered: one control per kind of thing, with the
 * variants of a kind behind it.
 *
 * A button apiece put five along a side bar narrow enough that they
 * wrapped, and read as five unrelated things when a cylindrical mirror
 * is a mirror. Grouped, the row says what can be added - a mirror, a
 * lens, a source - and the variant is a second choice made only by
 * someone who wants it. A kind with nothing to choose between is a
 * plain button; a menu of one would be a question with one answer.
 *
 * The order within a menu puts the ordinary one first.
 */
var ADD_GROUPS = [
    {label: 'Mirror', title: 'Add a mirror',
     types: ['Mirror', 'CyMirror'],
     names: ['Spherical', 'Cylindrical']},
    {label: 'Lens', title: 'Add a lens',
     types: ['Lens', 'CyLens'],
     names: ['Spherical', 'Cylindrical']},
    {label: 'Source', title: 'Add a laser source',
     types: ['Source']}
];

function addableType(type) {
    for (var i = 0; i < ADDABLE_TYPES.length; i++) {
        if (ADDABLE_TYPES[i].type === type) { return ADDABLE_TYPES[i]; }
    }
    return null;
}

/*
 * Ways of aiming the selected optics.
 *
 * A drag can put an element approximately anywhere, and Ctrl-drag
 * squares it onto a beam that already exists. What is left is the
 * angles a bench is laid out by before there is a beam to point at:
 * facing from one place towards another, or bisecting the corner at
 * a place light is to be folded at - and the quarter turn that a
 * 45 degree steering mirror is specified by.
 *
 * The click order carries the direction in both: the face looks
 * towards the second place of a pair, and into the corner of a
 * three.
 *
 * Aiming leaves the element where it is. Which way it faces and where
 * it stands are two questions, and the second already has answers:
 * Ctrl-drag, the Center rows, Along beam / Move by.
 */
var ALIGN_ITEMS = [
    {label: 'Line 2 points', points: 2, key: 'a',
     title: 'Click two points; the optics faces from the first '
            + 'towards the second'},
    {label: 'Bisect 3 points', points: 3, key: 'b',
     title: 'Click from, at, to; the optics faces the bisector, '
            + 'folding light from the first point to the last'},
    {label: 'Turn +45°', turn: 45, key: ']',
     title: 'Turn the optics a quarter turn counterclockwise'},
    {label: 'Turn −45°', turn: -45, key: '[',
     title: 'Turn the optics a quarter turn clockwise'}
];

/*
 * The shapes a part can be drawn from, as the editor offers them.
 * The labels are the button row; the type is what an add message
 * names, and what shape_from_dict builds on the Python side.
 */
var SHAPE_KINDS = [
    {type: 'rectangle', label: '+ Rect'},
    {type: 'circle', label: '+ Circle'},
    {type: 'line', label: '+ Line'},
    {type: 'polyline', label: '+ Poly'},
    {type: 'arc', label: '+ Arc'},
    {type: 'text', label: '+ Text'}
];

/*
 * The rows the panel shows for each kind of shape, and how each row
 * maps to the serialized shape a message carries.
 *
 * Lengths are in millimetres and angles in degrees, as everywhere
 * else a part is dimensioned; the shape itself is in metres and
 * radians, like the rest of gtrace.
 */
var SHAPE_FIELDS = {
    rectangle: [
        {key: 'x', label: 'Corner x', unit: 'mm'},
        {key: 'y', label: 'Corner y', unit: 'mm'},
        {key: 'width', label: 'Width', unit: 'mm'},
        {key: 'height', label: 'Height', unit: 'mm'}
    ],
    circle: [
        {key: 'cx', label: 'Center x', unit: 'mm'},
        {key: 'cy', label: 'Center y', unit: 'mm'},
        {key: 'radius', label: 'Radius', unit: 'mm'}
    ],
    line: [
        {key: 'x1', label: 'From x', unit: 'mm'},
        {key: 'y1', label: 'From y', unit: 'mm'},
        {key: 'x2', label: 'To x', unit: 'mm'},
        {key: 'y2', label: 'To y', unit: 'mm'}
    ],
    arc: [
        {key: 'cx', label: 'Center x', unit: 'mm'},
        {key: 'cy', label: 'Center y', unit: 'mm'},
        {key: 'radius', label: 'Radius', unit: 'mm'},
        {key: 'startangle', label: 'From', unit: '°'},
        {key: 'stopangle', label: 'To', unit: '°'}
    ],
    text: [
        {key: 'text', label: 'Text', text: true},
        {key: 'x', label: 'At x', unit: 'mm'},
        {key: 'y', label: 'At y', unit: 'mm'},
        {key: 'height', label: 'Size', unit: 'mm'},
        {key: 'rotation', label: 'Angle', unit: '°'}
    ],
    polyline: [
        {key: 'points', label: 'Vertices', readonly: true}
    ]
};

/*
 * What a row of a shape panel reads, from the serialized shape.
 */
function shapeFieldValue(s, key) {
    switch (key) {
    case 'x': return (s.point ? s.point[0] : 0) / MM;
    case 'y': return (s.point ? s.point[1] : 0) / MM;
    case 'cx': return (s.center ? s.center[0] : 0) / MM;
    case 'cy': return (s.center ? s.center[1] : 0) / MM;
    case 'x1': return (s.start ? s.start[0] : 0) / MM;
    case 'y1': return (s.start ? s.start[1] : 0) / MM;
    case 'x2': return (s.stop ? s.stop[0] : 0) / MM;
    case 'y2': return (s.stop ? s.stop[1] : 0) / MM;
    case 'width': return s.width / MM;
    case 'height': return s.height / MM;
    case 'radius': return s.radius / MM;
    case 'startangle': return normAngle(s.startangle) * DEG;
    case 'stopangle': return normAngle(s.stopangle) * DEG;
    case 'rotation': return normAngle(s.rotation) * DEG;
    case 'text': return s.text;
    case 'points': return (s.x || []).length + ' points';
    default: return s[key];
    }
}

/*
 * The attributes a row sets, as the serialized shape spells them.
 * The pairs that make a point are sent whole, since that is what the
 * shape carries - the other half comes from the shape on show.
 */
function shapeFieldAttrs(s, key, value) {
    var attrs = {};
    switch (key) {
    case 'x': attrs.point = [value * MM, s.point[1]]; break;
    case 'y': attrs.point = [s.point[0], value * MM]; break;
    case 'cx': attrs.center = [value * MM, s.center[1]]; break;
    case 'cy': attrs.center = [s.center[0], value * MM]; break;
    case 'x1': attrs.start = [value * MM, s.start[1]]; break;
    case 'y1': attrs.start = [s.start[0], value * MM]; break;
    case 'x2': attrs.stop = [value * MM, s.stop[1]]; break;
    case 'y2': attrs.stop = [s.stop[0], value * MM]; break;
    case 'width':
    case 'height':
    case 'radius': attrs[key] = value * MM; break;
    case 'startangle':
    case 'stopangle':
    case 'rotation': attrs[key] = value / DEG; break;
    case 'text': attrs.text = value; break;
    default: attrs[key] = value;
    }
    return attrs;
}

/*
 * The corners of the box a serialized shape occupies, for the mark
 * that shows which one the panel is editing. An arc is bounded by its
 * whole circle, as it is everywhere else in gtrace: looser than the
 * arc, never smaller.
 */
function shapeBounds(s) {
    var xs = [], ys = [], i;
    switch (s.type) {
    case 'line': xs = [s.start[0], s.stop[0]]; ys = [s.start[1], s.stop[1]];
        break;
    case 'polyline': xs = s.x.slice(); ys = s.y.slice(); break;
    case 'rectangle':
        xs = [s.point[0], s.point[0] + s.width];
        ys = [s.point[1], s.point[1] + s.height];
        break;
    case 'circle':
    case 'arc':
        xs = [s.center[0] - s.radius, s.center[0] + s.radius];
        ys = [s.center[1] - s.radius, s.center[1] + s.radius];
        break;
    case 'text': xs = [s.point[0]]; ys = [s.point[1]]; break;
    default: return null;
    }
    if (!xs.length) { return null; }
    var minx = Math.min.apply(null, xs), maxx = Math.max.apply(null, xs);
    var miny = Math.min.apply(null, ys), maxy = Math.max.apply(null, ys);
    return [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]];
}

/*
 * Where a point goes when the body it belongs to is turned by da
 * about a pivot. What the outline preview of an aim is built from,
 * since an optics turns about its anchor point and so its centre
 * travels.
 */
function turnAbout(p, pivot, da) {
    var ca = Math.cos(da), sa = Math.sin(da);
    var ox = p[0] - pivot[0], oy = p[1] - pivot[1];
    return [pivot[0] + ox * ca - oy * sa, pivot[1] + ox * sa + oy * ca];
}

/*
 * The angle to face along the line through two points.
 *
 * A line has two normals, and the click order says which: the face
 * ends up looking from the first place towards the second. So the
 * two places clicked the other way about turn the element right
 * round, which is how a face is flipped - and it means the order is
 * something to mean rather than something to ignore.
 */
function acrossAngle(p1, p2) {
    return Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
}

/*
 * The angle to face at the middle of three points: the bisector of
 * the corner, pointing back at both of the others.
 *
 * That is where a mirror folding light from the first point to the
 * last has to look, since the angle it makes with each arm is then
 * the same - which is what the law of reflection says. Null when the
 * three points make no corner: two of them in the same place, or all
 * three in a line, where the bisector is not defined.
 */
function bisectorAngle(p1, p2, p3) {
    var d1 = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
    var d2 = Math.hypot(p3[0] - p2[0], p3[1] - p2[1]);
    if (!d1 || !d2) { return null; }
    var ux = (p1[0] - p2[0]) / d1 + (p3[0] - p2[0]) / d2;
    var uy = (p1[1] - p2[1]) / d1 + (p3[1] - p2[1]) / d2;
    if (Math.hypot(ux, uy) < 1e-9) { return null; }
    return Math.atan2(uy, ux);
}

/*
 * The laser drawn at the start of a source beam, in screen pixels.
 *
 * A source is drawn at all because nothing else in the picture says
 * which beams the user put there and which the trace produced: they are
 * all lines, and the one the laser emits looks exactly like the one it
 * became after a mirror. The box is where the light comes from, and it
 * is the handle the source is edited by.
 *
 * In screen pixels rather than metres because a layout is anywhere from
 * a bench to a kilometre across, and a body sized in millimetres would
 * be a dot on one and would fill the other. gtrace draws optics at their
 * optical size and nothing at its mechanical size, and a laser given a
 * footprint would be the first exception to that; this is a marker, not
 * a part.
 *
 * The shape runs backwards from the origin: the body sits behind the
 * point the beam leaves from, so the drawing is not covered by the very
 * thing it is pointing at.
 */
var SOURCE_BODY = 30;      // how far back the body reaches
var SOURCE_HALFW = 11;     // half its width
var SOURCE_NOSE = 4;       // the length of the aperture stub
var SOURCE_NOSE_HALFW = 4; // and half of its width

/*
 * The outline of the laser, in the beam's own frame: u along the
 * direction it fires, v across it, both in screen pixels.
 */
var SOURCE_SHAPE = [
    [-SOURCE_BODY, -SOURCE_HALFW],
    [-SOURCE_NOSE, -SOURCE_HALFW],
    [-SOURCE_NOSE, -SOURCE_NOSE_HALFW],
    [0, -SOURCE_NOSE_HALFW],
    [0, SOURCE_NOSE_HALFW],
    [-SOURCE_NOSE, SOURCE_NOSE_HALFW],
    [-SOURCE_NOSE, SOURCE_HALFW],
    [-SOURCE_BODY, SOURCE_HALFW]
];

/*
 * Where a point of the laser outline falls on screen, given where the
 * origin of the beam is, which way it fires, and how far the shape has
 * been let grow (see sourceGrowth).
 */
function sourcePoint(uv, originPx, dirVect, k) {
    // The screen has y downwards while the scene has it upwards, so the
    // across-axis is taken from the direction with the sign that gives
    // the same handedness the drawing already has.
    var ux = dirVect[0], uy = -dirVect[1];
    var u = uv[0] * k, v = uv[1] * k;
    return [originPx[0] + u * ux - v * uy,
            originPx[1] + u * uy + v * ux];
}

/*
 * How much bigger than its nominal size the laser has to be drawn.
 *
 * The box is in screen pixels so that it stays legible across a layout
 * that may be a bench or a kilometre. That holds only while the beam is
 * a line. Zoom in far enough and the drawn envelope is wider than the
 * aperture it is supposed to be coming out of, which is a picture of
 * something that cannot happen - so past that point the box grows with
 * the view instead, and the aperture goes on matching the beam.
 *
 * The threshold is exactly where the two meet: the width of the beam
 * where it leaves, drawn as the display draws it, against the width of
 * the nose. Below it the factor is 1 and nothing has changed.
 */
function sourceGrowth(s, display, scale) {
    var d = display || {};
    if (d.drawMainWidth === false || !s.width) { return 1; }
    var sigma = d.sigma_main === undefined ? 2.7 : d.sigma_main;
    var mode = d.width_mode || 'x';
    var w = mode === 'y' ? s.width[1]
        : mode === 'avg' ? (s.width[0] + s.width[1]) / 2
        : s.width[0];
    // Half the drawn envelope, in screen pixels, against half the nose.
    return Math.max(1, sigma * w * scale / SOURCE_NOSE_HALFW);
}

/*
 * Whether a screen point falls on the laser drawn for a source.
 *
 * The same shape the drawing uses, at the same growth factor, so that
 * what is clickable is what is visible. The test is in screen pixels
 * for as long as the shape is: an element sized in metres would stop
 * being clickable as soon as the view was zoomed, which is a mistake
 * this viewer has already made once with the Ctrl-drag snap.
 */
function sourceHit(px, py, originPx, dirVect, k) {
    var dx = px - originPx[0], dy = py - originPx[1];
    var ux = dirVect[0], uy = -dirVect[1];
    var u = (dx * ux + dy * uy) / k;
    var v = (-dx * uy + dy * ux) / k;
    if (u > 0 || u < -SOURCE_BODY) { return false; }
    var half = u < -SOURCE_NOSE ? SOURCE_HALFW : SOURCE_NOSE_HALFW;
    return Math.abs(v) <= half;
}

function Viewer(container, scene, options) {
    this.container = container;
    this.scene = scene || {canvas: {layers: []}, beams: []};
    this.opts = options || {};
    this.fontSize = this.opts.fontSize || 12;

    this.scale = 1;         // screen pixels per scene unit
    this.cx = 0;            // scene coordinate at the center of the view
    this.cy = 0;
    this.width = 1;
    this.height = 1;
    // Set for real by _applyTransform. Here so that anything drawn in
    // screen coordinates before the first framing gets a number rather
    // than a NaN, which SVG reports as an error and then ignores.
    this.tx = 0;
    this.ty = 0;

    this.fitMargin = 0.06;  // fraction of the view left clear around the scene
    this.fitPending = false; // a fit waiting for the view to have a size

    this.labels = [];       // {el, x, y, rotation, layer}
    this.layerGroups = {};  // layer name -> {geom, label, visible}
    this.pinned = null;     // pinned readout {beam, d, point, rank, count}
    this.hover = null;
    this.cycle = 0;         // index into the bundle of overlapping beams
    this.lastClick = null;
    // The same pair again for picking the beam to move an optics along.
    // Kept apart from the readout's so that the two cycles do not step
    // each other on: they are answers to different questions asked at
    // the same place.
    this.slideBeam = null;  // {name, index} of the chosen beam
    this.slideCycle = 0;
    this.lastSlideClick = null;

    // Editing is available only when the transport gave us somewhere to
    // send the edits: the notebook widget and the live server do, the
    // static HTML file does not, so that file stays read-only.
    this.onEdit = this.opts.onEdit || null;
    this.hoverOptic = null;
    this.dragOptic = null;

    // The lasers standing for the registered sources: which one is
    // selected, which the cursor is over, which is being dragged, and
    // the SVG drawn for each.
    this.selectedSource = null;
    this.hoverSource = null;
    this.dragSource = null;
    this.sourceEls = [];
    this.sourceFallback = null;

    // The mechanics - the hardware the trace never sees. Their shapes
    // are in the canvas like the substrates of the optics; this is the
    // selection state their outlines and panel work from.
    this.selectedMech = null;
    this.hoverMech = null;
    this.dragMech = null;
    this.mechFallback = null;
    // A corner of a resizable body being dragged to a new size, and
    // where its handles were last drawn (screen coordinates, for the
    // mousedown hit test).
    this.dragMechResize = null;
    this._handlePts = null;

    // Measuring. The tool is a mode because it takes two clicks, and
    // between them the picture has to answer to the cursor rather than
    // to what is under it.
    this.measuring = false;
    this.measureFrom = null;  // scene point of the first click
    this.measureTo = null;    // scene point of the second
    this.measureOffset = 0;   // how far aside the line is being carried
    this.snapped = null;      // the snap point under the cursor, or null
    this.selectedDim = null;  // name of the selected dimension
    this.dimEls = {};         // dimension name -> its SVG elements
    this.pendingEls = null;   // the SVG of the dimension being placed

    // Aiming the selected optics. A mode, like measuring and for the
    // same reason: it takes two or three clicks, and between them a
    // click has to mean "this place" rather than whatever clicking
    // there would otherwise have meant.
    this.aligning = null;     // {optic, want, points: [[x, y], ...]}
    this.alignPreview = null; // where the next click would land

    // Editing a part: which of its shapes the panel is showing, by
    // index into scene.shapes. An index is a place in the list rather
    // than a thing, so it is re-read against every scene.
    this.selectedShape = null;
    this._shapeFieldsKind = null;

    VIEWERS.push(this);
    this._build();
    this._renderScene();
    this._refreshDisplayPanel();
    this._refreshUndo();
    this._bindEvents();
    this.fit();
}

//{{{ DOM construction

Viewer.prototype._build = function () {
    var self = this;
    var root = htmlEl('div', 'gt-root');

    // --- stage (drawing area) ---
    var stage = htmlEl('div', 'gt-stage');
    this.svg = svgEl('svg', {'class': 'gt-svg'});
    this.sceneGroup = svgEl('g', {'class': 'gt-scene'});
    this.labelGroup = svgEl('g', {'class': 'gt-labels'});
    // Dimensions sit above the drawing and below the overlay: they are
    // notes on the picture rather than part of it, but they are standing
    // marks rather than an answer to where the cursor is.
    this.dimGroup = svgEl('g', {'class': 'gt-dims'});
    // The lasers. Part of the picture rather than a mark on it - a
    // source is a thing in the layout, not an answer to where the
    // cursor is - but drawn in screen coordinates like the labels and
    // the dimensions, and so kept in a group of its own.
    this.sourceGroup = svgEl('g', {'class': 'gt-sources'});
    this.overlayGroup = svgEl('g', {'class': 'gt-overlay'});
    this.svg.appendChild(this.sceneGroup);
    this.svg.appendChild(this.labelGroup);
    this.svg.appendChild(this.dimGroup);
    this.svg.appendChild(this.sourceGroup);
    this.svg.appendChild(this.overlayGroup);
    stage.appendChild(this.svg);

    this.statusBar = htmlEl('div', 'gt-status');
    stage.appendChild(this.statusBar);

    // Folding the side bar away gives the drawing the whole width, which
    // in a notebook cell is most of what there is. The button rides on
    // the drawing rather than in the panel, since a button inside the
    // thing it hides cannot bring it back.
    this.sideToggle = htmlEl('button', 'gt-sidetoggle', '»');
    this.sideToggle.title = 'Hide the side panel';
    this.sideToggle.addEventListener('click', function () {
        self.toggleSide();
    });
    stage.appendChild(this.sideToggle);
    root.appendChild(stage);

    // --- side bar ---
    var side = htmlEl('div', 'gt-side');

    // The head is two rows of buttons: the ones that put something into
    // the layout, then the ones that act on it or on the view. They are
    // kept to their own rows rather than left to wrap wherever they fit
    // - which row a button lands on would otherwise depend on how wide
    // the panel happened to be.
    //
    // The layout has no heading here. In a notebook it is already
    // labelled by the cell that made it, and a written page carries its
    // name in the browser tab; a line of the side bar spent repeating
    // it is a line not spent on the readout.
    var head = htmlEl('div', 'gt-head');

    // Editing a part rather than looking at a bench: the buttons put
    // shapes down instead of optics, and the panels below deal in
    // shapes. Everything the two have in common - zoom, pan, undo,
    // layers - is the same code either way.
    var editing = !!this.scene.editor;

    if (editing && this.opts.onEdit) {
        var shapeRow = htmlEl('div', 'gt-btnrow');
        SHAPE_KINDS.forEach(function (spec) {
            var btn = htmlEl('button', 'gt-btn', spec.label);
            btn.title = 'Add a ' + spec.type + ' at the origin';
            btn.addEventListener('click', function () {
                self.addShape(spec.type);
            });
            shapeRow.appendChild(btn);
        });
        head.appendChild(shapeRow);
    }

    if (!editing && this.opts.onEdit) {
        var addRow = htmlEl('div', 'gt-btnrow');
        this.addMenus = [];
        ADD_GROUPS.forEach(function (g) {
            var only = g.types.length === 1 ? addableType(g.types[0]) : null;
            var btn = htmlEl('button', 'gt-btn', '+ ' + g.label);
            if (only) {
                btn.title = only.title + ' at the centre of the view';
                btn.addEventListener('click', function () {
                    self.addOptics(only.type);
                });
                addRow.appendChild(btn);
                return;
            }

            // The variants sit in a menu the button opens. It is a
            // child of the button's own wrapper rather than of the
            // page, so it travels with the button when the side bar
            // scrolls without anything having to compute where it went.
            var wrap = htmlEl('div', 'gt-add');
            btn.className += ' gt-addbtn';
            btn.title = g.title + ' at the centre of the view';
            var menu = htmlEl('div', 'gt-menu');
            menu.style.display = 'none';
            g.types.forEach(function (type, i) {
                var t = addableType(type);
                if (!t) { return; }
                var item = htmlEl('button', 'gt-menuitem',
                                  (g.names && g.names[i]) || t.label);
                item.title = t.title + ' at the centre of the view';
                item.addEventListener('click', function () {
                    self.closeAddMenus();
                    self.addOptics(t.type);
                });
                menu.appendChild(item);
            });
            btn.addEventListener('click', function () {
                var open = menu.style.display === 'none';
                self.closeAddMenus();
                if (open) {
                    menu.style.display = '';
                    btn.classList.add('gt-open');
                }
            });
            wrap.appendChild(btn);
            wrap.appendChild(menu);
            addRow.appendChild(wrap);
            self.addMenus.push({button: btn, menu: menu, wrap: wrap});
        });

        // The hardware, behind one more button of the same row. Its
        // variants are not classes but library models, and the library
        // rides in the scene - so the menu is filled by
        // _refreshHardwareMenu, and refilled whenever a new scene
        // brings a new library.
        var hwrap = htmlEl('div', 'gt-add');
        var hbtn = htmlEl('button', 'gt-btn gt-addbtn', '+ Hardware');
        hbtn.title = 'Add hardware from the model library at the centre '
            + 'of the view';
        var hmenu = htmlEl('div', 'gt-menu');
        hmenu.style.display = 'none';
        hbtn.addEventListener('click', function () {
            var open = hmenu.style.display === 'none';
            self.closeAddMenus();
            if (open) {
                hmenu.style.display = '';
                hbtn.classList.add('gt-open');
            }
        });
        hwrap.appendChild(hbtn);
        hwrap.appendChild(hmenu);
        addRow.appendChild(hwrap);
        self.addMenus.push({button: hbtn, menu: hmenu, wrap: hwrap});
        this.hardwareMenu = {button: hbtn, menu: hmenu, wrap: hwrap};
        this._refreshHardwareMenu();

        head.appendChild(addRow);
    }

    var viewRow = htmlEl('div', 'gt-btnrow');
    if (this.opts.onEdit) {
        // Undo and redo are Python's: it holds the layout, so it is the
        // only thing that knows what the layout was. Disabled until the
        // scene says there is something to go back - or forward - to.
        this.undoBtn = htmlEl('button', 'gt-btn', 'Undo');
        this.undoBtn.title = 'Undo the last edit';
        this.undoBtn.addEventListener('click', function () { self.undo(); });
        viewRow.appendChild(this.undoBtn);
        this.redoBtn = htmlEl('button', 'gt-btn', 'Redo');
        this.redoBtn.title = 'Redo the last undone edit';
        this.redoBtn.addEventListener('click', function () { self.redo(); });
        viewRow.appendChild(this.redoBtn);
    }
    // Measuring needs no Python: the points to snap to are in the scene
    // and the distance between two of them is arithmetic. A viewer with
    // nowhere to send edits keeps the measurement to itself - see
    // _addLocalDimension for what that costs. It is offered while
    // editing a part too: how far one hole is from another is exactly
    // what a part is drawn from.
    this.measureBtn = htmlEl('button', 'gt-btn', 'Measure');
    this.measureBtn.title = 'Measure between two points, then place '
        + 'the dimension line';
    this.measureBtn.addEventListener('click', function () {
        self.toggleMeasure();
    });
    viewRow.appendChild(this.measureBtn);

    // Aiming the selected optics. A menu, since it offers four ways
    // of saying the same kind of thing, and only where there is a
    // Python to send the turn to. Nothing to aim in a part: its
    // shapes are written in the frame, not turned in it.
    if (this.opts.onEdit && !editing) {
        this.addMenus = this.addMenus || [];
        var awrap = htmlEl('div', 'gt-add');
        this.alignBtn = htmlEl('button', 'gt-btn gt-addbtn', 'Align');
        var amenu = htmlEl('div', 'gt-menu');
        amenu.style.display = 'none';
        ALIGN_ITEMS.forEach(function (spec) {
            var item = htmlEl('button', 'gt-menuitem', spec.label);
            item.title = spec.title + '   (' + spec.key + ')';
            item.addEventListener('click', function () {
                self.closeAddMenus();
                if (spec.turn) { self.turnSelected(spec.turn); }
                else { self.startAlign(spec.points); }
            });
            amenu.appendChild(item);
        });
        this.alignBtn.addEventListener('click', function () {
            var open = amenu.style.display === 'none';
            self.closeAddMenus();
            if (open) {
                amenu.style.display = '';
                self.alignBtn.classList.add('gt-open');
            }
        });
        awrap.appendChild(this.alignBtn);
        awrap.appendChild(amenu);
        viewRow.appendChild(awrap);
        this.addMenus.push({button: this.alignBtn, menu: amenu, wrap: awrap});
        this._refreshAlign();
    }

    var fitBtn = htmlEl('button', 'gt-btn', 'Fit');
    fitBtn.title = 'Frame the whole layout';
    fitBtn.addEventListener('click', function () { self.fit(); });
    viewRow.appendChild(fitBtn);
    head.appendChild(viewRow);
    side.appendChild(head);

    // Readout panel. It shows either the beam under the cursor or the
    // properties of the selected optics, whichever was picked last.
    var rpanel = htmlEl('div', 'gt-panel');
    var rtitle = htmlEl('div', 'gt-panel-title');
    this.panelTitle = htmlEl('span', null, 'Beam readout');
    rtitle.appendChild(this.panelTitle);
    this.pinLabel = htmlEl('span', 'gt-pin', '');
    rtitle.appendChild(this.pinLabel);
    rpanel.appendChild(rtitle);
    this.readoutBody = htmlEl('div', 'gt-readout');
    this.opticBody = htmlEl('div', 'gt-props');
    this.dimBody = htmlEl('div', 'gt-props');
    this.sourceBody = htmlEl('div', 'gt-props');
    this.mechBody = htmlEl('div', 'gt-props');
    this.shapeBody = htmlEl('div', 'gt-props');
    rpanel.appendChild(this.readoutBody);
    rpanel.appendChild(this.opticBody);
    rpanel.appendChild(this.dimBody);
    rpanel.appendChild(this.sourceBody);
    rpanel.appendChild(this.mechBody);
    rpanel.appendChild(this.shapeBody);
    side.appendChild(rpanel);
    this._buildReadout();
    this._buildOpticPanel();
    this._buildDimPanel();
    this._buildSourcePanel();
    this._buildMechPanel();
    if (editing) {
        this._buildShapePanel();
        this._showPanel('shape');
    } else {
        this._showPanel('beam');
    }

    // Two file panels, kept apart because they deal in two different
    // things. The layout is the model - saving it and loading it back
    // is the same system either way. The DXF is a drawing of it, going
    // out to something that will never send it back. Sharing a panel,
    // or worse a file name, invites Load to be pressed on a drawing.
    //
    // Both are written by Python: the page has no business touching the
    // disk.
    // Putting the part on the library shelf: the same register_model
    // any cell would call, with the name and the line of description
    // it takes. The part itself is the user's object and is already
    // being edited in place; this is only what gives other layouts a
    // way to ask for one.
    if (this.onEdit && editing) {
        var mpanel = htmlEl('div', 'gt-panel');
        mpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Model library'));
        var mbody = htmlEl('div', 'gt-file');
        this.modelInput = htmlEl('input', 'gt-input gt-input-text');
        this.modelInput.type = 'text';
        this.modelInput.spellcheck = false;
        this.modelInput.value = (this.scene.editor
                                 && this.scene.editor.model_name) || '';
        this.modelInput.title = 'The name other layouts ask for this part by';
        mbody.appendChild(this.modelInput);
        this.modelDesc = htmlEl('input', 'gt-input gt-input-text');
        this.modelDesc.type = 'text';
        this.modelDesc.spellcheck = false;
        this.modelDesc.placeholder = 'one line, for the menu';
        mbody.appendChild(this.modelDesc);
        var mrow = htmlEl('div', 'gt-filebuttons');
        var saveModelBtn = htmlEl('button', 'gt-btn', 'Save to library');
        saveModelBtn.title = 'Register these shapes under that name';
        saveModelBtn.addEventListener('click', function () {
            self.saveModel();
        });
        mrow.appendChild(saveModelBtn);
        mbody.appendChild(mrow);
        mpanel.appendChild(mbody);
        side.appendChild(mpanel);
    }

    if (this.onEdit && !editing) {
        var layoutPath = this.opts.layoutPath || 'layout.json';

        var fpanel = htmlEl('div', 'gt-panel');
        fpanel.appendChild(htmlEl('div', 'gt-panel-title',
                                  'Optical layout (JSON)'));
        var fbody = htmlEl('div', 'gt-file');
        this.pathInput = htmlEl('input', 'gt-input gt-input-text');
        this.pathInput.type = 'text';
        this.pathInput.spellcheck = false;
        this.pathInput.value = layoutPath;
        this.pathInput.title = 'Relative to where the kernel is running';
        fbody.appendChild(this.pathInput);
        var frow = htmlEl('div', 'gt-filebuttons');
        var saveBtn = htmlEl('button', 'gt-btn', 'Save');
        saveBtn.title = 'Write the optical layout to this file';
        saveBtn.addEventListener('click', function () { self.saveLayout(); });
        var loadBtn = htmlEl('button', 'gt-btn', 'Load');
        loadBtn.title = 'Replace the optical layout with the one in this file';
        loadBtn.addEventListener('click', function () { self.loadLayout(); });
        frow.appendChild(saveBtn);
        frow.appendChild(loadBtn);
        fbody.appendChild(frow);
        fpanel.appendChild(fbody);
        side.appendChild(fpanel);

        // The drawing, for whatever comes after gtrace in an
        // engineering workflow. Its name starts from the layout's, so
        // the two match without being typed twice, and is then the
        // user's own: they are not the same file and need not share a
        // stem.
        var xpanel = htmlEl('div', 'gt-panel');
        xpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Drawing (DXF)'));
        var xbody = htmlEl('div', 'gt-file');
        this.dxfInput = htmlEl('input', 'gt-input gt-input-text');
        this.dxfInput.type = 'text';
        this.dxfInput.spellcheck = false;
        this.dxfInput.value = this.opts.dxfPath
            || withExtension(layoutPath, '.dxf');
        this.dxfInput.title = 'Relative to where the kernel is running';
        xbody.appendChild(this.dxfInput);
        var xrow = htmlEl('div', 'gt-filebuttons');
        var dxfBtn = htmlEl('button', 'gt-btn', 'Export');
        dxfBtn.title = 'Write the drawing to this DXF file';
        dxfBtn.addEventListener('click', function () { self.exportDXF(); });
        xrow.appendChild(dxfBtn);
        xbody.appendChild(xrow);
        xpanel.appendChild(xbody);
        side.appendChild(xpanel);
    }

    // Display panel. These change how Python draws the scene, so they
    // exist only when there is a Python to ask - and only where there
    // are beams for them to be about.
    if (this.onEdit && !editing) {
        var dpanel = htmlEl('div', 'gt-panel');
        dpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Beam width'));
        var dbody = htmlEl('div', 'gt-display');
        this.displayControls = {};
        [{key: 'sigma', label: 'Envelope',
          options: [['1', '1σ'], ['2.7', '2.7σ  (1 ppm)'], ['3', '3σ']]},
         {key: 'width_mode', label: 'Direction',
          options: [['x', 'x'], ['y', 'y'], ['avg', 'average']]}
        ].forEach(function (spec) {
            var row = htmlEl('label', 'gt-displayrow');
            row.appendChild(htmlEl('span', 'gt-key', spec.label));
            var sel = htmlEl('select', 'gt-select');
            spec.options.forEach(function (o) {
                var opt = htmlEl('option', null, o[1]);
                opt.value = o[0];
                sel.appendChild(opt);
            });
            sel.addEventListener('change', function () {
                self._commitDisplay(spec.key, sel.value);
            });
            row.appendChild(sel);
            dbody.appendChild(row);
            self.displayControls[spec.key] = sel;
        });
        dpanel.appendChild(dbody);
        side.appendChild(dpanel);

        // How deep the trace goes. Editable through the protocol since
        // Stage 2b and until now with nothing to reach it, which is the
        // one setting anyone chasing stray light wants to hand.
        var tpanel = htmlEl('div', 'gt-panel');
        tpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Tracing rules'));
        var built = buildFieldTable(
            RULE_FIELDS, true,
            function (key, el) { self._commitRuleField(key, el); },
            function () { self._refreshRulesPanel(); });
        this.ruleFields = built.fields;
        tpanel.appendChild(built.table);
        side.appendChild(tpanel);
        this._refreshRulesPanel();
    }

    // layer panel
    var lpanel = htmlEl('div', 'gt-panel');
    lpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Layers'));
    this.layerBody = htmlEl('div', 'gt-layerlist');
    lpanel.appendChild(this.layerBody);
    side.appendChild(lpanel);

    // help
    var hpanel = htmlEl('div', 'gt-panel gt-help');
    hpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Controls'));
    var ul = htmlEl('ul');
    var rows = editing
        ? [['Wheel', 'zoom at cursor'],
           ['Drag', 'pan'],
           ['Click a shape in the list', 'edit its numbers'],
           ['+ Rect / + Circle / …', 'put one down at the origin'],
           ['Copy', 'a second one, just beside it'],
           ['Remove', 'take it away'],
           ['↑ / ↓', 'draw it earlier or later'],
           ['Measure, or m', 'measure between two points'],
           ['f', 'fit to view'],
           ['Undo, or Ctrl + Z', 'put the last edit back'],
           ['Save to library', 'register the part under a name']]
        : [['Wheel', 'zoom at cursor'],
                ['Drag', 'pan'],
                ['Move over a beam', 'live readout'],
                ['Click', 'pin the readout'],
                ['Click again', 'cycle overlapping beams and hardware'],
                ['Click an optics', 'show its properties'],
                ['Click a laser', 'show the source it stands for'],
                ['Click hardware', 'show its pose'],
                ['f', 'fit to view'],
                ['Measure, or m', 'measure between two points'],
                ['Esc', 'clear selection']];
    if (this.opts.onEdit && !editing) {
        rows.push(['Drag an optics or a laser', 'move it'],
                  ['Drag near a screw hole', 'land the anchor on it '
                   + '(Alt rides free)'],
                  ['Drag selected hardware', 'move it'],
                  ['Drag a corner handle', 'cut a breadboard to size'],
                  ['+ Hardware', 'add a part from the model library'],
                  ['Attached to', 'seat a mount on an optics; '
                   + '(free) detaches it in place'],
                  ['Ctrl + drag', 'drop it square on a beam'],
                  ['Shift + drag', 'rotate it'],
                  ['Align, or a', 'face from one point towards another'],
                  ['Align, or b', 'face the bisector of three points'],
                  ['[ and ]', 'turn it a quarter turn'],
                  ['Ctrl + click a beam', 'move the selected optics along it'],
                  ['Edit a property', 'apply it to the layout'],
                  ['+ Mirror / + Lens / + Source',
                   'add one at the centre of the view'],
                  ['+ Mirror, + Lens',
                   'open for the cylindrical variant'],
                  ['Remove', 'delete the selection'],
                  ['DXF', 'write the drawing out for CAD'],
                  ['Undo, or Ctrl + Z', 'put the last edit back'],
                  ['Redo, or Ctrl + Shift + Z', 'take the undo back']);
    }
    rows.forEach(function (row) {
        var li = htmlEl('li');
        li.appendChild(htmlEl('b', null, row[0]));
        li.appendChild(document.createTextNode(' — ' + row[1]));
        ul.appendChild(li);
    });
    hpanel.appendChild(ul);
    side.appendChild(hpanel);

    root.appendChild(side);
    this.side = side;
    this.container.appendChild(root);
    this.root = root;

    // A grip along the bottom edge, for dragging the viewer taller. Only
    // where the embedder said how tall it may be in the first place: a
    // written page fills the window, and a grip there would only fight
    // it. See _bindEvents for the drag, and opts.onResize for what
    // becomes of the height afterwards.
    if (this.opts.resizable) {
        this.resizeGrip = htmlEl('div', 'gt-resize');
        this.resizeGrip.title = 'Drag to change the height';
        this.container.appendChild(this.resizeGrip);
    }
};

/*
 * Shut any open add menu. Called whenever something else is pressed,
 * and by the item that was chosen.
 */
Viewer.prototype.closeAddMenus = function () {
    (this.addMenus || []).forEach(function (m) {
        m.menu.style.display = 'none';
        m.button.classList.remove('gt-open');
    });
};

/*
 * Whether a menu is open, and whether an element is inside one.
 */
Viewer.prototype._inAddMenu = function (node) {
    var found = false;
    (this.addMenus || []).forEach(function (m) {
        if (m.wrap.contains(node)) { found = true; }
    });
    return found;
};

/*
 * Fold the side panel away, or bring it back.
 */
Viewer.prototype.toggleSide = function (on) {
    var show = on === undefined ? this.side.style.display === 'none' : !!on;
    this.side.style.display = show ? '' : 'none';
    this.sideToggle.textContent = show ? '»' : '«';
    this.sideToggle.title = show ? 'Hide the side panel'
                                 : 'Show the side panel';
    this.sideToggle.classList.toggle('gt-sidetoggle-folded', !show);
    // The drawing area just changed width. Without a ResizeObserver
    // nothing would notice, and the view would keep the shape it had.
    this._resize();
    return show;
};

/*
 * The readout table. Rows with two values show the x (horizontal) and
 * y (vertical) transverse directions side by side.
 */
var READOUT_ROWS = [
    {key: 'beam', label: 'Beam', span: true},
    {key: 'layer', label: 'Layer', span: true},
    {key: 'dir', label: 'Direction', span: true},
    {key: 'point', label: 'Point', span: true},
    {key: 'dist', label: 'Distance', span: true},
    {key: 'w', label: 'Radius w'},
    {key: 'R', label: 'ROC R'},
    {key: 'q', label: 'q [m]'},
    {key: 'w0', label: 'Waist w₀'},
    {key: 'waist', label: 'To waist'},
    {key: 'zR', label: 'zᴿ'},
    {key: 'Gouy', label: 'Gouy'},
    {key: 'P', label: 'Power', span: true},
    {key: 'wl', label: 'Wavelength', span: true},
    {key: 'n', label: 'Index n', span: true},
    {key: 'optDist', label: 'Optical dist.', span: true},
    {key: 'stray', label: 'Stray order', span: true}
];

Viewer.prototype._buildReadout = function () {
    var table = htmlEl('table');
    var header = htmlEl('tr', 'gt-rowhead');
    header.appendChild(htmlEl('th', null, ''));
    header.appendChild(htmlEl('th', null, 'x'));
    header.appendChild(htmlEl('th', null, 'y'));
    table.appendChild(header);

    this.cells = {};
    var self = this;
    READOUT_ROWS.forEach(function (row) {
        var tr = htmlEl('tr');
        tr.appendChild(htmlEl('td', 'gt-key', row.label));
        if (row.span) {
            var td = htmlEl('td', 'gt-val', '-');
            td.setAttribute('colspan', '2');
            tr.appendChild(td);
            self.cells[row.key] = [td];
        } else {
            var tdx = htmlEl('td', 'gt-val', '-');
            var tdy = htmlEl('td', 'gt-val', '-');
            tr.appendChild(tdx);
            tr.appendChild(tdy);
            self.cells[row.key] = [tdx, tdy];
        }
        table.appendChild(tr);
    });
    this.readoutBody.appendChild(table);
    this.readoutHeader = header;
};

/*
 * Properties of an optics, shown in the same panel as the beam readout.
 *
 * The fields the user thinks in are not always the traits the model
 * keeps: an angle is natural in degrees, and a mirror is specified by
 * its radius of curvature rather than by the inverse the code stores.
 * The conversion lives here, in the two functions below, so that the
 * edit messages still speak the model's language.
 */
var OPTIC_FIELDS = [
    {key: 'name', label: 'Name', text: true},
    {key: 'type', label: 'Type', readonly: true},
    // First of the numbers, because for a lens it is the one that
    // matters: the two radii below are how the model holds it, but the
    // focal length is what the lens is for. In millimetres, which is
    // how a catalogue lists one and how anyone speaks of it, unlike
    // everything else on a bench of this size. Only a lens has one, and
    // the row hides itself for anything else. 'inf' is refused rather
    // than sent: a lens with no power is a flat window, which is a
    // different element and not something to arrive at by typing.
    {key: 'f', label: 'Focal length', unit: 'mm', optional: true,
     finite: true},
    {key: 'cx', label: 'Center x', unit: 'm'},
    {key: 'cy', label: 'Center y', unit: 'm'},
    {key: 'angle', label: 'Angle', unit: '°'},
    // Once an element is square on a beam, the one placement left is
    // how far along that beam it sits, and that is a number rather
    // than something to find with a mouse: a lens goes where the mode
    // matching says. The two rows are one control - which beam, and
    // how far - and they hide together when no beam passes through the
    // element. The distance is relative and in millimetres: it is an
    // adjustment, and an adjustment on a bench is spoken of in mm.
    {key: 'slide_beam', label: 'Along beam', optional: true,
     choices: [], dynamicChoices: true},
    {key: 'slide_by', label: 'Move by', unit: 'mm', optional: true},
    // The size of the substrate, in millimetres: that is how a blank is
    // ordered and how anyone speaks of one, and a 1 inch mirror reading
    // 0.0254 is arithmetic rather than a specification. Where it stands
    // on the bench stays in metres, since that is a distance across the
    // table rather than a dimension of the part.
    {key: 'diameter', label: 'Diameter', unit: 'mm'},
    {key: 'thickness', label: 'Thickness', unit: 'mm'},
    {key: 'wedgeAngle', label: 'Wedge', unit: '°'},
    {key: 'rocHR', label: 'ROC HR', unit: 'm'},
    {key: 'rocAR', label: 'ROC AR', unit: 'm'},
    // The point the element is held by: what stays put when a
    // curvature changes, and what it turns about. A mirror pins its HR
    // face so the beam spot on it does not move; a lens pins its
    // middle, since the beam goes through.
    {key: 'anchor_point', label: 'Anchor', optional: true,
     choices: [['HRcenter', 'HR center'], ['center', 'substrate center']]},
    {key: 'n', label: 'Index n'},
    {key: 'Refl_HR', label: 'Refl HR'},
    {key: 'Trans_HR', label: 'Trans HR'},
    {key: 'Refl_AR', label: 'Refl AR'},
    {key: 'Trans_AR', label: 'Trans AR'},
    // How this element is to be traced. These belong to the element in
    // the same way its coatings do: see Optics.max_stray_order.
    {group: 'Tracing'},
    {key: 'max_stray_order', label: 'Max stray order', nullable: true},
    {key: 'HRtransmissive', label: 'HR transmissive', bool: true},
    {key: 'HRreflective', label: 'HR reflective', bool: true},
    {key: 'term_on_HR', label: 'Terminate on HR', bool: true},
    {key: 'term_on_HR_order', label: 'Term. on HR order'},
    // Only a CyMirror has this; the row hides itself otherwise. Two
    // values exist, so it is a choice rather than something to type.
    {key: 'curve_direction', label: 'Curve direction', optional: true,
     choices: [['h', 'horizontal'], ['v', 'vertical']]}
];

var DEG = 180 / Math.PI;
var MM = 0.001;

/*
 * The point an optics is held by: what stays put when its curvature
 * changes, and what it therefore turns about. ``anchor_point`` names it -
 * the apex of the front face for a mirror, the middle of the substrate
 * for a lens. An optics from a scene old enough not to carry one is
 * held by its front face, which is what a mirror does.
 */
function opticAnchorIsCenter(o) {
    return o.anchor_point === 'center' && !!o.center;
}

function opticAnchorPoint(o) {
    if (opticAnchorIsCenter(o)) { return o.center; }
    return o.HRcenter || o.center || [0, 0];
}

function opticFieldValue(o, key) {
    var c = o.center || o.HRcenter || [0, 0];
    switch (key) {
    case 'cx': return c[0];
    case 'cy': return c[1];
    case 'angle': return normAngle(o.normAngleHR || 0) * DEG;
    case 'wedgeAngle': return (o.wedgeAngle || 0) * DEG;
    // In millimetres. An element that does not carry the attribute at
    // all reads as absent rather than as a NaN.
    case 'diameter':
    case 'thickness':
        return o[key] === undefined || o[key] === null
            ? o[key] : o[key] / MM;
    case 'rocHR': return o.inv_ROC_HR ? 1 / o.inv_ROC_HR : Infinity;
    case 'rocAR': return o.inv_ROC_AR ? 1 / o.inv_ROC_AR : Infinity;
    case 'f':
        // Python sends the power, which is finite even for a substrate
        // with no power left in it; JSON could not have carried the
        // infinite focal length that one has. An element with no entry
        // at all is not a lens, and reads as absent so the row hides.
        if (o.inv_f === undefined || o.inv_f === null) { return undefined; }
        return o.inv_f ? 1 / (o.inv_f * MM) : Infinity;
    case 'max_stray_order':
        // An optics that does not carry the setting reads as unset,
        // which is the same thing as far as the panel is concerned.
        return o.max_stray_order === undefined ? null : o.max_stray_order;
    default: return o[key];
    }
}

/*
 * The edit message that sets one field of an optics.
 */
function opticFieldMessage(o, key, value) {
    var c = o.center || o.HRcenter || [0, 0];
    var attrs = {};
    switch (key) {
    case 'cx':
        return {op: 'move', target: o.name, center: [value, c[1]]};
    case 'cy':
        return {op: 'move', target: o.name, center: [c[0], value]};
    case 'angle':
        return {op: 'rotate', target: o.name, normAngleHR: value / DEG};
    case 'wedgeAngle':
        attrs.wedgeAngle = value / DEG;
        break;
    case 'diameter':
    case 'thickness':
        // The panel is in millimetres; the model, like everything else
        // in gtrace, is in metres.
        attrs[key] = value * MM;
        break;
    case 'rocHR':
        // A flat surface is an infinite radius, which is the inverse
        // being zero. Anything non-finite means flat.
        attrs.inv_ROC_HR = isFinite(value) && value !== 0 ? 1 / value : 0;
        break;
    case 'rocAR':
        attrs.inv_ROC_AR = isFinite(value) && value !== 0 ? 1 / value : 0;
        break;
    case 'f':
        // The panel is in millimetres; the model, like everything else
        // in gtrace, is in metres.
        attrs.f = value * MM;
        break;
    default:
        attrs[key] = value;
    }
    return {op: 'set', target: o.name, attrs: attrs};
}

/*
 * Render a number for an editable field: full precision, so that what
 * is read back is exactly what the model holds.
 */
function fmtField(v) {
    // A field that may be unset shows 'auto', meaning the layout-wide
    // value applies. Empty would read as "nothing here".
    if (v === null || v === undefined) { return 'auto'; }
    if (typeof v !== 'number') { return String(v); }
    if (!isFinite(v)) { return v > 0 ? 'inf' : '-inf'; }
    return String(v);
}

/*
 * Parse a field. Returns NaN for anything unusable, and null for the
 * spellings that mean "leave it to the layout".
 */
function parseField(s) {
    var t = String(s).trim().toLowerCase();
    if (t === 'inf' || t === 'infinity' || t === '∞') { return Infinity; }
    if (t === '-inf' || t === '-infinity' || t === '-∞') { return -Infinity; }
    if (t === '' || t === 'auto' || t === 'none' || t === '-') { return null; }
    var v = Number(t);
    return isNaN(v) ? NaN : v;
}

/*
 * Properties of a source beam.
 *
 * A laser is not specified by a q-parameter. It is specified by how
 * wide its waist is and where that waist sits, which is exactly the
 * pair GaussianBeam.waist() reports, so those are the rows - and Python
 * converts, because what a waist means is the model's to say. The scene
 * carries both, so nothing here has to work one out from the other.
 */
var NM = 1e-9;

var SOURCE_FIELDS = [
    {key: 'name', label: 'Name', text: true},
    {key: 'type', label: 'Type', readonly: true},
    {key: 'px', label: 'Position x', unit: 'm'},
    {key: 'py', label: 'Position y', unit: 'm'},
    {key: 'angle', label: 'Direction', unit: '°'},
    // The beam it puts out. In millimetres and nanometres: a waist is
    // spoken of in mm and a wavelength in nm, and a panel that made
    // either of them read 0.0002 would be arithmetic rather than a
    // specification.
    {group: 'Beam'},
    {key: 'w0x', label: 'Waist size x', unit: 'mm'},
    {key: 'w0y', label: 'Waist size y', unit: 'mm'},
    // Measured from the laser forward along the beam, the way
    // GaussianBeam.waist() reports it: positive is downstream, and a
    // waist behind the output is negative.
    {key: 'dx', label: 'Waist pos x', unit: 'm'},
    {key: 'dy', label: 'Waist pos y', unit: 'm'},
    {key: 'wl', label: 'Wavelength', unit: 'nm'},
    {key: 'P', label: 'Power', unit: 'W'},
    {key: 'n', label: 'Index n'},
    // Only used while the beam reaches nothing: the trace cuts a beam
    // at whatever it hits, so this is how far a source fires into an
    // empty bench. It is here because that is precisely the state a
    // layout is in while it is being built.
    {key: 'length', label: 'Free length', unit: 'm'}
];

function sourceFieldValue(s, key) {
    switch (key) {
    case 'type': return 'Source';
    case 'px': return s.pos[0];
    case 'py': return s.pos[1];
    case 'angle': return normAngle(s.dirAngle || 0) * DEG;
    case 'w0x': return s.waist_size[0] / MM;
    case 'w0y': return s.waist_size[1] / MM;
    case 'dx': return s.waist_pos[0];
    case 'dy': return s.waist_pos[1];
    case 'wl': return s.wl / NM;
    default: return s[key];
    }
}

/*
 * The edit message that sets one field of a source.
 */
function sourceFieldMessage(s, key, value) {
    var attrs = {};
    switch (key) {
    case 'px':
        return {op: 'move', target: s.name, pos: [value, s.pos[1]]};
    case 'py':
        return {op: 'move', target: s.name, pos: [s.pos[0], value]};
    case 'angle':
        return {op: 'rotate', target: s.name, dirAngle: value / DEG};
    case 'w0x': attrs.waist_size_x = value * MM; break;
    case 'w0y': attrs.waist_size_y = value * MM; break;
    case 'dx': attrs.waist_pos_x = value; break;
    case 'dy': attrs.waist_pos_y = value; break;
    case 'wl': attrs.wl = value * NM; break;
    default:
        attrs[key] = value;
    }
    return {op: 'set', target: s.name, attrs: attrs};
}

/*
 * Properties of a mechanics: its pose, and the labels that say what it
 * is. The shapes are not here - they are the body itself, drawn
 * through the canvas, and a front end moves the body rather than
 * redrawing it.
 */
var MECH_FIELDS = [
    {key: 'name', label: 'Name', text: true},
    {key: 'type', label: 'Type', readonly: true},
    // The catalogue model the shapes came from, when there is one. A
    // label rather than a link: the saved shapes are the truth.
    {key: 'model', label: 'Model', readonly: true, optional: true},
    // The optics this body stands on, or nothing. Editable as a
    // choice: picking an optics seats the mount on it at the model's
    // designed position - a mount is built around its optic, so where
    // it belongs on the host is the library's to say, not the drop
    // point's - and picking free detaches it where it is. While it
    // stands on something, the pose rows below are the host's doing
    // and are disabled. The choices are filled by _refreshMechPanel,
    // since they are the optics of the moment.
    {key: 'attached', label: 'Attached to', optional: true,
     choices: [], dynamicChoices: true},
    // Where it stands on that host, in the host's frame: the origin
    // seats at the substrate centre, x runs along the HR normal. The
    // rows show only while attached - a free body's place is its pose
    // - and are the adjustment an attached body still owns: the
    // model's designed position is the default, and these move the
    // body off it deliberately. In millimetres and degrees, like the
    // other adjustments.
    {key: 'ox', label: 'Offset x', unit: 'mm', optional: true},
    {key: 'oy', label: 'Offset y', unit: 'mm', optional: true},
    {key: 'oangle', label: 'Offset angle', unit: '°', optional: true},
    {key: 'cx', label: 'Center x', unit: 'm'},
    {key: 'cy', label: 'Center y', unit: 'm'},
    {key: 'angle', label: 'Angle', unit: '°'},
    // Only a parametric body - a breadboard - has a size to set; the
    // rows hide themselves for hardware drawn by hand, whose shapes
    // are all anyone knows about it. In millimetres, like every other
    // dimension of a part.
    {key: 'width', label: 'Width', unit: 'mm', optional: true},
    {key: 'height', label: 'Height', unit: 'mm', optional: true},
    {key: 'layer', label: 'Layer', readonly: true}
];

function mechFieldValue(m, key) {
    switch (key) {
    case 'type': return 'Mechanics';
    case 'model': return m.model === null ? undefined : m.model;
    case 'attached':
        return m.attached_to === null || m.attached_to === undefined
            ? undefined : m.attached_to;
    case 'cx': return m.center[0];
    case 'cy': return m.center[1];
    case 'angle': return normAngle(m.rotationAngle || 0) * DEG;
    case 'ox':
        return m.offset ? m.offset[0] / MM : undefined;
    case 'oy':
        return m.offset ? m.offset[1] / MM : undefined;
    case 'oangle':
        return m.offset_angle === null || m.offset_angle === undefined
            ? undefined : normAngle(m.offset_angle) * DEG;
    case 'width':
    case 'height':
        return m[key] === null || m[key] === undefined
            ? undefined : m[key] / MM;
    default: return m[key];
    }
}

/*
 * The edit message that sets one field of a mechanics.
 */
function mechFieldMessage(m, key, value) {
    switch (key) {
    case 'cx':
        return {op: 'move', target: m.name, center: [value, m.center[1]]};
    case 'cy':
        return {op: 'move', target: m.name, center: [m.center[0], value]};
    case 'angle':
        return {op: 'rotate', target: m.name, rotationAngle: value / DEG};
    case 'ox':
        return {op: 'set', target: m.name,
                attrs: {offset: [value * MM, m.offset[1]]}};
    case 'oy':
        return {op: 'set', target: m.name,
                attrs: {offset: [m.offset[0], value * MM]}};
    case 'oangle':
        return {op: 'set', target: m.name,
                attrs: {offset_angle: value / DEG}};
    case 'width':
    case 'height':
        // The panel is in millimetres; the model, as everywhere in
        // gtrace, in metres.
        var size = {};
        size[key] = value * MM;
        return {op: 'set', target: m.name, attrs: size};
    }
    var attrs = {};
    attrs[key] = value;
    return {op: 'set', target: m.name, attrs: attrs};
}

/*
 * The corners of a rectangle of a given pose, counterclockwise from
 * the lower left. What the resize handles sit on and the resize
 * preview is drawn from: a parametric body's outline is exactly this
 * rectangle, centred on its local origin.
 */
function rectCorners(center, w, h, angle) {
    var ca = Math.cos(angle), sa = Math.sin(angle);
    return [[-w / 2, -h / 2], [w / 2, -h / 2],
            [w / 2, h / 2], [-w / 2, h / 2]].map(function (p) {
        return [center[0] + p[0] * ca - p[1] * sa,
                center[1] + p[0] * sa + p[1] * ca];
    });
}

/*
 * How deep the trace goes. Not a property of any one element - the cap
 * an element may put on its own ghosts is in OPTIC_FIELDS - but of the
 * layout, and the pair anyone chasing stray light reaches for first.
 */
var RULE_FIELDS = [
    {key: 'order', label: 'Order'},
    {key: 'power_threshold', label: 'Power threshold'},
    // What a beam that reaches nothing is drawn as. It applies to the
    // beams the trace makes; a source with nothing in front of it uses
    // its own Free length instead.
    {key: 'open_beam_length', label: 'Open beam', unit: 'm'}
];

/*
 * A dimension in the panel: where its ends are, and what it comes to.
 *
 * The two ends are editable, so that a measurement placed by eye can be
 * given exact coordinates afterwards. What it comes to is not: the
 * length is the answer, not an input, and typing over it would only
 * raise the question of which end was supposed to move.
 */
var DIM_FIELDS = [
    {key: 'name', label: 'Name', text: true},
    {key: 'p1x', label: 'From x', unit: 'm'},
    {key: 'p1y', label: 'From y', unit: 'm'},
    {key: 'p2x', label: 'To x', unit: 'm'},
    {key: 'p2y', label: 'To y', unit: 'm'},
    // Where the line is drawn, not what is measured. In millimetres,
    // like the other rows that are an adjustment rather than a place:
    // it is nudged until the line clears whatever it was covering.
    {key: 'offset', label: 'Line offset', unit: 'mm'},
    {group: 'Measurement'},
    {key: 'length', label: 'Distance', readonly: true},
    {key: 'angle', label: 'Direction', readonly: true},
    // Only when the span runs inside a substrate, which is the one case
    // where an optical distance is a distance of anything. The rows hide
    // themselves otherwise; see Dimension.measure in layout.py.
    {key: 'inside', label: 'Inside', readonly: true, optional: true},
    {key: 'n', label: 'Index n', readonly: true, optional: true},
    {key: 'optical', label: 'Optical dist.', readonly: true, optional: true}
];

Viewer.prototype._buildDimPanel = function () {
    var self = this;
    var table = htmlEl('table');
    this.dimFields = {};

    DIM_FIELDS.forEach(function (f) {
        var tr = htmlEl('tr');
        if (f.group) {
            var th = htmlEl('td', 'gt-group', f.group);
            th.setAttribute('colspan', '2');
            tr.appendChild(th);
            table.appendChild(tr);
            return;
        }
        tr.appendChild(htmlEl('td', 'gt-key',
                              f.label + (f.unit ? ' [' + f.unit + ']' : '')));
        var td = htmlEl('td', 'gt-val');
        var rec = {row: tr, optional: !!f.optional};
        if (f.readonly || !self.onEdit) {
            var span = htmlEl('span', 'gt-static', '-');
            td.appendChild(span);
            rec.el = span;
            rec.editable = false;
        } else {
            var input = htmlEl('input', 'gt-input');
            input.type = 'text';
            if (f.text) { input.className += ' gt-input-text'; }
            input.spellcheck = false;
            input.addEventListener('change', function () {
                self._commitDimField(f.key, input);
            });
            input.addEventListener('keydown', function (ev) {
                if (ev.key === 'Escape') {
                    self._refreshDimPanel();
                    input.blur();
                    ev.stopPropagation();
                }
            });
            td.appendChild(input);
            rec.el = input;
            rec.editable = true;
        }
        tr.appendChild(td);
        table.appendChild(tr);
        self.dimFields[f.key] = rec;
    });

    this.dimBody.appendChild(table);

    // Built even where there is nothing to send: a viewer with no Python
    // behind it can still take back a measurement it drew itself.
    // _refreshDimPanel decides whether it is on offer.
    this.dimFoot = htmlEl('div', 'gt-props-foot');
    var delBtn = htmlEl('button', 'gt-btn gt-btn-danger', 'Remove');
    delBtn.title = 'Remove this dimension';
    delBtn.addEventListener('click', function () { self.removeSelected(); });
    this.dimFoot.appendChild(delBtn);
    this.dimBody.appendChild(this.dimFoot);
};

/*
 * Build a table of property rows from a field list.
 *
 * Shared by the optics panel and the source panel, which differ in what
 * they show and not at all in how a row works. The two callbacks are
 * what a row does: commit(key, element) when the value is entered, and
 * revert() when Escape says to put back what the model holds.
 *
 * Returns a map from field key to a record the refresh loop works
 * through: which row it is, what kind of control, and whether the row
 * hides itself when the thing on show has no such property.
 */
function buildFieldTable(fields, editable, commit, revert) {
    var table = htmlEl('table');
    var recs = {};

    fields.forEach(function (f) {
        var tr = htmlEl('tr');

        if (f.group) {
            var th = htmlEl('td', 'gt-group', f.group);
            th.setAttribute('colspan', '2');
            tr.appendChild(th);
            table.appendChild(tr);
            return;
        }

        tr.appendChild(htmlEl('td', 'gt-key',
                              f.label + (f.unit ? ' [' + f.unit + ']' : '')));
        var td = htmlEl('td', 'gt-val');
        var rec = {row: tr, optional: !!f.optional};

        if (f.readonly || !editable) {
            // Nothing to edit: either the class of the element, or a
            // viewer with no Python behind it.
            var span = htmlEl('span', 'gt-static', '-');
            td.appendChild(span);
            rec.el = span;
            rec.editable = false;
            rec.kind = f.bool ? 'bool' : 'static';
        } else if (f.bool) {
            var box = htmlEl('input', 'gt-check');
            box.type = 'checkbox';
            box.addEventListener('change', function () { commit(f.key, box); });
            td.appendChild(box);
            rec.el = box;
            rec.editable = true;
            rec.kind = 'bool';
        } else if (f.choices) {
            var sel = htmlEl('select', 'gt-select gt-select-prop');
            f.choices.forEach(function (c) {
                var opt = htmlEl('option', null, c[1]);
                opt.value = c[0];
                sel.appendChild(opt);
            });
            sel.addEventListener('change', function () { commit(f.key, sel); });
            td.appendChild(sel);
            rec.el = sel;
            rec.editable = true;
            rec.kind = 'choice';
        } else {
            var input = htmlEl('input', 'gt-input');
            input.type = 'text';
            if (f.text) { input.className += ' gt-input-text'; }
            input.spellcheck = false;
            input.addEventListener('change', function () {
                commit(f.key, input);
            });
            input.addEventListener('keydown', function (ev) {
                if (ev.key === 'Escape') {
                    revert();
                    input.blur();
                    ev.stopPropagation();
                }
            });
            td.appendChild(input);
            rec.el = input;
            rec.editable = true;
            rec.kind = f.text ? 'text' : 'num';
        }
        tr.appendChild(td);
        table.appendChild(tr);
        recs[f.key] = rec;
    });

    return {table: table, fields: recs};
}

/*
 * Fill a table built by buildFieldTable from whatever it is showing.
 *
 * `value(key)` answers for the thing on show, or the caller passes null
 * for "nothing selected" and every row empties. A row marked optional
 * disappears entirely when its value is absent, which is how a panel
 * shared by several classes shows only what the one in front of it has.
 */
function refreshFieldTable(recs, fields, value, skip) {
    for (var key in recs) {
        if (skip && skip.indexOf(key) >= 0) { continue; }
        var f = recs[key];
        var v = value ? value(key) : null;

        if (f.optional) {
            f.row.style.display = (v === undefined || v === null) ? 'none' : '';
        }

        // Never overwrite the field the user is working in.
        if (f.editable && document.activeElement === f.el) { continue; }

        if (f.kind === 'bool') {
            if (f.editable) { f.el.checked = !!v; }
            else { f.el.textContent = value ? (v ? 'yes' : 'no') : '-'; }
        } else if (f.kind === 'choice') {
            f.el.value = v === undefined || v === null ? '' : String(v);
        } else if (f.editable) {
            f.el.value = value ? fmtField(v) : '';
        } else {
            f.el.textContent = value ? fmtField(v) : '-';
        }
    }
}

Viewer.prototype._buildOpticPanel = function () {
    var self = this;
    var built = buildFieldTable(
        OPTIC_FIELDS, !!this.onEdit,
        function (key, el) { self._commitOpticField(key, el); },
        function () { self._refreshOpticPanel(); });
    this.opticFields = built.fields;
    this.opticBody.appendChild(built.table);

    if (this.onEdit) {
        var foot = htmlEl('div', 'gt-props-foot');
        var delBtn = htmlEl('button', 'gt-btn gt-btn-danger', 'Remove');
        delBtn.title = 'Remove this optics from the layout';
        delBtn.addEventListener('click', function () { self.removeSelected(); });
        foot.appendChild(delBtn);
        this.opticBody.appendChild(foot);
    }
};

/*
 * The source panel. The same rows as any other, over a different list.
 */
Viewer.prototype._buildSourcePanel = function () {
    var self = this;
    var built = buildFieldTable(
        SOURCE_FIELDS, !!this.onEdit,
        function (key, el) { self._commitSourceField(key, el); },
        function () { self._refreshSourcePanel(); });
    this.sourceFields = built.fields;
    this.sourceBody.appendChild(built.table);

    if (this.onEdit) {
        var foot = htmlEl('div', 'gt-props-foot');
        var delBtn = htmlEl('button', 'gt-btn gt-btn-danger', 'Remove');
        delBtn.title = 'Remove this source from the layout';
        delBtn.addEventListener('click', function () { self.removeSelected(); });
        foot.appendChild(delBtn);
        this.sourceBody.appendChild(foot);
    }
};

/*
 * The mechanics panel. The same rows again, over the hardware.
 */
Viewer.prototype._buildMechPanel = function () {
    var self = this;
    var built = buildFieldTable(
        MECH_FIELDS, !!this.onEdit,
        function (key, el) { self._commitMechField(key, el); },
        function () { self._refreshMechPanel(); });
    this.mechFields = built.fields;
    this.mechBody.appendChild(built.table);

    if (this.onEdit) {
        var foot = htmlEl('div', 'gt-props-foot');
        var delBtn = htmlEl('button', 'gt-btn gt-btn-danger', 'Remove');
        delBtn.title = 'Remove this mechanics from the layout';
        delBtn.addEventListener('click', function () { self.removeSelected(); });
        foot.appendChild(delBtn);
        this.mechBody.appendChild(foot);
    }
};

/*
 * The shape panel: the list of what the part is drawn from, and the
 * numbers of whichever one is selected.
 *
 * The list is the part - the order is the order they are drawn in -
 * so it is where a shape is picked, moved earlier or later, copied
 * and taken away. The rows below it are rebuilt whenever the
 * selection changes kind, since a circle and a rectangle have
 * nothing in common to keep.
 */
Viewer.prototype._buildShapePanel = function () {
    var self = this;

    this.shapeList = htmlEl('div', 'gt-shapelist');
    this.shapeBody.appendChild(this.shapeList);

    this.shapeRows = htmlEl('div', 'gt-shaperows');
    this.shapeBody.appendChild(this.shapeRows);

    if (this.onEdit) {
        var foot = htmlEl('div', 'gt-props-foot');
        [['Copy', 'Add a second one just beside it',
          function () { self.duplicateShape(); }],
         ['↑', 'Draw it earlier, under the others',
          function () { self.moveShape(-1); }],
         ['↓', 'Draw it later, over the others',
          function () { self.moveShape(1); }],
         ['Remove', 'Take this shape out of the part',
          function () { self.removeShape(); }]
        ].forEach(function (spec, i) {
            var btn = htmlEl('button',
                             i === 3 ? 'gt-btn gt-btn-danger' : 'gt-btn',
                             spec[0]);
            btn.title = spec[1];
            btn.addEventListener('click', spec[2]);
            foot.appendChild(btn);
        });
        this.shapeBody.appendChild(foot);
    }
    this._refreshShapePanel();
};

/*
 * The shapes of the part being edited, or an empty list.
 */
Viewer.prototype._shapes = function () {
    return this.scene.shapes || [];
};

Viewer.prototype._selectedShape = function () {
    var shapes = this._shapes();
    if (this.selectedShape === null || this.selectedShape === undefined) {
        return null;
    }
    return shapes[this.selectedShape] || null;
};

Viewer.prototype._selectShape = function (index) {
    this.selectedShape = index;
    this._refreshShapePanel();
    this._updateOverlay();
};

Viewer.prototype._refreshShapePanel = function () {
    var self = this;
    if (!this.shapeList) { return; }
    var shapes = this._shapes();

    // The list. Rebuilt outright: it is a dozen rows at most, and an
    // index means a place rather than a thing, so keeping rows across
    // an edit would only invite them to point at the wrong shape.
    this.shapeList.textContent = '';
    if (!shapes.length) {
        this.shapeList.appendChild(
            htmlEl('div', 'gt-note', 'No shapes yet - add one above.'));
    }
    shapes.forEach(function (s, i) {
        var row = htmlEl('button', 'gt-shaperow', (i + 1) + '.  ' + s.type);
        row.classList.toggle('gt-selected', i === self.selectedShape);
        row.addEventListener('click', function () { self._selectShape(i); });
        self.shapeList.appendChild(row);
    });

    // The numbers. A new table whenever the kind changes, since the
    // rows themselves are different.
    var s = this._selectedShape();
    var kind = s ? s.type : null;
    if (kind !== this._shapeFieldsKind) {
        this.shapeRows.textContent = '';
        this.shapeFields = null;
        this._shapeFieldsKind = kind;
        if (kind && SHAPE_FIELDS[kind]) {
            var built = buildFieldTable(
                SHAPE_FIELDS[kind], !!this.onEdit,
                function (key, el) { self._commitShapeField(key, el); },
                function () { self._refreshShapePanel(); });
            this.shapeFields = built.fields;
            this.shapeRows.appendChild(built.table);
        }
    }
    if (this.shapeFields) {
        refreshFieldTable(this.shapeFields, SHAPE_FIELDS[kind],
                          s ? function (key) {
                              return shapeFieldValue(s, key);
                          } : null);
    }
};

Viewer.prototype._commitShapeField = function (key, input) {
    var s = this._selectedShape();
    if (!s || !this.onEdit) { return; }
    var field = null;
    (SHAPE_FIELDS[s.type] || []).forEach(function (f) {
        if (f.key === key) { field = f; }
    });

    var value;
    if (field && field.text) {
        value = String(input.value);
        if (!value.trim()) { this._refreshShapePanel(); return; }
    } else {
        value = parseField(input.value);
        // Every number a shape takes is finite: a drawing with an
        // infinity in it takes the whole view with it the first time
        // anything is framed.
        if (typeof value !== 'number' || !isFinite(value)) {
            this._refreshShapePanel();
            return;
        }
        if (value === shapeFieldValue(s, key)) { return; }
    }
    this.onEdit({op: 'set_shape', index: this.selectedShape,
                 attrs: shapeFieldAttrs(s, key, value)});
};

/*
 * Put a new shape down at the origin, and select it: what was just
 * asked for is what the panel should be showing.
 */
Viewer.prototype.addShape = function (type) {
    if (!this.onEdit) { return null; }
    var msg = {op: 'add_shape', type: type};
    this.selectedShape = this._shapes().length;
    this.onEdit(msg);
    return msg;
};

Viewer.prototype.removeShape = function () {
    var s = this._selectedShape();
    if (!s || !this.onEdit) { return null; }
    var msg = {op: 'remove_shape', index: this.selectedShape};
    // The list closes up, so the selection follows what is left.
    if (this.selectedShape >= this._shapes().length - 1) {
        this.selectedShape = this._shapes().length - 2;
    }
    if (this.selectedShape < 0) { this.selectedShape = null; }
    this.onEdit(msg);
    return msg;
};

Viewer.prototype.duplicateShape = function () {
    var s = this._selectedShape();
    if (!s || !this.onEdit) { return null; }
    var msg = {op: 'duplicate_shape', index: this.selectedShape};
    // The copy lands just after the original, and is what the panel
    // then shows: it is the one about to be moved somewhere.
    this.selectedShape = this.selectedShape + 1;
    this.onEdit(msg);
    return msg;
};

/*
 * Draw this shape earlier or later than its neighbours - which is
 * what puts one over another where they overlap.
 */
Viewer.prototype.moveShape = function (by) {
    var s = this._selectedShape();
    if (!s || !this.onEdit) { return null; }
    var to = this.selectedShape + by;
    if (to < 0 || to >= this._shapes().length) { return null; }
    var msg = {op: 'move_shape', index: this.selectedShape, to: to};
    this.selectedShape = to;
    this.onEdit(msg);
    return msg;
};

/*
 * Register the part under the name in the panel.
 */
Viewer.prototype.saveModel = function (name, description) {
    if (!this.onEdit) { return null; }
    name = (name || (this.modelInput && this.modelInput.value) || '').trim();
    if (!name) { return null; }
    if (description === undefined) {
        description = (this.modelDesc && this.modelDesc.value) || '';
    }
    var msg = {op: 'save_model', name: name, description: description};
    this.onEdit(msg);
    return msg;
};

/*
 * Add an optics at the centre of the current view.
 *
 * The name is chosen here rather than by Python, so that the viewer can
 * select the new element as soon as the scene comes back without
 * needing a reply channel. Everything else is left to the layout, which
 * fills the gaps from the optics already registered.
 */
Viewer.prototype.addOptics = function (type, params) {
    if (!this.onEdit) { return null; }
    var spec = null;
    for (var i = 0; i < ADDABLE_TYPES.length; i++) {
        if (ADDABLE_TYPES[i].type === type) { spec = ADDABLE_TYPES[i]; }
    }
    if (!spec) { return null; }

    var name = this._freshOpticName(spec.prefix);
    // A source stands at a point and fires along a direction; an optics
    // stands with a face turned. The default pose differs to match:
    // a new mirror faces back down the -x axis, where the beams
    // already in a layout tend to come from, and a new laser fires
    // along +x, which is where the rest of a bench is built from.
    var pose = spec.source
        ? {pos: [this.cx, this.cy], dirAngle: 0}
        : {HRcenter: [this.cx, this.cy], normAngleHR: Math.PI};
    var msg = {op: 'add', type: spec.type, name: name,
               params: Object.assign(pose, spec.params || {}, params || {})};
    // Optimistic: the scene that comes back will contain it, and the
    // selection resolves the name then.
    if (spec.source) {
        this.selectedSource = name;
        this.selectedOptic = null;
    } else {
        this.selectedOptic = name;
        this.selectedSource = null;
    }
    this.onEdit(msg);
    return msg;
};

/*
 * Kept for callers that only ever wanted the ordinary kind.
 */
Viewer.prototype.addMirror = function (params) {
    return this.addOptics('Mirror', params);
};

/*
 * Fill the + Hardware menu from the library the scene carries. The
 * shapes stay on the Python side; the menu deals in names, and the
 * layout builds the body when one is chosen.
 */
Viewer.prototype._refreshHardwareMenu = function () {
    var self = this;
    var hm = this.hardwareMenu;
    if (!hm) { return; }
    var lib = this.scene.mechlib || [];
    hm.wrap.style.display = lib.length ? '' : 'none';
    hm.menu.textContent = '';
    lib.forEach(function (entry) {
        var item = htmlEl('button', 'gt-menuitem', entry.name);
        item.title = entry.description || '';
        item.addEventListener('click', function () {
            self.closeAddMenus();
            self.addHardware(entry.name);
        });
        hm.menu.appendChild(item);
    });
};

/*
 * Add a library model at the centre of the current view. The name is
 * chosen here, like a new optics' name, so the viewer can select what
 * it asked for as soon as the scene comes back.
 */
Viewer.prototype.addHardware = function (model) {
    if (!this.onEdit) { return null; }
    var name = this._freshOpticName('H');
    var msg = {op: 'add', type: 'Mechanics', name: name,
               params: {model: model, center: [this.cx, this.cy]}};
    this.selectedMech = name;
    this.selectedOptic = null;
    this.selectedSource = null;
    this.selectedDim = null;
    this.onEdit(msg);
    return msg;
};

Viewer.prototype._freshOpticName = function (prefix) {
    prefix = prefix || 'M';
    // Optics, sources and dimensions share one namespace, so a name is
    // only free if none of the three has it. Python would refuse a
    // clash anyway; asking for one and having it turned down would
    // leave the optimistic selection pointing at nothing.
    var taken = {};
    [this.scene.optics, this.scene.sources, this.scene.dimensions,
     this.scene.mechanics]
        .forEach(function (list) {
            (list || []).forEach(function (o) { taken[o.name] = true; });
        });
    var i = 1;
    while (taken[prefix + i]) { i++; }
    return prefix + i;
};

/*
 * Write the layout to the file named in the panel.
 */
Viewer.prototype.saveLayout = function (path) {
    if (!this.onEdit) { return null; }
    path = (path || (this.pathInput && this.pathInput.value) || '').trim();
    if (!path) { return null; }
    var msg = {op: 'save', path: path};
    this.onEdit(msg);
    return msg;
};

/*
 * Write the drawing to the DXF file named in its own panel, for
 * whatever comes after gtrace in an engineering workflow.
 *
 * Python does the drawing and the writing; the page only says where.
 */
Viewer.prototype.exportDXF = function (path) {
    if (!this.onEdit) { return null; }
    path = (path || (this.dxfInput && this.dxfInput.value) || '').trim();
    if (!path) { return null; }
    // The field is the user's, so an extension they typed is left
    // alone; one they did not type is filled in, since the panel this
    // field sits in has already said what kind of file it is.
    var msg = {op: 'export', format: 'dxf',
               path: withDefaultExtension(path, '.dxf')};
    this.onEdit(msg);
    return msg;
};

/*
 * Split a path into the part before the last separator and the part
 * after it, with the index of the extension's dot in that last part, or
 * -1 when it has none. A leading dot is a hidden file, not an
 * extension.
 */
function splitExtension(path) {
    var cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
    var name = path.slice(cut + 1);
    var dot = name.lastIndexOf('.');
    return {head: path.slice(0, cut + 1), name: name,
            dot: dot > 0 ? dot : -1};
}

/*
 * A path with its extension replaced, or given one if it had none.
 * Used to suggest a drawing's name from the layout's.
 */
function withExtension(path, ext) {
    var p = splitExtension(path);
    if (p.dot < 0) { return path + ext; }
    return p.head + p.name.slice(0, p.dot) + ext;
}

/*
 * A path with an extension added only if it has none. Used where the
 * user typed the name themselves: replacing what they wrote would be
 * presumptuous, and leaving a name with no extension at all would not
 * be a DXF file to anything that opens one.
 */
function withDefaultExtension(path, ext) {
    return splitExtension(path).dot < 0 ? path + ext : path;
}

/*
 * Replace the layout with the one in that file.
 *
 * What comes back is a whole new scene which may be somewhere else
 * entirely, so the view is fitted to it rather than left where the
 * previous layout happened to be.
 */
Viewer.prototype.loadLayout = function (path) {
    if (!this.onEdit) { return null; }
    path = (path || (this.pathInput && this.pathInput.value) || '').trim();
    if (!path) { return null; }
    this.selectedOptic = null;
    this.pinned = null;
    this.fitOnNextScene = true;
    var msg = {op: 'load', path: path};
    this.onEdit(msg);
    return msg;
};

/*
 * Ask Python to put the layout back as it was before the last edit.
 *
 * The selection is left alone: undoing a move means looking at the same
 * element again, in its old place. If the element was one the undone
 * edit had removed, it comes back and the name resolves to it again;
 * if it was one the undone edit had added, it goes, and the panel falls
 * back to the readout when the scene arrives without it.
 */
Viewer.prototype.undo = function () {
    if (!this.onEdit || !this.scene.can_undo) { return null; }
    var msg = {op: 'undo'};
    this.onEdit(msg);
    return msg;
};

/*
 * Ask Python to put back the state the last undo stepped out of. The
 * selection is left alone for the same reasons as in undo().
 */
Viewer.prototype.redo = function () {
    if (!this.onEdit || !this.scene.can_redo) { return null; }
    var msg = {op: 'redo'};
    this.onEdit(msg);
    return msg;
};

Viewer.prototype._refreshUndo = function () {
    if (this.undoBtn) { this.undoBtn.disabled = !this.scene.can_undo; }
    if (this.redoBtn) { this.redoBtn.disabled = !this.scene.can_redo; }
};

/*
 * Remove whatever the panel is showing. One button serves both kinds
 * because the message is the same: 'remove' names its target, and the
 * layout resolves it across optics and dimensions alike.
 */
Viewer.prototype.removeSelected = function () {
    var target = this.panelKind === 'dimension' ? this.selectedDim
        : this.panelKind === 'source' ? this.selectedSource
        : this.panelKind === 'mech' ? this.selectedMech
        : this.selectedOptic;
    if (!target) { return null; }

    // A dimension the viewer drew itself is the viewer's to take back.
    // Anything else needs Python, which owns the layout: a read-only
    // viewer must not appear to change what it was handed.
    var local = this.panelKind === 'dimension' && this._selectedDim()
        && this._selectedDim().local;
    if (!this.onEdit && !local) { return null; }

    var msg = {op: 'remove', target: target};
    this.selectedOptic = null;
    this.selectedDim = null;
    this.selectedSource = null;
    this.selectedMech = null;
    this._showPanel('beam');
    if (local && !this.onEdit) {
        this.scene.dimensions = this.scene.dimensions.filter(
            function (d) { return d.name !== target; });
        this._renderDimensions();
        this._updateDimensions();
        this._updateOverlay();
        return msg;
    }
    this.onEdit(msg);
    return msg;
};

/*
 * The display controls, unlike everything else in the side bar, do not
 * describe the scene: they ask Python to draw it differently. One
 * envelope width is offered for both beam kinds - two would suggest the
 * two envelopes mean different things, which is exactly what we spent
 * an earlier change getting rid of.
 */
Viewer.prototype._commitDisplay = function (key, value) {
    if (!this.onEdit) { return null; }
    var params = {};
    if (key === 'sigma') {
        params.sigma_main = Number(value);
        params.sigma_stray = Number(value);
    } else {
        params[key] = value;
    }
    var msg = {op: 'draw', params: params};
    this.onEdit(msg);
    return msg;
};

/*
 * The tracing rules, which describe the layout rather than the drawing:
 * changing one re-traces, so the picture that comes back has more or
 * fewer beams in it than the one that went out.
 */
Viewer.prototype._refreshRulesPanel = function () {
    if (!this.ruleFields) { return; }
    var r = this.scene.rules;
    refreshFieldTable(this.ruleFields, RULE_FIELDS,
                      r ? function (key) { return r[key]; } : null);
};

Viewer.prototype._commitRuleField = function (key, input) {
    var r = this.scene.rules;
    if (!r || !this.onEdit) { return; }
    var value = parseField(input.value);
    if (typeof value !== 'number' || !isFinite(value)) {
        this._refreshRulesPanel();
        return;
    }
    if (value === r[key]) { return; }
    var rules = {};
    rules[key] = value;
    this.onEdit({op: 'rules', rules: rules});
};

/*
 * Put the controls where the scene says the drawing actually stands.
 */
Viewer.prototype._refreshDisplayPanel = function () {
    if (!this.displayControls) { return; }
    var d = this.scene.display || {};
    if (d.sigma_main !== undefined) {
        this.displayControls.sigma.value = String(d.sigma_main);
    }
    if (d.width_mode !== undefined) {
        this.displayControls.width_mode.value = d.width_mode;
    }
};

/*
 * Show one of the two panels.
 */
var PANEL_TITLES = {optic: 'Optics properties', dimension: 'Dimension',
                    source: 'Source properties',
                    mech: 'Mechanics properties', shape: 'Shapes',
                    beam: 'Beam readout'};

Viewer.prototype._showPanel = function (kind) {
    this.panelKind = kind;
    this.readoutBody.style.display = kind === 'beam' ? '' : 'none';
    this.opticBody.style.display = kind === 'optic' ? '' : 'none';
    this.dimBody.style.display = kind === 'dimension' ? '' : 'none';
    this.sourceBody.style.display = kind === 'source' ? '' : 'none';
    this.mechBody.style.display = kind === 'mech' ? '' : 'none';
    this.shapeBody.style.display = kind === 'shape' ? '' : 'none';
    this.panelTitle.textContent = PANEL_TITLES[kind] || PANEL_TITLES.beam;
    if (kind !== 'beam') { this.pinLabel.textContent = ''; }
};

Viewer.prototype._selectedOptic = function () {
    if (!this.selectedOptic) { return null; }
    var optics = this.scene.optics || [];
    for (var i = 0; i < optics.length; i++) {
        if (optics[i].name === this.selectedOptic) { return optics[i]; }
    }
    return null;
};

/*
 * The beams that pass through an optics, nearest to the point it is
 * held by first. The same reach a Ctrl-drag snaps over, so what can be
 * slid along is what could have been dropped onto.
 */
Viewer.prototype._beamsThrough = function (o) {
    if (!o) { return []; }
    var c = opticAnchorPoint(o);
    var rad = opticRadius(o);
    return this._pickAll(c[0], c[1], rad).filter(function (h) {
        // A beam that both begins and ends inside the substrate is one
        // of the element's own internal reflections. It passes through
        // the element in the arithmetic sense, but there is no axis
        // there to move an element along, and no part of it outside the
        // element to point at either.
        var b = h.beam;
        return Math.max(Math.hypot(b.pos[0] - c[0], b.pos[1] - c[1]),
                        Math.hypot(b.end[0] - c[0], b.end[1] - c[1])) > rad;
    });
};

/*
 * Fill the beam picker from the beams currently through the element,
 * keeping the one already chosen if it is still among them.
 *
 * Beams are offered by index, since two of them can share a name, but
 * remembered by name: an index is only meaningful against one trace,
 * and every edit produces a new one.
 */
Viewer.prototype._refreshSlideBeams = function (o) {
    var hits = this._beamsThrough(o);
    var sel = this.opticFields.slide_beam.el;
    var seen = {}, i;
    while (sel.firstChild) { sel.removeChild(sel.firstChild); }
    for (i = 0; i < hits.length; i++) {
        var name = hits[i].beam.name;
        seen[name] = (seen[name] || 0) + 1;
        var opt = htmlEl('option', null,
                         seen[name] > 1 ? name + ' (' + seen[name] + ')'
                                        : name);
        opt.value = String(hits[i].index);
        sel.appendChild(opt);
    }
    if (!hits.length) { this.slideBeam = null; return hits; }

    // The exact beam first, then any of that name. Two beams can share
    // one, so matching on the name alone would undo a choice made by
    // clicking through the bundle; but an index is only meaningful
    // against one trace, so the name is what carries the choice across
    // the re-trace that follows every edit.
    var want = this.slideBeam;
    var chosen = -1;
    for (i = 0; want && i < hits.length; i++) {
        if (hits[i].index === want.index && hits[i].beam.name === want.name) {
            chosen = i;
            break;
        }
    }
    for (i = 0; want && chosen < 0 && i < hits.length; i++) {
        if (hits[i].beam.name === want.name) { chosen = i; }
    }
    if (chosen < 0) { chosen = 0; }
    this.slideBeam = {name: hits[chosen].beam.name, index: hits[chosen].index};
    sel.value = String(hits[chosen].index);
    return hits;
};

/*
 * Name one of the beams under the cursor in the Along beam row.
 *
 * Only a beam that passes through the selected optics can be chosen,
 * which is the same set the row offers: sliding along a beam that
 * misses the element is not a thing to mean. A click on any other beam
 * is left to fall through to the readout, as an unmodified click would.
 *
 * The hits come in nearest-first, so the deeper ones are considered
 * only when the nearest is not one of the element's own.
 */
Viewer.prototype._chooseSlideBeam = function (hits, px, py) {
    var o = this._selectedOptic();
    if (!o || !this.onEdit) { return false; }
    var through = {};
    this._beamsThrough(o).forEach(function (h) { through[h.index] = true; });
    var bundle = hits.filter(function (h) { return through[h.index]; });
    if (!bundle.length) { return false; }

    // Clicking the same place again steps to the next beam of the
    // bundle, as it does for the readout. Beams routinely lie on top of
    // one another - a beam and its return share a line, and a stray
    // often runs along a main beam - so pointing cannot tell them
    // apart, and the one meant is frequently not the nearest.
    var same = this.lastSlideClick
        && Math.abs(px - this.lastSlideClick[0]) < 5
        && Math.abs(py - this.lastSlideClick[1]) < 5;
    this.slideCycle = same ? (this.slideCycle + 1) % bundle.length : 0;
    this.lastSlideClick = [px, py];

    var pick = bundle[this.slideCycle];
    this.slideBeam = {name: pick.beam.name, index: pick.index};
    return true;
};

/*
 * Move the selected optics along the chosen beam. The distance is in
 * millimetres, as the panel shows it, and positive downstream.
 */
Viewer.prototype.slideSelected = function (mm) {
    var o = this._selectedOptic();
    if (!o || !this.onEdit || !this.slideBeam) { return null; }
    if (typeof mm !== 'number' || !isFinite(mm) || mm === 0) { return null; }
    var msg = {op: 'slide', target: o.name, beam: this.slideBeam.name,
               beam_index: this.slideBeam.index, distance: mm * MM};
    this.onEdit(msg);
    return msg;
};

Viewer.prototype._refreshOpticPanel = function () {
    var o = this._selectedOptic();
    var fields = this.opticFields;

    // The two beam rows describe neither the optics nor anything in the
    // scene: which beam to slide along is a choice made here, and the
    // distance is an instruction rather than a value. They are filled
    // in first, and skipped by the loop below.
    if (fields.slide_beam) {
        var hits = this.onEdit ? this._refreshSlideBeams(o) : [];
        var shown = hits.length > 0;
        fields.slide_beam.row.style.display = shown ? '' : 'none';
        fields.slide_by.row.style.display = shown ? '' : 'none';
        if (fields.slide_by.editable
                && document.activeElement !== fields.slide_by.el) {
            fields.slide_by.el.value = '0';
        }
    }

    // A row for something this class does not have - the curvature
    // direction of a plain mirror - is not shown at all.
    refreshFieldTable(fields, OPTIC_FIELDS,
                      o ? function (key) { return opticFieldValue(o, key); }
                        : null,
                      ['slide_beam', 'slide_by']);
};

Viewer.prototype._selectedSource = function () {
    if (!this.selectedSource) { return null; }
    var sources = this.scene.sources || [];
    for (var i = 0; i < sources.length; i++) {
        if (sources[i].name === this.selectedSource) { return sources[i]; }
    }
    return null;
};

Viewer.prototype._refreshSourcePanel = function () {
    var s = this._selectedSource();
    refreshFieldTable(this.sourceFields, SOURCE_FIELDS,
                      s ? function (key) { return sourceFieldValue(s, key); }
                        : null);
};

Viewer.prototype._selectSource = function (source) {
    this.selectedSource = source ? source.name : null;
    if (source) {
        this.selectedOptic = null;
        this.selectedDim = null;
        this.selectedMech = null;
        this._refreshSourcePanel();
        this._showPanel('source');
    } else {
        this._showPanel('beam');
    }
    this._updateOverlay();
};

Viewer.prototype._commitSourceField = function (key, input) {
    var s = this._selectedSource();
    if (!s || !this.onEdit) { return; }

    if (key === 'name') {
        this.renameSelected(input.value);
        return;
    }

    var field = null;
    for (var i = 0; i < SOURCE_FIELDS.length; i++) {
        if (SOURCE_FIELDS[i].key === key) { field = SOURCE_FIELDS[i]; }
    }

    if (field && field.text) {
        var text = String(input.value).trim();
        if (!text || text === sourceFieldValue(s, key)) {
            this._refreshSourcePanel();
            return;
        }
        this.onEdit(sourceFieldMessage(s, key, text));
        return;
    }

    var value = parseField(input.value);
    // Every number a source takes is finite. An infinite waist or
    // wavelength would not survive the trip - JSON has no infinity, and
    // what arrives on the Python side is a null the model cannot use -
    // and nothing here means "leave it to the layout" either.
    if (typeof value !== 'number' || !isFinite(value)) {
        this._refreshSourcePanel();
        return;
    }
    if (value === sourceFieldValue(s, key)) { return; }
    this.onEdit(sourceFieldMessage(s, key, value));
};

Viewer.prototype._selectedMech = function () {
    if (!this.selectedMech) { return null; }
    var mechs = this.scene.mechanics || [];
    for (var i = 0; i < mechs.length; i++) {
        if (mechs[i].name === this.selectedMech) { return mechs[i]; }
    }
    return null;
};

Viewer.prototype._refreshMechPanel = function () {
    var m = this._selectedMech();
    refreshFieldTable(this.mechFields, MECH_FIELDS,
                      m ? function (key) { return mechFieldValue(m, key); }
                        : null,
                      ['attached']);

    // The Attached to row is filled by hand: its choices are the
    // optics of the moment, and what "no value" means differs by
    // viewer. Editable, the row is always on offer - a free mount is
    // exactly the one with an attachment to make - with free as one
    // of the choices. Read-only, a free body simply has no row.
    var af = this.mechFields.attached;
    if (af) {
        var host = (m && m.attached_to) || '';
        if (af.editable) {
            af.row.style.display = m ? '' : 'none';
            while (af.el.firstChild) { af.el.removeChild(af.el.firstChild); }
            var free = htmlEl('option', null, '(free)');
            free.value = '';
            af.el.appendChild(free);
            (this.scene.optics || []).forEach(function (o) {
                var opt = htmlEl('option', null, o.name);
                opt.value = o.name;
                af.el.appendChild(opt);
            });
            af.el.value = host;
            af.el.title = host
                ? 'Standing on ' + host + ' - pick (free) to detach it '
                  + 'where it is'
                : 'Pick an optics to seat this body on, at its designed '
                  + 'position';
        } else {
            af.row.style.display = (m && host) ? '' : 'none';
            af.el.textContent = host || '-';
        }
    }

    // An attached body has no pose of its own: the rows show where the
    // host put it, and refuse the keyboard rather than letting a value
    // be typed only for Python to turn it down.
    var attached = !!(m && m.attached_to);
    var self = this;
    ['cx', 'cy', 'angle'].forEach(function (key) {
        var f = self.mechFields[key];
        if (!f || !f.editable) { return; }
        f.el.disabled = attached;
        f.el.title = attached
            ? 'Attached to ' + m.attached_to + ' - move the optics instead'
            : '';
    });
};

Viewer.prototype._selectMech = function (mech) {
    this.selectedMech = mech ? mech.name : null;
    if (mech) {
        this.selectedOptic = null;
        this.selectedDim = null;
        this.selectedSource = null;
        this._refreshMechPanel();
        this._showPanel('mech');
    } else {
        this._showPanel('beam');
    }
    this._updateOverlay();
};

Viewer.prototype._commitMechField = function (key, input) {
    var m = this._selectedMech();
    if (!m || !this.onEdit) { return; }

    if (key === 'name') {
        this.renameSelected(input.value);
        return;
    }

    // The attachment: an optics name to seat the body on, or empty
    // for free. Attaching moves the body to the model's designed
    // position on the host - Python owns that, through the local
    // origin convention - and detaching frees it where it is.
    if (key === 'attached') {
        var current = m.attached_to || '';
        if (input.value === current) { return; }
        this.onEdit({op: 'set', target: m.name,
                     attrs: {attached_to: input.value === ''
                             ? null : input.value}});
        return;
    }

    var value = parseField(input.value);
    // Every number a mechanics takes is finite: a pose has no
    // 'infinity' and no 'leave it to the layout'.
    if (typeof value !== 'number' || !isFinite(value)) {
        this._refreshMechPanel();
        return;
    }
    if (value === mechFieldValue(m, key)) { return; }
    this.onEdit(mechFieldMessage(m, key, value));
};

Viewer.prototype._selectOptic = function (optic) {
    this.selectedOptic = optic ? optic.name : null;
    if (optic) {
        this.selectedMech = null;
        this._refreshOpticPanel();
        this._showPanel('optic');
    } else {
        this._showPanel('beam');
    }
    this._updateOverlay();
};

Viewer.prototype._commitOpticField = function (key, input) {
    var o = this._selectedOptic();
    if (!o || !this.onEdit) { return; }

    if (key === 'name') {
        this.renameSelected(input.value);
        return;
    }

    // Which beam to slide along is remembered here, not sent anywhere:
    // nothing in the model records it.
    if (key === 'slide_beam') {
        var hits = this._beamsThrough(o);
        for (var h = 0; h < hits.length; h++) {
            if (String(hits[h].index) === input.value) {
                this.slideBeam = {name: hits[h].beam.name,
                                  index: hits[h].index};
            }
        }
        return;
    }
    // A distance to move by, not a value to hold: the field goes back
    // to zero whether or not the move is accepted, so that leaning on
    // Enter does not walk the element down the bench.
    if (key === 'slide_by') {
        var by = parseField(input.value);
        input.value = '0';
        if (typeof by === 'number' && isFinite(by)) { this.slideSelected(by); }
        return;
    }

    var field = null;
    for (var i = 0; i < OPTIC_FIELDS.length; i++) {
        if (OPTIC_FIELDS[i].key === key) { field = OPTIC_FIELDS[i]; }
    }

    if (field && field.bool) {
        if (input.checked === !!opticFieldValue(o, key)) { return; }
        this.onEdit(opticFieldMessage(o, key, input.checked));
        return;
    }

    if (field && (field.text || field.choices)) {
        var text = String(input.value).trim();
        if (!text || text === opticFieldValue(o, key)) {
            this._refreshOpticPanel();
            return;
        }
        this.onEdit(opticFieldMessage(o, key, text));
        return;
    }

    var value = parseField(input.value);
    var unusable = (typeof value === 'number' && isNaN(value))
        || (value === null && !(field && field.nullable))
        // An infinity would not survive the trip: JSON has none, and
        // what arrives on the Python side is a null the model cannot
        // use. A field that cannot mean it must not send it.
        || (field && field.finite && typeof value === 'number'
            && !isFinite(value));
    if (unusable) {
        // Not a usable value: put back what the model actually holds.
        this._refreshOpticPanel();
        return;
    }
    if (value === opticFieldValue(o, key)) { return; }
    this.onEdit(opticFieldMessage(o, key, value));
};

/*
 * Rename the selected optics.
 *
 * The name is the identity the layout resolves edits by, so this is its
 * own operation rather than one more property to set. Python decides
 * whether the new name is free; the viewer follows it optimistically so
 * that the next edit addresses the right element, and revertSelection()
 * puts it back if the rename was refused.
 */
Viewer.prototype.renameSelected = function (name) {
    // Whichever panel is up names the thing being renamed. The message
    // is the same either way - the layout resolves a target across
    // optics, sources, dimensions and mechanics alike - so only which
    // selection to carry optimistically differs.
    var source = this.panelKind === 'source';
    var mech = this.panelKind === 'mech';
    var o = source ? this._selectedSource()
        : mech ? this._selectedMech() : this._selectedOptic();
    if (!o || !this.onEdit) { return null; }
    name = String(name).trim();
    if (!name || name === o.name) {
        if (source) { this._refreshSourcePanel(); }
        else if (mech) { this._refreshMechPanel(); }
        else { this._refreshOpticPanel(); }
        return null;
    }
    var msg = {op: 'rename', target: o.name, name: name};
    if (source) {
        this.sourceFallback = o.name;
        this.selectedSource = name;
    } else if (mech) {
        this.mechFallback = o.name;
        this.selectedMech = name;
    } else {
        this.selectionFallback = o.name;
        this.selectedOptic = name;
    }
    this.onEdit(msg);
    return msg;
};

/*
 * Undo an optimistic selection change after Python refused the edit.
 */
Viewer.prototype.revertSelection = function () {
    if (this.selectionFallback && !this._selectedOptic()) {
        this.selectedOptic = this.selectionFallback;
        this._showPanel('optic');
    }
    if (this.dimFallback && !this._selectedDim()) {
        this.selectedDim = this.dimFallback;
        this._showPanel('dimension');
    }
    if (this.sourceFallback && !this._selectedSource()) {
        this.selectedSource = this.sourceFallback;
        this._showPanel('source');
    }
    if (this.mechFallback && !this._selectedMech()) {
        this.selectedMech = this.mechFallback;
        this._showPanel('mech');
    }
    this.selectionFallback = null;
    this.dimFallback = null;
    this.sourceFallback = null;
    this.mechFallback = null;
    this._refreshOpticPanel();
    this._refreshDimPanel();
    this._refreshSourcePanel();
    this._refreshMechPanel();
    this._updateOverlay();
};

//}}}

//{{{ Measuring

/*
 * How near the cursor has to be, in screen pixels, for a measurement to
 * take a marked point rather than the cursor itself. Wider than the beam
 * pick radius: a measurement is meant to land on something exactly, and
 * the points on offer are far apart compared with the beams, so being
 * generous costs nothing.
 */
var SNAP_RADIUS = 16;

/*
 * How near the cursor has to be to a dimension line to take hold of it,
 * in screen pixels.
 */
var DIM_PICK = 8;

/*
 * Two marked points closer together than this, in metres, are the same
 * point as far as snapping is concerned, and the first of them wins.
 * Well below anything on a bench and well above what a trace rounds to.
 */
var SNAP_TIE = 1e-9;

/*
 * How short the viewer may be dragged, in screen pixels. Below this the
 * side panel is taller than the drawing and neither is usable.
 */
var MIN_HEIGHT = 240;

/*
 * The points a measurement may be taken from.
 *
 * The optics contribute theirs through the scene: corners, the apex of
 * each face and the middle, all worked out by Python where the wedge and
 * the sagitta of a curved face are already understood. Beams contribute
 * their two ends, which are carried literally in the scene and so need
 * no geometry here - deriving them would be a second description of
 * something already stated.
 *
 * Hidden layers are left out. A layer switched off is one the user has
 * said they are not looking at, and snapping to an invisible point would
 * put an end of the measurement somewhere nothing appears to be.
 */
Viewer.prototype._snapCandidates = function () {
    var out = (this.scene.snap || []).slice();
    var beams = this.scene.beams || [];
    for (var i = 0; i < beams.length; i++) {
        var b = beams[i];
        var g = this.layerGroups[b.layer];
        if (g && !g.visible) { continue; }
        out.push({point: b.pos, kind: 'beam', label: b.name + ' start'});
        out.push({point: b.end, kind: 'beam', label: b.name + ' end'});
    }
    return out;
};

/*
 * The marked point nearest a scene point, or null if none is near
 * enough. The reach is in screen pixels, so it is the same on the screen
 * however far the view is zoomed in - the alternative would be a tool
 * that stops snapping as soon as the drawing is enlarged.
 */
Viewer.prototype._snapAt = function (x, y) {
    var reach = SNAP_RADIUS / this.scale;
    var best = null, bestD = reach;
    var cands = this._snapCandidates();
    for (var i = 0; i < cands.length; i++) {
        var p = cands[i].point;
        var d = Math.hypot(x - p[0], y - p[1]);
        // A later candidate has to be nearer by a real distance, not by
        // rounding. A beam ends on the face it hit, so its end and the
        // apex of that face are the same point to within what the trace
        // could resolve; the optics comes first in the list and should
        // win, both because it is the exact value the model holds and
        // because 'M2 HR' says more than 'b0 end'.
        if (d < bestD - SNAP_TIE) { best = cands[i]; bestD = d; }
    }
    return best;
};

/*
 * Where the next click of a measurement would land: the marked point
 * under the cursor if there is one, the cursor itself if not.
 */
Viewer.prototype._measurePoint = function (x, y) {
    var s = this._snapAt(x, y);
    this.snapped = s;
    return s ? [s.point[0], s.point[1]] : [x, y];
};

/*
 * Arm or disarm the measuring tool.
 *
 * It is a mode rather than a modifier because it takes three clicks with
 * the picture answering to the cursor between them, and because those
 * clicks have to mean "here" rather than whatever clicking there would
 * otherwise have meant.
 */
Viewer.prototype.toggleMeasure = function (on) {
    this.measuring = on === undefined ? !this.measuring : !!on;
    // The two tools are both modes, and a click cannot mean "measure
    // here" and "face this way" at once.
    if (this.measuring) { this.cancelAlign(); }
    this.measureFrom = null;
    this.measureTo = null;
    this.measureOffset = 0;
    this.snapped = null;
    if (this.measureBtn) {
        this.measureBtn.classList.toggle('gt-btn-on', this.measuring);
    }
    this.svg.classList.toggle('gt-measuring',
                              this.measuring || !!this.aligning);
    this._updateOverlay();
    this._updateStatus();
    return this.measuring;
};

/*
 * How far, in screen pixels, the cursor has to be off the line between
 * the two points before the dimension line is carried aside at all.
 * Inside it the offset is zero, so a line drawn straight between them -
 * which is what a short measurement inside an element wants - can be
 * had without hitting it exactly.
 */
var OFFSET_DEADZONE = 5;

/*
 * A click while measuring. The first two fix the points being measured;
 * the third fixes where the dimension line is drawn, and asks Python
 * for it.
 *
 * There is a third click because the two points worth measuring between
 * are usually the two the drawing is busiest around - along a beam, or
 * through an element - and a line drawn straight between them lands on
 * top of what it is measuring. Carrying it aside is a choice about the
 * drawing, so it is made by eye, like the rest of the drawing.
 *
 * The tool disarms itself at the end rather than staying up for another
 * measurement: a mode that stays on until it is switched off is a mode
 * that gets left on, and the button is right there.
 */
Viewer.prototype._onMeasureClick = function (x, y) {
    if (!this.measureFrom) {
        this.measureFrom = this._measurePoint(x, y);
        this._updateOverlay();
        this._updateStatus();
        return null;
    }
    if (!this.measureTo) {
        var pt = this._measurePoint(x, y);
        if (this.measureFrom[0] === pt[0] && this.measureFrom[1] === pt[1]) {
            // Both ends in the same place is not a measurement. Left
            // armed, so the click can be tried again somewhere else.
            return null;
        }
        this.measureTo = pt;
        this.measureOffset = 0;
        this._updateOverlay();
        this._updateStatus();
        return null;
    }
    var name = this._freshDimName();
    var msg = {op: 'add', type: 'Dimension', name: name,
               params: {p1: [this.measureFrom[0], this.measureFrom[1]],
                        p2: [this.measureTo[0], this.measureTo[1]],
                        offset: this._offsetAt(x, y)}};
    this.toggleMeasure(false);
    this.selectedOptic = null;
    this.selectedDim = name;
    if (this.onEdit) {
        this.onEdit(msg);
    } else {
        this._addLocalDimension(name, msg.params);
    }
    return msg;
};

/*
 * Add a dimension to the scene in hand, without telling anyone.
 *
 * This is what a viewer with nowhere to send edits does: a written HTML
 * file, or a widget made read-only. The two points and the distance
 * between them are arithmetic, and the points to snap to are already in
 * the scene, so measuring needs no Python - which is the whole reason a
 * file you can mail to a collaborator is worth having.
 *
 * Two things it cannot do, both because Python is what would have done
 * them. There is no optical distance: whether a span runs inside a
 * substrate is a question about the surfaces, and those live in the
 * model rather than in the drawing. And the measurement is not saved -
 * it lasts as long as the page, and a scene pushed from Python replaces
 * it along with everything else.
 */
Viewer.prototype._addLocalDimension = function (name, params) {
    var p1 = params.p1, p2 = params.p2;
    var off = params.offset || 0;
    var vx = p2[0] - p1[0], vy = p2[1] - p1[1];
    var len = Math.hypot(vx, vy);
    var nx = len ? -vy / len * off : 0;
    var ny = len ? vx / len * off : 0;
    var dim = {type: 'Dimension', name: name, p1: p1, p2: p2, offset: off,
               line: [[p1[0] + nx, p1[1] + ny], [p2[0] + nx, p2[1] + ny]],
               length: len, optical: null, inside: null, n: null,
               // Marks it as the viewer's own, so that Remove offers to
               // take back what the reader drew and nothing else.
               local: true};
    if (!this.scene.dimensions) { this.scene.dimensions = []; }
    this.scene.dimensions.push(dim);
    this._renderDimensions();
    this._refreshDimPanel();
    this._showPanel('dimension');
    this._updateDimensions();
    this._updateOverlay();
    return dim;
};

/*
 * Where the cursor puts the dimension line: its distance from the line
 * between the two points, signed to the left of the way they run.
 *
 * Not snapped to anything. The points being measured are exact and the
 * marked points are there for them; where the line is drawn is a matter
 * of where there is room, and nothing in the model has an opinion.
 */
Viewer.prototype._offsetAt = function (x, y) {
    var a = this.measureFrom, b = this.measureTo;
    if (!a || !b) { return 0; }
    var vx = b[0] - a[0], vy = b[1] - a[1];
    var len = Math.hypot(vx, vy);
    if (!len) { return 0; }
    var off = ((x - a[0]) * (-vy) + (y - a[1]) * vx) / len;
    return Math.abs(off) * this.scale < OFFSET_DEADZONE ? 0 : off;
};

/*
 * The dimension as it would be added, for the preview. The same shape
 * Python sends back, so the preview and the result are drawn by the
 * same code.
 */
Viewer.prototype._pendingDim = function () {
    var a = this.measureFrom, b = this.measureTo;
    if (!a || !b) { return null; }
    var off = this.measureOffset || 0;
    var vx = b[0] - a[0], vy = b[1] - a[1];
    var len = Math.hypot(vx, vy);
    var nx = len ? -vy / len * off : 0;
    var ny = len ? vx / len * off : 0;
    return {name: null, p1: a, p2: b, offset: off, length: len,
            line: [[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny]],
            optical: null, inside: null, n: null};
};

/*
 * A name no element of the scene is using. Chosen here, like the name of
 * a new optics, so that the viewer can select what it just asked for as
 * soon as the scene comes back.
 */
Viewer.prototype._freshDimName = function () {
    var taken = {};
    // Everything in the one namespace, the same list _freshOpticName
    // walks: a dimension named after a source or a breadboard would be
    // refused by Python, stranding the optimistic selection.
    [this.scene.optics, this.scene.sources, this.scene.dimensions,
     this.scene.mechanics]
        .forEach(function (list) {
            (list || []).forEach(function (o) { taken[o.name] = true; });
        });
    var i = 1;
    while (taken['D' + i]) { i++; }
    return 'D' + i;
};

Viewer.prototype._selectedDim = function () {
    if (!this.selectedDim) { return null; }
    var dims = this.scene.dimensions || [];
    for (var i = 0; i < dims.length; i++) {
        if (dims[i].name === this.selectedDim) { return dims[i]; }
    }
    return null;
};

Viewer.prototype._selectDim = function (dim) {
    this.selectedDim = dim ? dim.name : null;
    this.selectedOptic = null;
    this.selectedMech = null;
    this.pinned = null;
    if (dim) {
        this._refreshDimPanel();
        this._showPanel('dimension');
    } else {
        this._showPanel('beam');
    }
    // Which dimension is marked is part of how they are drawn, and the
    // drawing is what has to agree with the panel.
    this._updateDimensions();
    this._updateOverlay();
};

/*
 * The dimension nearest a scene point, or null.
 *
 * Aimed at the dimension line, which is where it was put to be read,
 * and not at the span between the measured points - that one usually
 * runs along a beam or through an element, and taking hold of it there
 * is what carrying the line aside was for. Measured to the segment
 * rather than to its ends, so it can be picked anywhere along its
 * length.
 */
Viewer.prototype._pickDimension = function (x, y) {
    var dims = this.scene.dimensions || [];
    // Tighter than the snapping reach: this one is taken from whatever
    // lies underneath, so it should ask for aim rather than accept the
    // neighbourhood.
    var reach = DIM_PICK / this.scale;
    var best = null, bestD = reach;
    for (var i = 0; i < dims.length; i++) {
        var ends = dimLineEnds(dims[i]);
        var d = distToSegment(x, y, ends[0], ends[1]);
        if (d <= bestD) { best = dims[i]; bestD = d; }
    }
    return best;
};

Viewer.prototype._refreshDimPanel = function () {
    var d = this._selectedDim();
    var f = this.dimFields;
    var values = {};
    if (this.dimFoot) {
        this.dimFoot.style.display =
            (d && (this.onEdit || d.local)) ? '' : 'none';
    }
    if (d) {
        values = {name: d.name,
                  p1x: d.p1[0], p1y: d.p1[1],
                  p2x: d.p2[0], p2y: d.p2[1],
                  offset: (d.offset || 0) / MM,
                  length: fmtLen(d.length),
                  angle: fmtDeg(Math.atan2(d.p2[1] - d.p1[1],
                                           d.p2[0] - d.p1[0])),
                  inside: d.inside,
                  n: d.n === null || d.n === undefined ? null : fmtNum(d.n, 6),
                  optical: d.optical === null || d.optical === undefined
                      ? null : fmtLen(d.optical)};
    }
    for (var key in f) {
        var rec = f[key];
        var v = values[key];
        if (rec.optional) {
            rec.row.style.display = (v === undefined || v === null)
                ? 'none' : '';
        }
        if (rec.editable && document.activeElement === rec.el) { continue; }
        if (rec.editable) {
            rec.el.value = d ? (typeof v === 'number' ? fmtField(v)
                                : String(v === undefined ? '' : v)) : '';
        } else {
            rec.el.textContent = (d && v !== undefined && v !== null)
                ? String(v) : '-';
        }
    }
};

Viewer.prototype._commitDimField = function (key, input) {
    var d = this._selectedDim();
    if (!d || !this.onEdit) { return; }

    if (key === 'name') {
        var name = String(input.value).trim();
        if (!name || name === d.name) { this._refreshDimPanel(); return; }
        var rmsg = {op: 'rename', target: d.name, name: name};
        this.dimFallback = d.name;
        this.selectedDim = name;
        this.onEdit(rmsg);
        return;
    }

    var value = parseField(input.value);
    if (typeof value !== 'number' || !isFinite(value)) {
        this._refreshDimPanel();
        return;
    }

    if (key === 'offset') {
        this.onEdit({op: 'set', target: d.name,
                     attrs: {offset: value * MM}});
        return;
    }

    // Which end, and which of its two coordinates. The message carries
    // the whole end, since that is what the model holds.
    var end = key.charAt(0) === 'p' && key.charAt(1) === '1' ? 'p1' : 'p2';
    var point = [d[end][0], d[end][1]];
    point[key.charAt(2) === 'x' ? 0 : 1] = value;
    var attrs = {};
    attrs[end] = point;
    this.onEdit({op: 'set', target: d.name, attrs: attrs});
};

/*
 * Distance from a point to a segment, in scene units.
 */
function distToSegment(x, y, a, b) {
    var vx = b[0] - a[0], vy = b[1] - a[1];
    var len2 = vx * vx + vy * vy;
    var t = len2 > 0 ? ((x - a[0]) * vx + (y - a[1]) * vy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(x - (a[0] + vx * t), y - (a[1] + vy * t));
}

//}}}

//{{{ Aiming

/*
 * Arm the aiming tool for the selected optics: the next two or three
 * clicks name the places it is to face by.
 *
 * The points snap to the same marks a measurement takes - the faces
 * and corners of the elements, the screw holes of a breadboard, the
 * ends of the beams - which is what makes this exact rather than a
 * steadier drag.
 */
Viewer.prototype.startAlign = function (points) {
    var o = this._selectedOptic();
    if (!o || !this.onEdit) { return null; }
    if (this.measuring) { this.toggleMeasure(false); }
    this.aligning = {optic: o.name, want: points, points: []};
    this.alignPreview = null;
    this.snapped = null;
    this.svg.classList.add('gt-measuring');
    this._updateOverlay();
    this._updateStatus();
    return this.aligning;
};

Viewer.prototype.cancelAlign = function () {
    if (!this.aligning) { return false; }
    this.aligning = null;
    this.alignPreview = null;
    this.snapped = null;
    this.svg.classList.remove('gt-measuring');
    this._updateOverlay();
    this._updateStatus();
    return true;
};

/*
 * What the aim comes to, from the points taken so far plus wherever
 * the cursor is. Null while there are too few points, or where the
 * points name no answer - see bisectorAngle.
 */
Viewer.prototype._alignAngle = function (cursor) {
    var al = this.aligning;
    if (!al) { return null; }
    var pts = cursor ? al.points.concat([cursor]) : al.points;
    if (pts.length < al.want) { return null; }
    if (al.want === 2) {
        return acrossAngle(pts[0], pts[1]);
    }
    return bisectorAngle(pts[0], pts[1], pts[2]);
};

/*
 * A click while aiming. The last one turns the optics and puts the
 * tool away; a mode left armed is a mode left on.
 */
Viewer.prototype._onAlignClick = function (x, y) {
    var al = this.aligning;
    var pt = this._measurePoint(x, y);
    var last = al.points[al.points.length - 1];
    if (last && last[0] === pt[0] && last[1] === pt[1]) {
        // The same place twice names no direction. Left armed, so the
        // click can be tried again somewhere else.
        return null;
    }
    al.points.push(pt);
    if (al.points.length < al.want) {
        this._updateOverlay();
        this._updateStatus();
        return null;
    }
    var angle = this._alignAngle(null);
    var target = al.optic;
    this.cancelAlign();
    if (angle === null) { return null; }
    var msg = {op: 'rotate', target: target, normAngleHR: angle};
    this.onEdit(msg);
    return msg;
};

/*
 * Turn the selected optics by so many degrees from where it faces
 * now - the quarter turn a steering mirror is specified by.
 */
Viewer.prototype.turnSelected = function (deg) {
    var o = this._selectedOptic();
    if (!o || !this.onEdit) { return null; }
    var msg = {op: 'rotate', target: o.name,
               normAngleHR: (o.normAngleHR || 0) + deg / DEG};
    this.onEdit(msg);
    return msg;
};

/*
 * There is nothing to aim while nothing is selected.
 */
Viewer.prototype._refreshAlign = function () {
    if (this.alignBtn) {
        this.alignBtn.disabled = !this._selectedOptic();
    }
};

//}}}

//{{{ Scene rendering

/*
 * Convert a shape dict into an SVG element in scene coordinates.
 */
function shapeToSVG(s) {
    switch (s.type) {
    case 'line':
        return svgEl('line', {x1: s.start[0], y1: s.start[1],
                              x2: s.stop[0], y2: s.stop[1]});
    case 'polyline':
        var pts = [];
        for (var i = 0; i < s.x.length; i++) { pts.push(s.x[i] + ',' + s.y[i]); }
        return svgEl('polyline', {points: pts.join(' ')});
    case 'rectangle':
        // 'point' is the lower left corner; the scene group is y-up so
        // the SVG rect covers [point, point + (width, height)] directly.
        return svgEl('rect', {x: s.point[0], y: s.point[1],
                              width: s.width, height: s.height});
    case 'circle':
        return svgEl('circle', {cx: s.center[0], cy: s.center[1], r: s.radius});
    case 'arc':
        return arcToSVG(s);
    default:
        return null;
    }
}

/*
 * An Arc runs counterclockwise from startangle to stopangle (radians).
 * Inside the y-up scene group, increasing angle is the SVG positive
 * sweep direction, hence sweep-flag = 1.
 */
function arcToSVG(s) {
    var span = (s.stopangle - s.startangle) % (2 * Math.PI);
    if (span < 0) { span += 2 * Math.PI; }
    if (span < 1e-12 || Math.abs(span - 2 * Math.PI) < 1e-12) {
        return svgEl('circle', {cx: s.center[0], cy: s.center[1], r: s.radius});
    }
    var x0 = s.center[0] + s.radius * Math.cos(s.startangle);
    var y0 = s.center[1] + s.radius * Math.sin(s.startangle);
    var x1 = s.center[0] + s.radius * Math.cos(s.startangle + span);
    var y1 = s.center[1] + s.radius * Math.sin(s.startangle + span);
    var large = span > Math.PI ? 1 : 0;
    return svgEl('path', {d: 'M ' + x0 + ' ' + y0 + ' A ' + s.radius + ' ' +
                             s.radius + ' 0 ' + large + ' 1 ' + x1 + ' ' + y1});
}

Viewer.prototype._renderScene = function () {
    var self = this;
    var layers = (this.scene.canvas && this.scene.canvas.layers) || [];
    var hidden = this.opts.hiddenLayers || [];

    layers.forEach(function (ly) {
        var color = layerColor(ly.color);
        var geom = svgEl('g', {'class': 'gt-layer', stroke: color});
        var labels = svgEl('g', {'class': 'gt-layer', fill: color});

        ly.shapes.forEach(function (s) {
            if (s.type === 'text') {
                var t = svgEl('text', {'font-size': self.fontSize});
                t.textContent = s.text;
                labels.appendChild(t);
                self.labels.push({el: t, x: s.point[0], y: s.point[1],
                                  rotation: s.rotation, layer: ly.name});
            } else {
                var e = shapeToSVG(s);
                if (e) { geom.appendChild(e); }
            }
        });

        self.sceneGroup.appendChild(geom);
        self.labelGroup.appendChild(labels);
        self.layerGroups[ly.name] = {geom: geom, label: labels,
                                     color: color, visible: true,
                                     count: ly.shapes.length};
        self._addLayerToggle(ly.name, color, ly.shapes.length);
        if (hidden.indexOf(ly.name) >= 0) { self.setLayerVisible(ly.name, false); }
    });

    // Dimensions are drawn in screen coordinates, like the labels and
    // for the same reason: the ticks and the numbers have to keep their
    // size whatever the zoom, or a measurement of a lens would vanish
    // next to one across the bench.
    this._renderDimensions();
    this._renderSources();

    // Overlay elements for the readout marker, the highlighted beam and
    // the arrow telling which way that beam travels.
    this.highlight = svgEl('line', {'class': 'gt-highlight'});
    this.arrow = svgEl('path', {'class': 'gt-arrow'});
    this.marker = svgEl('circle', {'class': 'gt-marker', r: 4});
    this.outline = svgEl('polygon', {'class': 'gt-optic-outline'});
    // The outline of a mechanics: its own element, so that a hovered
    // optics and a selected breadboard can both be marked at once.
    this.mechOutline = svgEl('polygon', {'class': 'gt-optic-outline'});
    // The origin of the part being edited, and the box round the
    // shape the panel is showing. The origin is the point that comes
    // to sit at the host's substrate centre, so a part is drawn
    // around it and it has to be visible to draw around.
    this.originMark = svgEl('path', {'class': 'gt-origin'});
    this.originMark.style.display = 'none';
    this.shapeMark = svgEl('polygon', {'class': 'gt-optic-outline gt-selected'});
    this.shapeMark.style.display = 'none';

    // The corner handles a resizable body is cut by. Four, built once,
    // shown only while such a body is selected.
    this.mechHandles = [];
    for (var hi = 0; hi < 4; hi++) {
        var handle = svgEl('rect', {'class': 'gt-mech-handle',
                                    width: 7, height: 7});
        handle.style.display = 'none';
        this.mechHandles.push(handle);
    }
    this._handlePts = null;
    // The beam the Along beam row names. A name like 'b0:M1t1' says
    // nothing about which line in the picture it is, so the choice is
    // drawn. Underneath everything else, since it is a standing mark
    // rather than an answer to where the cursor is.
    this.slideMark = svgEl('line', {'class': 'gt-slide-beam'});
    // Which way that beam travels, which is the sign of Move by. Two
    // beams lying on the same line often run opposite ways, so stepping
    // through them has to show more than that the mark moved.
    this.slideArrow = svgEl('path', {'class': 'gt-slide-arrow'});
    // The measurement being placed, and the marked point the next click
    // would take.
    this.rubber = svgEl('line', {'class': 'gt-rubber'});
    this.snapMark = svgEl('circle', {'class': 'gt-snap', r: 5});
    // The places an aim has been given so far, joined to the cursor.
    // A polyline rather than a line: bisecting takes three points, so
    // there are two arms to show.
    this.alignPath = svgEl('polyline', {'class': 'gt-rubber'});
    // The preview of the dimension being placed lives in this group
    // too, so it goes with it and has to be built again.
    this.pendingEls = null;
    this.overlayGroup.appendChild(this.slideMark);
    this.overlayGroup.appendChild(this.slideArrow);
    this.overlayGroup.appendChild(this.originMark);
    this.overlayGroup.appendChild(this.shapeMark);
    this.overlayGroup.appendChild(this.rubber);
    this.overlayGroup.appendChild(this.alignPath);
    this.overlayGroup.appendChild(this.snapMark);
    this.overlayGroup.appendChild(this.mechOutline);
    for (var hj = 0; hj < this.mechHandles.length; hj++) {
        this.overlayGroup.appendChild(this.mechHandles[hj]);
    }
    this.overlayGroup.appendChild(this.outline);
    this.overlayGroup.appendChild(this.highlight);
    this.overlayGroup.appendChild(this.arrow);
    this.overlayGroup.appendChild(this.marker);
    this._showMarker(false);
    this.outline.style.display = 'none';
    this.mechOutline.style.display = 'none';
    this.slideMark.style.display = 'none';
    this.slideArrow.style.display = 'none';
    this.rubber.style.display = 'none';
    this.alignPath.style.display = 'none';
    this.snapMark.style.display = 'none';
};

/*
 * Build one polygon per source. Where they are drawn is worked out
 * every frame by _updateSources, since the shape keeps its size on
 * screen while the origin follows the scene.
 */
Viewer.prototype._renderSources = function () {
    var self = this;
    this.sourceEls = [];
    (this.scene.sources || []).forEach(function (s) {
        var poly = svgEl('polygon', {'class': 'gt-source'});
        // A source names the beam it emits, and that beam is drawn on a
        // layer; hiding the layer hides the beam, so the laser goes
        // with it rather than being left pointing at nothing.
        self.sourceGroup.appendChild(poly);
        self.sourceEls.push({el: poly, source: s});
    });
    // Deliberately not placed here. Building the scene happens before
    // the view has been framed, so there is no transform yet to put a
    // screen-space shape through; _applyTransform places them, and it
    // runs on the way out of every path that gets here.
};

/*
 * Put the lasers where the view now stands.
 */
Viewer.prototype._updateSources = function () {
    var self = this;
    (this.sourceEls || []).forEach(function (rec) {
        var s = rec.source;
        var g = self.layerGroups[s.layer];
        if (g && !g.visible) { rec.el.style.display = 'none'; return; }
        rec.el.style.display = '';
        // While one is being dragged it follows the cursor rather than
        // the scene: Python has not been told yet, and will not be
        // until the button comes up.
        var d = self.dragSource;
        var pos = (d && d.source.name === s.name) ? d.pos : s.pos;
        var ang = (d && d.source.name === s.name) ? d.angle : s.dirAngle;
        var dir = [Math.cos(ang), Math.sin(ang)];
        var o = self.sceneToScreen(pos[0], pos[1]);
        var k = self._sourceGrowth(s);
        rec.el.setAttribute('points', SOURCE_SHAPE.map(function (uv) {
            var p = sourcePoint(uv, o, dir, k);
            return p[0] + ',' + p[1];
        }).join(' '));
        rec.el.classList.toggle('gt-selected',
                                s.name === self.selectedSource);
        rec.el.classList.toggle('gt-hover',
                                !!self.hoverSource
                                && self.hoverSource.name === s.name);
        rec.el.classList.toggle('gt-dragging', !!d
                                && d.source.name === s.name);
    });
};

/*
 * The source whose laser a screen point falls on, or null. Later
 * sources win, so the one drawn on top is the one picked.
 */
Viewer.prototype._pickSource = function (px, py) {
    var found = null;
    (this.sourceEls || []).forEach(function (rec) {
        if (rec.el.style.display === 'none') { return; }
        var s = rec.source;
        var o = this.sceneToScreen(s.pos[0], s.pos[1]);
        if (sourceHit(px, py, o, s.dirVect, this._sourceGrowth(s))) {
            found = s;
        }
    }, this);
    return found;
};

/*
 * How much the laser for this source is grown by, at the current zoom.
 * One place, so that the drawing and the pick cannot disagree.
 */
Viewer.prototype._sourceGrowth = function (s) {
    return sourceGrowth(s, this.scene.display, this.scale);
};

/*
 * Length of the ticks at the ends of a dimension line, and how far the
 * numbers sit off it. In screen pixels.
 */
var DIM_TICK = 6;
var DIM_TEXT_GAP = 7;

/*
 * The extension lines that carry the measured points out to wherever
 * the dimension line has been placed: a small gap at the point being
 * measured, so that the line does not cover the very thing it points
 * at, and a small overshoot past the dimension line, as on any
 * engineering drawing. In screen pixels.
 */
var DIM_EXT_GAP = 3;
var DIM_EXT_OVERSHOOT = 4;

/*
 * Build the SVG for the dimensions in the scene.
 *
 * Each one is a dimension line with a tick across each end, an
 * extension line carrying each measured point out to it, the physical
 * distance written above the line and, when the span runs inside a
 * substrate, the optical distance below. Above and below rather than
 * side by side, so that the two numbers cannot be read as one.
 */
Viewer.prototype._renderDimensions = function () {
    var self = this;
    this.dimEls = {};
    this.dimGroup.textContent = '';
    (this.scene.dimensions || []).forEach(function (d) {
        self.dimEls[d.name] = self._buildDimEls(d, self.dimGroup);
    });
};

/*
 * The SVG for one dimension, appended to a group. Used both for the
 * dimensions in the scene and for the one being placed, which is drawn
 * the same way so that what is previewed is what will appear.
 */
Viewer.prototype._buildDimEls = function (d, parent, cls) {
    var g = svgEl('g', {'class': 'gt-dim' + (cls ? ' ' + cls : '')});
    var rec = {
        e1: svgEl('line', {'class': 'gt-dim-ext'}),
        e2: svgEl('line', {'class': 'gt-dim-ext'}),
        line: svgEl('line', {'class': 'gt-dim-line'}),
        t1: svgEl('line', {'class': 'gt-dim-tick'}),
        t2: svgEl('line', {'class': 'gt-dim-tick'}),
        label: svgEl('text', {'class': 'gt-dim-label',
                              'font-size': this.fontSize}),
        optical: svgEl('text', {'class': 'gt-dim-label gt-dim-optical',
                                'font-size': this.fontSize}),
        group: g,
        dim: d
    };
    ['e1', 'e2', 'line', 't1', 't2', 'label', 'optical'].forEach(
        function (k) { g.appendChild(rec[k]); });
    parent.appendChild(g);
    this._setDimText(rec, d);
    return rec;
};

Viewer.prototype._setDimText = function (rec, d) {
    rec.label.textContent = fmtLen(d.length);
    if (d.optical === null || d.optical === undefined) {
        rec.optical.style.display = 'none';
    } else {
        rec.optical.style.display = '';
        rec.optical.textContent = fmtLen(d.optical) + ' optical';
    }
};

/*
 * The two ends of a dimension's line. Python works them out and sends
 * them, so a front end does not have to know which way round the offset
 * goes; the fallback is for a dimension being previewed, which Python
 * has not been told about yet.
 */
function dimLineEnds(d) {
    if (d.line) { return [d.line[0], d.line[1]]; }
    return [d.p1, d.p2];
}

/*
 * Place the dimensions for the current view. Called whenever the
 * transform changes, like the labels.
 */
Viewer.prototype._updateDimensions = function () {
    for (var name in this.dimEls) {
        this._placeDim(this.dimEls[name], name === this.selectedDim);
    }
};

Viewer.prototype._placeDim = function (rec, selected) {
    var d = rec.dim;
    var ends = dimLineEnds(d);
    var a = this.sceneToScreen(ends[0][0], ends[0][1]);
    var b = this.sceneToScreen(ends[1][0], ends[1][1]);
    var p1 = this.sceneToScreen(d.p1[0], d.p1[1]);
    var p2 = this.sceneToScreen(d.p2[0], d.p2[1]);
    var dx = b[0] - a[0], dy = b[1] - a[1];
    var len = Math.hypot(dx, dy) || 1;
    var ux = dx / len, uy = dy / len;
    var nx = -uy, ny = ux;                     // across the line

    rec.line.setAttribute('x1', a[0]);
    rec.line.setAttribute('y1', a[1]);
    rec.line.setAttribute('x2', b[0]);
    rec.line.setAttribute('y2', b[1]);
    setTick(rec.t1, a, nx, ny);
    setTick(rec.t2, b, nx, ny);

    // An extension line runs from just clear of the point being
    // measured to just past the dimension line. With no offset there is
    // nothing to carry and the pair would be a smudge on the ticks, so
    // they go away and what is left is a line drawn straight between
    // the two points.
    setExt(rec.e1, p1, a);
    setExt(rec.e2, p2, b);

    // The numbers go at the middle, turned to lie along the line and
    // kept the right way up: a dimension running right to left would
    // otherwise be written upside down.
    var mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
    var deg = Math.atan2(dy, dx) * 180 / Math.PI;
    if (deg > 90 || deg < -90) { deg += 180; }
    var rot = 'rotate(' + deg + ',' + mx + ',' + my + ')';
    place(rec.label, mx, my, rot, -DIM_TEXT_GAP);
    place(rec.optical, mx, my, rot, DIM_TEXT_GAP + this.fontSize * 0.8);

    rec.group.classList.toggle('gt-selected', !!selected);

    function setTick(el, p, tnx, tny) {
        el.setAttribute('x1', p[0] + tnx * DIM_TICK);
        el.setAttribute('y1', p[1] + tny * DIM_TICK);
        el.setAttribute('x2', p[0] - tnx * DIM_TICK);
        el.setAttribute('y2', p[1] - tny * DIM_TICK);
    }
    function setExt(el, from, to) {
        var vx = to[0] - from[0], vy = to[1] - from[1];
        var d0 = Math.hypot(vx, vy);
        if (d0 <= DIM_EXT_GAP + DIM_EXT_OVERSHOOT) {
            el.style.display = 'none';
            return;
        }
        el.style.display = '';
        var ex = vx / d0, ey = vy / d0;
        el.setAttribute('x1', from[0] + ex * DIM_EXT_GAP);
        el.setAttribute('y1', from[1] + ey * DIM_EXT_GAP);
        el.setAttribute('x2', to[0] + ex * DIM_EXT_OVERSHOOT);
        el.setAttribute('y2', to[1] + ey * DIM_EXT_OVERSHOOT);
    }
    function place(el, tx, ty, trot, ody) {
        el.setAttribute('x', tx);
        el.setAttribute('y', ty + ody);
        el.setAttribute('transform', trot);
    }
};

Viewer.prototype._addLayerToggle = function (name, color, count) {
    var self = this;
    var row = htmlEl('label', 'gt-layerrow');
    var cb = htmlEl('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.addEventListener('change', function () {
        self.setLayerVisible(name, cb.checked);
    });
    var swatch = htmlEl('span', 'gt-swatch');
    swatch.style.background = color;
    row.appendChild(cb);
    row.appendChild(swatch);
    row.appendChild(htmlEl('span', 'gt-layername', name));
    if (!count) {
        // The layer was declared but nothing was drawn into it. Say so,
        // otherwise it reads as a bug in the viewer.
        row.className += ' gt-layer-empty';
        row.appendChild(htmlEl('span', 'gt-note', 'empty'));
        row.title = 'This layer contains no shape.';
    }
    this.layerBody.appendChild(row);
    this.layerGroups[name].checkbox = cb;
};

Viewer.prototype.setLayerVisible = function (name, visible) {
    var g = this.layerGroups[name];
    if (!g) { return; }
    g.visible = visible;
    g.geom.style.display = visible ? '' : 'none';
    g.label.style.display = visible ? '' : 'none';
    if (g.checkbox) { g.checkbox.checked = visible; }
    // The lasers are not on the layer groups - they are drawn in screen
    // coordinates, like the labels - so they have to be told. A laser
    // left standing over a hidden beam would be pointing at nothing.
    this._updateSources();
};

//}}}

//{{{ Coordinate transforms

Viewer.prototype.sceneToScreen = function (x, y) {
    return [x * this.scale + this.tx, -y * this.scale + this.ty];
};

Viewer.prototype.screenToScene = function (px, py) {
    return [(px - this.tx) / this.scale, (this.ty - py) / this.scale];
};

Viewer.prototype._applyTransform = function () {
    this.tx = this.width / 2 - this.cx * this.scale;
    this.ty = this.height / 2 + this.cy * this.scale;
    this.sceneGroup.setAttribute('transform',
        'translate(' + this.tx + ',' + this.ty + ') ' +
        'scale(' + this.scale + ',' + (-this.scale) + ')');

    // Labels live in screen coordinates so that they stay readable.
    for (var i = 0; i < this.labels.length; i++) {
        var lb = this.labels[i];
        var p = this.sceneToScreen(lb.x, lb.y);
        lb.el.setAttribute('x', p[0]);
        lb.el.setAttribute('y', p[1]);
        if (lb.rotation) {
            lb.el.setAttribute('transform', 'rotate(' +
                (-lb.rotation * 180 / Math.PI) + ',' + p[0] + ',' + p[1] + ')');
        }
    }

    this._updateDimensions();
    // The lasers are refreshed by _updateOverlay, which every path that
    // changes what is highlighted already calls.
    this._updateOverlay();
    this._updateStatus();
};

Viewer.prototype._measure = function () {
    var rect = this.svg.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.svg.setAttribute('viewBox', '0 0 ' + this.width + ' ' + this.height);
};

Viewer.prototype._resize = function () {
    this._measure();
    // A fit that could not be done for want of a viewport is done now.
    // See fit() for why that happens.
    if (this.fitPending && this.width > 1 && this.height > 1) {
        this.fitPending = false;
        this._fitToBBox();
        return;
    }
    this._applyTransform();
};

//}}}

//{{{ Bounding box and fit

Viewer.prototype.bbox = function () {
    var minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    function add(x, y) {
        if (x < minx) { minx = x; }
        if (x > maxx) { maxx = x; }
        if (y < miny) { miny = y; }
        if (y > maxy) { maxy = y; }
    }
    ((this.scene.canvas && this.scene.canvas.layers) || []).forEach(function (ly) {
        ly.shapes.forEach(function (s) {
            switch (s.type) {
            case 'line': add(s.start[0], s.start[1]); add(s.stop[0], s.stop[1]); break;
            case 'polyline':
                for (var i = 0; i < s.x.length; i++) { add(s.x[i], s.y[i]); }
                break;
            case 'rectangle':
                add(s.point[0], s.point[1]);
                add(s.point[0] + s.width, s.point[1] + s.height);
                break;
            case 'circle':
            case 'arc':
                add(s.center[0] - s.radius, s.center[1] - s.radius);
                add(s.center[0] + s.radius, s.center[1] + s.radius);
                break;
            case 'text': add(s.point[0], s.point[1]); break;
            }
        });
    });
    (this.scene.beams || []).forEach(function (b) {
        add(b.pos[0], b.pos[1]); add(b.end[0], b.end[1]);
    });
    // Dimensions are framed with everything else: a measurement taken to
    // a point off the end of the bench is still part of what there is to
    // look at, and Fit leaving it off the screen would read as its
    // having been lost.
    (this.scene.dimensions || []).forEach(function (d) {
        add(d.p1[0], d.p1[1]); add(d.p2[0], d.p2[1]);
        var ends = dimLineEnds(d);
        add(ends[0][0], ends[0][1]); add(ends[1][0], ends[1][1]);
    });
    if (!isFinite(minx)) { return {minx: -1, miny: -1, maxx: 1, maxy: 1}; }
    return {minx: minx, miny: miny, maxx: maxx, maxy: maxy};
};

Viewer.prototype._fitToBBox = function () {
    var bb = this.bbox();
    var w = Math.max(bb.maxx - bb.minx, 1e-9);
    var h = Math.max(bb.maxy - bb.miny, 1e-9);
    var m = this.fitMargin;
    this.scale = Math.min(this.width / w, this.height / h) * (1 - 2 * m);
    this.cx = (bb.minx + bb.maxx) / 2;
    this.cy = (bb.miny + bb.maxy) / 2;
    this._applyTransform();
};

/*
 * Frame the whole scene.
 *
 * This needs to know how big the view is, and there is one situation
 * where it cannot: a notebook. anywidget calls render() before the
 * output area has been laid out, so the element measures zero, and a
 * scale worked out from that is wrong by whatever the real size turns
 * out to be - three orders of magnitude, in practice, which puts the
 * whole drawing in a dot.
 *
 * So when there is nothing to fit into, the fit is remembered instead
 * of being done wrongly, and _resize() carries it out as soon as a real
 * size arrives. The static HTML page never takes this path: its script
 * runs after the body is laid out, so the first measurement is real.
 */
Viewer.prototype.fit = function (margin) {
    this.fitMargin = margin === undefined ? 0.06 : margin;
    this._measure();
    if (this.width <= 1 || this.height <= 1) {
        this.fitPending = true;
        return;
    }
    this.fitPending = false;
    this._fitToBBox();
};

//}}}

//{{{ Events

Viewer.prototype._bindEvents = function () {
    var self = this;
    var dragging = false, moved = 0, lastX = 0, lastY = 0;

    // Every listener is recorded so that destroy() can take it back off
    // again. A notebook cell can be re-run any number of times, and a
    // viewer that outlives its output area would keep reacting to the
    // mouse and the keyboard for the rest of the session.
    this._listeners = [];
    function on(target, type, fn, opts) {
        target.addEventListener(type, fn, opts);
        self._listeners.push([target, type, fn, opts]);
    }

    if (global.ResizeObserver) {
        this._ro = new ResizeObserver(function () { self._resize(); });
        this._ro.observe(this.svg);
    } else {
        on(global, 'resize', function () { self._resize(); });
    }

    // Dragging the bottom edge changes the height of the viewer. A
    // notebook cell is a letterbox and a bench drawing is not, so this
    // is the control that matters most for actually working in one.
    //
    // The height is the embedder's - it owns the element the viewer was
    // mounted into - so the drag sets it directly and reports it on
    // release. The widget writes it back to its traitlet, which is what
    // makes the new height survive a re-render and readable from Python.
    if (this.resizeGrip) {
        var resizing = false, startY = 0, startH = 0;
        on(this.resizeGrip, 'mousedown', function (ev) {
            if (ev.button !== 0) { return; }
            resizing = true;
            startY = ev.clientY;
            startH = self.container.getBoundingClientRect().height;
            self.resizeGrip.classList.add('gt-resizing');
            ev.preventDefault();
        });
        on(global, 'mousemove', function (ev) {
            if (!resizing) { return; }
            var h = Math.max(MIN_HEIGHT, startH + ev.clientY - startY);
            self.container.style.height = h + 'px';
            self._resize();
            ev.preventDefault();
        });
        on(global, 'mouseup', function () {
            if (!resizing) { return; }
            resizing = false;
            self.resizeGrip.classList.remove('gt-resizing');
            if (self.opts.onResize) {
                self.opts.onResize(
                    Math.round(self.container.getBoundingClientRect().height));
            }
        });
    }

    // An open add menu is shut by pressing anywhere else - including in
    // another viewer, which is why this is on the window rather than on
    // the root. Listened for on the way down, so that the press which
    // shuts the menu still reaches whatever it landed on.
    on(global, 'mousedown', function (ev) {
        if (!self._inAddMenu(ev.target)) { self.closeAddMenus(); }
    }, true);

    // Keyboard shortcuts act on the viewer the pointer is over, so that
    // several viewers in one notebook do not all answer the same key.
    this.pointerInside = false;
    on(this.root, 'mouseenter', function () { self.pointerInside = true; });
    on(this.root, 'mouseleave', function () { self.pointerInside = false; });

    on(this.svg, 'wheel', function (ev) {
        ev.preventDefault();
        var r = self.svg.getBoundingClientRect();
        var px = ev.clientX - r.left, py = ev.clientY - r.top;
        var pt = self.screenToScene(px, py);
        var factor = Math.pow(1.0015, -ev.deltaY);
        self.scale *= factor;
        // Keep the scene point under the cursor fixed.
        self.cx = pt[0] - (px - self.width / 2) / self.scale;
        self.cy = pt[1] + (py - self.height / 2) / self.scale;
        self._applyTransform();
    }, {passive: false});

    on(this.svg, 'mousedown', function (ev) {
        if (ev.button !== 0) { return; }
        var r = self.svg.getBoundingClientRect();
        var px = ev.clientX - r.left, py = ev.clientY - r.top;
        var pt = self.screenToScene(px, py);

        // A resize handle first: it is UI chrome drawn on top of the
        // picture, and only exists while a resizable body is selected.
        if (self.onEdit && !self.measuring && !self.aligning) {
            var hidx = self._pickMechHandle(px, py);
            if (hidx >= 0 && self._selectedMech()) {
                self._beginMechResize(self._selectedMech(), hidx);
                dragging = true; moved = 0;
                lastX = ev.clientX; lastY = ev.clientY;
                self.svg.classList.add('gt-dragging');
                ev.preventDefault();
                return;
            }
        }

        // Grabbing an optics or a laser starts an edit; grabbing
        // anywhere else pans.
        //
        // Not while measuring: a click then means "measure here", and
        // dragging an element out from under the cursor mid-measurement
        // would move the very thing being measured. Nor on a dimension,
        // which is picked ahead of the element under it - a press that
        // grabbed the element would never let the release get as far as
        // selecting the dimension.
        //
        // The laser is tested before the optics, in the same order the
        // click pipeline uses, so that a press and a click never take
        // hold of different things.
        var grabbable = self.onEdit && !self.measuring && !self.aligning
            && !self._pickDimension(pt[0], pt[1]);
        var s = grabbable ? self._pickSource(px, py) : null;
        var o = (grabbable && !s) ? self._pickOptic(pt[0], pt[1]) : null;
        // A mechanics is grabbed only while it is the selection. A
        // breadboard can cover most of the bench, and a press on it
        // usually means "pan the view" - so the first click selects,
        // and only then does dragging move the hardware. An attached
        // body is never grabbed: it goes where its host goes, and its
        // host is right there to be dragged.
        var h = (grabbable && !s && !o && self.selectedMech)
            ? self._pickMech(pt[0], pt[1]) : null;
        if (h && (h.name !== self.selectedMech || h.attached_to)) {
            h = null;
        }
        if (s) {
            self._beginSourceDrag(s, pt, ev.shiftKey);
            ev.preventDefault();
        } else if (o) {
            self._beginOpticDrag(o, pt, ev.shiftKey);
            ev.preventDefault();
        } else if (h) {
            self._beginMechDrag(h, pt, ev.shiftKey);
            ev.preventDefault();
        }
        dragging = true; moved = 0;
        lastX = ev.clientX; lastY = ev.clientY;
        self.svg.classList.add('gt-dragging');
    });

    on(global, 'mousemove', function (ev) {
        var r = self.svg.getBoundingClientRect();
        if (self.dragMechResize) {
            moved += Math.abs(ev.clientX - lastX) + Math.abs(ev.clientY - lastY);
            lastX = ev.clientX; lastY = ev.clientY;
            self._updateMechResize(
                self.screenToScene(ev.clientX - r.left, ev.clientY - r.top));
            return;
        }
        if (self.dragSource) {
            moved += Math.abs(ev.clientX - lastX) + Math.abs(ev.clientY - lastY);
            lastX = ev.clientX; lastY = ev.clientY;
            self._updateSourceDrag(
                self.screenToScene(ev.clientX - r.left, ev.clientY - r.top));
            return;
        }
        if (self.dragOptic) {
            moved += Math.abs(ev.clientX - lastX) + Math.abs(ev.clientY - lastY);
            lastX = ev.clientX; lastY = ev.clientY;
            self._updateOpticDrag(
                self.screenToScene(ev.clientX - r.left, ev.clientY - r.top),
                ev.ctrlKey, ev.altKey);
            return;
        }
        if (self.dragMech) {
            moved += Math.abs(ev.clientX - lastX) + Math.abs(ev.clientY - lastY);
            lastX = ev.clientX; lastY = ev.clientY;
            self._updateMechDrag(
                self.screenToScene(ev.clientX - r.left, ev.clientY - r.top));
            return;
        }
        if (dragging) {
            var dx = ev.clientX - lastX, dy = ev.clientY - lastY;
            moved += Math.abs(dx) + Math.abs(dy);
            lastX = ev.clientX; lastY = ev.clientY;
            self.cx -= dx / self.scale;
            self.cy += dy / self.scale;
            self._applyTransform();
            return;
        }
        if (ev.clientX < r.left || ev.clientX > r.right ||
            ev.clientY < r.top || ev.clientY > r.bottom) { return; }
        self._onHover(ev.clientX - r.left, ev.clientY - r.top);
    });

    on(global, 'mouseup', function (ev) {
        if (self.dragMechResize) {
            dragging = false;
            self.svg.classList.remove('gt-dragging');
            var rz = self.svg.getBoundingClientRect();
            self._updateMechResize(
                self.screenToScene(ev.clientX - rz.left, ev.clientY - rz.top));
            self._endMechResize(moved >= 4);
            return;
        }
        if (self.dragSource) {
            dragging = false;
            self.svg.classList.remove('gt-dragging');
            var rs = self.svg.getBoundingClientRect();
            if (moved < 4) {
                // A grab that went nowhere is a click on the laser,
                // which selects it. Let the click pipeline do that, so
                // that press-and-click agree on what was pointed at.
                self.dragSource = null;
                self._onClick(ev.clientX - rs.left, ev.clientY - rs.top,
                              ev.ctrlKey);
                return;
            }
            // Take the pose from where the cursor actually is at the
            // moment of release, rather than from the last movement
            // event, which may have been a pixel or two short.
            self._updateSourceDrag(
                self.screenToScene(ev.clientX - rs.left, ev.clientY - rs.top));
            self._endSourceDrag();
            return;
        }
        if (self.dragMech) {
            dragging = false;
            self.svg.classList.remove('gt-dragging');
            var rm = self.svg.getBoundingClientRect();
            if (moved < 4) {
                // A grab that went nowhere is a click on the selected
                // hardware, which keeps it selected. Let the click
                // pipeline say so, as it does for the others.
                self.dragMech = null;
                self._onClick(ev.clientX - rm.left, ev.clientY - rm.top,
                              ev.ctrlKey);
                return;
            }
            self._updateMechDrag(
                self.screenToScene(ev.clientX - rm.left, ev.clientY - rm.top));
            self._endMechDrag();
            return;
        }
        if (self.dragOptic) {
            dragging = false;
            self.svg.classList.remove('gt-dragging');
            var ru = self.svg.getBoundingClientRect();
            if (moved < 4) {
                // A grab that went nowhere is a click. Hand it to the
                // click pipeline, whose repeated-click cycle can step
                // from the element to the beams under it.
                self.dragOptic = null;
                self._onClick(ev.clientX - ru.left, ev.clientY - ru.top,
                              ev.ctrlKey);
                return;
            }
            // Re-read the pose at the moment of release: Ctrl or Alt
            // may have been pressed or let go since the last movement,
            // and it is the state on release the user is answering for.
            self._updateOpticDrag(
                self.screenToScene(ev.clientX - ru.left, ev.clientY - ru.top),
                ev.ctrlKey, ev.altKey);
            self._endOpticDrag();
            return;
        }
        if (!dragging) { return; }
        dragging = false;
        self.svg.classList.remove('gt-dragging');
        if (moved < 4) {
            var r = self.svg.getBoundingClientRect();
            self._onClick(ev.clientX - r.left, ev.clientY - r.top,
                          ev.ctrlKey);
        }
    });

    on(global, 'keydown', function (ev) {
        if (VIEWERS.length > 1 && !self.pointerInside) { return; }
        // Not while a property field has the keyboard.
        if (ev.target && ev.target.classList
            && ev.target.classList.contains('gt-input')) { return; }
        if (ev.key === 'f' || ev.key === 'F') { self.fit(); }
        // Ctrl+Z and Ctrl+Shift+Z / Ctrl+Y, but only with the pointer
        // over this viewer: the page around a notebook cell has its own
        // undo, and taking the keys from everywhere would undo edits
        // the user meant for that. Both spellings of redo are bound
        // because which one is the habit depends on the platform.
        if ((ev.key === 'z' || ev.key === 'Z') && (ev.ctrlKey || ev.metaKey)
                && self.pointerInside) {
            if (ev.shiftKey) {
                if (self.redo()) { ev.preventDefault(); }
            } else if (self.undo()) { ev.preventDefault(); }
        }
        if ((ev.key === 'y' || ev.key === 'Y') && (ev.ctrlKey || ev.metaKey)
                && !ev.shiftKey && self.pointerInside) {
            if (self.redo()) { ev.preventDefault(); }
        }
        // Measuring: 'm' arms the tool, Escape puts it away. Escape
        // clears the selection too, as it always has - a measurement
        // half placed is one more thing it is letting go of.
        if ((ev.key === 'm' || ev.key === 'M') && !ev.ctrlKey && !ev.metaKey) {
            self.toggleMeasure();
        }
        // Aiming the selected optics: the two ways of naming an angle
        // by places, and the quarter turn. See ALIGN_ITEMS, which
        // carries the same keys into the menu's tooltips.
        if (!ev.ctrlKey && !ev.metaKey) {
            if (ev.key === 'a' || ev.key === 'A') { self.startAlign(2); }
            if (ev.key === 'b' || ev.key === 'B') { self.startAlign(3); }
            if (ev.key === ']') { self.turnSelected(45); }
            if (ev.key === '[') { self.turnSelected(-45); }
        }
        if (ev.key === 'Escape') {
            // An open menu is the innermost thing Escape can close, so
            // it goes first and the selection is left alone.
            var wasOpen = (self.addMenus || []).some(function (m) {
                return m.menu.style.display !== 'none';
            });
            self.closeAddMenus();
            if (wasOpen) { return; }
            // An aim half taken is the next innermost thing to let go
            // of, and letting go of it should not also clear the
            // selection it was being taken for.
            if (self.cancelAlign()) { return; }
            if (self.measuring) { self.toggleMeasure(false); }
            self.pinned = null;
            self.selectedOptic = null;
            self.selectedDim = null;
            self.selectedSource = null;
            self.selectedMech = null;
            self._setReadout(null);
            self._showPanel('beam');
            self._updateOverlay();
        }
    });
};

//}}}

//{{{ Editing

/*
 * Dragging an optics.
 *
 * The drag is previewed locally with an outline and only committed on
 * release, as one edit message. Python owns the model: it applies the
 * edit, re-traces and sends back a whole new scene. Nothing here tries
 * to guess what the beams will do.
 */
/*
 * How near a beam has to pass, in screen pixels, for a Ctrl-drag to
 * snap onto it. A floor rather than the whole rule: the beam counts as
 * caught if it passes anywhere within the element itself, and this only
 * adds a little reach around a small element in a wide view.
 */
var SNAP_TOL = 12;

Viewer.prototype._beginOpticDrag = function (optic, scenePt, rotate) {
    var c = optic.center || optic.HRcenter;
    // An optics turns about the point it is held by, not about the
    // middle of its substrate: a mirror swings about the apex of its
    // HR face, so that turning it does not walk the beam spot off it.
    // That is the same point Python keeps fixed, so the outline shown
    // while dragging is where the element actually ends up.
    var pivot = opticAnchorPoint(optic);
    this.dragOptic = {
        optic: optic,
        rotate: !!rotate,
        grab: scenePt,
        center0: [c[0], c[1]],
        pivot: [pivot[0], pivot[1]],
        angle0: optic.normAngleHR || 0,
        // Angle of the grab point as seen from the pivot, so that the
        // optics turns with the cursor instead of jumping to it.
        grabAngle: Math.atan2(scenePt[1] - pivot[1], scenePt[0] - pivot[0]),
        center: [c[0], c[1]],
        angle: optic.normAngleHR || 0,
        snap: null
    };
    this._updateOpticOutline(optic, this.dragOptic.center,
                             this.dragOptic.angle);
};

/*
 * Where the middle of the substrate goes when the element is turned by
 * da about the point it is held by.
 */
function _centreAfterTurn(d, da, pivot) {
    var ca = Math.cos(da), sa = Math.sin(da);
    var ox = d.center0[0] - d.pivot[0], oy = d.center0[1] - d.pivot[1];
    return [pivot[0] + ox * ca - oy * sa, pivot[1] + ox * sa + oy * ca];
}

Viewer.prototype._updateOpticDrag = function (scenePt, snap, free) {
    var d = this.dragOptic;
    if (!d) { return; }
    if (d.rotate) {
        var a = Math.atan2(scenePt[1] - d.pivot[1], scenePt[0] - d.pivot[0]);
        d.angle = d.angle0 + (a - d.grabAngle);
        d.center = _centreAfterTurn(d, d.angle - d.angle0, d.pivot);
    } else {
        var dx = scenePt[0] - d.grab[0], dy = scenePt[1] - d.grab[1];
        d.center = [d.center0[0] + dx, d.center0[1] + dy];
        d.snap = null;
        d.hole = null;
        // Held down, Ctrl asks for the element to be put on the beam it
        // was dropped over properly - square to it, and centred on it -
        // rather than merely near it. The preview shows the answer, so
        // that the modifier is not a leap of faith.
        //
        // What has to be over the beam is the element, not the cursor.
        // An element is grabbed wherever the user took hold of it, and
        // zoomed in that can be a long way from the point it is held
        // by; testing the cursor made the whole thing stop working
        // above a certain zoom. So the test is at the held point, and
        // a beam passing anywhere within the element's own footprint
        // counts - which is what "dropped it on that beam" means, and
        // does not shrink as the view is zoomed in.
        if (snap) {
            var ax = d.pivot[0] + dx, ay = d.pivot[1] + dy;
            var tol = Math.max(opticRadius(d.optic), SNAP_TOL / this.scale);
            var hit = this._pick(ax, ay, tol);
            if (hit) {
                var v = hit.beam.dirVect;
                d.angle = Math.atan2(-v[1], -v[0]);
                d.center = _centreAfterTurn(d, d.angle - d.angle0, hit.point);
                d.snap = {beam: hit.beam.name, index: hit.index,
                          point: hit.point};
            }
        } else if (!free) {
            // Riding over a screw hole, the anchor point lands on it
            // exactly: the holes are where a bench actually puts
            // things, and the anchor is the point the element is held
            // by. Alt rides free. Only a nudge - the reach is small
            // against the grid - so anywhere off the holes still
            // means where it says.
            var hx = d.pivot[0] + dx, hy = d.pivot[1] + dy;
            var hole = this._nearestHole(hx, hy);
            if (hole) {
                d.center = [d.center[0] + hole.point[0] - hx,
                            d.center[1] + hole.point[1] - hy];
                d.hole = hole;
            }
        }
    }
    this._updateOpticOutline(d.optic, d.center, d.angle);
    this._updateOverlay();
    this._updateStatus();
};

/*
 * How far a screw hole reaches for a dragged anchor, at most, in
 * metres. The screen-pixel reach shrinks it further when zoomed in;
 * the cap is what keeps a whole 25 mm grid from being one big magnet
 * when zoomed out - it stays well under half a pitch, so the space
 * between holes still exists.
 */
var HOLE_SNAP_MAX = 0.008;

Viewer.prototype._nearestHole = function (x, y) {
    var reach = Math.min(SNAP_RADIUS / this.scale, HOLE_SNAP_MAX);
    var best = null, bestD = reach;
    var snaps = this.scene.snap || [];
    for (var i = 0; i < snaps.length; i++) {
        if (snaps[i].kind !== 'hole') { continue; }
        var p = snaps[i].point;
        var d = Math.hypot(x - p[0], y - p[1]);
        if (d < bestD) { best = snaps[i]; bestD = d; }
    }
    return best;
};

/*
 * Dragging a source.
 *
 * Kept apart from the optics drag rather than folded into it: the two
 * share the shape of the gesture and nothing else. An optics is held by
 * an anchor that is not its middle, is squared onto beams with Ctrl,
 * and lands as a 'move' of its substrate centre; a laser is held where
 * its light comes from, has nothing to be squared onto - it is what the
 * beams are square to - and lands as a 'move' of that point.
 */
Viewer.prototype._beginSourceDrag = function (source, scenePt, rotate) {
    var p = source.pos;
    this.dragSource = {
        source: source,
        rotate: !!rotate,
        grab: scenePt,
        pos0: [p[0], p[1]],
        angle0: source.dirAngle || 0,
        // Where the grab point stands as seen from the laser, so that
        // turning follows the cursor instead of jumping to it.
        grabAngle: Math.atan2(scenePt[1] - p[1], scenePt[0] - p[0]),
        pos: [p[0], p[1]],
        angle: source.dirAngle || 0
    };
    this._updateOverlay();
};

Viewer.prototype._updateSourceDrag = function (scenePt) {
    var d = this.dragSource;
    if (!d) { return; }
    if (d.rotate) {
        // About the point the light leaves from, which is the one the
        // model keeps fixed when dirAngle is assigned: the beam swings
        // and the laser stays where it was put.
        var a = Math.atan2(scenePt[1] - d.pos0[1], scenePt[0] - d.pos0[0]);
        d.angle = d.angle0 + (a - d.grabAngle);
    } else {
        d.pos = [d.pos0[0] + scenePt[0] - d.grab[0],
                 d.pos0[1] + scenePt[1] - d.grab[1]];
    }
    this._updateOverlay();
    this._updateStatus();
};

Viewer.prototype._endSourceDrag = function () {
    var d = this.dragSource;
    this.dragSource = null;
    if (!d) { return; }
    this._selectSource(d.source);
    if (!this.onEdit) { return; }
    this.onEdit(d.rotate
        ? {op: 'rotate', target: d.source.name, dirAngle: d.angle}
        : {op: 'move', target: d.source.name, pos: d.pos});
};

/*
 * Dragging a mechanics.
 *
 * The same gesture as the others - drag to move, Shift-drag to turn -
 * over the simplest body of the three: a mechanics turns about its own
 * center, which is the local origin its shapes are drawn from, so the
 * preview is its outline carried and turned and nothing more.
 */
Viewer.prototype._beginMechDrag = function (mech, scenePt, rotate) {
    // Belt and braces: the mousedown handler already refuses these.
    if (mech.attached_to) { return; }
    var c = mech.center;
    this.dragMech = {
        mech: mech,
        rotate: !!rotate,
        grab: scenePt,
        center0: [c[0], c[1]],
        angle0: mech.rotationAngle || 0,
        grabAngle: Math.atan2(scenePt[1] - c[1], scenePt[0] - c[0]),
        center: [c[0], c[1]],
        angle: mech.rotationAngle || 0
    };
    this._updateOverlay();
};

Viewer.prototype._updateMechDrag = function (scenePt) {
    var d = this.dragMech;
    if (!d) { return; }
    if (d.rotate) {
        var a = Math.atan2(scenePt[1] - d.center0[1],
                           scenePt[0] - d.center0[0]);
        d.angle = d.angle0 + (a - d.grabAngle);
    } else {
        d.center = [d.center0[0] + scenePt[0] - d.grab[0],
                    d.center0[1] + scenePt[1] - d.grab[1]];
    }
    this._updateOverlay();
    this._updateStatus();
};

Viewer.prototype._endMechDrag = function () {
    var d = this.dragMech;
    this.dragMech = null;
    if (!d) { return; }
    this._selectMech(d.mech);
    if (!this.onEdit) { return; }
    this.onEdit(d.rotate
        ? {op: 'rotate', target: d.mech.name, rotationAngle: d.angle}
        : {op: 'move', target: d.mech.name, center: d.center});
};

Viewer.prototype._endOpticDrag = function () {
    var d = this.dragOptic;
    this.dragOptic = null;
    if (!d) { return; }
    // Show the properties of whatever was just moved.
    this._selectOptic(d.optic);
    if (!this.onEdit) { return; }

    var msg;
    if (d.snap) {
        // Python does the geometry, from the beam objects themselves
        // rather than from the copy of them in this scene.
        msg = {op: 'align', target: d.optic.name, beam: d.snap.beam,
               beam_index: d.snap.index, point: d.snap.point};
    } else if (d.rotate) {
        // Python turns an optics about its anchor point, which is the
        // pivot the preview turned it about, so the angle says it all.
        msg = {op: 'rotate', target: d.optic.name, normAngleHR: d.angle};
    } else {
        // 'center' is the middle of the substrate, which is the trait
        // the outline was built from, so the optics lands where it was
        // dropped.
        msg = {op: 'move', target: d.optic.name, center: d.center};
    }
    this.onEdit(msg);
};

//}}}

//{{{ Teardown

/*
 * Detach the viewer: remove every listener it installed and take its DOM
 * out of the page. anywidget calls this when the output area goes away.
 */
Viewer.prototype.destroy = function () {
    (this._listeners || []).forEach(function (l) {
        l[0].removeEventListener(l[1], l[2], l[3]);
    });
    this._listeners = [];
    if (this._ro) { this._ro.disconnect(); this._ro = null; }
    var i = VIEWERS.indexOf(this);
    if (i >= 0) { VIEWERS.splice(i, 1); }
    if (this.root && this.root.parentNode) {
        this.root.parentNode.removeChild(this.root);
    }
    // A sibling of the root rather than a child of it, so it has to be
    // taken out on its own.
    if (this.resizeGrip && this.resizeGrip.parentNode) {
        this.resizeGrip.parentNode.removeChild(this.resizeGrip);
    }
};

/*
 * All visible beams within tol of a scene point, nearest first.
 *
 * Several beams routinely lie on top of each other: a beam and its
 * counter-propagating return share the same line, and stray beams often
 * run along the main beam. They cannot be told apart by position, so
 * the caller cycles through this list on repeated clicks.
 */
Viewer.prototype._pickAll = function (sx, sy, tol) {
    var hits = [];
    var beams = this.scene.beams || [];
    for (var i = 0; i < beams.length; i++) {
        var b = beams[i];
        var g = this.layerGroups[b.layer];
        if (g && !g.visible) { continue; }
        var pr = projectOnBeam(b, sx, sy);
        if (pr.dist <= tol) {
            hits.push({beam: b, d: pr.d, point: pr.foot, dist: pr.dist,
                       index: i});
        }
    }
    hits.sort(function (a, b) {
        return a.dist !== b.dist ? a.dist - b.dist : a.index - b.index;
    });
    for (var j = 0; j < hits.length; j++) {
        hits[j].rank = j;
        hits[j].count = hits.length;
    }
    return hits;
};

/*
 * The visible beam closest to a scene point, or null.
 */
Viewer.prototype._pick = function (sx, sy, tol) {
    var hits = this._pickAll(sx, sy, tol);
    return hits.length ? hits[0] : null;
};

/*
 * The optics under a scene point, if any. Only the enclosing circle of
 * the substrate is tested: this is a grab handle, not a rendering.
 */
Viewer.prototype._pickOptic = function (sx, sy) {
    var best = null, bestD = Infinity;
    var optics = this.scene.optics || [];
    for (var i = 0; i < optics.length; i++) {
        var o = optics[i];
        var c = o.center || o.HRcenter;
        if (!c) { continue; }
        var d = Math.hypot(sx - c[0], sy - c[1]);
        if (d <= opticRadius(o) && d < bestD) { best = o; bestD = d; }
    }
    return best;
};

/*
 * The mechanics under a scene point, or null. Tested against the
 * outline polygon by area rather than against an enclosing circle: a
 * breadboard is huge, and a circle around it would cover the bench.
 * Of several hits the smallest wins, so a mount standing on a
 * breadboard is not shadowed by it.
 */
Viewer.prototype._pickMech = function (sx, sy) {
    var best = null, bestArea = Infinity;
    var mechs = this.scene.mechanics || [];
    for (var i = 0; i < mechs.length; i++) {
        var m = mechs[i];
        var g = this.layerGroups[m.layer];
        if (g && !g.visible) { continue; }
        if (!m.outline || m.outline.length < 3) { continue; }
        if (!pointInPolygon(sx, sy, m.outline)) { continue; }
        var area = polygonArea(m.outline);
        if (area < bestArea) { best = m; bestArea = area; }
    }
    return best;
};

/*
 * Where the outline of a mechanics falls for a trial pose. The scene
 * carries the outline for the pose Python knows; a drag turns and
 * carries those same points, so the preview costs no geometry.
 */
Viewer.prototype._mechOutlinePoints = function (m, center, angle) {
    var c0 = m.center, a0 = m.rotationAngle || 0;
    var cx = center === undefined ? c0[0] : center[0];
    var cy = center === undefined ? c0[1] : center[1];
    var da = (angle === undefined ? a0 : angle) - a0;
    var ca = Math.cos(da), sa = Math.sin(da);
    return (m.outline || []).map(function (p) {
        var ox = p[0] - c0[0], oy = p[1] - c0[1];
        return [cx + ox * ca - oy * sa, cy + ox * sa + oy * ca];
    });
};

Viewer.prototype._updateMechOutline = function (m, center, angle) {
    if (!m) { this.mechOutline.style.display = 'none'; return; }
    var self = this;
    var pts = this._mechOutlinePoints(m, center, angle).map(function (p) {
        var s = self.sceneToScreen(p[0], p[1]);
        return s[0] + ',' + s[1];
    });
    this.mechOutline.setAttribute('points', pts.join(' '));
    this.mechOutline.classList.toggle('gt-dragging', !!this.dragMech);
    this.mechOutline.classList.toggle(
        'gt-selected',
        !this.dragMech && !this.hoverMech && m.name === this.selectedMech);
    this.mechOutline.style.display = '';
};

/*
 * The outline as a bare polygon of world points: what the resize
 * preview draws, since mid-resize there is no body of that size to
 * derive one from yet.
 */
Viewer.prototype._setMechOutlinePts = function (worldPts, dragging) {
    var self = this;
    this.mechOutline.setAttribute('points', worldPts.map(function (p) {
        var s = self.sceneToScreen(p[0], p[1]);
        return s[0] + ',' + s[1];
    }).join(' '));
    this.mechOutline.classList.toggle('gt-dragging', !!dragging);
    this.mechOutline.classList.remove('gt-selected');
    this.mechOutline.style.display = '';
};

/*
 * Stand the corner handles on a rectangle, remembering where they are
 * for the mousedown hit test.
 */
Viewer.prototype._placeMechHandles = function (center, w, h, angle) {
    var self = this;
    this._handlePts = rectCorners(center, w, h, angle).map(function (p) {
        return self.sceneToScreen(p[0], p[1]);
    });
    this.mechHandles.forEach(function (el, i) {
        el.setAttribute('x', self._handlePts[i][0] - 3.5);
        el.setAttribute('y', self._handlePts[i][1] - 3.5);
        el.style.display = '';
    });
};

Viewer.prototype._hideMechHandles = function () {
    this._handlePts = null;
    (this.mechHandles || []).forEach(function (el) {
        el.style.display = 'none';
    });
};

/*
 * The handle under a screen point, or -1. A little reach beyond the
 * drawn square, since a grip that has to be hit to the pixel is a
 * grip that gets missed.
 */
Viewer.prototype._pickMechHandle = function (px, py) {
    if (!this._handlePts) { return -1; }
    for (var i = 0; i < this._handlePts.length; i++) {
        if (Math.abs(px - this._handlePts[i][0]) <= 6
                && Math.abs(py - this._handlePts[i][1]) <= 6) {
            return i;
        }
    }
    return -1;
};

/*
 * Dragging a corner of a resizable body. The opposite corner stays
 * put - that is what dragging a corner of anything means - and on
 * release Python re-drills the board at the new size, which is why
 * this is not a scale: the holes keep their diameter and their pitch.
 */
var MECH_MIN_SIZE = 0.01;

Viewer.prototype._beginMechResize = function (mech, corner) {
    if (!mech) { return; }
    var a = mech.rotationAngle || 0;
    this.dragMechResize = {
        mech: mech,
        angle: a,
        center: [mech.center[0], mech.center[1]],
        width: mech.width,
        height: mech.height,
        fixed: rectCorners(mech.center, mech.width, mech.height,
                           a)[(corner + 2) % 4]
    };
    this._updateOverlay();
};

Viewer.prototype._updateMechResize = function (scenePt) {
    var r = this.dragMechResize;
    if (!r) { return; }
    function rot(p, ang) {
        var c = Math.cos(ang), s = Math.sin(ang);
        return [p[0] * c - p[1] * s, p[0] * s + p[1] * c];
    }
    // Both the cursor and the fixed corner into the body's own frame,
    // where the rectangle is axis-aligned and the arithmetic is two
    // absolute values.
    var u = rot(scenePt, -r.angle);
    var f = rot(r.fixed, -r.angle);
    var w = Math.max(MECH_MIN_SIZE, Math.abs(u[0] - f[0]));
    var h = Math.max(MECH_MIN_SIZE, Math.abs(u[1] - f[1]));
    // The centre is half a size from the fixed corner, towards the
    // cursor - which keeps that corner fixed even when the size hits
    // its floor.
    var sx = u[0] >= f[0] ? 1 : -1;
    var sy = u[1] >= f[1] ? 1 : -1;
    r.width = w;
    r.height = h;
    r.center = rot([f[0] + sx * w / 2, f[1] + sy * h / 2], r.angle);
    this._updateOverlay();
    this._updateStatus();
};

Viewer.prototype._endMechResize = function (commit) {
    var r = this.dragMechResize;
    this.dragMechResize = null;
    if (!r) { return; }
    this._updateOverlay();
    // A press on a handle that never moved decided nothing.
    if (!commit || !this.onEdit) { return; }
    this.onEdit({op: 'set', target: r.mech.name,
                 attrs: {width: r.width, height: r.height,
                         center: r.center}});
};

Viewer.prototype._onHover = function (px, py) {
    var pt = this.screenToScene(px, py);
    this.cursor = pt;

    // While aiming, as while measuring, the question is only where
    // the next click lands - and what the optics would then face.
    if (this.aligning) {
        this.alignPreview = this._measurePoint(pt[0], pt[1]);
        this.hoverOptic = null;
        this.hoverSource = null;
        this.hoverMech = null;
        this.hover = null;
        this._updateOverlay();
        this._updateStatus();
        return;
    }

    // While measuring, nothing under the cursor is being pointed at: the
    // question is only where the next click lands. For the first two
    // that is the nearest marked point if there is one; for the third it
    // is how far aside the dimension line goes, which snaps to nothing.
    if (this.measuring) {
        if (this.measureTo) {
            this.measureOffset = this._offsetAt(pt[0], pt[1]);
            this.snapped = null;
            this.measurePreview = null;
        } else {
            this.measurePreview = this._measurePoint(pt[0], pt[1]);
        }
        this.hoverOptic = null;
        this.hoverSource = null;
        this.hoverMech = null;
        this.hover = null;
        this._updateOverlay();
        this._updateStatus();
        return;
    }
    this.measurePreview = null;

    // An optics or a laser under the cursor takes precedence: it is
    // what the next mousedown would act on, so say so before the user
    // presses. The laser comes first for the same reason a dimension
    // does - it is a small mark that a large element would otherwise
    // shadow, and a source usually sits at the end of its own beam.
    this.hoverSource = this._pickSource(px, py);
    this.hoverOptic = this.hoverSource ? null
        : this._pickOptic(pt[0], pt[1]);
    // The hardware comes last, as it does in the click order: it is
    // the largest thing in the picture, and everything else stands on
    // or in front of it.
    this.hoverMech = (this.hoverSource || this.hoverOptic) ? null
        : this._pickMech(pt[0], pt[1]);
    var over = this.hoverOptic || this.hoverSource;
    // A selected mechanics is grabbable; an unselected one only
    // selectable, and an attached one never - it goes where its host
    // goes. The cursor says which - see the mousedown handler for why
    // a breadboard is not grabbed until it is selected.
    var mechGrab = this.hoverMech && this.onEdit
        && this.hoverMech.name === this.selectedMech
        && !this.hoverMech.attached_to;
    this.svg.classList.toggle('gt-over-optic',
                              (!!over && !!this.onEdit) || !!mechGrab);
    this.svg.classList.toggle('gt-over-pickable',
                              (!!over && !this.onEdit)
                              || (!!this.hoverMech && !mechGrab));

    var hit = this._pick(pt[0], pt[1], 12 / this.scale);
    this.hover = hit;
    if (!this.pinned && this.panelKind !== 'optic') { this._setReadout(hit); }
    this._updateOverlay();
    this._updateStatus();
};

/*
 * Outline shown while an optics is hovered or dragged. Drawn in screen
 * coordinates from the scene-space corners.
 */
Viewer.prototype._updateOpticOutline = function (o, center, angle) {
    if (!o) { this.outline.style.display = 'none'; return; }
    var self = this;
    var pts = opticOutline(o, center, angle).map(function (p) {
        var s = self.sceneToScreen(p[0], p[1]);
        return s[0] + ',' + s[1];
    });
    this.outline.setAttribute('points', pts.join(' '));
    this.outline.classList.toggle('gt-dragging', !!this.dragOptic);
    this.outline.classList.toggle(
        'gt-selected',
        !this.dragOptic && !this.hoverOptic && o.name === this.selectedOptic);
    this.outline.style.display = '';
};

Viewer.prototype._onClick = function (px, py, pickBeamFor) {
    var pt = this.screenToScene(px, py);

    if (this.aligning) {
        this._onAlignClick(pt[0], pt[1]);
        return;
    }

    if (this.measuring) {
        this._onMeasureClick(pt[0], pt[1]);
        return;
    }

    // A dimension is picked before the optics and the beams under it.
    // It has to be: the measurement this whole tool exists for - the
    // optical thickness of a substrate - lies wholly inside an element,
    // and an element that took precedence would leave it unreachable.
    // It costs little the other way, since a dimension is a line a few
    // pixels wide and an element is an area: the element is still there
    // to be clicked anywhere off the line.
    var dim = this._pickDimension(pt[0], pt[1]);
    if (dim) {
        this.pinned = null;
        this.selectedOptic = null;
        this._selectDim(dim);
        return;
    }
    if (this.panelKind === 'dimension') {
        this.selectedDim = null;
        this._showPanel('beam');
    }

    // Then the lasers, ahead of the optics and the beams for the same
    // reason: a source sits at the end of its own beam and often right
    // against the first element, and a box of a few dozen pixels that
    // an element could shadow would be unreachable. It is small, so
    // the element is still there to be clicked anywhere off it.
    var source = this._pickSource(px, py);
    if (source) {
        this.pinned = null;
        this.lastClick = null;
        this.cycle = 0;
        this._selectSource(source);
        return;
    }
    if (this.panelKind === 'source') {
        this.selectedSource = null;
        this._showPanel('beam');
    }

    // Clicking an optics selects it and shows its properties. This works
    // whether or not the viewer is editable: reading is always allowed.
    // Near a surface the element and the beams that end on it overlap,
    // and the element, being an area, would shadow them for good - so
    // clicking the same spot again steps from the element into the
    // bundle of beams under it and back around, exactly as repeated
    // clicks already walk a bundle of overlapping beams.
    //
    // The hardware under the element takes the last turn of that walk.
    // It has to be in the cycle: a mount stands where its mirror
    // stands, so the mirror's own pick circle covers it entirely, and
    // there is no spot to click that reaches the mount any other way.
    var optic = this._pickOptic(pt[0], pt[1]);
    if (optic) {
        var under = this._pickAll(pt[0], pt[1], 12 / this.scale);
        var mechUnder = this._pickMech(pt[0], pt[1]);
        var slots = 1 + under.length + (mechUnder ? 1 : 0);
        var again = !pickBeamFor && this.lastClick &&
            Math.abs(px - this.lastClick[0]) < 5 &&
            Math.abs(py - this.lastClick[1]) < 5;
        this.cycle = again ? (this.cycle + 1) % slots : 0;
        this.lastClick = [px, py];
        if (this.cycle === 0) {
            this.pinned = null;
            this.selectedDim = null;
            this._selectOptic(optic);
            return;
        }
        if (this.cycle > under.length) {
            this.pinned = null;
            this.selectedOptic = null;
            this._selectMech(mechUnder);
            return;
        }
        this.selectedOptic = null;
        this.selectedMech = null;
        this._showPanel('beam');
        this.pinned = under[this.cycle - 1];
        this._setReadout(this.pinned);
        this._updateOverlay();
        return;
    }

    var hits = this._pickAll(pt[0], pt[1], 12 / this.scale);

    // With Ctrl held and an optics selected, a click on a beam names it
    // in the Along beam row instead of doing anything to the readout.
    // Ctrl already means "this optics, against that beam" in a drag, so
    // it means the same here, and the selection is left alone: picking
    // the beam to move along is part of working on that element, not a
    // move away from it.
    if (pickBeamFor && this.panelKind === 'optic' && hits.length) {
        // lastClick is deliberately left alone: it belongs to the
        // readout's cycle, and a plain click after this one should
        // start that cycle at the nearest beam rather than partway in.
        if (this._chooseSlideBeam(hits, px, py)) {
            this._refreshOpticPanel();
            this._updateOverlay();
            return;
        }
    }

    // The hardware is picked after everything else, as the largest
    // thing in the picture: a beam crossing a breadboard would be
    // unreachable the other way round, and the board is still there to
    // be clicked anywhere its beams are not.
    if (!hits.length) {
        var mech = this._pickMech(pt[0], pt[1]);
        if (mech) {
            this.pinned = null;
            this.cycle = 0;
            this.lastClick = [px, py];
            this._selectMech(mech);
            return;
        }
    }
    if (this.panelKind === 'mech') {
        this.selectedMech = null;
        this._showPanel('beam');
    }

    if (this.panelKind === 'optic') {
        // Leaving the optics: back to the beam readout.
        this.selectedOptic = null;
        this._showPanel('beam');
    }
    if (!hits.length) {
        this.pinned = null;
        this.cycle = 0;
    } else {
        // Clicking again at the same spot steps to the next beam of the
        // bundle, which is the only way to reach a beam that is hidden
        // underneath another one.
        var same = this.lastClick &&
            Math.abs(px - this.lastClick[0]) < 5 &&
            Math.abs(py - this.lastClick[1]) < 5;
        this.cycle = same ? (this.cycle + 1) % hits.length : 0;
        this.pinned = hits[this.cycle];
    }
    this.lastClick = [px, py];
    this._setReadout(this.pinned);
    this._updateOverlay();
};

//}}}

//{{{ Overlay and status

Viewer.prototype._showMarker = function (on) {
    this.marker.style.display = on ? '' : 'none';
    this.highlight.style.display = on ? '' : 'none';
    this.arrow.style.display = on ? '' : 'none';
};

/* Arrowhead geometry, in screen pixels. */
var ARROW_GAP = 8;      // clearance between the marker and the tail
var ARROW_LENGTH = 13;
var ARROW_HALFWIDTH = 5;

/*
 * An arrowhead just ahead of the readout point, pointing the way the
 * beam travels. Two beams sharing a line are told apart by this arrow:
 * clicking again swaps to the counter-propagating one and the arrow
 * flips over.
 */
Viewer.prototype._arrowPath = function (px, py, dirVect) {
    // The scene group is y-up and the screen is y-down, so the screen
    // direction is the scene direction with its y component negated.
    var ux = dirVect[0], uy = -dirVect[1];
    var vx = -uy, vy = ux;                       // unit normal
    var bx = px + ux * ARROW_GAP;
    var by = py + uy * ARROW_GAP;
    var tx = bx + ux * ARROW_LENGTH;
    var ty = by + uy * ARROW_LENGTH;
    return 'M ' + tx + ' ' + ty +
           ' L ' + (bx + vx * ARROW_HALFWIDTH) + ' ' + (by + vy * ARROW_HALFWIDTH) +
           ' L ' + (bx - vx * ARROW_HALFWIDTH) + ' ' + (by - vy * ARROW_HALFWIDTH) +
           ' Z';
};

/*
 * How far the arms of the origin cross reach, in screen pixels. A
 * mark rather than a shape: it keeps its size at any zoom, since what
 * it says is "here is zero" and that is true at every scale.
 */
var ORIGIN_ARM = 14;

Viewer.prototype._updateEditorMarks = function () {
    if (!this.scene.editor) {
        this.originMark.style.display = 'none';
        this.shapeMark.style.display = 'none';
        return;
    }
    var o = this.sceneToScreen(0, 0);
    this.originMark.setAttribute(
        'd', 'M ' + (o[0] - ORIGIN_ARM) + ' ' + o[1] +
             ' L ' + (o[0] + ORIGIN_ARM) + ' ' + o[1] +
             ' M ' + o[0] + ' ' + (o[1] - ORIGIN_ARM) +
             ' L ' + o[0] + ' ' + (o[1] + ORIGIN_ARM));
    this.originMark.style.display = '';

    var s = this._selectedShape();
    var box = s ? shapeBounds(s) : null;
    if (!box) {
        this.shapeMark.style.display = 'none';
        return;
    }
    var self = this;
    this.shapeMark.setAttribute('points', box.map(function (p) {
        var q = self.sceneToScreen(p[0], p[1]);
        return q[0] + ',' + q[1];
    }).join(' '));
    this.shapeMark.style.display = '';
};

Viewer.prototype._updateOverlay = function () {
    // Set while an aim is being taken; see the aiming block below and
    // the outline that reads it.
    var alignOutline = null;

    // The origin of the part, and the box round the shape on show.
    this._updateEditorMarks();

    // The lasers stand where the scene says, or where a drag has them.
    // Done here rather than in _applyTransform so that every path which
    // changes what is selected or hovered brings them along.
    this._updateSources();

    // The measurement being placed. Drawn here rather than through the
    // scene because Python has not been told about it yet - there is
    // nothing to tell until the last click - so this is the one piece of
    // a dimension the viewer works out for itself.
    //
    // Between the first two clicks it is a bare line to the cursor;
    // after them it is the dimension itself, drawn by the same code that
    // draws the finished ones, so that what is previewed is what will
    // appear.
    var pending = this._pendingDim();
    if (pending) {
        this.rubber.style.display = 'none';
        if (!this.pendingEls) {
            this.pendingEls = this._buildDimEls(pending, this.overlayGroup,
                                                'gt-dim-pending');
        }
        this.pendingEls.dim = pending;
        this._setDimText(this.pendingEls, pending);
        this.pendingEls.group.style.display = '';
        this._placeDim(this.pendingEls, false);
    } else {
        if (this.pendingEls) { this.pendingEls.group.style.display = 'none'; }
        if (this.measureFrom && this.cursor) {
            var to = this.measurePreview || this.cursor;
            var m0 = this.sceneToScreen(this.measureFrom[0],
                                        this.measureFrom[1]);
            var m1 = this.sceneToScreen(to[0], to[1]);
            this.rubber.setAttribute('x1', m0[0]);
            this.rubber.setAttribute('y1', m0[1]);
            this.rubber.setAttribute('x2', m1[0]);
            this.rubber.setAttribute('y2', m1[1]);
            this.rubber.style.display = '';
        } else {
            this.rubber.style.display = 'none';
        }
    }
    // The places an aim has been given, joined up to the cursor, and
    // the optics outlined as it would face. Both are the whole of what
    // makes the tool answerable: the angle is arithmetic on points the
    // user cannot otherwise see the effect of.
    if (this.aligning) {
        var apts = this.aligning.points.slice();
        if (this.alignPreview) { apts.push(this.alignPreview); }
        if (apts.length > 1) {
            var self2 = this;
            this.alignPath.setAttribute('points', apts.map(function (p) {
                var s = self2.sceneToScreen(p[0], p[1]);
                return s[0] + ',' + s[1];
            }).join(' '));
            this.alignPath.style.display = '';
        } else {
            this.alignPath.style.display = 'none';
        }
        var ao = this._selectedOptic();
        var aangle = this._alignAngle(this.alignPreview);
        if (ao && aangle !== null) {
            // Turning is about the anchor point, so the centre
            // travels; the preview has to travel with it or it would
            // promise a place the element will not land in. Kept for
            // the outline below, which is drawn in one place so that
            // a drag, an aim and a plain selection cannot each set it
            // and the last one win.
            var apivot = opticAnchorPoint(ao);
            alignOutline = {
                optic: ao,
                centre: turnAbout(ao.center || ao.HRcenter, apivot,
                                  aangle - (ao.normAngleHR || 0)),
                angle: aangle
            };
        }
    } else {
        this.alignPath.style.display = 'none';
    }

    // Where the next click would land, when that is a marked point
    // rather than the cursor. Without it the tool is guesswork: the
    // snap is invisible until the measurement is already made. The
    // same mark shows the screw hole a dragged anchor has caught on.
    var snapPt = ((this.measuring || this.aligning) && this.snapped)
        ? this.snapped.point
        : (this.dragOptic && this.dragOptic.hole)
            ? this.dragOptic.hole.point : null;
    if (snapPt) {
        var s = this.sceneToScreen(snapPt[0], snapPt[1]);
        this.snapMark.setAttribute('cx', s[0]);
        this.snapMark.setAttribute('cy', s[1]);
        this.snapMark.style.display = '';
    } else {
        this.snapMark.style.display = 'none';
    }

    if (this.dragOptic) {
        this._updateOpticOutline(this.dragOptic.optic, this.dragOptic.center,
                                 this.dragOptic.angle);
    } else if (alignOutline) {
        this._updateOpticOutline(alignOutline.optic, alignOutline.centre,
                                 alignOutline.angle);
    } else {
        // The selected optics stays outlined so that the panel and the
        // drawing agree on what is being looked at.
        this._updateOpticOutline(this.hoverOptic || this._selectedOptic());
    }

    // The hardware outline, by the same rules on its own element -
    // or, mid-resize, the rectangle being cut. The corner handles
    // stand on the selected resizable body, and follow the preview.
    if (this.dragMechResize) {
        var rz = this.dragMechResize;
        this._setMechOutlinePts(
            rectCorners(rz.center, rz.width, rz.height, rz.angle), true);
        this._placeMechHandles(rz.center, rz.width, rz.height, rz.angle);
    } else if (this.dragMech) {
        this._updateMechOutline(this.dragMech.mech, this.dragMech.center,
                                this.dragMech.angle);
        this._hideMechHandles();
    } else {
        this._updateMechOutline(this.hoverMech || this._selectedMech());
        var selMech = this._selectedMech();
        if (this.onEdit && selMech && selMech.resizable
                && !selMech.attached_to) {
            this._placeMechHandles(selMech.center, selMech.width,
                                   selMech.height,
                                   selMech.rotationAngle || 0);
        } else {
            this._hideMechHandles();
        }
    }

    // The beam the Along beam row names, marked along its whole length
    // so that the name in the panel and a line in the picture are the
    // same thing. Only while that panel is up: it belongs to the
    // element being worked on, not to the drawing.
    var sb = (this.panelKind === 'optic' && this.slideBeam)
        ? (this.scene.beams || [])[this.slideBeam.index] : null;
    if (sb && sb.name === this.slideBeam.name) {
        var q0 = this.sceneToScreen(sb.pos[0], sb.pos[1]);
        var q1 = this.sceneToScreen(sb.end[0], sb.end[1]);
        this.slideMark.setAttribute('x1', q0[0]);
        this.slideMark.setAttribute('y1', q0[1]);
        this.slideMark.setAttribute('x2', q1[0]);
        this.slideMark.setAttribute('y2', q1[1]);
        this.slideMark.style.display = '';
        // Halfway along, where there is room for it and where it cannot
        // be mistaken for the readout's arrow at the cursor.
        this.slideArrow.setAttribute(
            'd', this._arrowPath((q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2,
                                 sb.dirVect));
        this.slideArrow.style.display = '';
    } else {
        this.slideMark.style.display = 'none';
        this.slideArrow.style.display = 'none';
    }

    // Whether there is anything to aim can change with any click.
    this._refreshAlign();

    var hit = this.pinned || this.hover;
    if (!hit) { this._showMarker(false); return; }
    var b = hit.beam;
    var p0 = this.sceneToScreen(b.pos[0], b.pos[1]);
    var p1 = this.sceneToScreen(b.end[0], b.end[1]);
    var pm = this.sceneToScreen(hit.point[0], hit.point[1]);
    this.highlight.setAttribute('x1', p0[0]);
    this.highlight.setAttribute('y1', p0[1]);
    this.highlight.setAttribute('x2', p1[0]);
    this.highlight.setAttribute('y2', p1[1]);
    this.arrow.setAttribute('d', this._arrowPath(pm[0], pm[1], b.dirVect));
    this.marker.setAttribute('cx', pm[0]);
    this.marker.setAttribute('cy', pm[1]);
    this.marker.classList.toggle('gt-pinned', !!this.pinned);
    this.arrow.classList.toggle('gt-pinned', !!this.pinned);
    this._showMarker(true);
};

Viewer.prototype._updateStatus = function () {
    var s = this.dragSource;
    if (s) {
        this.statusBar.textContent = s.rotate
            ? s.source.name + ':  ' + fmtDeg(normAngle(s.angle)) +
              '   (was ' + fmtDeg(normAngle(s.angle0)) + ')'
            : s.source.name + ':  ' + fmtLen(s.pos[0]) + ',  ' +
              fmtLen(s.pos[1]);
        return;
    }
    var h = this.dragMech;
    if (h) {
        this.statusBar.textContent = h.rotate
            ? h.mech.name + ':  ' + fmtDeg(normAngle(h.angle)) +
              '   (was ' + fmtDeg(normAngle(h.angle0)) + ')'
            : h.mech.name + ':  ' + fmtLen(h.center[0]) + ',  ' +
              fmtLen(h.center[1]);
        return;
    }
    var d = this.dragOptic;
    if (d) {
        if (d.snap) {
            this.statusBar.textContent =
                d.optic.name + ':  square onto ' + d.snap.beam + ' at  ' +
                fmtLen(d.snap.point[0]) + ',  ' + fmtLen(d.snap.point[1]);
        } else {
            this.statusBar.textContent = d.rotate
                ? d.optic.name + ':  ' + fmtDeg(normAngle(d.angle)) +
                  '   (was ' + fmtDeg(normAngle(d.angle0)) + ')'
                : d.optic.name + ':  ' + fmtLen(d.center[0]) + ',  ' +
                  fmtLen(d.center[1]) +
                  (d.hole ? '   on ' + d.hole.label : '');
        }
        return;
    }
    if (this.aligning) {
        var al = this.aligning;
        var awhere = this.snapped ? this.snapped.label
            : (this.cursor ? fmtLen(this.cursor[0]) + ',  '
                             + fmtLen(this.cursor[1]) : '');
        var ordinal = ['first', 'second', 'third'][al.points.length]
            || 'next';
        var aangle2 = this._alignAngle(this.alignPreview);
        var ao2 = this._selectedOptic();
        this.statusBar.textContent =
            'Align ' + al.optic + ':  click the ' + ordinal + ' point' +
            (awhere ? '     at  ' + awhere : '') +
            (aangle2 === null ? ''
             : '     → ' + fmtDeg(normAngle(aangle2)) +
               (ao2 ? '   (was ' + fmtDeg(normAngle(ao2.normAngleHR || 0))
                      + ')' : '')) +
            '     (Esc to cancel)';
        return;
    }
    if (this.measuring) {
        var where = this.snapped ? this.snapped.label
            : (this.cursor ? fmtLen(this.cursor[0]) + ',  '
                             + fmtLen(this.cursor[1]) : '');
        var tail = '     (Esc to cancel)';
        if (this.measureTo) {
            // The last click places the line rather than measuring
            // anything, so the distance is settled and shown as such.
            this.statusBar.textContent =
                'Measure:  ' + fmtLen(Math.hypot(
                    this.measureTo[0] - this.measureFrom[0],
                    this.measureTo[1] - this.measureFrom[1])) +
                '     place the line:  offset ' +
                fmtLen(this.measureOffset || 0) + tail;
        } else if (this.measureFrom) {
            var to = this.measurePreview || this.cursor || this.measureFrom;
            this.statusBar.textContent =
                'Measure:  ' + fmtLen(Math.hypot(to[0] - this.measureFrom[0],
                                                 to[1] - this.measureFrom[1])) +
                '     to  ' + where + tail;
        } else {
            this.statusBar.textContent =
                'Measure:  click the first point' +
                (where ? '     at  ' + where : '') + tail;
        }
        return;
    }
    var parts = [];
    if (this.cursor) {
        parts.push('x = ' + fmtLen(this.cursor[0]) +
                   ',  y = ' + fmtLen(this.cursor[1]));
    }
    parts.push('scale: ' + fmtNum(this.scale, 3) + ' px/m');
    var n = (this.scene.beams || []).length;
    parts.push(n + ' beam' + (n === 1 ? '' : 's'));
    this.statusBar.textContent = parts.join('     ');
};

//}}}

//{{{ Readout

Viewer.prototype._setReadout = function (hit) {
    var c = this.cells;
    function set(key, xval, yval) {
        c[key][0].textContent = xval;
        if (c[key].length > 1) { c[key][1].textContent = yval; }
    }
    if (!hit) {
        for (var k in c) {
            c[k][0].textContent = '-';
            if (c[k].length > 1) { c[k][1].textContent = '-'; }
        }
        this.pinLabel.textContent = '';
        this.readoutBody.classList.add('gt-empty');
        return;
    }
    this.readoutBody.classList.remove('gt-empty');
    var tag = this.pinned ? 'pinned' : '';
    if (hit.count > 1) {
        // Tell the user that more beams pass through this point.
        tag += (tag ? ' ' : '') + (hit.rank + 1) + '/' + hit.count;
    }
    this.pinLabel.textContent = tag;

    var b = hit.beam;
    var p = beamParamsAt(b, hit.d);
    set('beam', b.name);
    set('layer', b.layer);
    set('dir', fmtDeg(normAngle(b.dirAngle)) + '   (' +
               b.dirVect[0].toFixed(3) + ', ' + b.dirVect[1].toFixed(3) + ')');
    set('point', fmtLen(hit.point[0]) + ',  ' + fmtLen(hit.point[1]));
    set('dist', fmtLen(hit.d) + ' of ' + fmtLen(b.length));
    set('w', fmtLen(p.wx), fmtLen(p.wy));
    set('R', fmtLen(p.Rx), fmtLen(p.Ry));
    set('q', fmtQ(p.qx), fmtQ(p.qy));
    set('w0', fmtLen(p.w0x), fmtLen(p.w0y));
    set('waist', fmtLen(p.waistx), fmtLen(p.waisty));
    set('zR', fmtLen(p.zRx), fmtLen(p.zRy));
    set('Gouy', fmtDeg(p.Gouyx), fmtDeg(p.Gouyy));
    set('P', fmtNum(b.P) + ' W');
    set('wl', fmtLen(b.wl));
    set('n', fmtNum(b.n, 4));
    set('optDist', fmtLen(p.optDist));
    set('stray', String(b.stray_order));
};

//}}}

//{{{ Public API

/*
 * Replace the scene of a mounted viewer, keeping the current view.
 * Used by the notebook widget and the live server to push updates.
 */
Viewer.prototype.setScene = function (scene) {
    var visible = {};
    for (var name in this.layerGroups) {
        visible[name] = this.layerGroups[name].visible;
    }
    this.scene = scene;
    this.pinned = null;
    this.hover = null;
    this.hoverOptic = null;
    this.hoverSource = null;
    this.hoverMech = null;
    this.dragOptic = null;
    this.dragSource = null;
    this.dragMech = null;
    this.dragMechResize = null;
    // An aim is answered by the scene that comes back, and a scene
    // arriving for any other reason is a layout that may not hold the
    // element being aimed at all.
    this.aligning = null;
    this.alignPreview = null;
    this.cycle = 0;
    this.lastClick = null;
    this.labels = [];
    this.layerGroups = {};
    this.sceneGroup.textContent = '';
    this.labelGroup.textContent = '';
    this.dimGroup.textContent = '';
    this.sourceGroup.textContent = '';
    this.overlayGroup.textContent = '';
    this.layerBody.textContent = '';
    this.opts.hiddenLayers = Object.keys(visible).filter(function (k) {
        return !visible[k];
    });
    this._renderScene();
    this._refreshDisplayPanel();
    this._refreshRulesPanel();
    this._refreshHardwareMenu();
    this._refreshUndo();
    this._setReadout(null);

    // Editing a part: the selection is an index, and the list it
    // indexes has just been replaced. Anything past the end - the
    // shape that was removed, or an undone add - falls back to
    // nothing selected.
    if (this.scene.editor) {
        if (this.selectedShape !== null
                && this.selectedShape >= this._shapes().length) {
            this.selectedShape = this._shapes().length
                ? this._shapes().length - 1 : null;
        }
        this._refreshShapePanel();
        this._showPanel('shape');
        if (this.modelInput && this.scene.editor.model_name
                && document.activeElement !== this.modelInput) {
            this.modelInput.value = this.scene.editor.model_name;
        }
        this._applyTransform();
        return;
    }

    // A scene arriving after an edit describes the same optics, so keep
    // the selection and show the values Python came back with. Getting
    // one means the edit went through, so any optimistic rename stands.
    this.selectionFallback = null;
    this.dimFallback = null;
    this.sourceFallback = null;
    this.mechFallback = null;
    if (this._selectedDim()) {
        this._refreshDimPanel();
        this._showPanel('dimension');
    } else if (this._selectedSource()) {
        this._refreshSourcePanel();
        this._showPanel('source');
    } else if (this._selectedMech()) {
        this._refreshMechPanel();
        this._showPanel('mech');
    } else if (this._selectedOptic()) {
        this._refreshOpticPanel();
        this._showPanel('optic');
    } else if (this.panelKind === 'optic') {
        this.selectedOptic = null;
        this._showPanel('beam');
    } else if (this.panelKind === 'source') {
        // The source it was showing is gone - removed, or undone.
        this.selectedSource = null;
        this._showPanel('beam');
    } else if (this.panelKind === 'dimension') {
        // The dimension it was showing is gone - removed, or undone.
        this.selectedDim = null;
        this._showPanel('beam');
    } else if (this.panelKind === 'mech') {
        // The mechanics it was showing is gone - removed, or undone.
        this.selectedMech = null;
        this._showPanel('beam');
    }

    // A loaded layout can be anywhere; frame it rather than leaving the
    // view over wherever the previous one happened to be.
    if (this.fitOnNextScene) {
        this.fitOnNextScene = false;
        this.fit();
    } else {
        this._applyTransform();
    }
};

var GTraceViewer = {
    mount: function (container, scene, options) {
        return new Viewer(container, scene, options);
    },
    Viewer: Viewer,
    beamParamsAt: beamParamsAt,
    projectOnBeam: projectOnBeam,
    // What an embedder may not make the viewer shorter than, whether by
    // dragging the grip or by working a height out for itself. Exported
    // so that the widget does not carry a second copy of the number.
    MIN_HEIGHT: MIN_HEIGHT
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GTraceViewer;
}
global.GTraceViewer = GTraceViewer;

//}}}

})(typeof globalThis !== 'undefined' ? globalThis : window);
