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
     params: {curve_direction: 'h'}}
];

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

    this.labels = [];       // {el, x, y, rotation, layer}
    this.layerGroups = {};  // layer name -> {geom, label, visible}
    this.pinned = null;     // pinned readout {beam, d, point, rank, count}
    this.hover = null;
    this.cycle = 0;         // index into the bundle of overlapping beams
    this.lastClick = null;

    // Editing is available only when the transport gave us somewhere to
    // send the edits: the notebook widget and the live server do, the
    // static HTML file does not, so that file stays read-only.
    this.onEdit = this.opts.onEdit || null;
    this.hoverOptic = null;
    this.dragOptic = null;

    VIEWERS.push(this);
    this._build();
    this._renderScene();
    this._refreshDisplayPanel();
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
    this.overlayGroup = svgEl('g', {'class': 'gt-overlay'});
    this.svg.appendChild(this.sceneGroup);
    this.svg.appendChild(this.labelGroup);
    this.svg.appendChild(this.overlayGroup);
    stage.appendChild(this.svg);

    this.statusBar = htmlEl('div', 'gt-status');
    stage.appendChild(this.statusBar);
    root.appendChild(stage);

    // --- side bar ---
    var side = htmlEl('div', 'gt-side');

    var head = htmlEl('div', 'gt-head');
    head.appendChild(htmlEl('div', 'gt-title', this.opts.title || 'gtrace'));
    var buttons = htmlEl('div', 'gt-buttons');
    if (this.opts.onEdit) {
        ADDABLE_TYPES.forEach(function (t) {
            var btn = htmlEl('button', 'gt-btn', '+ ' + t.label);
            btn.title = t.title + ' at the centre of the view';
            btn.addEventListener('click', function () {
                self.addOptics(t.type);
            });
            buttons.appendChild(btn);
        });
    }
    var fitBtn = htmlEl('button', 'gt-btn', 'Fit');
    fitBtn.addEventListener('click', function () { self.fit(); });
    buttons.appendChild(fitBtn);
    head.appendChild(buttons);
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
    rpanel.appendChild(this.readoutBody);
    rpanel.appendChild(this.opticBody);
    side.appendChild(rpanel);
    this._buildReadout();
    this._buildOpticPanel();
    this._showPanel('beam');

    // Layout file panel. Editing in the browser is only worth anything
    // if the result can be taken out again, and the file has to be
    // written by Python: the page has no business touching the disk.
    if (this.onEdit) {
        var fpanel = htmlEl('div', 'gt-panel');
        fpanel.appendChild(htmlEl('div', 'gt-panel-title', 'Layout file'));
        var fbody = htmlEl('div', 'gt-file');
        this.pathInput = htmlEl('input', 'gt-input gt-input-text');
        this.pathInput.type = 'text';
        this.pathInput.spellcheck = false;
        this.pathInput.value = this.opts.layoutPath || 'layout.json';
        this.pathInput.title = 'Relative to where the kernel is running';
        fbody.appendChild(this.pathInput);
        var frow = htmlEl('div', 'gt-filebuttons');
        var saveBtn = htmlEl('button', 'gt-btn', 'Save');
        saveBtn.title = 'Write the layout to this file';
        saveBtn.addEventListener('click', function () { self.saveLayout(); });
        var loadBtn = htmlEl('button', 'gt-btn', 'Load');
        loadBtn.title = 'Replace the layout with the one in this file';
        loadBtn.addEventListener('click', function () { self.loadLayout(); });
        frow.appendChild(saveBtn);
        frow.appendChild(loadBtn);
        fbody.appendChild(frow);
        fpanel.appendChild(fbody);
        side.appendChild(fpanel);
    }

    // Display panel. These change how Python draws the scene, so they
    // exist only when there is a Python to ask.
    if (this.onEdit) {
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
    var rows = [['Wheel', 'zoom at cursor'],
                ['Drag', 'pan'],
                ['Move over a beam', 'live readout'],
                ['Click', 'pin the readout'],
                ['Click again', 'cycle overlapping beams'],
                ['Click an optics', 'show its properties'],
                ['f', 'fit to view'],
                ['Esc', 'clear selection']];
    if (this.opts.onEdit) {
        rows.push(['Drag an optics', 'move it'],
                  ['Shift + drag', 'rotate it'],
                  ['Edit a property', 'apply it to the layout'],
                  ['+ Mirror / + CyMirror',
                   'add one at the centre of the view'],
                  ['Remove', 'delete the selected optics']);
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
    this.container.appendChild(root);
    this.root = root;
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
    {key: 'cx', label: 'Center x', unit: 'm'},
    {key: 'cy', label: 'Center y', unit: 'm'},
    {key: 'angle', label: 'Angle', unit: '°'},
    {key: 'diameter', label: 'Diameter', unit: 'm'},
    {key: 'thickness', label: 'Thickness', unit: 'm'},
    {key: 'wedgeAngle', label: 'Wedge', unit: '°'},
    {key: 'rocHR', label: 'ROC HR', unit: 'm'},
    {key: 'rocAR', label: 'ROC AR', unit: 'm'},
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
    {key: 'term_on_HR', label: 'Terminate on HR', bool: true},
    {key: 'term_on_HR_order', label: 'Term. on HR order'},
    // Only a CyMirror has this; the row hides itself otherwise. Two
    // values exist, so it is a choice rather than something to type.
    {key: 'curve_direction', label: 'Curve direction', optional: true,
     choices: [['h', 'horizontal'], ['v', 'vertical']]}
];

var DEG = 180 / Math.PI;

function opticFieldValue(o, key) {
    var c = o.center || o.HRcenter || [0, 0];
    switch (key) {
    case 'cx': return c[0];
    case 'cy': return c[1];
    case 'angle': return normAngle(o.normAngleHR || 0) * DEG;
    case 'wedgeAngle': return (o.wedgeAngle || 0) * DEG;
    case 'rocHR': return o.inv_ROC_HR ? 1 / o.inv_ROC_HR : Infinity;
    case 'rocAR': return o.inv_ROC_AR ? 1 / o.inv_ROC_AR : Infinity;
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
    case 'rocHR':
        // A flat surface is an infinite radius, which is the inverse
        // being zero. Anything non-finite means flat.
        attrs.inv_ROC_HR = isFinite(value) && value !== 0 ? 1 / value : 0;
        break;
    case 'rocAR':
        attrs.inv_ROC_AR = isFinite(value) && value !== 0 ? 1 / value : 0;
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

Viewer.prototype._buildOpticPanel = function () {
    var self = this;
    var table = htmlEl('table');
    this.opticFields = {};

    OPTIC_FIELDS.forEach(function (f) {
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
            box.addEventListener('change', function () {
                self._commitOpticField(f.key, box);
            });
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
            sel.addEventListener('change', function () {
                self._commitOpticField(f.key, sel);
            });
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
                self._commitOpticField(f.key, input);
            });
            input.addEventListener('keydown', function (ev) {
                if (ev.key === 'Escape') {
                    self._refreshOpticPanel();
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
        self.opticFields[f.key] = rec;
    });

    this.opticBody.appendChild(table);

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
    var msg = {op: 'add', type: spec.type, name: name,
               params: Object.assign({
                   HRcenter: [this.cx, this.cy],
                   normAngleHR: Math.PI
               }, spec.params || {}, params || {})};
    // Optimistic: the scene that comes back will contain it, and
    // _selectedOptic() resolves the name then.
    this.selectedOptic = name;
    this.onEdit(msg);
    return msg;
};

/*
 * Kept for callers that only ever wanted the ordinary kind.
 */
Viewer.prototype.addMirror = function (params) {
    return this.addOptics('Mirror', params);
};

Viewer.prototype._freshOpticName = function (prefix) {
    prefix = prefix || 'M';
    var taken = {};
    (this.scene.optics || []).forEach(function (o) { taken[o.name] = true; });
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

Viewer.prototype.removeSelected = function () {
    if (!this.onEdit || !this.selectedOptic) { return null; }
    var msg = {op: 'remove', target: this.selectedOptic};
    this.selectedOptic = null;
    this._showPanel('beam');
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
Viewer.prototype._showPanel = function (kind) {
    this.panelKind = kind;
    var optic = kind === 'optic';
    this.readoutBody.style.display = optic ? 'none' : '';
    this.opticBody.style.display = optic ? '' : 'none';
    this.panelTitle.textContent = optic ? 'Optics properties' : 'Beam readout';
    if (optic) { this.pinLabel.textContent = ''; }
};

Viewer.prototype._selectedOptic = function () {
    if (!this.selectedOptic) { return null; }
    var optics = this.scene.optics || [];
    for (var i = 0; i < optics.length; i++) {
        if (optics[i].name === this.selectedOptic) { return optics[i]; }
    }
    return null;
};

Viewer.prototype._refreshOpticPanel = function () {
    var o = this._selectedOptic();
    var fields = this.opticFields;
    for (var key in fields) {
        var f = fields[key];
        var v = o ? opticFieldValue(o, key) : null;

        // A row for something this class does not have - the curvature
        // direction of a plain mirror - is not shown at all.
        if (f.optional) {
            f.row.style.display = (v === undefined || v === null) ? 'none' : '';
        }

        // Never overwrite the field the user is working in.
        if (f.editable && document.activeElement === f.el) { continue; }

        if (f.kind === 'bool') {
            if (f.editable) { f.el.checked = !!v; }
            else { f.el.textContent = o ? (v ? 'yes' : 'no') : '-'; }
        } else if (f.kind === 'choice') {
            f.el.value = v === undefined || v === null ? '' : String(v);
        } else if (f.editable) {
            f.el.value = o ? fmtField(v) : '';
        } else {
            f.el.textContent = o ? fmtField(v) : '-';
        }
    }
};

Viewer.prototype._selectOptic = function (optic) {
    this.selectedOptic = optic ? optic.name : null;
    if (optic) {
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
        || (value === null && !(field && field.nullable));
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
    var o = this._selectedOptic();
    if (!o || !this.onEdit) { return null; }
    name = String(name).trim();
    if (!name || name === o.name) {
        this._refreshOpticPanel();
        return null;
    }
    var msg = {op: 'rename', target: o.name, name: name};
    this.selectionFallback = o.name;
    this.selectedOptic = name;
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
    this.selectionFallback = null;
    this._refreshOpticPanel();
    this._updateOverlay();
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

    // Overlay elements for the readout marker, the highlighted beam and
    // the arrow telling which way that beam travels.
    this.highlight = svgEl('line', {'class': 'gt-highlight'});
    this.arrow = svgEl('path', {'class': 'gt-arrow'});
    this.marker = svgEl('circle', {'class': 'gt-marker', r: 4});
    this.outline = svgEl('polygon', {'class': 'gt-optic-outline'});
    this.overlayGroup.appendChild(this.outline);
    this.overlayGroup.appendChild(this.highlight);
    this.overlayGroup.appendChild(this.arrow);
    this.overlayGroup.appendChild(this.marker);
    this._showMarker(false);
    this.outline.style.display = 'none';
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

    this._updateOverlay();
    this._updateStatus();
};

Viewer.prototype._resize = function () {
    var rect = this.svg.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.svg.setAttribute('viewBox', '0 0 ' + this.width + ' ' + this.height);
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
    if (!isFinite(minx)) { return {minx: -1, miny: -1, maxx: 1, maxy: 1}; }
    return {minx: minx, miny: miny, maxx: maxx, maxy: maxy};
};

Viewer.prototype.fit = function (margin) {
    this._resize();
    var bb = this.bbox();
    var w = Math.max(bb.maxx - bb.minx, 1e-9);
    var h = Math.max(bb.maxy - bb.miny, 1e-9);
    var m = margin === undefined ? 0.06 : margin;
    this.scale = Math.min(this.width / w, this.height / h) * (1 - 2 * m);
    this.cx = (bb.minx + bb.maxx) / 2;
    this.cy = (bb.miny + bb.maxy) / 2;
    this._applyTransform();
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
        var pt = self.screenToScene(ev.clientX - r.left, ev.clientY - r.top);

        // Grabbing an optics starts an edit; grabbing anywhere else pans.
        var o = self.onEdit ? self._pickOptic(pt[0], pt[1]) : null;
        if (o) {
            self._beginOpticDrag(o, pt, ev.shiftKey);
            ev.preventDefault();
        }
        dragging = true; moved = 0;
        lastX = ev.clientX; lastY = ev.clientY;
        self.svg.classList.add('gt-dragging');
    });

    on(global, 'mousemove', function (ev) {
        var r = self.svg.getBoundingClientRect();
        if (self.dragOptic) {
            moved += Math.abs(ev.clientX - lastX) + Math.abs(ev.clientY - lastY);
            lastX = ev.clientX; lastY = ev.clientY;
            self._updateOpticDrag(
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
        if (self.dragOptic) {
            dragging = false;
            self.svg.classList.remove('gt-dragging');
            self._endOpticDrag(moved >= 4);
            return;
        }
        if (!dragging) { return; }
        dragging = false;
        self.svg.classList.remove('gt-dragging');
        if (moved < 4) {
            var r = self.svg.getBoundingClientRect();
            self._onClick(ev.clientX - r.left, ev.clientY - r.top);
        }
    });

    on(global, 'keydown', function (ev) {
        if (VIEWERS.length > 1 && !self.pointerInside) { return; }
        // Not while a property field has the keyboard.
        if (ev.target && ev.target.classList
            && ev.target.classList.contains('gt-input')) { return; }
        if (ev.key === 'f' || ev.key === 'F') { self.fit(); }
        if (ev.key === 'Escape') {
            self.pinned = null;
            self.selectedOptic = null;
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
Viewer.prototype._beginOpticDrag = function (optic, scenePt, rotate) {
    var c = optic.center || optic.HRcenter;
    this.dragOptic = {
        optic: optic,
        rotate: !!rotate,
        grab: scenePt,
        center0: [c[0], c[1]],
        angle0: optic.normAngleHR || 0,
        // Angle of the grab point as seen from the pivot, so that the
        // optics turns with the cursor instead of jumping to it.
        grabAngle: Math.atan2(scenePt[1] - c[1], scenePt[0] - c[0]),
        center: [c[0], c[1]],
        angle: optic.normAngleHR || 0
    };
    this._updateOpticOutline(optic, this.dragOptic.center,
                             this.dragOptic.angle);
};

Viewer.prototype._updateOpticDrag = function (scenePt) {
    var d = this.dragOptic;
    if (!d) { return; }
    if (d.rotate) {
        var a = Math.atan2(scenePt[1] - d.center0[1],
                           scenePt[0] - d.center0[0]);
        d.angle = d.angle0 + (a - d.grabAngle);
    } else {
        d.center = [d.center0[0] + scenePt[0] - d.grab[0],
                    d.center0[1] + scenePt[1] - d.grab[1]];
    }
    this._updateOpticOutline(d.optic, d.center, d.angle);
    this._updateStatus();
};

Viewer.prototype._endOpticDrag = function (moved) {
    var d = this.dragOptic;
    this.dragOptic = null;
    if (!d) { return; }
    if (!moved) {
        // A grab that went nowhere is a click: select it instead.
        this.pinned = null;
        this._selectOptic(d.optic);
        return;
    }
    // Show the properties of whatever was just moved.
    this._selectOptic(d.optic);
    if (!this.onEdit) { return; }

    // 'center' is the middle of the substrate, which is the trait the
    // outline was built from, so the optics lands where it was dropped.
    var msg = d.rotate
        ? {op: 'rotate', target: d.optic.name, normAngleHR: d.angle}
        : {op: 'move', target: d.optic.name, center: d.center};
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

Viewer.prototype._onHover = function (px, py) {
    var pt = this.screenToScene(px, py);
    this.cursor = pt;

    // An optics under the cursor takes precedence: it is what the next
    // mousedown would act on, so say so before the user presses.
    this.hoverOptic = this._pickOptic(pt[0], pt[1]);
    this.svg.classList.toggle('gt-over-optic',
                              !!this.hoverOptic && !!this.onEdit);
    this.svg.classList.toggle('gt-over-pickable',
                              !!this.hoverOptic && !this.onEdit);

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

Viewer.prototype._onClick = function (px, py) {
    var pt = this.screenToScene(px, py);

    // Clicking an optics selects it and shows its properties. This works
    // whether or not the viewer is editable: reading is always allowed.
    var optic = this._pickOptic(pt[0], pt[1]);
    if (optic) {
        this.pinned = null;
        this._selectOptic(optic);
        return;
    }

    var hits = this._pickAll(pt[0], pt[1], 12 / this.scale);
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

Viewer.prototype._updateOverlay = function () {
    if (this.dragOptic) {
        this._updateOpticOutline(this.dragOptic.optic, this.dragOptic.center,
                                 this.dragOptic.angle);
    } else {
        // The selected optics stays outlined so that the panel and the
        // drawing agree on what is being looked at.
        this._updateOpticOutline(this.hoverOptic || this._selectedOptic());
    }

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
    var d = this.dragOptic;
    if (d) {
        this.statusBar.textContent = d.rotate
            ? d.optic.name + ':  ' + fmtDeg(normAngle(d.angle)) +
              '   (was ' + fmtDeg(normAngle(d.angle0)) + ')'
            : d.optic.name + ':  ' + fmtLen(d.center[0]) + ',  ' +
              fmtLen(d.center[1]);
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
    this.dragOptic = null;
    this.cycle = 0;
    this.lastClick = null;
    this.labels = [];
    this.layerGroups = {};
    this.sceneGroup.textContent = '';
    this.labelGroup.textContent = '';
    this.overlayGroup.textContent = '';
    this.layerBody.textContent = '';
    this.opts.hiddenLayers = Object.keys(visible).filter(function (k) {
        return !visible[k];
    });
    this._renderScene();
    this._refreshDisplayPanel();
    this._setReadout(null);

    // A scene arriving after an edit describes the same optics, so keep
    // the selection and show the values Python came back with. Getting
    // one means the edit went through, so any optimistic rename stands.
    this.selectionFallback = null;
    if (this._selectedOptic()) {
        this._refreshOpticPanel();
        this._showPanel('optic');
    } else if (this.panelKind === 'optic') {
        this.selectedOptic = null;
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
    projectOnBeam: projectOnBeam
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GTraceViewer;
}
global.GTraceViewer = GTraceViewer;

//}}}

})(typeof globalThis !== 'undefined' ? globalThis : window);
