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
 * Format a q parameter, held as a [real, imag] pair, in meters.
 */
function fmtQ(q) {
    if (!q) { return '-'; }
    var im = q[1];
    return fmtNum(q[0], 4) + (im < 0 ? ' - ' : ' + ') +
           fmtNum(Math.abs(im), 4) + 'i m';
}

function fmtDeg(rad) {
    if (rad === null || rad === undefined || isNaN(rad)) { return '-'; }
    return (rad * 180 / Math.PI).toFixed(2) + '°';
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

function Viewer(container, scene, options) {
    this.container = container;
    this.scene = scene || {canvas: {layers: []}, beams: []};
    this.opts = options || {};
    this.fontSize = this.opts.fontSize || 11;

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

    this._build();
    this._renderScene();
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
    var fitBtn = htmlEl('button', 'gt-btn', 'Fit');
    fitBtn.addEventListener('click', function () { self.fit(); });
    buttons.appendChild(fitBtn);
    head.appendChild(buttons);
    side.appendChild(head);

    // readout panel
    var rpanel = htmlEl('div', 'gt-panel');
    var rtitle = htmlEl('div', 'gt-panel-title');
    rtitle.appendChild(htmlEl('span', null, 'Beam readout'));
    this.pinLabel = htmlEl('span', 'gt-pin', '');
    rtitle.appendChild(this.pinLabel);
    rpanel.appendChild(rtitle);
    this.readoutBody = htmlEl('div', 'gt-readout');
    rpanel.appendChild(this.readoutBody);
    side.appendChild(rpanel);
    this._buildReadout();

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
    [['Wheel', 'zoom at cursor'],
     ['Drag', 'pan'],
     ['Move over a beam', 'live readout'],
     ['Click', 'pin the readout'],
     ['Click again', 'cycle overlapping beams'],
     ['f', 'fit to view'],
     ['Esc', 'clear readout']].forEach(function (row) {
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
    {key: 'point', label: 'Point', span: true},
    {key: 'dist', label: 'Distance', span: true},
    {key: 'w', label: 'Radius w'},
    {key: 'R', label: 'ROC R'},
    {key: 'q', label: 'q'},
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

    // Overlay elements for the readout marker and the highlighted beam.
    this.highlight = svgEl('line', {'class': 'gt-highlight'});
    this.marker = svgEl('circle', {'class': 'gt-marker', r: 4});
    this.overlayGroup.appendChild(this.highlight);
    this.overlayGroup.appendChild(this.marker);
    this._showMarker(false);
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

    if (global.ResizeObserver) {
        this._ro = new ResizeObserver(function () { self._resize(); });
        this._ro.observe(this.svg);
    } else {
        global.addEventListener('resize', function () { self._resize(); });
    }

    this.svg.addEventListener('wheel', function (ev) {
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

    this.svg.addEventListener('mousedown', function (ev) {
        if (ev.button !== 0) { return; }
        dragging = true; moved = 0;
        lastX = ev.clientX; lastY = ev.clientY;
        self.svg.classList.add('gt-dragging');
    });

    global.addEventListener('mousemove', function (ev) {
        var r = self.svg.getBoundingClientRect();
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

    global.addEventListener('mouseup', function (ev) {
        if (!dragging) { return; }
        dragging = false;
        self.svg.classList.remove('gt-dragging');
        if (moved < 4) {
            var r = self.svg.getBoundingClientRect();
            self._onClick(ev.clientX - r.left, ev.clientY - r.top);
        }
    });

    global.addEventListener('keydown', function (ev) {
        if (ev.key === 'f' || ev.key === 'F') { self.fit(); }
        if (ev.key === 'Escape') { self.pinned = null; self._setReadout(null); }
    });
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

Viewer.prototype._onHover = function (px, py) {
    var pt = this.screenToScene(px, py);
    this.cursor = pt;
    var hit = this._pick(pt[0], pt[1], 12 / this.scale);
    this.hover = hit;
    if (!this.pinned) { this._setReadout(hit); }
    this._updateOverlay();
    this._updateStatus();
};

Viewer.prototype._onClick = function (px, py) {
    var pt = this.screenToScene(px, py);
    var hits = this._pickAll(pt[0], pt[1], 12 / this.scale);
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
};

Viewer.prototype._updateOverlay = function () {
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
    this.marker.setAttribute('cx', pm[0]);
    this.marker.setAttribute('cy', pm[1]);
    this.marker.classList.toggle('gt-pinned', !!this.pinned);
    this._showMarker(true);
};

Viewer.prototype._updateStatus = function () {
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
    this._setReadout(null);
    this._applyTransform();
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
