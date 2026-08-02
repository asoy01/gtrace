/*
 * Stage 1 verification, JavaScript side.
 *
 * Loads viewer.js and checks that beamParamsAt() / projectOnBeam()
 * reproduce the values gtrace computed in Python (stage1_reference.json).
 */

const path = require('path');
const fs = require('fs');

const REPO = process.argv[2];
const REF = process.argv[3];

const GTraceViewer = require(path.join(REPO, 'gtrace', 'draw', 'viewer', 'viewer.js'));
const ref = JSON.parse(fs.readFileSync(REF, 'utf8'));
const beams = ref.scene.beams;

let npass = 0, nfail = 0, worst = {name: '-', err: 0};

function rel(a, b) {
    if (!isFinite(a) && !isFinite(b)) { return 0; }
    const scale = Math.max(Math.abs(a), Math.abs(b));
    if (scale === 0) { return 0; }
    return Math.abs(a - b) / scale;
}

function check(name, cond, detail) {
    if (cond) { npass++; }
    else { nfail++; console.log('  FAIL  ' + name + '  ' + (detail || '')); }
}

function agree(name, got, want, tol) {
    // Infinities travel as strings because JSON has no Infinity literal.
    if (typeof want === 'string') {
        const wantInf = want === 'inf' ? Infinity : -Infinity;
        check(name, got === wantInf, 'got ' + got + ' want ' + want);
        return;
    }
    const e = rel(got, want);
    if (e > worst.err) { worst = {name: name, err: e}; }
    check(name, e <= tol, 'got ' + got + ' want ' + want + ' (rel ' + e + ')');
}

console.log('--- viewer.js exports ---');
check('beamParamsAt exported', typeof GTraceViewer.beamParamsAt === 'function');
check('projectOnBeam exported', typeof GTraceViewer.projectOnBeam === 'function');
check('mount exported', typeof GTraceViewer.mount === 'function');

console.log('--- beam parameters vs gtrace (' + ref.samples.length + ' samples) ---');
const TOL = 1e-12;
ref.samples.forEach(function (s, i) {
    const b = beams[s.index];
    const p = GTraceViewer.beamParamsAt(b, s.d);
    const tag = '#' + i + ' beam ' + b.name + ' d=' + s.d;
    agree(tag + ' wx', p.wx, s.wx, TOL);
    agree(tag + ' wy', p.wy, s.wy, TOL);
    agree(tag + ' Rx', p.Rx, s.Rx, TOL);
    agree(tag + ' Ry', p.Ry, s.Ry, TOL);
    agree(tag + ' Re(qx)', p.qx[0], s.qx[0], TOL);
    agree(tag + ' Im(qx)', p.qx[1], s.qx[1], TOL);
    agree(tag + ' Re(qy)', p.qy[0], s.qy[0], TOL);
    agree(tag + ' Im(qy)', p.qy[1], s.qy[1], TOL);
    agree(tag + ' Gouyx', p.Gouyx, s.Gouyx, 1e-10);
    agree(tag + ' Gouyy', p.Gouyy, s.Gouyy, 1e-10);
    agree(tag + ' optDist', p.optDist, s.optDist, 1e-12);
});

console.log('--- waist relations ---');
ref.samples.slice(0, 40).forEach(function (s, i) {
    const b = beams[s.index];
    const p = GTraceViewer.beamParamsAt(b, s.d);
    // Propagating to the waist must give exactly the waist radius.
    const atWaist = GTraceViewer.beamParamsAt(b, s.d + p.waistx);
    agree('#' + i + ' w at waist == w0', atWaist.wx, p.w0x, 1e-9);
    // Rayleigh range: w0^2 = 2 zR / k
    const k = 2 * Math.PI * b.n / b.wl;
    agree('#' + i + ' zR consistency', p.w0x * p.w0x, 2 * p.zRx / k, 1e-12);
});

console.log('--- hit test projection (' + ref.picks.length + ' points) ---');
ref.picks.forEach(function (q, i) {
    const b = beams[q.index];
    const pr = GTraceViewer.projectOnBeam(b, q.point[0], q.point[1]);
    agree('pick #' + i + ' d', pr.d, q.d, 1e-9);
    agree('pick #' + i + ' dist', pr.dist, q.dist, 1e-9);
});

console.log('');
console.log(npass + ' passed, ' + nfail + ' failed');
console.log('worst relative error: ' + worst.err.toExponential(3) + '  (' + worst.name + ')');
process.exit(nfail ? 1 : 0);
