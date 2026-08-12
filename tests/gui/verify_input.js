/*
 * What a panel row makes of what is typed into it.
 *
 * A row used to take a number and nothing else. It now takes the sum
 * as well as the answer - 2*25.4 rather than 50.8 - and a value may
 * carry a unit of its own, which converts it into the row's unit:
 * 1[in] in a millimetre row is 25.4.
 *
 * Two things want checking hardest. The first is that the arithmetic
 * is arithmetic: precedence, brackets, a leading minus, and every
 * spelling that is not an expression turned away rather than guessed
 * at. The second is that a unit converts by kind - a length typed
 * into an angle is refused, and a row with no unit of its own takes
 * no unit at all - since a row that quietly took 1[in] as 1 would be
 * worse than one that refused it.
 *
 * Run by run_all.py, like verify_stage1.js, and needs nothing but
 * Node: parseField is exported from viewer.js for exactly this.
 */

const path = require('path');

const REPO = process.argv[2];
const GTraceViewer = require(
    path.join(REPO, 'gtrace', 'draw', 'viewer', 'viewer.js'));
const parseField = GTraceViewer.parseField;

let npass = 0, nfail = 0;

function check(name, cond, detail) {
    if (cond) { npass++; }
    else { nfail++; console.log('  FAIL  ' + name + '  ' + (detail || '')); }
}

function near(a, b) {
    if (a === b) { return true; }
    if (typeof a !== 'number' || typeof b !== 'number') { return false; }
    return Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));
}

function value(text, unit, want) {
    const got = parseField(text, unit);
    check('"' + text + '" in [' + (unit || 'none') + '] is ' + want,
          near(got, want), 'got ' + got);
}

function refused(text, unit, why) {
    const got = parseField(text, unit);
    check('refuses ' + JSON.stringify(text) + ' - ' + why,
          typeof got === 'number' && isNaN(got), 'got ' + got);
}

// --- a bare number is what it always was ---
value('50.8', 'mm', 50.8);
value('-3', 'mm', -3);
value('0', 'mm', 0);
value('1e-3', 'm', 1e-3);
value('1E-3', 'm', 1e-3);
value('.5', 'mm', 0.5);
value('5.', 'mm', 5);
value('  7  ', 'mm', 7);

// The words a row uses for "no value", and the infinities a curvature
// row is written with. None of them goes through the arithmetic.
check('an empty row is null', parseField('', 'mm') === null);
check("'auto' is null", parseField('auto', 'mm') === null);
check("'none' is null", parseField('none', 'mm') === null);
check("'-' is null", parseField('-', 'mm') === null);
check("'inf' is infinity", parseField('inf', 'm') === Infinity);
check("'-inf' is minus infinity", parseField('-inf', 'm') === -Infinity);
check("'∞' is infinity", parseField('∞', 'm') === Infinity);

// --- the four operations ---
value('2*25.4', 'mm', 50.8);
value('300/4', 'mm', 75);
value('12.7+6.35', 'mm', 19.05);
value('50-12.7', 'mm', 37.3);
value('2 * 25.4', 'mm', 50.8);

// Precedence, which is the whole reason this is parsed rather than
// read left to right.
value('1+2*3', 'mm', 7);
value('2*3+1', 'mm', 7);
value('1-2*3', 'mm', -5);
value('12/4/3', 'mm', 1);
value('12-4-3', 'mm', 5);
value('(1+2)*3', 'mm', 9);
value('2*(3+1)', 'mm', 8);
value('((2))', 'mm', 2);
value('-(2+3)', 'mm', -5);
value('-2*-3', 'mm', 6);
value('+5', 'mm', 5);

// --- units convert into the row's own ---
value('1[in]', 'mm', 25.4);
value('1[inch]', 'mm', 25.4);
value('1[in]', 'm', 0.0254);
value('1[m]', 'mm', 1000);
value('1[cm]', 'mm', 10);
value('1[um]', 'nm', 1000);
value('1064[nm]', 'nm', 1064);
value('1[mil]', 'mm', 0.0254);
value('1[ft]', 'mm', 304.8);
value('25.4[mm]', 'mm', 25.4);
value('1[IN]', 'mm', 25.4);
value('1[ In ]', 'mm', 25.4);

// Angles are a kind of their own, and a row reading degrees takes
// radians as the degrees they come to.
value('1[rad]', '°', 180 / Math.PI);
value('90[deg]', '°', 90);
value('90[°]', '°', 90);
value('1[mrad]', '°', 1e-3 * 180 / Math.PI);

// So is power, which one row is read in.
value('0.5[w]', 'W', 0.5);
value('500[mw]', 'W', 0.5);
value('1[kw]', 'W', 1000);

// --- a unit and the arithmetic together ---
// The unit converts the number it follows; everything after that is
// ordinary arithmetic in the row's unit.
value('1[in]+2', 'mm', 27.4);
value('2*1[in]', 'mm', 50.8);
value('1[in]*2', 'mm', 50.8);
value('1[in]-1[mm]', 'mm', 24.4);
value('(1+1)[in]', 'mm', 50.8);
value('(1[in]+1[in])/2', 'mm', 25.4);
value('-1[in]', 'mm', -25.4);
value('1[m]/2', 'mm', 500);

// --- what is refused ---
refused('wide', 'mm', 'a word that is not a number');
refused('2*', 'mm', 'an operation with nothing to work on');
refused('*2', 'mm', 'an operation with nothing on the left');
refused('(1+2', 'mm', 'a bracket left open');
refused('1+2)', 'mm', 'a bracket never opened');
refused('1 2', 'mm', 'two numbers with nothing between them');
refused('1[bogus]', 'mm', 'a unit nothing answers to');
refused('1[in', 'mm', 'a unit bracket left open');
refused('[in]', 'mm', 'a unit with no number');
refused('1[]', 'mm', 'an empty unit');
refused('1//2', 'mm', 'an operation twice over');

// A unit only converts to a unit of its own kind, and a row without
// one has nothing to convert into.
refused('1[in]', '°', 'a length typed into an angle');
refused('1[rad]', 'mm', 'an angle typed into a length');
refused('1[w]', 'mm', 'a power typed into a length');
refused('1[in]', null, 'a unit in a row that has none');
refused('1[in]', undefined, 'a unit in a row that has none at all');
check('a bare number is still taken by a row with no unit',
      parseField('7', null) === 7);

// The table itself: every entry is a kind and a factor, so that a
// conversion is always defined and never a guess.
const kinds = {};
Object.keys(GTraceViewer.INPUT_UNITS).forEach(function (k) {
    const u = GTraceViewer.INPUT_UNITS[k];
    kinds[u[0]] = true;
    check('the unit ' + k + ' has a kind and a positive factor',
          typeof u[0] === 'string' && typeof u[1] === 'number' && u[1] > 0,
          JSON.stringify(u));
    check('  and ' + k + ' is spelled in lower case', k === k.toLowerCase());
});
check('the kinds are length, angle and power',
      Object.keys(kinds).sort().join(',') === 'angle,length,power',
      Object.keys(kinds).join(','));

// Every unit a panel row is labelled with has to be in the table, or
// a value typed with a unit into that row could never be converted.
['m', 'mm', 'nm', '°', 'W'].forEach(function (u) {
    check('a row labelled [' + u + '] can take units',
          !!GTraceViewer.INPUT_UNITS[u.toLowerCase()]);
});

console.log('');
console.log(npass + ' passed, ' + nfail + ' failed');
process.exit(nfail ? 1 : 0);
