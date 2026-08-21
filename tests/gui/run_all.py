'''
Run every GUI verification suite and report a summary.

    pixi run python tests/gui/run_all.py

The order matters: a suite reads what an earlier one wrote. Stage 1
produces the reference values the JavaScript check compares against and
the HTML pages the browser checks load; Stage 2 writes the ESM module
that every widget-side browser check imports; Stage 2b writes the scene
they drive. Running one on its own is fine as long as its inputs are
already in the work directory from a previous run.
'''

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK, SKIP, find_chrome, find_node

HERE = os.path.dirname(os.path.abspath(__file__))

#: (script, what it covers). Order is a dependency order, not a
#: preference.
SUITES = [
    ('verify_beam.py',
     "a beam's width and reduced q agree with the q it was given"),
    ('verify_geometry.py',
     'the geometry kernel: intersections, rotation, and the surface matrices'),
    ('verify_surfaces.py',
     'where the two faces of a substrate are, against an independent arc'),
    ('verify_lens.py',
     'Lens: the focal length ordered, measured by tracing a ray through it'),
    ('verify_cylindrical.py',
     'CyMirror: one plane focused, the other left alone, against Siegman'),
    ('verify_cylens.py',
     'CyLens: the focal length ordered lands in one plane, the other is a window'),
    ('verify_stray.py',
     'what a stray order is and what order caps: ghosts made, not ghosts met'),
    ('verify_dimensions.py',
     'substrate corners, what a span runs inside, and the dimension model'),
    ('verify_rect.py',
     'a rectangle that is turned: its corners, and everything downstream'),
    ('verify_rect.js',
     'the page and gtrace agreeing about where a turned rectangle is'),
    ('verify_rect_browser.py',
     'headless browser: where a turned rectangle is actually painted'),
    ('verify_stage1.py',
     'renderHTML: self-containment, the embedded scene, every entry point'),
    ('verify_stage1.js',
     'viewer.js physics against gtrace, over every beam of a traced system'),
    ('verify_input.js',
     'what a panel row makes of what is typed into it: arithmetic and units'),
    ('verify_browser.py',
     'headless browser: the DOM the viewer builds, and the real HTML file'),
    ('verify_interact.py',
     'headless browser: hover, pin, cycling, layer visibility, zoom'),
    ('verify_stage2.py',
     'the widget: ESM assembly, traitlets, backend selection'),
    ('verify_stage2_browser.py',
     'headless browser: the ESM driven as anywidget drives it'),
    ('verify_stage2b.py',
     'the edit protocol, save/load, and the model invariants behind them'),
    ('verify_stage2b_browser.py',
     'headless browser: dragging an optics, and what Python makes of it'),
    ('verify_props_browser.py',
     'headless browser: the properties panel, the controls, add and remove'),
    ('verify_measure_browser.py',
     'headless browser: the measuring tool, snapping and the dimension panel'),
    ('verify_source.py',
     'sources: the waist a laser is specified by, and the protocol reaching it'),
    ('verify_source_browser.py',
     'headless browser: the laser drawn for a source, and editing it'),
    ('verify_mechanics.py',
     'mechanics: the bodies the trace never sees, and the protocol reaching it'),
    ('verify_mech_browser.py',
     'headless browser: picking a body by its area, and editing its pose'),
    ('verify_mech_shape_browser.py',
     "headless browser: editing the one shape of a hand-drawn body"),
    ('verify_assembly.py',
     'one element following another: the joint, the settle, what it refuses'),
    ('verify_assembly_parts.py',
     'an element and the parts that hold it, built in one call'),
    ('verify_align_browser.py',
     'headless browser: aiming an optics by places, and the quarter turn'),
    ('verify_editor.py',
     'the shape editor: the protocol behind Mechanics.edit(), and what it refuses'),
    ('verify_editor_browser.py',
     'headless browser: the editor side bar, and the messages its rows send'),
    ('verify_editor_drag_browser.py',
     'headless browser: drawing a part by hand - picking, carrying, grips'),
]

#: Checks that live with the rest of the tests but are worth running in
#: the same breath, since the GUI leans on both.
EXTRA = [
    ('test_beam_propagation.py', 'the propagation convention of GaussianBeam'),
    ('test_gtrace.py', 'the DXF renderer still runs end to end'),
]

def run(cmd, cwd=None):
    # The suites print µ, °, » and the like. Captured output is a pipe,
    # and on a Japanese Windows console Python encodes a pipe as cp932
    # unless told otherwise - so without this a suite dies of
    # UnicodeEncodeError in its own PASS line and reads as a failure of
    # gtrace. Harmless to node and to the browser.
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=env)
    return p

def summarise(out):
    '''
    Pull the trailing 'N passed, M failed' line out of a suite's output.
    '''
    for line in reversed((out or '').splitlines()):
        if 'passed,' in line and 'failed' in line:
            return line.strip()
    return ''

def main():
    node = find_node()
    chrome = find_chrome()
    print('gtrace   : %s' % REPO)
    print('work dir : %s' % WORK)
    print('browser  : %s' % (chrome or 'not found - browser checks will skip'))
    print('node     : %s' % (node or 'not found - the JS check will skip'))
    print()

    results = []
    for name, what in SUITES:
        sys.stdout.write('%-28s ' % name)
        sys.stdout.flush()

        if name.endswith('.js'):
            if node is None:
                print('SKIP  (no node)')
                results.append((name, 'skip', ''))
                continue
            p = run([node, os.path.join(HERE, name), REPO,
                     os.path.join(WORK, 'stage1_reference.json')])
        else:
            p = run([sys.executable, '-W', 'ignore', os.path.join(HERE, name)])

        line = summarise(p.stdout)
        if p.returncode == SKIP:
            print('SKIP  %s' % summarise(p.stdout))
            results.append((name, 'skip', ''))
        elif p.returncode == 0:
            print('ok    %s' % line)
            results.append((name, 'ok', line))
        else:
            print('FAIL  %s' % line)
            results.append((name, 'fail', line))
            for l in (p.stdout or '').splitlines():
                if l.startswith('  FAIL'):
                    print('        ' + l.strip())
            if not line:
                print((p.stderr or '').strip()[-2000:])

    print()
    for name, what in EXTRA:
        sys.stdout.write('%-28s ' % name)
        sys.stdout.flush()
        p = run([sys.executable, '-W', 'ignore', name],
                cwd=os.path.join(REPO, 'tests'))
        if p.returncode == 0:
            print('ok')
            results.append((name, 'ok', ''))
        else:
            print('FAIL')
            print((p.stderr or '').strip()[-2000:])
            results.append((name, 'fail', ''))

    failed = [r for r in results if r[1] == 'fail']
    skipped = [r for r in results if r[1] == 'skip']
    print()
    print('%d suites ok, %d failed, %d skipped'
          % (len(results) - len(failed) - len(skipped), len(failed),
             len(skipped)))
    for name, _, line in failed:
        print('  FAILED: %s  %s' % (name, line))
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
