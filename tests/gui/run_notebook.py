'''
Execute the OpticalLayout demo notebook and strip the saved widget
state.

    pixi run python tests/gui/run_notebook.py

nbconvert stores the whole anywidget model in metadata.widgets, and that
model carries a copy of viewer.js. Left in, it puts a hundred kilobytes
of the front end into git and makes every change to viewer.js show up as
a notebook diff as well. The dangling widget-view outputs go too, so a
static reader sees the plain repr rather than a broken widget.
'''

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO

NOTEBOOK = os.path.join(REPO, 'tests', 'OpticalLayout_demo.ipynb')
WIDGET_VIEW = 'application/vnd.jupyter.widget-view+json'

def main(path=NOTEBOOK):
    p = subprocess.run(
        [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
         '--inplace', path],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode:
        print(p.stdout)
        print(p.stderr)
        return p.returncode

    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    had_state = nb['metadata'].pop('widgets', None) is not None
    views = 0
    errors = 0
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                errors += 1
            data = out.get('data')
            if data and WIDGET_VIEW in data:
                del data[WIDGET_VIEW]
                views += 1

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print('executed %s' % os.path.basename(path))
    print('  error outputs      : %d' % errors)
    print('  widget state        : %s' % ('stripped' if had_state else 'none'))
    print('  widget-view outputs : %d stripped' % views)
    print('  size                : %d bytes' % os.path.getsize(path))
    return 1 if errors else 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
