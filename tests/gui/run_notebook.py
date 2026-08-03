'''
Execute the tutorial notebook and strip the saved widget state.

    pixi run python tests/gui/run_notebook.py

This doubles as a check on the tutorial: it is the only suite that runs
the documented code end to end, so an example that has drifted away from
the library shows up here as an error output rather than in a reader's
session.

nbconvert stores the whole anywidget model in metadata.widgets, and that
model carries a copy of viewer.js. Left in, it puts a hundred kilobytes
of the front end into git and makes every change to viewer.js show up as
a notebook diff as well.

The widget outputs themselves go too, rather than being reduced to their
text/plain part. That part is an object repr carrying a memory address:
it renders on the documentation page as a line of noise, and it differs
on every run, so it would also make a spurious diff each time. A reader
of the rendered page sees the screenshots next to the cell instead.

Two more things would otherwise change on every run, for reasons that
have nothing to do with the notebook:

  * nbconvert records how long each cell took, as wall-clock timestamps
    in cell.metadata.execution;
  * a cell's printed output arrives as one stream message or as several,
    depending on how the writes happened to be flushed, so the same text
    is stored split in different places from one run to the next.

Both are removed, which makes the executed notebook reproducible: run
this twice without touching anything and the file does not change, so a
diff against it means the code or the library really did.
'''

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO

NOTEBOOK = os.path.join(REPO, 'docs', 'source', 'tutorial',
                        'gtrace-tutorial.ipynb')
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
    timings = 0
    for cell in nb['cells']:
        if cell.get('metadata', {}).pop('execution', None) is not None:
            timings += 1

        kept = []
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'error':
                errors += 1
            data = out.get('data')
            if data and WIDGET_VIEW in data:
                views += 1
                continue
            # Join a stream onto the one before it when both go to the
            # same place, so that where the flushes fell does not show.
            if (kept and out.get('output_type') == 'stream'
                    and kept[-1].get('output_type') == 'stream'
                    and kept[-1].get('name') == out.get('name')):
                kept[-1]['text'] = kept[-1]['text'] + out['text']
                continue
            kept.append(out)
        if 'outputs' in cell:
            cell['outputs'] = kept

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print('executed %s' % os.path.basename(path))
    print('  error outputs      : %d' % errors)
    print('  widget state        : %s' % ('stripped' if had_state else 'none'))
    print('  widget outputs      : %d dropped' % views)
    print('  cell timings        : %d stripped' % timings)
    print('  size                : %d bytes' % os.path.getsize(path))
    return 1 if errors else 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
