'''
Shared plumbing for the GUI verification suites.

The suites write a good deal of scratch - generated HTML pages, scenes,
screenshots - which belongs next to them but not in the repository, and
several of them drive a real browser, which lives in a different place
on every machine. Both are handled here so that the checks themselves
stay about gtrace.
'''

import os
import shutil
import sys

#: Root of the gtrace checkout.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Where the suites put everything they generate. Ignored by git; safe
#: to delete at any time.
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_work')
os.makedirs(WORK, exist_ok=True)

#: Exit code a suite uses when it cannot run at all, as opposed to
#: running and finding something wrong. run_all reports these apart.
SKIP = 77

_CHROME_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
]

def find_chrome():
    '''
    Path to a Chrome-like browser, or None.

    Set GTRACE_CHROME to point at a particular one.
    '''
    override = os.environ.get('GTRACE_CHROME')
    if override:
        return override if os.path.exists(override) else None
    for name in ['google-chrome', 'chromium', 'chrome']:
        found = shutil.which(name)
        if found:
            return found
    for path in _CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

def require_chrome():
    '''
    Return the browser to drive, or leave the suite with SKIP.
    '''
    chrome = find_chrome()
    if chrome is None:
        print('SKIP: no Chrome-like browser found. Set GTRACE_CHROME to '
              'the executable to run the browser checks.')
        sys.exit(SKIP)
    return chrome

def find_node():
    '''
    Path to node, or None.
    '''
    return shutil.which('node')
