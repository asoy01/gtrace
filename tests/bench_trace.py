#!/usr/bin/env python
'''
Measure how long a non-sequential trace takes, and print numbers that
identify the result.

Two things are printed after every run: the number of beams, and the sum
of their optical path lengths. Both are properties of the trace and not
of the timing, so running this before and after a change shows whether
the change moved the physics. The sum is printed to 12 digits because
that is where a reordered floating-point calculation starts to differ.

Two workloads are available, and they do not have the same bottleneck.
The tutorial layout is small (3 optics), where copying beams dominates.
A real interferometer has tens of optics, where the hit test dominates
because the work grows as beams x optics x 4 faces. Measure the one you
are trying to speed up.

Usage
-----
    python tests/bench_trace.py
    python tests/bench_trace.py --order 5 --threshold 1e-5 --profile
    python tests/bench_trace.py --pickle path/to/bKAGRA_Full_Obj.pkl

The pickle form takes a file holding a dict with 'OpticsList' and
'input_beam', which is how the KAGRA layouts are stored.
'''

import argparse
import cProfile
import os
import pickle
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtrace.nonsequential import non_seq_trace


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTORIAL = os.path.join(REPO, 'docs', 'source', 'tutorial',
                        'tutorial_layout.json')


def load_tutorial():
    '''The optics and the source beam of the tutorial layout.'''
    from gtrace.layout import OpticalLayout
    layout = OpticalLayout.load(TUTORIAL)
    layout._settle_assemblies()
    if not layout.sources:
        raise SystemExit('%s has no source' % TUTORIAL)
    return layout.optics, layout.sources[0], 'tutorial layout'


def load_pickle(path):
    '''The optics and the source beam of a pickled layout.'''
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    try:
        optics = obj['OpticsList']
        source = obj['input_beam']
    except (TypeError, KeyError):
        raise SystemExit("%s does not hold 'OpticsList' and 'input_beam'"
                         % path)
    return optics, source, os.path.basename(path)


def run(optics, source, order, threshold):
    '''One trace. Returns the beams.'''
    return non_seq_trace(optics, source.copy(), order=order,
                         power_threshold=threshold)


def describe(beams):
    '''The two numbers that identify a trace result.'''
    total = sum(b.optDist for b in beams)
    return len(beams), total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--order', type=int, default=5,
                   help='highest stray_order a beam may reach (default 5)')
    p.add_argument('--threshold', type=float, default=1e-5,
                   help='power threshold (default 1e-5)')
    p.add_argument('--repeat', type=int, default=3,
                   help='how many times to time the trace (default 3)')
    p.add_argument('--pickle', default=None,
                   help="a pickled layout holding 'OpticsList' and 'input_beam'")
    p.add_argument('--profile', action='store_true',
                   help='profile one trace and print the top of the profile')
    p.add_argument('--top', type=int, default=25,
                   help='how many lines of the profile to print (default 25)')
    args = p.parse_args()

    if args.pickle:
        optics, source, name = load_pickle(args.pickle)
    else:
        optics, source, name = load_tutorial()

    print('%s: %d optics, order=%d, power_threshold=%g'
          % (name, len(optics), args.order, args.threshold))

    times = []
    beams = None
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        beams = run(optics, source, args.order, args.threshold)
        times.append(time.perf_counter() - t0)

    count, total = describe(beams)
    #Every run is printed, not just the best one. The deepcopy in
    #GaussianBeam.copy() used to make the third run several times
    #slower than the first, and an average would have hidden that.
    print('time:  ' + '  '.join('%.3f s' % t for t in times)
          + '   (best %.3f s)' % min(times))
    print('beams: %d' % count)
    print('sum of optDist: %.12f' % total)

    if args.profile:
        print()
        prof = cProfile.Profile()
        prof.enable()
        run(optics, source, args.order, args.threshold)
        prof.disable()
        stats = pstats.Stats(prof)
        stats.sort_stats('cumulative').print_stats(args.top)


if __name__ == '__main__':
    main()
