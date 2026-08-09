'''
Source beams: the channel that tells them from the trace, the waist a
front end edits them by, and the edit protocol reaching them at last.

Until now the layout could be edited in every part but the one the light
comes from. The optics could be added, removed, renamed and set; the
sources could only be registered from Python and never touched again -
and, worse, nothing in a scene said which beams were sources. A source
is traced from a copy of itself, so its own beam is in 'beams' looking
exactly like the ones the trace produced from it.

Two things carry the weight here and are checked hardest.

The first is the waist. A GaussianBeam holds q-parameters; a laser is
specified by how wide its waist is and where that waist sits. The
conversion is on the Python side, next to GaussianBeam.waist() which is
the model's own statement of what a waist is, so that a browser does not
end up holding a second description of it - the sort of duplicate that
put the AR surface a sagitta out of place until 2026-08-03. What must
hold is that the two directions are exact inverses, that setting one
half of the pair leaves the other alone, and that the y axis is not
disturbed by an edit to x.

The second is the namespace. Optics and dimensions already shared one,
because an edit message names its target and nothing else. Sources have
now joined it, so a name is free only if none of the three has it.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json

import numpy as np

import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.layout import (OpticalLayout, TraceRules, EditError,
                           MAX_RULE_ORDER, DEFAULT_SOURCE_WAIST,
                           DEFAULT_SOURCE_WL, EDITABLE_SOURCE_ATTRS,
                           rayleigh_range, q_from_waist, source_waist,
                           source_scene_dict, source_to_dict,
                           source_from_dict)
from gtrace.unit import *

npass = 0
nfail = 0

def check(name, cond, detail=''):
    global npass, nfail
    if cond:
        npass += 1
        print('  PASS  %s %s' % (name, detail))
    else:
        nfail += 1
        print('  FAIL  %s %s' % (name, detail))

def refused(layout, msg, why):
    '''
    Check that an edit message is rejected without side effects.
    '''
    before = json.dumps(layout.to_dict())
    try:
        layout.apply_edit(msg)
    except EditError as e:
        check('refuses %s' % why, True, '(%s)' % str(e)[:60])
    except Exception as e:
        check('refuses %s' % why, False,
              '(raised %s instead)' % type(e).__name__)
        return
    else:
        check('refuses %s' % why, False, '(it went through)')
        return
    check('  and leaves the layout alone',
          json.dumps(layout.to_dict()) == before)

def fresh(**kw):
    '''
    A layout with one mirror in front of one source.
    '''
    L = OpticalLayout(name='sources', rules=TraceRules(order=3,
                                                       power_threshold=1e-6))
    L.add_optics(opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=np.pi,
                            diameter=1*inch, name='M1'))
    L.add_source(GaussianBeam(pos=[0.0, 0.0], dirAngle=0.0,
                              q0=q_from_waist(kw.get('w0', 0.2*mm),
                                              kw.get('d', 0.0),
                                              kw.get('wl', 1064*nm)),
                              wl=kw.get('wl', 1064*nm), name='b0'))
    return L


print('--- the waist and the q-parameter are inverses ---')

# Every combination worth trying: a waist in front of the laser, behind
# it, and at it, over three orders of magnitude of size and two of
# wavelength. The tolerance is that of the arithmetic, not of a model:
# both directions are closed-form.
for wl in [1064*nm, 532*nm, 10.6*um]:
    for n in [1.0, 1.45]:
        for w0 in [20*um, 0.2*mm, 2*mm]:
            for d in [-0.5, -1*mm, 0.0, 1*mm, 0.5, 12.0]:
                q = q_from_waist(w0, d, wl, n)
                b = GaussianBeam(q0=q, wl=wl, n=n, name='t')
                w = source_waist(b)
                ok = (abs(w['waist_size'][0] - w0) <= 1e-12 * w0
                      and abs(w['waist_pos'][0] - d) <= 1e-9 * max(abs(d), 1e-3))
                check('w0=%g d=%g wl=%g n=%g round trips' % (w0, d, wl, n), ok,
                      '' if ok else str(w))

# The Rayleigh range is the imaginary part, which is what makes the
# beam have a width at all. Stated here from the definition rather than
# from the function under test.
for w0, wl, n in [(0.2*mm, 1064*nm, 1.0), (1*mm, 532*nm, 1.45)]:
    zR = np.pi * n * w0 * w0 / wl
    check('zR is pi n w0^2 / wl',
          abs(rayleigh_range(w0, wl, n) - zR) < 1e-18, str(zR))
    check('  and is the imaginary part of q at the waist',
          abs(q_from_waist(w0, 0.0, wl, n).imag - zR) < 1e-18)

# The sign convention, spelt out because it is the one thing about a
# waist position that can be got backwards: waist() reports the distance
# forward along the beam, so a waist ahead of the laser is positive.
b = GaussianBeam(q0=q_from_waist(0.2*mm, 0.3, 1064*nm), wl=1064*nm, name='t')
check('a waist ahead of the source reads positive',
      source_waist(b)['waist_pos'][0] > 0,
      str(source_waist(b)['waist_pos'][0]))
check('  and the beam is narrower there than at the source',
      b.width(0.3)[0] < b.width(0.0)[0],
      '%g vs %g' % (b.width(0.3)[0], b.width(0.0)[0]))
check('  and the model agrees where the waist is',
      abs(b.waist()['Waist Position'][0] - 0.3) < 1e-12)


print('--- the scene says which beams are sources ---')

L = fresh()
scene = L.scene_dict()
check('the scene carries a sources channel', 'sources' in scene)
check('one entry per registered source', len(scene['sources']) == 1,
      str(len(scene['sources'])))
s = scene['sources'][0]
check('named as the source is', s['name'] == 'b0', s['name'])
for key in ['pos', 'dirVect', 'dirAngle', 'length', 'wl', 'P', 'n',
            'qx', 'qy', 'width', 'waist_size', 'waist_pos', 'layer']:
    check('it carries %s' % key, key in s)
# The width where the light leaves. A front end drawing an aperture
# there cannot draw it narrower than the beam coming out of it.
check('the width is the beam width at the origin, not at the waist',
      abs(s['width'][0] - L.get_source('b0').width(0.0)[0]) < 1e-18,
      '%s vs %s' % (s['width'][0], L.get_source('b0').width(0.0)[0]))
check('the waist travels with it, worked out rather than stored',
      abs(s['waist_size'][0] - 0.2*mm) < 1e-15, str(s['waist_size']))
check('every value is JSON-compatible',
      json.loads(json.dumps(scene['sources'])) == scene['sources'])

# The point of the channel: the source's own beam is in 'beams' too, and
# nothing there tells it from the beams the trace made.
names = [b['name'] for b in scene['beams']]
check('the source beam is among the traced beams as well', 'b0' in names,
      str(names[:4]))
check('  which is why the channel is needed', len(names) > 1, str(len(names)))

# The tracing rules travel too, so a front end can show what it is
# looking at rather than guessing.
check('the scene carries the tracing rules', 'rules' in scene)
check('  as the layout holds them', scene['rules']['order'] == 3,
      str(scene['rules']))


print('--- editing a source through the protocol ---')

L = fresh()
L.apply_edit({'op': 'move', 'target': 'b0', 'pos': [0.1, 0.02]})
check('move puts the laser where it says',
      np.allclose(L.get_source('b0').pos, [0.1, 0.02]),
      str(L.get_source('b0').pos))
L.apply_edit({'op': 'rotate', 'target': 'b0', 'dirAngle': np.deg2rad(30)})
check('rotate aims it',
      abs(L.get_source('b0').dirAngle - np.deg2rad(30)) < 1e-15,
      str(L.get_source('b0').dirAngle))
check('  and the direction vector follows',
      np.allclose(L.get_source('b0').dirVect,
                  [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))]))
L.apply_edit({'op': 'rotate', 'target': 'b0', 'dirVect': [0.0, 2.0]})
check('rotate takes a vector too, and normalizes it',
      np.allclose(L.get_source('b0').dirVect, [0.0, 1.0]),
      str(L.get_source('b0').dirVect))

# Rotating a source turns it about the point the light leaves from,
# which is the only thing it could turn about: that point is the source.
L = fresh()
L.apply_edit({'op': 'move', 'target': 'b0', 'pos': [0.3, 0.4]})
L.apply_edit({'op': 'rotate', 'target': 'b0', 'dirAngle': 1.1})
check('turning a source leaves it where it stands',
      np.allclose(L.get_source('b0').pos, [0.3, 0.4]),
      str(L.get_source('b0').pos))

print('--- the waist rows edit one half at a time ---')

L = fresh()
before = source_waist(L.get_source('b0'))
L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'waist_size_x': 0.5*mm}})
after = source_waist(L.get_source('b0'))
check('setting the x waist size takes',
      abs(after['waist_size'][0] - 0.5*mm) < 1e-18, str(after['waist_size']))
check('  without moving the waist',
      after['waist_pos'][0] == before['waist_pos'][0],
      '%g -> %g' % (before['waist_pos'][0], after['waist_pos'][0]))
check('  and without touching y',
      after['waist_size'][1] == before['waist_size'][1],
      '%g -> %g' % (before['waist_size'][1], after['waist_size'][1]))

L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'waist_pos_x': 0.25}})
moved = source_waist(L.get_source('b0'))
check('setting the x waist position takes',
      abs(moved['waist_pos'][0] - 0.25) < 1e-15, str(moved['waist_pos']))
check('  without resizing the waist',
      abs(moved['waist_size'][0] - 0.5*mm) < 1e-18,
      str(moved['waist_size']))
check('  and without touching y',
      moved['waist_pos'][1] == before['waist_pos'][1])

L.apply_edit({'op': 'set', 'target': 'b0',
              'attrs': {'waist_size_y': 0.7*mm, 'waist_pos_y': -0.1}})
both = source_waist(L.get_source('b0'))
check('the y rows work the same way',
      abs(both['waist_size'][1] - 0.7*mm) < 1e-18
      and abs(both['waist_pos'][1] + 0.1) < 1e-15, str(both))
check('  and x is where it was left',
      both['waist_size'][0] == moved['waist_size'][0]
      and both['waist_pos'][0] == moved['waist_pos'][0])

# A q-parameter says nothing on its own: what width it comes to depends
# on the wavelength. So changing the wavelength has to keep one of the
# two, and through this protocol it keeps the waist - which is what the
# laser is specified by, and what the panel is showing. The index of
# refraction already behaves this way in the model itself, whose handler
# holds the reduced q fixed and so preserves the waist size; the two
# would otherwise disagree.
L = fresh()
L.apply_edit({'op': 'set', 'target': 'b0',
              'attrs': {'waist_size_x': 1*mm, 'wl': 532*nm,
                        'waist_size_y': 1*mm}})
w = source_waist(L.get_source('b0'))
check('a batch of waist and wavelength lands on the waist asked for',
      abs(w['waist_size'][0] - 1*mm) < 1e-15
      and abs(w['waist_size'][1] - 1*mm) < 1e-15, str(w['waist_size']))
check('  and the wavelength took',
      L.get_source('b0').wl == 532*nm, str(L.get_source('b0').wl))

# The same either way round, which is what matters for a panel: each row
# a user types is a message of its own, so the order is theirs and not
# the protocol's to arrange.
first = fresh()
first.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'wl': 532*nm}})
first.apply_edit({'op': 'set', 'target': 'b0',
                  'attrs': {'waist_size_x': 1*mm, 'waist_pos_x': 0.2}})
second = fresh()
second.apply_edit({'op': 'set', 'target': 'b0',
                   'attrs': {'waist_size_x': 1*mm, 'waist_pos_x': 0.2}})
second.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'wl': 532*nm}})
check('typing the wavelength before or after the waist comes to the same',
      source_waist(first.get_source('b0'))
      == source_waist(second.get_source('b0')),
      '%s vs %s' % (source_waist(first.get_source('b0')),
                    source_waist(second.get_source('b0'))))
check('  and that is the waist that was typed',
      abs(source_waist(second.get_source('b0'))['waist_size'][0] - 1*mm) < 1e-15
      and abs(source_waist(second.get_source('b0'))['waist_pos'][0] - 0.2)
          < 1e-12,
      str(source_waist(second.get_source('b0'))))

# Keeping the waist means the divergence changes instead, which is the
# whole content of the choice: the same waist at half the wavelength
# spreads half as fast. Read far enough out for the Rayleigh range to
# have stopped mattering, since that is where the ratio is exactly two.
green = fresh()
red_far = green.get_source('b0').width(100.0)[0]
green.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'wl': 532*nm}})
green_far = green.get_source('b0').width(100.0)[0]
check('halving the wavelength halves the spread in the far field',
      abs(green_far / red_far - 0.5) < 1e-3, '%g -> %g' % (red_far, green_far))

# Assigning the trait directly in Python is untouched. That is the
# model's own convention - there is no handler for the wavelength at all
# - and this is the edit protocol, which deals in the waist throughout.
direct = fresh()
q_before = complex(direct.get_source('b0').qx)
direct.get_source('b0').wl = 532*nm
check('assigning b.wl in Python still keeps the q-parameter',
      complex(direct.get_source('b0').qx) == q_before,
      str(direct.get_source('b0').qx))
# The cached widths are recomputed, which neither the wavelength (no
# handler at all) nor the index (a handler that notifies nothing) does
# on its own.
b = L.get_source('b0')
check('  and the cached width is not left describing the old light',
      abs(b.wx - b.width(0.0)[0]) < 1e-18, '%g vs %g' % (b.wx, b.width(0.0)[0]))

L = fresh()
L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'n': 1.45}})
b = L.get_source('b0')
check('setting the index leaves the cached widths right',
      abs(b.wx - b.width(0.0)[0]) < 1e-18 and abs(b.wy - b.width(0.0)[1]) < 1e-18,
      '%g vs %g' % (b.wx, b.width(0.0)[0]))

print('--- q-parameters may still be set directly ---')

L = fresh()
L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'qx': [-0.25, 0.75]}})
check('a q arrives as a pair and lands as a complex',
      complex(L.get_source('b0').qx) == complex(-0.25, 0.75),
      str(L.get_source('b0').qx))
check('  and reads back as the waist it describes',
      abs(source_waist(L.get_source('b0'))['waist_pos'][0] - 0.25) < 1e-15)


print('--- adding, renaming and removing ---')

L = fresh()
L.apply_edit({'op': 'add', 'type': 'Source', 'name': 'S1',
              'params': {'pos': [0.0, 0.2], 'dirAngle': 0.0}})
check('a source can be added from a message',
      [s.name for s in L.sources] == ['b0', 'S1'],
      str([s.name for s in L.sources]))
s1 = L.get_source('S1')
check('  with the catalogue waist when none is asked for',
      abs(source_waist(s1)['waist_size'][0] - DEFAULT_SOURCE_WAIST) < 1e-18,
      str(source_waist(s1)['waist_size'][0]))
check('  and the catalogue wavelength', s1.wl == DEFAULT_SOURCE_WL,
      str(s1.wl))
check('  a round beam, since nothing said otherwise',
      s1.qx == s1.qy, '%s / %s' % (s1.qx, s1.qy))

# Nothing is inherited from the source already there. A q-parameter
# describes a waist measured from a point the new source does not stand
# at, so carrying one over would be worse than a default.
L2 = fresh(w0=2*mm, wl=532*nm)
L2.apply_edit({'op': 'add', 'type': 'Source', 'name': 'S1'})
check('a new source inherits nothing from the one already there',
      L2.get_source('S1').wl == DEFAULT_SOURCE_WL
      and abs(source_waist(L2.get_source('S1'))['waist_size'][0]
              - DEFAULT_SOURCE_WAIST) < 1e-18,
      str(L2.get_source('S1').wl))

L.apply_edit({'op': 'add', 'type': 'Source',
              'params': {'waist_size': 1*mm, 'waist_pos': 0.05,
                         'wl': 532*nm, 'P': 0.2, 'n': 1.0,
                         'pos': [1.0, 1.0], 'dirVect': [0.0, -1.0]}})
made = L.sources[-1]
check('a source can be added without naming it', made.name == 'S2',
      made.name)
w = source_waist(made)
check('  and every parameter arrives',
      abs(w['waist_size'][0] - 1*mm) < 1e-18
      and abs(w['waist_pos'][0] - 0.05) < 1e-15
      and made.wl == 532*nm and made.P == 0.2
      and np.allclose(made.dirVect, [0.0, -1.0]), str(w))

L.apply_edit({'op': 'rename', 'target': 'S1', 'name': 'aux'})
check('a source renames', [s.name for s in L.sources] == ['b0', 'aux', 'S2'],
      str([s.name for s in L.sources]))
L.apply_edit({'op': 'remove', 'target': 'aux'})
check('and is removed by name',
      [s.name for s in L.sources] == ['b0', 'S2'],
      str([s.name for s in L.sources]))

# A layout with no source at all is a picture of the optics. That is
# where every layout starts, so it is a state to be able to get back to.
L3 = fresh()
L3.apply_edit({'op': 'remove', 'target': 'b0'})
check('the last source may be removed', L3.sources == [], str(L3.sources))
check('  and the layout then traces to nothing',
      L3.trace() == [], str(L3.trace()))
check('  and still draws', L3.scene_dict()['sources'] == [])


print('--- one namespace, shared with the optics and the dimensions ---')

L = fresh()
refused(L, {'op': 'add', 'type': 'Source', 'name': 'M1'},
        'a source named after an optics')
refused(L, {'op': 'rename', 'target': 'b0', 'name': 'M1'},
        'renaming a source onto an optics')
L.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
              'params': {'p1': [0.0, 0.0], 'p2': [0.4, 0.0]}})
refused(L, {'op': 'add', 'type': 'Source', 'name': 'D1'},
        'a source named after a dimension')
try:
    L.add_optics(opt.Mirror(name='b0'))
    check('an optics named after a source is refused', False,
          '(it went through)')
except ValueError as e:
    check('an optics named after a source is refused', True,
          '(%s)' % str(e)[:50])

check('unique_source_name steps over every kind',
      L.unique_source_name('M') == 'M2', L.unique_source_name('M'))


print('--- what a source refuses ---')

L = fresh()
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'diameter': 0.1}},
        'an attribute a beam does not have')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'waist_size_x': 0.0}},
        'a waist of no size')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'waist_size_x': -1*mm}},
        'a negative waist')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'wl': 0.0}},
        'a wavelength of zero')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'n': -1.0}},
        'a negative index')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'P': -1.0}},
        'a negative power')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'length': 0.0}},
        'a beam of no length')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'qx': [0.0, -1.0]}},
        'a q with no Rayleigh range')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'qx': [0.0, 0.0]}},
        'a q that is nothing at all')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'pos': [0.0, None]}},
        'a position with a hole in it')
refused(L, {'op': 'set', 'target': 'b0', 'attrs': {'waist_pos_x': None}},
        'a waist position that is not a number')
refused(L, {'op': 'rotate', 'target': 'b0', 'dirVect': [0.0, 0.0]},
        'a direction that points nowhere')
refused(L, {'op': 'rotate', 'target': 'b0'},
        'a rotate that says nothing')
refused(L, {'op': 'move', 'target': 'b0'},
        'a move that says nothing')
refused(L, {'op': 'add', 'type': 'Source', 'params': {'diameter': 0.1}},
        'a construction parameter a source does not take')
refused(L, {'op': 'add', 'type': 'Source', 'params': {'waist_size': 0.0}},
        'a new source with no waist')

# Power alone may be zero: a beam is still a beam when the laser is off,
# and its geometry is what the layout is drawn from.
L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'P': 0.0}})
check('but a power of zero is allowed', L.get_source('b0').P == 0.0)

# The names are checked before any value is set, so a message that
# reaches for something a source does not have leaves it untouched
# rather than half changed.
L = fresh()
q0 = complex(L.get_source('b0').qx)
refused(L, {'op': 'set', 'target': 'b0',
            'attrs': {'waist_size_x': 1*mm, 'inv_ROC_HR': 1.0}},
        'a batch with one attribute that does not belong')
check('  and none of the batch was applied',
      complex(L.get_source('b0').qx) == q0, str(L.get_source('b0').qx))

# A source is not an optics, and the optics operations do not reach it:
# there is no beam to square a laser onto, and nothing to slide it
# along that is not itself.
refused(L, {'op': 'align', 'target': 'b0', 'beam': 'b0', 'beam_index': 0,
            'point': [0.2, 0.0]},
        'aligning a source to a beam')
refused(L, {'op': 'slide', 'target': 'b0', 'beam': 'b0', 'beam_index': 0,
            'distance': 0.1},
        'sliding a source along a beam')


print('--- the tracing rules ---')

L = fresh()
L.apply_edit({'op': 'rules', 'rules': {'order': 5}})
check('the order can be set', L.rules.order == 5, str(L.rules.order))
check('  and is a whole number', isinstance(L.rules.order, int),
      type(L.rules.order).__name__)
L.apply_edit({'op': 'rules', 'rules': {'power_threshold': 0.0}})
check('a threshold of zero is allowed - it means chase everything',
      L.rules.power_threshold == 0.0)
L.apply_edit({'op': 'rules', 'rules': {'open_beam_length': 2.5}})
check('the open beam length can be set', L.rules.open_beam_length == 2.5)

refused(L, {'op': 'rules', 'rules': {'order': 'lots'}},
        'an order that is not a number')
refused(L, {'op': 'rules', 'rules': {'order': 2.5}},
        'a fractional order')
refused(L, {'op': 'rules', 'rules': {'order': -1}},
        'a negative order')
refused(L, {'op': 'rules', 'rules': {'order': MAX_RULE_ORDER + 1}},
        'an order past what the trace can be asked for')
refused(L, {'op': 'rules', 'rules': {'power_threshold': -1e-6}},
        'a negative threshold')
refused(L, {'op': 'rules', 'rules': {'open_beam_length': 0.0}},
        'an open beam of no length')
refused(L, {'op': 'rules', 'rules': {'sigma_main': 3}},
        'a drawing option dressed as a rule')

# The rules decide how much of the picture there is, so changing one
# has to invalidate the trace rather than leave the old beams standing.
L = fresh()
few = len(L.scene_dict()['beams'])
L.apply_edit({'op': 'rules', 'rules': {'order': 12,
                                       'power_threshold': 1e-12}})
many = len(L.scene_dict()['beams'])
check('a deeper trace really is deeper', many > few, '%d -> %d' % (few, many))


print('--- undo restores the source itself ---')

L = fresh()
src = L.get_source('b0')
L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'waist_size_x': 1*mm}})
L.undo()
check('undo puts the waist back',
      abs(source_waist(L.get_source('b0'))['waist_size'][0] - 0.2*mm) < 1e-18,
      str(source_waist(L.get_source('b0'))['waist_size'][0]))
check('  onto the very object the user holds', L.get_source('b0') is src)

L.apply_edit({'op': 'rename', 'target': 'b0', 'name': 'laser'})
L.undo()
check('undoing a rename keeps the object', L.get_source('b0') is src)

L.apply_edit({'op': 'remove', 'target': 'b0'})
check('the source is gone', L.sources == [])
L.undo()
check('undoing a removal brings back the same object',
      L.sources and L.sources[0] is src, str(L.sources))
L.redo()
check('and redo takes it out again as itself', L.sources == [])
L.undo()
check('once more, still itself', L.sources[0] is src)

# An edit that was refused is not a step of the history: nothing
# changed, and an undo spent putting back what is already there is a
# press wasted.
L = fresh()
L.apply_edit({'op': 'move', 'target': 'b0', 'pos': [0.1, 0.0]})
try:
    L.apply_edit({'op': 'set', 'target': 'b0', 'attrs': {'wl': -1.0}})
except EditError:
    pass
L.undo()
check('a refused source edit costs no undo',
      np.allclose(L.get_source('b0').pos, [0.0, 0.0]),
      str(L.get_source('b0').pos))


print('--- save and load carry a source across ---')

L = fresh()
L.apply_edit({'op': 'set', 'target': 'b0',
              'attrs': {'waist_size_x': 0.4*mm, 'waist_pos_x': 0.15,
                        'waist_size_y': 0.6*mm, 'P': 0.5}})
L.apply_edit({'op': 'add', 'type': 'Source', 'name': 'S1',
              'params': {'pos': [0.0, 0.3]}})
path = os.path.join(WORK, 'sources.json')
L.save(path)

back = OpticalLayout.load(path)
check('both sources come back',
      [s.name for s in back.sources] == ['b0', 'S1'],
      str([s.name for s in back.sources]))
a, b = source_waist(L.get_source('b0')), source_waist(back.get_source('b0'))
check('  with the waist exactly as it was',
      a == b, '%s vs %s' % (a, b))
check('  and the power', back.get_source('b0').P == 0.5)

# Loading into a live layout keeps the objects that survived it, which
# is what lets the user's own variable go on naming the right source.
same = fresh()
held = same.get_source('b0')
same.update_from_file(path)
check('a load reuses the source that was already there',
      same.get_source('b0') is held)
check('  and updates it in place',
      abs(source_waist(held)['waist_size'][0] - 0.4*mm) < 1e-18,
      str(source_waist(held)['waist_size'][0]))

check('the scene dict of a loaded layout matches',
      back.scene_dict()['sources'] == L.scene_dict()['sources'])


print('--- source_to_dict and source_scene_dict describe the same beam ---')

b = fresh().get_source('b0')
saved = source_to_dict(b)
shown = source_scene_dict(b)
for key in ['name', 'pos', 'dirAngle', 'length', 'wl', 'P', 'n', 'qx', 'qy']:
    check('%s agrees between the two' % key,
          list(np.atleast_1d(saved[key])) == list(np.atleast_1d(shown[key])),
          '%s vs %s' % (saved[key], shown[key]))
check('only the scene one carries the derived waist',
      'waist_size' in shown and 'waist_size' not in saved)
check('and only the scene one carries the direction vector',
      'dirVect' in shown and 'dirVect' not in saved)
check('a saved source reads back as itself',
      source_to_dict(source_from_dict(saved)) == saved)


print('--- the whitelist is what it says ---')

check('the four waist names are editable',
      {'waist_size_x', 'waist_size_y', 'waist_pos_x', 'waist_pos_y'}
      <= EDITABLE_SOURCE_ATTRS)
check('the name is not, since renaming has its own operation',
      'name' not in EDITABLE_SOURCE_ATTRS)
check('nor is anything of an optics',
      not (EDITABLE_SOURCE_ATTRS & {'diameter', 'inv_ROC_HR', 'HRcenter',
                                    'anchor_point', 'f'}))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
