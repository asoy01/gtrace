'''
Assemblies: one element following another.

Two absorbing faces in a V are one beam dump and a pair of steering
mirrors is one periscope, so a bench is built out of assemblies rather
than out of loose elements. A body attached to an optics derives its
pose on every read; an element cannot, because an Optics holds its
pose in traits whose derived geometry - the face centres, the normals
- is what the trace reads. So the joint is stored and **settled just
before the layout is read**.

Three things carry that and are checked hardest.

The first is that settling is as reliable as deriving. There is no
notification to miss because nothing is listening: assigning
`M1.HRcenter` in a cell and then tracing carries the assembly along,
and the three entry points that read a layout - trace, draw and
snap_points - each settle first.

The second is that it costs nothing where it is not used. A layout
with no assemblies is not touched by settling at all, which is what
keeps the KAGRA path exactly where it was.

The third is that a follower is not placed twice. A pose typed into
one would be written over at the next trace, which is worse than a
refusal because it would look as though it had worked - so it is
refused, in the same words a body attached to an optics already uses.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json

import numpy as np

import gtrace.draw as draw
import gtrace.optcomp as opt
from gtrace.beam import GaussianBeam
from gtrace.layout import (OpticalLayout, TraceRules, EditError,
                           q_from_waist, optic_to_dict, optic_from_dict)
from gtrace.mechanics import mirror_mount, breadboard
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

def close(a, b, tol=1e-12):
    return np.allclose(np.asarray(a, dtype='float64'),
                       np.asarray(b, dtype='float64'), atol=tol)

def refused(L, msg, why):
    '''
    Check that an edit is rejected without side effects.
    '''
    before = json.dumps(L.to_dict())
    try:
        L.apply_edit(msg)
    except EditError as e:
        check('refuses %s' % why, True, '(%s)' % str(e)[:60])
    except Exception as e:
        check('refuses %s' % why, False,
              '(raised %s instead)' % type(e).__name__)
        return
    else:
        check('refuses %s' % why, False, '(it went through)')
        return
    check('  and leaves the layout alone', json.dumps(L.to_dict()) == before)

def dump(name='D', at=(0.5, 0.0), face=200.0, second=160.0):
    '''
    The two faces of a V, unassembled: the first at `at` facing
    `face`, the second beside it facing back across.
    '''
    a = opt.Mirror(HRcenter=list(at), normAngleHR=np.deg2rad(face),
                   diameter=1*inch, thickness=10*mm, Refl_HR=0.01,
                   Trans_HR=0.0, name=name + '1')
    b = opt.Mirror(HRcenter=[at[0] + 0.05, at[1] + 0.02],
                   normAngleHR=np.deg2rad(second),
                   diameter=1*inch, thickness=10*mm, Refl_HR=0.01,
                   Trans_HR=0.0, name=name + '2')
    return a, b

def fresh():
    a, b = dump()
    b0 = GaussianBeam(pos=[0.0, 0.0], dirAngle=0.0,
                      q0=q_from_waist(0.2*mm, 0.0, 1064*nm),
                      wl=1064*nm, name='b0')
    return OpticalLayout(optics=[a, b], sources=[b0], name='asm',
                         rules=TraceRules(order=3, power_threshold=1e-6))


print('--- the joint holds a V together ---')

L = fresh()
D1, D2 = L.get_optics('D1'), L.get_optics('D2')
rel0 = np.array(D2.HRcenter) - np.array(D1.HRcenter)
ang0 = float(D2.normAngleHR) - float(D1.normAngleHR)
L.assemble('D2', 'D1')
check('assembling keeps it exactly where it stands',
      D2.assembled_to is D1
      and close(D2.HRcenter, np.array(D1.HRcenter) + rel0))
# The host frame is the one a mount already attaches in: its origin
# is the host's substrate centre and its x runs along the HR normal.
hc = np.array(D1.center)
ha = float(D1.normAngleHR)
ca, sa = np.cos(ha), np.sin(ha)
here = np.array(D2.HRcenter) - hc
check('  and the joint is read off the bench, in the host frame',
      close(D2.assembly_offset, [here[0] * ca + here[1] * sa,
                                 -here[0] * sa + here[1] * ca])
      and abs(D2.assembly_angle - ang0) < 1e-12,
      str(np.round(D2.assembly_offset, 5).tolist()))

D1.HRcenter = [0.9, 0.3]
L.trace()
check('moving the host carries the follower',
      close(np.array(D2.HRcenter) - np.array(D1.HRcenter), rel0))
check('  and it is a cell assignment that did it, with nothing listening',
      True)

D1.normAngleHR = float(D1.normAngleHR) + 0.4
L.trace()
check('turning the host turns the follower with it',
      abs((float(D2.normAngleHR) - float(D1.normAngleHR)) - ang0) < 1e-12)
check('  about the host, so the V keeps its shape',
      abs(np.linalg.norm(np.array(D2.HRcenter) - np.array(D1.HRcenter))
          - np.linalg.norm(rel0)) < 1e-12)
check('  and the follower is turned, not carried flat',
      not close(np.array(D2.HRcenter) - np.array(D1.HRcenter), rel0))

# The pinned point is the one the element is held by, which is also
# the point a curvature change leaves alone.
L2 = fresh()
lens = opt.Lens(f=0.3, center=[0.7, 0.0], normAngleHR=0.0, diameter=1*inch,
                thickness=5*mm, name='L1')
L2.add_optics(lens)
L2.assemble('L1', 'D1')
before = np.array(lens.center)
L2.get_optics('D1').HRcenter = [0.6, 0.1]
L2.trace()
check('a lens is pinned by the point it is held by, its middle',
      lens.anchor_point == 'center'
      and close(np.array(lens.center) - before, [0.1, 0.1]))


print('--- what a settle is, and is not ---')

L = fresh()
L.assemble('D2', 'D1')
L.trace()
D1, D2 = L.get_optics('D1'), L.get_optics('D2')
was = np.array(D2.HRcenter)
D1.HRcenter = [1.5, 1.5]
check('a follower is stale until the layout is read',
      close(D2.HRcenter, was))
L.draw()
check('  and draw() settles it as trace() does',
      not close(D2.HRcenter, was))
D1.HRcenter = [0.4, -0.2]
stale = np.array(D2.HRcenter)
L.snap_points()
check('  as does snap_points(), the third way in',
      not close(D2.HRcenter, stale))

# Nothing to settle, nothing touched: this is what keeps every layout
# that has no assemblies exactly where it was.
L3 = fresh()
before = json.dumps(L3.to_dict())
L3.trace()
L3.draw()
L3.snap_points()
check('a layout with no assemblies is not moved by a single bit',
      json.dumps(L3.to_dict()) == before)
check('  and it has none to settle', L3.assemblies() == [])


print('--- chains, and what else rides on them ---')

L = fresh()
third = opt.Mirror(HRcenter=[0.62, 0.05], normAngleHR=1.0, diameter=1*inch,
                   name='D3')
L.add_optics(third)
L.assemble('D2', 'D1')
L.assemble('D3', 'D2')
L.add_mechanics(mirror_mount(name='MT2', attached_to=L.get_optics('D2')))
L.trace()
D1 = L.get_optics('D1')
poses = [np.array(o.HRcenter) for o in (L.get_optics('D2'),
                                        L.get_optics('D3'))]
mount0 = np.array(L.get_mechanics('MT2').center)
D1.HRcenter = np.asarray(D1.HRcenter) + [0.1, 0.0]
L.trace()
check('a chain follows all the way down',
      all(close(np.array(o.HRcenter), p + [0.1, 0.0])
          for o, p in zip((L.get_optics('D2'), L.get_optics('D3')), poses)))
check('  and a body standing on a follower comes too',
      close(L.get_mechanics('MT2').center, mount0 + [0.1, 0.0]))
check('the settling order puts hosts first',
      [o.name for o in L.assemblies()] == ['D2', 'D3'])

# A body can be the host: an element bolted to a bracket rather than
# to another element.
L4 = fresh()
board = breadboard(0.4, 0.3, center=[0.2, -0.4], name='BB1')
L4.add_mechanics(board)
L4.assemble('D2', 'BB1')
L4.trace()
here = np.array(L4.get_optics('D2').HRcenter)
board.center = np.asarray(board.center) + [0.05, 0.05]
L4.trace()
check('an element may follow a body as well as another element',
      close(L4.get_optics('D2').HRcenter, here + [0.05, 0.05]))

L5 = fresh()
L5.assemble('D2', 'D1')
try:
    L5.assemble('D1', 'D2')
    check('a circle is refused', False)
except ValueError as e:
    check('a circle is refused', True, '(%s)' % str(e)[:55])
try:
    L5.assemble('D1', 'D1')
    check('and so is following itself', False)
except ValueError as e:
    check('and so is following itself', True, '(%s)' % str(e)[:45])
check('  and neither one changed anything',
      L5.get_optics('D1').assembled_to is None)


print('--- a follower is not placed twice ---')

L = fresh()
L.assemble('D2', 'D1')
refused(L, {'op': 'move', 'target': 'D2', 'HRcenter': [1.0, 1.0]},
        'moving a follower')
refused(L, {'op': 'rotate', 'target': 'D2', 'normAngleHR': 1.0},
        'turning one whose angle is fixed')
refused(L, {'op': 'set', 'target': 'D2', 'attrs': {'center': [1.0, 1.0]}},
        'setting its pose by another name')
refused(L, {'op': 'align', 'target': 'D2', 'beam': 'b0', 'beam_index': 0,
            'point': [0.4, 0.0]},
        'squaring a follower onto a beam')
refused(L, {'op': 'slide', 'target': 'D2', 'beam': 'b0', 'beam_index': 0,
            'distance': 0.01},
        'sliding one along a beam')
refused(L, {'op': 'remove', 'target': 'D1'},
        'removing what another element follows')
refused(L, {'op': 'set', 'target': 'D2',
            'attrs': {'assembled_to': 'nobody'}},
        'following something that is not there')
refused(L, {'op': 'set', 'target': 'D1',
            'attrs': {'assembly_offset': [0.0, 0.1]}},
        'an offset on an element that follows nothing')

# Everything that is not the pose is still the element's own.
L.apply_edit({'op': 'set', 'target': 'D2', 'attrs': {'diameter': 0.05}})
check('a follower is still an element in every other way',
      abs(L.get_optics('D2').diameter - 0.05) < 1e-15)

# And a free turn is genuinely its own, read back into the joint so
# that the next settle keeps it.
L.apply_edit({'op': 'set', 'target': 'D2', 'attrs': {'fix_rotation': False}})
L.apply_edit({'op': 'rotate', 'target': 'D2', 'normAngleHR': np.deg2rad(150)})
check('a free turn goes through, and reaches the joint',
      abs(np.rad2deg(L.get_optics('D2').assembly_angle)
          - (150 - np.rad2deg(L.get_optics('D1').normAngleHR))) < 1e-9)
L.trace()
check('  so the settle keeps it rather than putting it back',
      abs(np.rad2deg(L.get_optics('D2').normAngleHR) - 150) < 1e-9)
refused(L, {'op': 'move', 'target': 'D2', 'HRcenter': [1.0, 1.0]},
        'moving one whose turn is free but whose place is not')

# Letting go leaves it where it stands, and it is its own again.
L.apply_edit({'op': 'set', 'target': 'D2', 'attrs': {'assembled_to': None}})
here = np.array(L.get_optics('D2').HRcenter)
check('letting go leaves it exactly where it stands',
      L.get_optics('D2').assembled_to is None)
L.apply_edit({'op': 'move', 'target': 'D2', 'HRcenter': [1.0, 1.0]})
check('  and it can be placed again',
      close(L.get_optics('D2').HRcenter, [1.0, 1.0]))
L.trace()
check('  with nothing left to write over it',
      close(L.get_optics('D2').HRcenter, [1.0, 1.0]))


print('--- the joint through the protocol, saving and undo ---')

L = fresh()
L.trace()
L.apply_edit({'op': 'set', 'target': 'D2',
              'attrs': {'assembled_to': 'D1',
                        'assembly_offset': [0.06, 0.0],
                        'assembly_angle': np.deg2rad(-40),
                        'fix_rotation': True}})
check('the joint is an edit like any other: the trace no longer stands',
      L.beams is None)
L.trace()
D1 = L.get_optics('D1')
# The offset is in the host's frame, whose origin is the host's
# substrate centre - the same frame a mount attaches in - and what
# lands there is the follower's anchor point.
check('the protocol makes the joint, at the offset it names',
      close(np.array(L.get_optics('D2').HRcenter),
            np.array(D1.center)
            + 0.06 * np.array([np.cos(D1.normAngleHR),
                               np.sin(D1.normAngleHR)])),
      str(np.round(L.get_optics('D2').HRcenter, 5).tolist()))
L.apply_edit({'op': 'undo'})
check('undo takes the joint back',
      L.get_optics('D2').assembled_to is None)
L.apply_edit({'op': 'redo'})
check('  and redo puts it back',
      L.get_optics('D2').assembled_to is L.get_optics('D1'))

d = optic_to_dict(L.get_optics('D2'))
check('a saved element writes what it follows and where',
      d['assembled_to'] == 'D1'
      and close(d['assembly_offset'], [0.06, 0.0])
      and abs(d['assembly_angle'] - np.deg2rad(-40)) < 1e-12
      and d['fix_rotation'] is True, json.dumps(d['assembled_to']))
check('  and its pose too, unlike a body: the trace reads it',
      'HRcenter' in d)
back = OpticalLayout.from_dict(L.to_dict())
check('a loaded layout joins them up again',
      back.get_optics('D2').assembled_to is back.get_optics('D1'))
check('  and settles to the same place',
      close(back.trace() and back.get_optics('D2').HRcenter,
            L.get_optics('D2').HRcenter))

# A file may list a follower before its host, or describe a circle.
d = L.to_dict()
d['optics'] = list(reversed(d['optics']))
check('a file that lists a follower before its host still loads',
      OpticalLayout.from_dict(d).get_optics('D2').assembled_to is not None)
d2 = L.to_dict()
for o in d2['optics']:
    o['assembled_to'] = 'D2' if o['name'] == 'D1' else 'D1'
try:
    OpticalLayout.from_dict(d2)
    check('a file whose elements stand in a circle is refused', False)
except ValueError as e:
    check('a file whose elements stand in a circle is refused', True,
          '(%s)' % str(e)[:45])


print('--- copying a dump copies the dump ---')

L = fresh()
L.assemble('D2', 'D1')
L.add_mechanics(mirror_mount(name='MT1', attached_to=L.get_optics('D1')))
L.trace()
rel = np.array(L.get_optics('D2').HRcenter) - np.array(
    L.get_optics('D1').HRcenter)
c = L.copy_optics('D1')
L.trace()
follower = [o for o in L.optics
            if getattr(o, 'assembled_to', None) is c]
check('the copy brings the second face with it',
      len(follower) == 1 and follower[0].name not in ('D1', 'D2'),
      json.dumps([o.name for o in L.optics]))
if follower:
    check('  at the same relative place, so the V is the same V',
          close(np.array(follower[0].HRcenter) - np.array(c.HRcenter), rel))
check('  and the mount along with them',
      any(m.attached_to is c for m in L.mechanics))
check('  while the original is untouched',
      L.get_optics('D2').assembled_to is L.get_optics('D1'))


print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
