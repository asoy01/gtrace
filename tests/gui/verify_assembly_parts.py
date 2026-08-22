'''
An element and the parts that hold it, built in one call.

A mirror on its own is something to trace with. What is bolted to a
bench is a mirror in a mount on a pedestal held down by a fork, and
that is four objects and three joints to get right - four steps of
undo in the viewer, and three offsets to look up. ``assembly()``
builds one, and the viewer's ``+ Assembly`` menu adds one.

What comes back is **split** - ``(optics, bodies)`` - since that is the
division everything downstream makes: a layout registers the two by
different doors, and the trace sees only the first. Splitting it here
is what saves every caller from sorting the list out again by class.

Three things carry it and are checked hardest.

The first is the stack. Every piece derives its pose from the one
below it, so **the element is the thing to move**: the checks here
move the mirror and require the mount, the pedestal and the fork to
have gone with it, with nothing notified and nothing kept in step by
hand. The pedestal stands in the hole the mount is bolted down
through, which is a named point of the mount rather than a number
written here.

The second is that it is one edit. The whole assembly arrives through
a single message, so it is one step of undo - and undoing it leaves
the layout exactly as it was, not three quarters of a mirror.

The third is the names. A part is known by what it is, so a two-inch
mirror comes down as M2 held by MT2 on P2 in FK2 - each piece taking
the first number free for its own kind, across the one namespace the
layout shares.
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
from gtrace.layout import (OpticalLayout, TraceRules, EditError, q_from_waist,
                           assembly, assembly_kinds, mirror_assembly,
                           lens_assembly)
from gtrace.mechanics import Mechanics, model_points
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

def fresh():
    b0 = GaussianBeam(q0=q_from_waist(1*mm, 0.0, 1064*nm), wl=1064*nm,
                      pos=[-0.5, 0.0], dirAngle=0.0, name='b0')
    return OpticalLayout(optics=[], sources=[b0],
                         rules=TraceRules(order=1, power_threshold=1e-3),
                         name='Assemblies')

print()
print('--- what there is ---')

kinds = assembly_kinds()
check('the kinds are the four a bench starts from',
      [k['kind'] for k in kinds] == ['MIRROR-1IN', 'MIRROR-2IN',
                                     'LENS-1IN', 'LENS-2IN'],
      json.dumps([k['kind'] for k in kinds]))
check('each says what to call it and what it is',
      all(k['label'] and k['description'] and k['prefix'] for k in kinds))
# A front end has one clicked point to give, and the two kinds do not
# stand by the same point of their element: a mirror by the centre of
# its front face, where light turns and where a layout measures to, a
# lens by its centre. Half a substrate apart, which is millimetres.
check('  and which point it stands by',
      {k['kind']: k['place'] for k in kinds}
      == {'MIRROR-1IN': 'HRcenter', 'MIRROR-2IN': 'HRcenter',
          'LENS-1IN': 'center', 'LENS-2IN': 'center'},
      json.dumps({k['kind']: k['place'] for k in kinds}))
check('  and the kinds are strict JSON, for the scene',
      json.loads(json.dumps(kinds)) == kinds)
try:
    assembly('NO-SUCH-KIND')
    ok = False
except ValueError:
    ok = True
check('a kind that is not there is refused', ok)

print()
print('--- the pieces, and what holds what ---')

# What each kind sits its mount at, in the optic's own frame. The two
# inch mount is bolted 5 mm further back than the drawing's designed
# position; the others sit where the model says.
WANT_OFFSET = {'MIRROR-1IN': [0.0, 0.0], 'MIRROR-2IN': [-5*mm, 0.0],
               'LENS-1IN': [0.0, 0.0], 'LENS-2IN': [0.0, 0.0]}

for kind, optic_class, mount_model in [('MIRROR-1IN', opt.Mirror, 'MOUNT-25'),
                                       ('MIRROR-2IN', opt.Mirror, 'MOUNT-50'),
                                       ('LENS-1IN', opt.Lens, 'HOLDER-25'),
                                       ('LENS-2IN', opt.Lens, 'HOLDER-50')]:
    optics, bodies = assembly(kind, center=[0.3, 0.1], angle=np.pi/2)
    pieces = optics + bodies
    optic = optics[0]
    check('%s builds an element and three bodies, split' % kind,
          len(optics) == 1 and len(bodies) == 3
          and isinstance(optic, optic_class)
          and all(isinstance(x, Mechanics) for x in bodies),
          str([[type(x).__name__ for x in optics],
               [type(x).__name__ for x in bodies]]))
    check('  the element stands where it was asked to',
          np.allclose(optic.center, [0.3, 0.1]),
          str(np.round(optic.center, 4).tolist()))
    check('  facing the way it was asked to',
          abs(optic.normAngleHR - np.pi/2) < 1e-12)
    mount, ped, fork = pieces[1], pieces[2], pieces[3]
    check('  the mount stands on the element',
          mount.attached_to is optic and mount.model == mount_model,
          str(mount.model))
    check('  seated where the kind says, in the optic\'s own frame',
          np.allclose(mount.offset, WANT_OFFSET[kind]),
          str(np.round(mount.offset, 5).tolist()))
    # And that offset read on the bench: x runs along the face normal,
    # so it is checked against the normal rather than against +x - the
    # element here is turned, so the two are not the same.
    normal = np.array([np.cos(optic.normAngleHR), np.sin(optic.normAngleHR)])
    across = np.array([-normal[1], normal[0]])
    away = np.asarray(mount.center) - np.asarray(optic.center)
    check('  which is where it stands, measured along the face normal',
          abs(np.dot(away, normal) - WANT_OFFSET[kind][0]) < 1e-12
          and abs(np.dot(away, across) - WANT_OFFSET[kind][1]) < 1e-12,
          '%.4f mm along, %.4f mm across'
          % (np.dot(away, normal)/mm, np.dot(away, across)/mm))
    check('  the pedestal stands on the mount',
          ped.attached_to is mount)
    check('  the fork stands on the pedestal, free to swing',
          fork.attached_to is ped and fork.fix_rotation is False)

# The pedestal in the post hole: the point the mount's model names,
# read from the library rather than written down here.
(optic,), (mount, ped, fork) = assembly('MIRROR-1IN', center=[0.0, 0.0],
                                       angle=0.0)
pieces = [optic, mount, ped, fork]
post = model_points('MOUNT-25')['post']
check('the pedestal stands in the hole the mount is bolted down through',
      np.allclose(ped.center, mount.to_world(post)),
      '%s vs %s' % (np.round(ped.center, 4).tolist(),
                    np.round(mount.to_world(post), 4).tolist()))
check('  and the fork closes on the pedestal',
      np.allclose(fork.center, ped.center))

print()
print('--- where the mount is bolted ---')

# The two inch mount sits 5 mm back. What the pedestal stands in is a
# point of the mount, so it moves with it: the post hole ends up 5 mm
# further back than the one inch stack's.
(m1,), (mt1, pd1, _) = assembly('MIRROR-1IN', center=[0.0, 0.0], angle=0.0)
(m2,), (mt2, pd2, _) = assembly('MIRROR-2IN', center=[0.0, 0.0], angle=0.0)
check('the one inch mount sits where its model says',
      np.allclose(mt1.offset, [0.0, 0.0]))
check('the two inch mount sits 5 mm back',
      np.allclose(mt2.offset, [-5*mm, 0.0]),
      str(np.round(mt2.offset, 5).tolist()))
check('  and the pedestal goes back with it',
      abs((pd2.center[0] - pd1.center[0]) + 5*mm) < 1e-12,
      '%.4f mm' % ((pd2.center[0] - pd1.center[0])/mm))
check('  the post hole still being the mount\'s own point',
      np.allclose(pd2.center, mt2.to_world(model_points('MOUNT-50')['post'])))

# A number given wins over the kind's.
(m3,), (mt3, _, _) = assembly('MIRROR-2IN', center=[0.0, 0.0], angle=0.0,
                              mount_offset=[0.0, 0.0])
check('an offset given wins over the kind\'s default',
      np.allclose(mt3.offset, [0.0, 0.0]))
(m4,), (mt4, _, _) = mirror_assembly(name='M4', mount_offset=[2*mm, 3*mm])
check('  and mirror_assembly takes one directly',
      np.allclose(mt4.offset, [2*mm, 3*mm]))
(l1,), (h1, _, _) = lens_assembly(name='L1', holder_offset=[-1*mm, 0.0])
check('a lens holder takes one the same way',
      np.allclose(h1.offset, [-1*mm, 0.0]))

# It is an offset like any other, so the panel still edits it and the
# stack still follows the element.
L = fresh()
L.add_assembly('MIRROR-2IN', center=[0.2, 0.0], angle=np.pi)
L.apply_edit({'op': 'set', 'target': 'MT1',
              'attrs': {'offset': [-8*mm, 0.0]}})
check('and the protocol can move it afterwards',
      np.allclose(L.get_mechanics('MT1').offset, [-8*mm, 0.0]))
check('  with the pedestal following',
      np.allclose(L.get_mechanics('P1').center,
                  L.get_mechanics('MT1').to_world(
                      model_points('MOUNT-50')['post'])))

print()
print('--- the element is the thing to move ---')

before = [np.array(x.center) for x in pieces[1:]]
optic.center = optic.center + [0.4, -0.2]
after = [np.array(x.center) for x in pieces[1:]]
check('moving the element carries the whole stack',
      all(np.allclose(a - b, [0.4, -0.2]) for a, b in zip(after, before)),
      str([np.round(a - b, 4).tolist() for a, b in zip(after, before)]))
optic.normAngleHR = np.pi / 3
check('  and turning it turns the stack with it',
      all(abs(x.rotationAngle - np.pi/3) < 1e-12 for x in pieces[1:]),
      str([round(x.rotationAngle, 4) for x in pieces[1:]]))
check('  the pedestal is still in the post hole',
      np.allclose(ped.center, mount.to_world(post)))

print()
print('--- through a layout ---')

L = fresh()
made_optics, made_bodies = L.add_assembly('MIRROR-1IN', center=[0.0, 0.0],
                                          angle=np.pi)
check('add_assembly registers the element and the parts',
      [o.name for o in L.optics] == ['M1']
      and [m.name for m in L.mechanics] == ['MT1', 'P1', 'FK1'],
      json.dumps([[o.name for o in L.optics],
                  [m.name for m in L.mechanics]]))
check('  and hands them back split, hosts first within each',
      [x.name for x in made_optics] == ['M1']
      and [x.name for x in made_bodies] == ['MT1', 'P1', 'FK1'])

L.add_assembly('MIRROR-2IN', center=[0.3, 0.0])
check('a second one takes the next number of every kind',
      [o.name for o in L.optics] == ['M1', 'M2']
      and [m.name for m in L.mechanics] == ['MT1', 'P1', 'FK1',
                                            'MT2', 'P2', 'FK2'],
      json.dumps([m.name for m in L.mechanics]))
L.add_assembly('LENS-1IN', center=[0.6, 0.0], f=0.15)
check('a lens brings a holder, and the pedestals go on counting',
      [m.name for m in L.mechanics][-3:] == ['HLD1', 'P3', 'FK3'],
      json.dumps([m.name for m in L.mechanics][-3:]))
check('  and the lens is the focal length it was ordered by',
      abs(L.get_optics('L1').f - 0.15) < 1e-9,
      '%.6f' % L.get_optics('L1').f)

L = fresh()
L.add_assembly('MIRROR-1IN', name='ITMX', center=[0.0, 0.0])
check('a name given is the element\'s',
      [o.name for o in L.optics] == ['ITMX']
      and [m.name for m in L.mechanics] == ['MT1', 'P1', 'FK1'])

L = fresh()
L.add_mechanics(Mechanics(shapes=[draw.Circle([0, 0], 0.01)], name='MT1'))
L.add_assembly('MIRROR-1IN', center=[0.0, 0.0])
check('a part name already taken is stepped over',
      [m.name for m in L.mechanics] == ['MT1', 'MT2', 'P1', 'FK1'],
      json.dumps([m.name for m in L.mechanics]))

print()
print('--- one message, one step of undo ---')

L = fresh()
L.apply_edit({'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-2IN',
              'name': 'M1', 'params': {'center': [0.2, 0.1],
                                       'angle': np.pi}})
check('the protocol builds the whole assembly',
      [o.name for o in L.optics] == ['M1']
      and [m.name for m in L.mechanics] == ['MT1', 'P1', 'FK1'])
check('  the element where the message said',
      np.allclose(L.get_optics('M1').center, [0.2, 0.1]))
L.apply_edit({'op': 'undo'})
check('undo takes the whole assembly away, in one step',
      L.optics == [] and L.mechanics == [],
      json.dumps([[o.name for o in L.optics],
                  [m.name for m in L.mechanics]]))
L.apply_edit({'op': 'redo'})
check('  and redo puts all four back',
      [o.name for o in L.optics] == ['M1']
      and [m.name for m in L.mechanics] == ['MT1', 'P1', 'FK1'])

# The scene a front end is handed.
scene = L.scene_dict()
check('the scene offers the kinds a front end can add',
      [a['kind'] for a in scene['assemblies']]
      == [k['kind'] for k in assembly_kinds()])
check('  and the layout it built is in it',
      len(scene['optics']) == 1 and len(scene['mechanics']) == 3)

print()
print('--- what the protocol refuses ---')

for msg, why in [
        ({'op': 'add', 'type': 'Assembly', 'kind': 'NO-SUCH'},
         'a kind that is not there'),
        ({'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-1IN',
          'params': {'colour': 'red'}}, 'a parameter it does not take'),
        ({'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-1IN',
          'name': '  '}, 'a name of nothing'),
        ({'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-1IN',
          'name': 'M1'}, 'a name already taken')]:
    L2 = fresh()
    L2.add_assembly('MIRROR-1IN', name='M1')
    n_optics = len(L2.optics)
    n_mech = len(L2.mechanics)
    try:
        L2.apply_edit(msg)
        ok = False
    except EditError:
        ok = True
    check('%s is refused' % why, ok, json.dumps(msg))
    check('  and nothing was half added',
          len(L2.optics) == n_optics and len(L2.mechanics) == n_mech,
          '%d optics, %d bodies' % (len(L2.optics), len(L2.mechanics)))

print()
print('--- saving, and what a saved one is ---')

L = fresh()
L.add_assembly('MIRROR-2IN', center=[0.1, 0.2], angle=0.7)
path = os.path.join(WORK, 'assembly_layout.json')
L.save(path)
L2 = OpticalLayout.load(path)
check('a saved assembly loads back whole',
      [o.name for o in L2.optics] == ['M1']
      and [m.name for m in L2.mechanics] == ['MT1', 'P1', 'FK1'])
check('  with the stack still standing on the element',
      L2.get_mechanics('MT1').attached_to is L2.get_optics('M1')
      and L2.get_mechanics('P1').attached_to is L2.get_mechanics('MT1')
      and L2.get_mechanics('FK1').attached_to is L2.get_mechanics('P1'))
m1 = L2.get_optics('M1')
p1 = L2.get_mechanics('P1')
was = np.array(p1.center)
m1.center = m1.center + [0.5, 0.0]
check('  and moving the element still carries it',
      np.allclose(p1.center - was, [0.5, 0.0]))

print()
print('--- a mirror stood by its front face ---')

# The point the click gives. It is not the substrate centre: the two
# are half a thickness apart along the face normal, and it is the
# front face a layout is built from.
for kind, thickness in [('MIRROR-1IN', 6*mm), ('MIRROR-2IN', 12.7*mm)]:
    (m,), (mt, pd, fk) = assembly(kind, HRcenter=[0.3, 0.1], angle=np.pi)
    check('%s puts the front face where it was asked to' % kind,
          np.allclose(m.HRcenter, [0.3, 0.1]),
          str(np.round(m.HRcenter, 6).tolist()))
    # Facing back down -x, so the glass runs the other way: +x.
    check('  with the substrate behind it, half a thickness back',
          np.allclose(m.center, [0.3 + thickness / 2, 0.1]),
          str(np.round(m.center, 6).tolist()))
    # Naming the place the other way round must build the same stack.
    # The mount is not always on the substrate centre - the two inch
    # one is bolted 5 mm back - so the claim is that nothing moves,
    # not that the mount is anywhere in particular.
    (m2,), parts2 = assembly(kind, center=m.center.tolist(), angle=np.pi)
    check('  and the stack is the one center would have built',
          np.allclose(m2.HRcenter, m.HRcenter)
          and all(np.allclose(a.center, b.center)
                  for a, b in zip(parts2, [mt, pd, fk])),
          str([np.round(x.center, 6).tolist() for x in parts2]))

# Two ways of saying where one thing goes. Taking both would leave the
# caller with no way of knowing which one was used.
try:
    assembly('MIRROR-1IN', center=[0.0, 0.0], HRcenter=[0.1, 0.0])
    both = False
except ValueError:
    both = True
check('center and HRcenter together are refused', both)
check('a mirror given neither still stands at the origin',
      np.allclose(mirror_assembly(name='M9')[0][0].center, [0.0, 0.0]))

# What the viewer sends when a mirror assembly is placed by a click.
L = fresh()
L.apply_edit({'op': 'add', 'type': 'Assembly', 'kind': 'MIRROR-2IN',
              'name': 'M7', 'params': {'HRcenter': [0.42, -0.13],
                                       'angle': np.pi}})
check('an add message may say HRcenter',
      np.allclose(L.get_optics('M7').HRcenter, [0.42, -0.13]),
      str(np.round(L.get_optics('M7').HRcenter, 6).tolist()))
check('  and the three parts come with it',
      len(L.mechanics) == 3, '(%d bodies)' % len(L.mechanics))

print()
print('--- what a piece left out means ---')

optics, bodies = mirror_assembly(name='M1', pedestal=None, fork=None)
check('a mirror with only a mount is one element and one body',
      len(optics) == 1 and len(bodies) == 1
      and isinstance(bodies[0], Mechanics))
optics, bodies = mirror_assembly(name='M1', mount=None)
check('and one with no mount stands its pedestal on the element itself',
      len(bodies) == 2 and bodies[0].attached_to is optics[0])
optics, bodies = lens_assembly(name='L1', holder=None, pedestal=None,
                               fork=None)
check('a lens with nothing to hold it is the lens alone',
      len(optics) == 1 and bodies == [])

print()
print('--- the beams still see one element ---')

L = fresh()
L.add_assembly('MIRROR-1IN', center=[0.0, 0.0], angle=np.pi,
               Refl_HR=0.9, Trans_HR=0.1)
L.trace()
n_beams = len(L.beams)
check('a trace runs with an assembly in it', n_beams > 0, '(%d beams)' % n_beams)
L.apply_edit({'op': 'set', 'target': 'FK1', 'attrs': {'offset_angle': 0.4}})
check('swinging the fork leaves the trace standing',
      L.beams is not None and len(L.beams) == n_beams,
      'a body is not something a beam can hit')

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
