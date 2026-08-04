'''
Stage 2b verification, Python side: the edit protocol.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, WORK

import json
import os
import sys

import numpy as np

import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.layout import (OpticalLayout, TraceRules, EditError,
                           EDITABLE_OPTIC_ATTRS, DEFAULT_LENS_F, UNDO_DEPTH)
from gtrace.nonsequential import non_seq_trace
from gtrace.draw.viewer import widget as wmod
from gtrace.unit import *

OUT = WORK

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

def refused(msg, why):
    '''
    Check that an edit message is rejected without side effects.
    Uses whatever `layout` currently refers to.
    '''
    try:
        layout.apply_edit(msg)
    except EditError as e:
        check('refuses %s' % why, True, '(%s)' % str(e)[:60])
        return
    except Exception as e:
        check('refuses %s' % why, False,
              '(raised %s instead)' % type(e).__name__)
        return
    check('refuses %s' % why, False, '(was applied)')

def make_layout():
    b0 = beam.GaussianBeam(q0=gauss.Rw2q(np.inf, 1*mm), wl=1064*nm,
                           pos=[0, 0], dirAngle=0, name='b0')

    def M(name, c, a, roc=0.0, rh=0.99, th=0.01):
        return opt.Mirror(HRcenter=c, normAngleHR=a, diameter=10*cm,
                          thickness=5*cm, wedgeAngle=deg2rad(0.25),
                          inv_ROC_HR=roc, Refl_HR=rh, Trans_HR=th,
                          Refl_AR=500*ppm, Trans_AR=1-500*ppm, n=1.45,
                          name=name)

    optics = [M('M1', [0.5, 0.0], deg2rad(135)),
              M('M2', [0.5, 0.4], deg2rad(-45), 1.0/2.0),
              M('M3', [0.9, 0.4], deg2rad(180), 0.0, 0.9, 0.1)]
    lay = OpticalLayout(optics=optics, sources=[b0],
                        rules=TraceRules(order=5, power_threshold=1e-4),
                        name='Stage 2b layout')
    return lay, optics

layout, (M1, M2, M3) = make_layout()

print('--- optics channel in the scene ---')
scene = layout.scene_dict()
check('scene has an optics list', 'optics' in scene, str(list(scene.keys())))
check('one entry per optics', len(scene['optics']) == 3,
      '(%d)' % len(scene['optics']))
o = scene['optics'][0]
check('carries the name', o['name'] == 'M1', o['name'])
check('carries the type', o['type'] == 'Mirror', o['type'])
for key in ['HRcenter', 'ARcenter', 'center', 'normAngleHR', 'diameter',
            'thickness', 'wedgeAngle', 'inv_ROC_HR', 'n']:
    check("carries '%s'" % key, key in o, str(o.get(key)))
check('centers agree with the object',
      np.allclose(o['center'], np.asarray(M1.center)),
      '%s vs %s' % (o['center'], list(np.asarray(M1.center))))
check('optics channel is JSON serializable',
      json.loads(json.dumps(scene['optics'])) == scene['optics'])

print('--- apply_edit: move ---')
layout.trace()
before = layout.beams[0].length
layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.6, 0.0]})
check('trace invalidated', layout.beams is None)
layout.trace()
check('the registered object moved',
      np.allclose(np.asarray(M1.HRcenter), [0.6, 0.0]),
      str(list(np.asarray(M1.HRcenter))))
check('the geometry followed',
      abs(layout.beams[0].length - 0.6) < 1e-9,
      '(%.4f -> %.4f)' % (before, layout.beams[0].length))

# 'center' is what the viewer sends, since that is what it outlines.
layout.apply_edit({'op': 'move', 'target': 'M1', 'center': [0.5, 0.02]})
check("move by 'center' works",
      np.allclose(np.asarray(M1.center), [0.5, 0.02]),
      str(list(np.asarray(M1.center))))
check('HRcenter followed center',
      not np.allclose(np.asarray(M1.HRcenter), [0.5, 0.02]),
      str(list(np.asarray(M1.HRcenter))))

print('--- apply_edit: rotate ---')
a0 = float(M1.normAngleHR)
layout.apply_edit({'op': 'rotate', 'target': 'M1',
                   'normAngleHR': deg2rad(130)})
check('the angle changed', abs(float(M1.normAngleHR) - deg2rad(130)) < 1e-12,
      '(%.4f -> %.4f rad)' % (a0, float(M1.normAngleHR)))
check('the normal vector followed',
      np.allclose(np.asarray(M1.normVectHR),
                  [np.cos(deg2rad(130)), np.sin(deg2rad(130))]),
      str(list(np.asarray(M1.normVectHR))))

print('--- apply_edit: set and rules ---')
layout.apply_edit({'op': 'set', 'target': 'M2',
                   'attrs': {'diameter': 0.15, 'inv_ROC_HR': 0.25}})
check('several attributes at once',
      abs(float(M2.diameter) - 0.15) < 1e-12
      and abs(float(M2.inv_ROC_HR) - 0.25) < 1e-12,
      '(d=%.4f, 1/R=%.4f)' % (float(M2.diameter), float(M2.inv_ROC_HR)))

layout.apply_edit({'op': 'rules', 'rules': {'power_threshold': 1e-6}})
check('a tracing rule changed',
      abs(layout.rules.power_threshold - 1e-6) < 1e-18,
      str(layout.rules.power_threshold))

print('--- max_stray_order lives on the optics ---')
layout, (M1, M2, M3) = make_layout()
layout.rules.power_threshold = 1e-8
check('unset by default', M1.max_stray_order is None,
      str(M1.max_stray_order))
n_free = len(layout.trace())

M1.max_stray_order = 0
layout.beams = None
n_capped = len(layout.trace())
check('capping one optics reduces the trace', n_capped < n_free,
      '(%d -> %d beams)' % (n_free, n_capped))
check('the trace-wide order is untouched', layout.rules.order == 5,
      str(layout.rules.order))

check('it survives a copy', M1.copy().max_stray_order == 0,
      str(M1.copy().max_stray_order))
check('it survives save/load',
      OpticalLayout.from_dict(layout.to_dict()).get_optics('M1')
      .max_stray_order == 0)
check('it reaches the viewer',
      [o for o in layout.scene_dict()['optics']
       if o['name'] == 'M1'][0]['max_stray_order'] == 0)
check('and an unset one comes through as null',
      [o for o in layout.scene_dict()['optics']
       if o['name'] == 'M2'][0]['max_stray_order'] is None)

layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'max_stray_order': None}})
check('it can be cleared from the front end', M1.max_stray_order is None)
layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'max_stray_order': 2}})
check('and set from the front end', M1.max_stray_order == 2)

check('per_optic_order is gone from the rules',
      not hasattr(layout.rules, 'per_optic_order'))
check('and from non_seq_trace',
      'per_optic_order' not in
      __import__('inspect').signature(non_seq_trace).parameters)

print('--- diameter keeps the sag consistent ---')
def sag_of(R, d):
    return -np.sign(R)*(abs(R) - np.sqrt(R**2 - (d/2.0)**2))

def curved(d):
    return opt.Mirror(HRcenter=[0, 0], normAngleHR=0.0, diameter=d,
                      thickness=5*cm, wedgeAngle=deg2rad(0.25),
                      inv_ROC_HR=1/2.0, inv_ROC_AR=1/3.0, name='C')

mc = curved(10*cm)
check('the sag is right to start with',
      abs(float(mc.sagHR) - sag_of(2.0, 0.1)) < 1e-15,
      '%.6f mm' % (float(mc.sagHR)*1e3))
mc.diameter = 20*cm
check('changing the diameter updates the HR sag',
      abs(float(mc.sagHR) - sag_of(2.0, 0.2)) < 1e-15,
      '%.6f mm (want %.6f)' % (float(mc.sagHR)*1e3, sag_of(2.0, 0.2)*1e3))
check('and the AR sag',
      abs(float(mc.sagAR) - sag_of(3.0, 0.2)) < 1e-15,
      '%.6f mm' % (float(mc.sagAR)*1e3))
ref = curved(20*cm)
check('landing where constructing it that way lands',
      abs(float(mc.sagHR) - float(ref.sagHR)) < 1e-15
      and abs(float(mc.sagAR) - float(ref.sagAR)) < 1e-15
      and np.allclose(np.asarray(mc.HRcenterC), np.asarray(ref.HRcenterC),
                      atol=1e-15))
check('a flat surface stays flat',
      float(opt.Mirror(HRcenter=[0, 0], diameter=0.1, inv_ROC_HR=0.0,
                       name='F').sagHR) == 0.0)

print('--- center is the middle of the substrate ---')
# HRcenter/ARcenter sit on the arcs; HRcenterC/ARcenterC are the centres
# of the chords, a sagitta away. thickness separates the chord planes,
# so the middle of the substrate is between those, and get_side_info
# builds the sides on exactly that assumption.
def wedged(roc_ar, angle=0.0):
    return opt.Mirror(HRcenter=[0, 0], normAngleHR=angle, diameter=20*cm,
                      thickness=5*cm, wedgeAngle=deg2rad(0.25),
                      inv_ROC_HR=1/2.0, inv_ROC_AR=roc_ar, name='W')

for angle in [0.0, deg2rad(30), deg2rad(137)]:
    w2 = wedged(1/3.0, angle)
    hcC = np.asarray(w2.HRcenterC)
    acC = np.asarray(w2.ARcenterC)
    nv = np.asarray(w2.normVectHR)
    check('center is between the chord planes at %.0f deg' % np.degrees(angle),
          np.allclose(np.asarray(w2.center), (hcC + acC)/2, atol=1e-15)
          and np.allclose(np.asarray(w2.center),
                          hcC - nv*float(w2.thickness)/2, atol=1e-15),
          str(list(np.asarray(w2.center).round(9))))

# The arcs stay a sagitta off their chords, which is what distinguishes
# HRcenter from HRcenterC in the first place.
w2 = wedged(1/3.0, deg2rad(30))
check('HRcenter is a sagitta off its chord centre',
      np.allclose(np.asarray(w2.HRcenterC),
                  np.asarray(w2.HRcenter) - float(w2.sagHR)*np.asarray(w2.normVectHR),
                  atol=1e-15))
check('ARcenter is a sagitta off its chord centre',
      np.allclose(np.asarray(w2.ARcenter),
                  np.asarray(w2.ARcenterC) + float(w2.sagAR)*np.asarray(w2.normVectAR),
                  atol=1e-15))

# Writing center is how a drag moves an optics, so reading it back and
# writing it must not shift anything.
w2 = wedged(1/3.0)
c0 = np.asarray(w2.center).copy()
h0 = np.asarray(w2.HRcenter).copy()
w2.center = [c0[0] + 1e-9, c0[1]]
check('nudging center by 1 nm moves the mirror by 1 nm',
      abs(float(np.asarray(w2.HRcenter)[0] - h0[0]) - 1e-9) < 1e-15,
      '%.3f nm' % (float(np.asarray(w2.HRcenter)[0] - h0[0])*1e9))

# A flat AR surface makes the two spellings identical, which is why no
# existing drawing moves.
flat = wedged(0.0, deg2rad(30))
check('with a flat AR surface the two agree exactly',
      np.allclose((np.asarray(flat.HRcenterC) + np.asarray(flat.ARcenter))/2,
                  (np.asarray(flat.HRcenterC) + np.asarray(flat.ARcenterC))/2,
                  atol=1e-15)
      and float(flat.sagAR) == 0.0)

print('--- drawing options ---')
layout, (M1, M2, M3) = make_layout()
# An astigmatic source, so that the three width modes are telling apart.
layout.sources[0].qy = gauss.Rw2q(np.inf, 2.5*mm)
layout.trace()

b = layout.beams[0]
wx, wy = b.width(0.0)
check('the test beam is astigmatic', abs(wx - wy) > 1e-4,
      'w_x=%.4f mm, w_y=%.4f mm' % (wx*1e3, wy*1e3))

def envelope_half_width():
    return abs(np.asarray(layout.draw().layers['main_beam_width']
                          .shapes[0].y)[0])

for mode, want_w in [('x', wx), ('y', wy), ('avg', (wx + wy)/2)]:
    for sigma in [1.0, 2.7, 3.0]:
        layout.draw_options = {'width_mode': mode, 'sigma_main': sigma,
                               'sigma_stray': sigma}
        got = envelope_half_width()
        check('envelope at %s, %sσ' % (mode, sigma),
              abs(got - want_w*sigma) < 1e-15,
              '%.6f mm' % (got*1e3))

layout.draw_options = {}
check("the default direction is 'x'",
      abs(envelope_half_width() - wx*2.7) < 1e-15,
      '(x=%.4f, y=%.4f, avg=%.4f mm)'
      % (wx*2.7*1e3, wy*2.7*1e3, (wx+wy)/2*2.7*1e3))

print('--- drawing options through the protocol ---')
n_beams = len(layout.beams)
beams_before = layout.beams
layout.apply_edit({'op': 'draw', 'params': {'width_mode': 'y',
                                            'sigma_main': 1.0,
                                            'sigma_stray': 1.0}})
check('the option was stored',
      layout.draw_options == {'width_mode': 'y', 'sigma_main': 1.0,
                              'sigma_stray': 1.0},
      str(layout.draw_options))
check('the trace was NOT invalidated', layout.beams is beams_before)
check('the drawing changed', abs(envelope_half_width() - wy*1.0) < 1e-15,
      '%.6f mm' % (envelope_half_width()*1e3))

scene = layout.scene_dict()
check('the scene reports the display settings in force',
      scene['display']['width_mode'] == 'y'
      and scene['display']['sigma_main'] == 1.0,
      str({k: scene['display'][k] for k in ['width_mode', 'sigma_main']}))
check('the beam count is unchanged', len(scene['beams']) == n_beams)

check('an explicit argument still wins over the stored option',
      abs(abs(np.asarray(layout.draw(width_mode='x')
                         .layers['main_beam_width'].shapes[0].y)[0])
          - wx*1.0) < 1e-15)

refused({'op': 'draw', 'params': {'width_mode': 'diagonal'}},
        'an unknown width mode')
refused({'op': 'draw', 'params': {'nonsense': 1}},
        'an unknown drawing option')
check('a refused draw option was not stored',
      'nonsense' not in layout.draw_options and
      layout.draw_options['width_mode'] == 'y', str(layout.draw_options))

raised = False
try:
    layout.draw(nonsense=1)
except TypeError:
    raised = True
check('draw() rejects a misspelt option', raised)

layout.draw_options = {}

print('--- the tracing flags reach the front end and back ---')
layout, (M1, M2, M3) = make_layout()
o1 = [o for o in layout.scene_dict()['optics'] if o['name'] == 'M1'][0]
for key, want in [('HRtransmissive', False), ('term_on_HR', False),
                  ('term_on_HR_order', 0)]:
    check("scene carries '%s'" % key, o1.get(key) == want,
          '%r' % (o1.get(key),))
check('the flags are real JSON types',
      isinstance(o1['HRtransmissive'], bool)
      and isinstance(o1['term_on_HR_order'], int),
      '%s / %s' % (type(o1['HRtransmissive']).__name__,
                   type(o1['term_on_HR_order']).__name__))

layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'HRtransmissive': True, 'term_on_HR': True,
                             'term_on_HR_order': 2}})
check('the flags can be set from the front end',
      bool(M1.HRtransmissive) and bool(M1.term_on_HR)
      and M1.term_on_HR_order == 2,
      '%s / %s / %s' % (M1.HRtransmissive, M1.term_on_HR, M1.term_on_HR_order))

# term_on_HR really terminates the beam at the HR surface.
layout.beams = None
n_term = len(layout.trace())
layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'term_on_HR': False}})
n_free2 = len(layout.trace())
check('terminating on HR shortens the trace', n_term < n_free2,
      '(%d terminated, %d not)' % (n_term, n_free2))
layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'HRtransmissive': False, 'term_on_HR_order': 0}})

print('--- curve_direction ---')
layout.apply_edit({'op': 'add', 'name': 'Cy', 'type': 'CyMirror',
                   'params': {'HRcenter': [0.2, 0.5], 'inv_ROC_HR': 0.5,
                              'curve_direction': 'v'}})
cy = layout.get_optics('Cy')
check('a CyMirror carries it into the scene',
      [o for o in layout.scene_dict()['optics']
       if o['name'] == 'Cy'][0]['curve_direction'] == 'v')
check('a plain Mirror has no such key',
      'curve_direction' not in
      [o for o in layout.scene_dict()['optics'] if o['name'] == 'M1'][0])
layout.apply_edit({'op': 'set', 'target': 'Cy',
                   'attrs': {'curve_direction': 'h'}})
check('it can be switched', cy.curve_direction == 'h', cy.curve_direction)
refused_choice = False
try:
    layout.apply_edit({'op': 'set', 'target': 'Cy',
                       'attrs': {'curve_direction': 'diagonal'}})
except EditError as e:
    refused_choice = True
    detail = str(e)
check('a value outside the allowed set is refused', refused_choice,
      detail if refused_choice else '(was applied)')
check('and the old value stands', cy.curve_direction == 'h')
layout.apply_edit({'op': 'remove', 'target': 'Cy'})

print('--- Lens through the protocol ---')
layout, (M1, M2, M3) = make_layout()
layout.trace()

layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                   'params': {'HRcenter': [0.6, 0.0],
                              'normAngleHR': np.pi}})
lens = layout.get_optics('L1')
check('a Lens can be created', type(lens).__name__ == 'Lens',
      type(lens).__name__)
check('it comes out at the default focal length',
      abs(float(lens.f) - DEFAULT_LENS_F) < 1e-9,
      'f=%.6f (default %.3f)' % (float(lens.f), DEFAULT_LENS_F))
check('it is where it was asked for',
      np.allclose(np.asarray(lens.HRcenter), [0.6, 0.0]),
      str(list(np.asarray(lens.HRcenter))))

# A mirror inherits its size and coatings from the layout; a lens must
# not. A "lens" wearing a mirror's 99% front face is one the main beam
# does not go through, and an aperture off a big mirror is a focal
# length the blank cannot be ground to.
check('a lens inherits no aperture from the mirrors',
      abs(float(lens.diameter) - float(M3.diameter)) > 1e-6
      and abs(float(lens.diameter) - 25.4*mm) < 1e-12,
      'd=%.5f (the mirrors are %.5f)'
      % (float(lens.diameter), float(M3.diameter)))
check('nor their coatings: a new lens reflects nothing at all',
      float(lens.Refl_HR) == 0.0 and float(lens.Refl_AR) == 0.0
      and float(M3.Refl_HR) > 0.5,
      'lens R=%g, mirror R=%g' % (float(lens.Refl_HR), float(M3.Refl_HR)))
check('the front face transmits', bool(lens.HRtransmissive))
check('and it has no wedge', float(lens.wedgeAngle) == 0.0)
check('its curvature anchor is the substrate centre',
      lens.ROC_anchor == 'center', str(lens.ROC_anchor))

sc = {o['name']: o for o in layout.scene_dict()['optics']}
check('the scene carries the power, not the focal length',
      'inv_f' in sc['L1'] and 'f' not in sc['L1'], str(sorted(sc['L1'])))
check('and it inverts to the focal length',
      abs(1.0/sc['L1']['inv_f'] - float(lens.f)) < 1e-9,
      '1/%.6f = %.6f' % (sc['L1']['inv_f'], 1.0/sc['L1']['inv_f']))
check('a mirror has no such key', 'inv_f' not in sc['M1'],
      str(sorted(sc['M1'])))
check('every optics reports its anchor',
      sc['M1']['ROC_anchor'] == 'HRcenter'
      and sc['L1']['ROC_anchor'] == 'center',
      '%s / %s' % (sc['M1']['ROC_anchor'], sc['L1']['ROC_anchor']))

# Nothing in a scene may be an infinity: JSON has none, and what would
# reach the browser is a token JSON.parse refuses.
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'Flat',
                   'params': {'inv_ROC_HR': 0.0, 'inv_ROC_AR': 0.0}})
flat = layout.get_optics('Flat')
check('a lens with no power has an infinite focal length',
      not np.isfinite(float(flat.f)), str(flat.f))
sc = {o['name']: o for o in layout.scene_dict()['optics']}
check('which the scene reports as zero power', sc['Flat']['inv_f'] == 0.0,
      str(sc['Flat']['inv_f']))
strict = True
try:
    json.dumps(layout.scene_dict(), allow_nan=False)
except ValueError:
    strict = False
check('the whole scene is strict JSON', strict)
layout.apply_edit({'op': 'remove', 'target': 'Flat'})

# Assigning a focal length re-solves both curvatures. The lens keeps its
# shape and stays where it is, which is what makes a mode-matching
# sweep work: L.f = ... in a loop, and nothing walks up the bench.
c_before = np.array(lens.center, dtype=float)
roc_before = (float(lens.inv_ROC_HR), float(lens.inv_ROC_AR))
shape_before = lens.shape
layout.trace()
layout.apply_edit({'op': 'set', 'target': 'L1', 'attrs': {'f': 0.25}})
check('setting f through the protocol re-solves the lens',
      abs(float(lens.f) - 0.25) < 1e-9, 'f=%.9f' % float(lens.f))
check('both curvatures moved',
      abs(float(lens.inv_ROC_HR) - roc_before[0]) > 1e-6
      and abs(float(lens.inv_ROC_AR) - roc_before[1]) > 1e-6)
check('the shape is kept', lens.shape == shape_before,
      '%s -> %s' % (shape_before, lens.shape))
check('and the lens did not move',
      np.abs(np.array(lens.center, dtype=float) - c_before).max() < 1e-15,
      str(list(np.array(lens.center, dtype=float) - c_before)))
check('the trace was invalidated', layout.beams is None)

# The whitelist can say which names are allowed, not which classes
# carry them. Only a lens has a focal length.
refused({'op': 'set', 'target': 'M1', 'attrs': {'f': 0.5}},
        'a focal length on a mirror')
check('and no stray attribute was left on it', not hasattr(M1, 'f'))

# A focal length the blank cannot be ground to. The solve happens
# before anything is assigned, so the lens is untouched.
f_before = float(lens.f)
roc_before = (float(lens.inv_ROC_HR), float(lens.inv_ROC_AR))
refused({'op': 'set', 'target': 'L1', 'attrs': {'f': 1e-6}},
        'a focal length no lens of that shape can have')
check('the lens is exactly as it was',
      float(lens.f) == f_before
      and (float(lens.inv_ROC_HR), float(lens.inv_ROC_AR)) == roc_before,
      'f=%.9f' % float(lens.f))

layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L2',
                   'params': {'f': 0.15, 'shape': 'convex-plano'}})
L2 = layout.get_optics('L2')
check('a lens can be ordered by shape as well',
      L2.shape == 'convex-plano' and abs(float(L2.f) - 0.15) < 1e-9,
      '%s, f=%.6f' % (L2.shape, float(L2.f)))
check('a flat face really is flat', float(L2.inv_ROC_AR) == 0.0)

# A meniscus is the one shape f alone does not determine: one radius is
# given and the other solved for. That third lens-only parameter has to
# reach the constructor too.
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'Men',
                   'params': {'f': 0.2, 'shape': 'meniscus',
                              'ROC_HR': -50*mm}})
men = layout.get_optics('Men')
check('a meniscus can be pinned by one radius',
      men.shape == 'convex-concave'
      and abs(1.0/float(men.inv_ROC_HR) + 50*mm) < 1e-12
      and abs(float(men.f) - 0.2) < 1e-9,
      '%s, R_HR=%.6f, f=%.6f'
      % (men.shape, 1.0/float(men.inv_ROC_HR), float(men.f)))
layout.apply_edit({'op': 'remove', 'target': 'Men'})

# Curvatures describe a lens completely, and Lens refuses a focal
# length on top of them rather than quietly preferring one. The
# default must not be imposed in that case.
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L3',
                   'params': {'inv_ROC_HR': -2.0, 'inv_ROC_AR': 2.0}})
L3 = layout.get_optics('L3')
check('a lens can be given its curvatures instead',
      float(L3.inv_ROC_HR) == -2.0 and float(L3.inv_ROC_AR) == 2.0,
      '%g / %g' % (float(L3.inv_ROC_HR), float(L3.inv_ROC_AR)))
check('and its focal length follows from them',
      np.isfinite(float(L3.f)) and float(L3.f) > 0, str(L3.f))

refused({'op': 'add', 'type': 'Mirror', 'name': 'Mf',
         'params': {'f': 0.15}}, 'a focal length on a new mirror')
refused({'op': 'add', 'type': 'Lens', 'name': 'Bad',
         'params': {'f': -0.05, 'thickness': 0.1*mm}},
        'a lens the blank cannot be ground to')
check('and it was not registered',
      'Bad' not in [o.name for o in layout.optics])
refused({'op': 'add', 'type': 'Prism', 'name': 'P1', 'params': {}},
        'an unknown optics type')

# The anchor is a choice, like curve_direction, and every optics has one.
layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'ROC_anchor': 'center'}})
check('the anchor can be changed', M1.ROC_anchor == 'center',
      str(M1.ROC_anchor))
refused({'op': 'set', 'target': 'M1', 'attrs': {'ROC_anchor': 'rim'}},
        'an anchor outside the allowed set')
check('and the old value stands', M1.ROC_anchor == 'center')
layout.apply_edit({'op': 'set', 'target': 'M1',
                   'attrs': {'ROC_anchor': 'HRcenter'}})

check('the layout still traces with lenses in it',
      len(layout.trace()) > 0, '(%d beams)' % len(layout.beams))

# A lens survives the file as a lens, by its curvatures. Re-solving
# from f would reshape one whose radii had been edited since.
lens_path = os.path.join(OUT, 'lens_roundtrip.json')
layout.apply_edit({'op': 'save', 'path': lens_path})
reloaded = OpticalLayout.load(lens_path)
r1 = reloaded.get_optics('L1')
check('a lens reloads as a lens', type(r1).__name__ == 'Lens',
      type(r1).__name__)
check('with the same focal length',
      abs(float(r1.f) - float(lens.f)) < 1e-12,
      '%.9f vs %.9f' % (float(r1.f), float(lens.f)))
check('the same curvatures',
      float(r1.inv_ROC_HR) == float(lens.inv_ROC_HR)
      and float(r1.inv_ROC_AR) == float(lens.inv_ROC_AR))
check('and the same anchor', r1.ROC_anchor == lens.ROC_anchor,
      str(r1.ROC_anchor))
check('the plano face survived too',
      float(reloaded.get_optics('L2').inv_ROC_AR) == 0.0)

print('--- align: putting an optics on a beam ---')
layout, (M1, M2, M3) = make_layout()
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                   'params': {'HRcenter': [0.2, 0.06], 'normAngleHR': 1.1}})
lens = layout.get_optics('L1')
layout.trace()
idx = [b.name for b in layout.beams].index('b0')
src = layout.beams[idx]
bpos = np.asarray(src.pos, dtype=float)
bdir = np.asarray(src.dirVect, dtype=float)

def _offsets(optics, point_attr):
    '''
    How far a point of an optics is across the source beam and how far
    along it. Across is what aligning has to drive to zero; along is
    what it has to leave alone.
    '''
    off = np.asarray(getattr(optics, point_attr), dtype=float) - bpos
    return abs(off[0]*bdir[1] - off[1]*bdir[0]), float(np.dot(off, bdir))

layout.apply_edit({'op': 'align', 'target': 'L1', 'beam': 'b0',
                   'beam_index': idx, 'point': [0.3, 0.05]})
across, along = _offsets(lens, 'center')
check('a lens lands on the beam axis', across < 1e-15, '(%.3e m)' % across)
check('square to the beam',
      abs(float(np.dot(np.asarray(lens.normVectHR), bdir)) + 1) < 1e-12,
      '(n.d = %.15f)' % float(np.dot(np.asarray(lens.normVectHR), bdir)))
# The one thing a drag does say is how far along the beam to put it.
check('at the distance along the beam it was dropped',
      abs(along - 0.3) < 1e-12, '(%.12f)' % along)
check('the trace was invalidated', layout.beams is None)
check('the layout still traces afterwards',
      len(layout.trace()) > 0, '(%d beams)' % len(layout.beams))

# A mirror is held by the apex of its HR face, and that is the point
# the beam has to arrive at: it is where the beam stops.
idx = [b.name for b in layout.beams].index('b0')
layout.apply_edit({'op': 'align', 'target': 'M1', 'beam': 'b0',
                   'beam_index': idx, 'point': [0.42, -0.07]})
across, along = _offsets(M1, 'HRcenter')
check('a mirror lands by the apex of its HR face', across < 1e-15,
      '(%.3e m)' % across)
check('and it too is square to the beam',
      abs(float(np.dot(np.asarray(M1.normVectHR), bdir)) + 1) < 1e-12)

# Past the end of the drawn beam there is no beam: aligning to a
# continuation of it would be aligning to something the layout does not
# say exists.
layout.trace()
idx = [b.name for b in layout.beams].index('b0')
length = float(layout.beams[idx].length)
layout.apply_edit({'op': 'align', 'target': 'L1', 'beam': 'b0',
                   'beam_index': idx, 'point': [length + 5.0, 0.0]})
across, along = _offsets(lens, 'center')
check('a point past the end of the beam is clamped to it',
      abs(along - length) < 1e-12, '(%.6f, beam is %.6f long)' % (along, length))

# The index is how the viewer names a beam, since two can share a name;
# the name is the check that the scene it was picked from is still the
# trace we have.
layout.trace()
layout.apply_edit({'op': 'align', 'target': 'L1', 'beam': 'b0',
                   'beam_index': 999, 'point': [0.25, 0.0]})
across, along = _offsets(lens, 'center')
check('an index that is out of date falls back to the name',
      across < 1e-15 and abs(along - 0.25) < 1e-12,
      '(%.3e m across, %.6f along)' % (across, along))

layout.trace()
refused({'op': 'align', 'target': 'L1', 'beam': 'ghost-beam',
         'beam_index': 999, 'point': [0.3, 0.0]}, 'a beam not in the trace')
refused({'op': 'align', 'target': 'L1', 'beam': 'b0', 'beam_index': 0},
        'an align with no point')
refused({'op': 'align', 'target': 'L1', 'beam': 'b0', 'beam_index': 0,
         'point': 'over there'}, 'a point that is not a pair of numbers')
refused({'op': 'align', 'target': 'nope', 'beam': 'b0', 'point': [0, 0]},
        'an unknown target')

print('--- slide: moving along a beam by a given distance ---')
layout, (M1, M2, M3) = make_layout()
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                   'params': {'HRcenter': [0.2, 0.06], 'normAngleHR': 1.1}})
lens = layout.get_optics('L1')
layout.trace()
idx = [b.name for b in layout.beams].index('b0')
bdir = np.asarray(layout.beams[idx].dirVect, dtype=float)
layout.apply_edit({'op': 'align', 'target': 'L1', 'beam': 'b0',
                   'beam_index': idx, 'point': [0.25, 0.0]})

c0 = np.asarray(lens.center, dtype=float).copy()
hr0 = np.asarray(lens.HRcenter, dtype=float).copy()
a0 = float(lens.normAngleHR)
layout.trace()
idx = [b.name for b in layout.beams].index('b0')
layout.apply_edit({'op': 'slide', 'target': 'L1', 'beam': 'b0',
                   'beam_index': idx, 'distance': 0.05})
moved = np.asarray(lens.center, dtype=float) - c0
check('the optics moves by exactly the distance given',
      abs(float(np.dot(moved, bdir)) - 0.05) < 1e-15,
      '(%.15f along the beam)' % float(np.dot(moved, bdir)))
check('and not at all across it',
      abs(moved[0]*bdir[1] - moved[1]*bdir[0]) < 1e-15,
      '(%.3e m)' % abs(moved[0]*bdir[1] - moved[1]*bdir[0]))
check('its orientation is untouched', float(lens.normAngleHR) == a0,
      '%.15f' % float(lens.normAngleHR))
# A translation moves every point of the substrate alike, so it makes
# no difference which one is nominally being moved.
check('the whole substrate went with it',
      np.allclose(np.asarray(lens.HRcenter, dtype=float) - hr0, moved,
                  atol=0),
      str(list(np.asarray(lens.HRcenter, dtype=float) - hr0 - moved)))
check('the trace was invalidated', layout.beams is None)

# Positive is downstream, so the opposite sign has to undo it exactly.
layout.trace()
idx = [b.name for b in layout.beams].index('b0')
layout.apply_edit({'op': 'slide', 'target': 'L1', 'beam': 'b0',
                   'beam_index': idx, 'distance': -0.05})
check('the opposite sign puts it back exactly',
      np.array_equal(np.asarray(lens.center, dtype=float), c0),
      str(list(np.asarray(lens.center, dtype=float) - c0)))

# Sliding a mirror that sits at an angle uses the beam's direction, not
# the mirror's: 'along the beam' is the beam's business.
layout.trace()
idx = [b.name for b in layout.beams].index('b0')
m1c = np.asarray(M1.center, dtype=float).copy()
layout.apply_edit({'op': 'slide', 'target': 'M1', 'beam': 'b0',
                   'beam_index': idx, 'distance': 0.02})
step = np.asarray(M1.center, dtype=float) - m1c
check('an optics at an angle still moves along the beam',
      np.allclose(step, bdir*0.02, atol=1e-15),
      str(list(step.round(15))))

layout.trace()
refused({'op': 'slide', 'target': 'L1', 'beam': 'b0', 'beam_index': 0},
        'a slide with no distance')
refused({'op': 'slide', 'target': 'L1', 'beam': 'b0', 'beam_index': 0,
         'distance': 'a long way'}, 'a distance that is not a number')
refused({'op': 'slide', 'target': 'L1', 'beam': 'b0', 'beam_index': 0,
         'distance': float('inf')}, 'an infinite distance')
refused({'op': 'slide', 'target': 'L1', 'beam': 'ghost-beam',
         'beam_index': 999, 'distance': 0.01}, 'a beam not in the trace')
refused({'op': 'slide', 'target': 'nope', 'beam': 'b0', 'distance': 0.01},
        'an unknown target')

print('--- a set applies its attributes in a fixed order ---')
# An edit message is a JSON object, and its key order is not something
# to rest on: the position handlers work from the orientation, so a
# position applied first lands off the old one.
lay_a, (Ma, _, _) = make_layout()
lay_a.apply_edit({'op': 'set', 'target': 'M1',
                  'attrs': {'center': [0.7, 0.1], 'normAngleHR': 0.3}})
lay_b, (Mb, _, _) = make_layout()
lay_b.apply_edit({'op': 'set', 'target': 'M1',
                  'attrs': {'normAngleHR': 0.3, 'center': [0.7, 0.1]}})
check('the same attributes in either order give the same optics',
      np.array_equal(np.asarray(Ma.center, dtype=float),
                     np.asarray(Mb.center, dtype=float))
      and float(Ma.normAngleHR) == float(Mb.normAngleHR),
      '%s vs %s' % (list(np.asarray(Ma.center).round(12)),
                    list(np.asarray(Mb.center).round(12))))
check('and the centre is where it was asked for',
      np.allclose(np.asarray(Ma.center, dtype=float), [0.7, 0.1], atol=1e-12),
      str(list(np.asarray(Ma.center).round(12))))
# This is what a lens gets when it is turned in the viewer: it is held
# by the middle of its substrate, so the middle is sent along with the
# angle and has to be applied second.
lay_c, _ = make_layout()
lay_c.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                  'params': {'HRcenter': [0.2, 0.06]}})
Lc = lay_c.get_optics('L1')
c_before = np.asarray(Lc.center, dtype=float).copy()
lay_c.apply_edit({'op': 'set', 'target': 'L1',
                  'attrs': {'center': list(c_before),
                            'normAngleHR': float(Lc.normAngleHR) + 0.4}})
check('turning a lens about its middle leaves the middle alone',
      np.allclose(np.asarray(Lc.center, dtype=float), c_before, atol=1e-15),
      str(list(np.asarray(Lc.center) - c_before)))
check('while the apex of its front face moves',
      not np.allclose(np.asarray(Lc.HRcenter, dtype=float),
                      [0.2, 0.06], atol=1e-9))

print('--- apply_edit: rename ---')
layout, (M1, M2, M3) = make_layout()
M1.max_stray_order = 0
layout.rules.power_threshold = 1e-8
n_before_rename = len(layout.trace())

layout.apply_edit({'op': 'rename', 'target': 'M1', 'name': 'PRM'})
check('the object was renamed', M1.name == 'PRM', M1.name)
check('the layout resolves the new name',
      layout.get_optics('PRM') is M1)
check('the old name is gone',
      'M1' not in [o.name for o in layout.optics],
      str([o.name for o in layout.optics]))
check('the trace was invalidated', layout.beams is None)

# The whole point of moving the setting onto the optics: a rename
# cannot detach it, because nothing is keyed by the name any more.
check('the per-optics cap survived the rename',
      M1.max_stray_order == 0, str(M1.max_stray_order))
check('and still has the same effect',
      len(layout.trace()) == n_before_rename,
      '(%d -> %d beams)' % (n_before_rename, len(layout.beams)))

layout.apply_edit({'op': 'rename', 'target': 'PRM', 'name': 'PRM'})
check('renaming to the same name is a no-op', M1.name == 'PRM')
layout.apply_edit({'op': 'rename', 'target': 'PRM', 'name': 'M1'})
check('and it can be renamed back', M1.name == 'M1')

print('--- apply_edit: add ---')
layout, (M1, M2, M3) = make_layout()
layout.trace()
n0 = len(layout.optics)

layout.apply_edit({'op': 'add', 'type': 'Mirror', 'name': 'M4',
                   'params': {'HRcenter': [0.7, 0.2],
                              'normAngleHR': np.pi}})
check('the optics was registered', len(layout.optics) == n0 + 1,
      '(%d -> %d)' % (n0, len(layout.optics)))
M4 = layout.get_optics('M4')
check('it is where it was asked for',
      np.allclose(np.asarray(M4.HRcenter), [0.7, 0.2]),
      str(list(np.asarray(M4.HRcenter))))
check('the trace was invalidated', layout.beams is None)
check('it appears in the scene',
      'M4' in [o['name'] for o in layout.scene_dict()['optics']])

# Defaults come from the optics already in the layout, not from the
# class, so a mirror added to a 10 cm system is a 10 cm mirror.
check('the size is inherited from the layout',
      abs(float(M4.diameter) - float(M3.diameter)) < 1e-15
      and abs(float(M4.thickness) - float(M3.thickness)) < 1e-15,
      'd=%.4f (class default is 0.25)' % float(M4.diameter))
check('the index is inherited', abs(float(M4.n) - float(M3.n)) < 1e-15)
check('but the surfaces are flat',
      float(M4.inv_ROC_HR) == 0.0 and float(M4.inv_ROC_AR) == 0.0,
      '1/R = %g' % float(M4.inv_ROC_HR))

layout.apply_edit({'op': 'add', 'params': {'HRcenter': [0.1, 0.3]}})
check('a name is generated when none is given',
      layout.optics[-1].name == 'M5', layout.optics[-1].name)
check('unique_optics_name skips the taken ones',
      layout.unique_optics_name() == 'M6', layout.unique_optics_name())

layout.apply_edit({'op': 'add', 'name': 'Curved', 'type': 'CyMirror',
                   'params': {'HRcenter': [0.2, 0.5],
                              'curve_direction': 'v', 'inv_ROC_HR': 0.5}})
cy = layout.get_optics('Curved')
check('a CyMirror can be created', type(cy).__name__ == 'CyMirror',
      type(cy).__name__)
check('its curve direction is set', cy.curve_direction == 'v',
      str(cy.curve_direction))

check('the layout still traces with the new optics',
      len(layout.trace()) > 0, '(%d beams)' % len(layout.beams))

print('--- apply_edit: remove ---')
n1 = len(layout.optics)
layout.trace()
layout.apply_edit({'op': 'remove', 'target': 'Curved'})
check('the optics is gone', len(layout.optics) == n1 - 1
      and 'Curved' not in [o.name for o in layout.optics],
      '(%d -> %d)' % (n1, len(layout.optics)))
check('the trace was invalidated', layout.beams is None)
check('the name becomes free again',
      layout.apply_edit({'op': 'add', 'name': 'Curved',
                         'params': {}}) is layout)
layout.apply_edit({'op': 'remove', 'target': 'Curved'})

print('--- undo ---')
# This section works on layouts of its own, and refused() reads whatever
# `layout` names, so borrow the name and give it back: the sections
# after this one carry on from the one built above.
_borrowed = (layout, M1, M2, M3)
layout, (M1, M2, M3) = make_layout()
check('a fresh layout has nothing to undo', not layout.can_undo)
check('and the scene says so', layout.scene_dict()['can_undo'] is False)
refused({'op': 'undo'}, 'an undo with nothing behind it')

home = np.asarray(M1.HRcenter, dtype=float).copy()
layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.8, 0.3]})
check('an edit gives it something to undo', layout.can_undo)
check('which the scene reports', layout.scene_dict()['can_undo'] is True)
layout.apply_edit({'op': 'undo'})
check('undoing a move puts it back exactly',
      np.array_equal(np.asarray(M1.HRcenter, dtype=float), home),
      str(list(np.asarray(M1.HRcenter, dtype=float) - home)))
# The whole design rests on the layout holding the user's own objects,
# so an undo must not swap them for new ones.
check('and it is still the same object', layout.get_optics('M1') is M1)
check('the trace was invalidated', layout.beams is None)
check('with nothing left behind it', not layout.can_undo)

# Several steps, undone in order.
layout.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'diameter': 0.2}})
layout.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'diameter': 0.3}})
layout.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'diameter': 0.4}})
layout.apply_edit({'op': 'undo'})
check('undo walks back one edit at a time',
      abs(float(M1.diameter) - 0.3) < 1e-15, str(float(M1.diameter)))
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'undo'})
check('all the way to where it started',
      abs(float(M1.diameter) - 0.1) < 1e-15, str(float(M1.diameter)))
refused({'op': 'undo'}, 'an undo past the beginning')

# Adding and removing, which change what is registered rather than a
# number on it.
n0 = len(layout.optics)
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                   'params': {'HRcenter': [0.2, 0.2]}})
layout.apply_edit({'op': 'undo'})
check('undoing an add takes the optics away again',
      len(layout.optics) == n0
      and 'L1' not in [o.name for o in layout.optics],
      str([o.name for o in layout.optics]))

layout.apply_edit({'op': 'remove', 'target': 'M2'})
layout.apply_edit({'op': 'undo'})
check('undoing a remove brings it back',
      'M2' in [o.name for o in layout.optics]
      and len(layout.optics) == n0, str([o.name for o in layout.optics]))
# The history holds the elements, not only their values, so what comes
# back is the object that was taken out - the M2 of the user's own code
# still names the registered optics.
check('as the very object that was removed',
      layout.get_optics('M2') is M2)

layout.apply_edit({'op': 'rename', 'target': 'M1', 'name': 'PRM'})
layout.apply_edit({'op': 'undo'})
check('undoing a rename gives the name back to the same object',
      M1.name == 'M1' and layout.get_optics('M1') is M1, str(M1.name))

layout.apply_edit({'op': 'rules', 'rules': {'order': 9}})
layout.apply_edit({'op': 'undo'})
check('the tracing rules are undone too', layout.rules.order == 5,
      str(layout.rules.order))
layout.apply_edit({'op': 'draw', 'params': {'width_mode': 'y'}})
layout.apply_edit({'op': 'undo'})
check('and so are the drawing options',
      layout.draw_options.get('width_mode') != 'y',
      str(layout.draw_options))

# A refused edit changes nothing, so it must not cost an undo step:
# pressing Undo after one would take back the edit before it instead.
layout, (M1, M2, M3) = make_layout()
layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.8, 0.3]})
depth = len(layout._history)
refused({'op': 'set', 'target': 'M1', 'attrs': {'nonsense': 1}},
        'an unknown attribute')
refused({'op': 'explode', 'target': 'M1'}, 'an unknown operation')
check('a refused edit leaves the history alone',
      len(layout._history) == depth, '(%d vs %d)' % (len(layout._history),
                                                     depth))
check('so undo still reaches the edit before it',
      layout.apply_edit({'op': 'undo'}) is layout
      and not layout.can_undo)

# Saving writes a file and changes nothing, so it is not a step either.
save_path = os.path.join(OUT, 'undo_save.json')
layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.7, 0.1]})
layout.apply_edit({'op': 'save', 'path': save_path})
check('nor does saving', len(layout._history) == 1,
      str(len(layout._history)))

# The bound is what keeps a long session from growing without limit.
layout, (M1, M2, M3) = make_layout()
for i in range(UNDO_DEPTH + 20):
    layout.apply_edit({'op': 'set', 'target': 'M1',
                       'attrs': {'diameter': 0.1 + i*0.001}})
check('the history is bounded', len(layout._history) == UNDO_DEPTH,
      '(%d, bound is %d)' % (len(layout._history), UNDO_DEPTH))
for i in range(UNDO_DEPTH):
    layout.apply_edit({'op': 'undo'})
check('and undoes exactly that many times', not layout.can_undo)

print('--- redo ---')
layout, (M1, M2, M3) = make_layout()
check('a fresh layout has nothing to redo', not layout.can_redo)
check('and the scene says so', layout.scene_dict()['can_redo'] is False)
refused({'op': 'redo'}, 'a redo with nothing ahead of it')

# Editing fills the undo side only. Redo is fed by undoing, never by
# doing: there is nothing ahead of an edit just made.
layout.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'diameter': 0.2}})
check('an edit gives it nothing to redo', not layout.can_redo)
refused({'op': 'redo'}, 'a redo straight after an edit')

layout.apply_edit({'op': 'undo'})
check('undoing does', layout.can_redo)
check('which the scene reports', layout.scene_dict()['can_redo'] is True)
check('and it stepped back', abs(float(M1.diameter) - 0.1) < 1e-15,
      str(float(M1.diameter)))
layout.trace()
layout.apply_edit({'op': 'redo'})
check('redoing puts the edit back exactly',
      abs(float(M1.diameter) - 0.2) < 1e-15, str(float(M1.diameter)))
check('on the same object', layout.get_optics('M1') is M1)
check('the trace was invalidated', layout.beams is None)
check('with nothing left ahead of it', not layout.can_redo)
check('and the edit back within reach of undo', layout.can_undo)

# Several steps, walked back and then forward again one at a time.
layout, (M1, M2, M3) = make_layout()
for d in (0.2, 0.3, 0.4):
    layout.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'diameter': d}})
for i in range(3):
    layout.apply_edit({'op': 'undo'})
check('three undos reach the start', abs(float(M1.diameter) - 0.1) < 1e-15,
      str(float(M1.diameter)))
layout.apply_edit({'op': 'redo'})
check('redo walks forward one edit at a time',
      abs(float(M1.diameter) - 0.2) < 1e-15, str(float(M1.diameter)))
layout.apply_edit({'op': 'redo'})
layout.apply_edit({'op': 'redo'})
check('all the way to where it left off',
      abs(float(M1.diameter) - 0.4) < 1e-15, str(float(M1.diameter)))
refused({'op': 'redo'}, 'a redo past the last edit')

# The turn that was not taken is not somewhere to return to: the states
# put aside describe elements the new edit may have moved on from.
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'undo'})
check('there is a branch to come back to', layout.can_redo)
layout.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'diameter': 0.7}})
check('a new edit discards it', not layout.can_redo)
check('and the scene reports that too',
      layout.scene_dict()['can_redo'] is False)
refused({'op': 'redo'}, 'a redo after the layout took another turn')

# A refused edit is not a turn taken, and saving is not a change of
# mind: neither may cost the branch.
layout.apply_edit({'op': 'undo'})
refused({'op': 'set', 'target': 'M1', 'attrs': {'nonsense': 1}},
        'an unknown attribute')
check('a refused edit leaves the redo side alone', layout.can_redo)
layout.apply_edit({'op': 'save',
                   'path': os.path.join(OUT, 'redo_save.json')})
check('nor does saving', layout.can_redo)
check('so redo still reaches it',
      layout.apply_edit({'op': 'redo'}) is layout
      and abs(float(M1.diameter) - 0.7) < 1e-15, str(float(M1.diameter)))

# What is registered, rather than a number on it. Redo restores as
# exactly as undo does, holding the elements themselves.
layout, (M1, M2, M3) = make_layout()
n0 = len(layout.optics)
layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                   'params': {'HRcenter': [0.2, 0.2]}})
added = layout.get_optics('L1')
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'redo'})
check('redoing an add puts the optics back',
      len(layout.optics) == n0 + 1 and 'L1' in [o.name for o in layout.optics],
      str([o.name for o in layout.optics]))
check('as the very object that was added',
      layout.get_optics('L1') is added)

layout.apply_edit({'op': 'remove', 'target': 'M2'})
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'redo'})
check('redoing a remove takes it away again',
      'M2' not in [o.name for o in layout.optics],
      str([o.name for o in layout.optics]))

layout.apply_edit({'op': 'rename', 'target': 'M1', 'name': 'PRM'})
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'redo'})
check('redoing a rename gives the name back to the same object',
      M1.name == 'PRM' and layout.get_optics('PRM') is M1, str(M1.name))

layout.apply_edit({'op': 'rules', 'rules': {'order': 9}})
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'redo'})
check('the tracing rules are redone too', layout.rules.order == 9,
      str(layout.rules.order))
layout.apply_edit({'op': 'draw', 'params': {'width_mode': 'y'}})
layout.apply_edit({'op': 'undo'})
layout.apply_edit({'op': 'redo'})
check('and so are the drawing options',
      layout.draw_options.get('width_mode') == 'y', str(layout.draw_options))

# Stepping back and forth over the same edit repeatedly must settle, not
# accumulate: each move consumes one side and feeds the other.
layout, (M1, M2, M3) = make_layout()
layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.8, 0.3]})
for i in range(5):
    layout.apply_edit({'op': 'undo'})
    layout.apply_edit({'op': 'redo'})
check('stepping back and forth does not pile up',
      len(layout._history) == 1 and len(layout._future) == 0,
      '(%d, %d)' % (len(layout._history), len(layout._future)))
check('and leaves the edit applied',
      np.allclose(np.asarray(M1.HRcenter, dtype=float), [0.8, 0.3]),
      str(list(np.asarray(M1.HRcenter, dtype=float))))

# The redo side is filled by undoing, so the same bound holds it.
layout, (M1, M2, M3) = make_layout()
for i in range(UNDO_DEPTH + 20):
    layout.apply_edit({'op': 'set', 'target': 'M1',
                       'attrs': {'diameter': 0.1 + i*0.001}})
for i in range(UNDO_DEPTH):
    layout.apply_edit({'op': 'undo'})
check('the redo side is bounded as well',
      len(layout._future) == UNDO_DEPTH,
      '(%d, bound is %d)' % (len(layout._future), UNDO_DEPTH))
for i in range(UNDO_DEPTH):
    layout.apply_edit({'op': 'redo'})
check('and redoes exactly that many times', not layout.can_redo)
check('ending where the edits had left it',
      abs(float(M1.diameter) - (0.1 + (UNDO_DEPTH + 19)*0.001)) < 1e-15,
      str(float(M1.diameter)))

layout, M1, M2, M3 = _borrowed

print('--- apply_edit: what is refused ---')

refused({'op': 'explode', 'target': 'M1'}, 'an unknown operation')
refused({'op': 'move', 'target': 'nope', 'center': [0, 0]}, 'an unknown target')
refused({'op': 'move', 'target': 'M1'}, 'a move with no coordinates')
refused({'op': 'set', 'target': 'M1', 'attrs': {'__class__': int}},
        'a dunder attribute')
refused({'op': 'set', 'target': 'M1', 'attrs': {'name': 'hacked'}},
        'renaming an optics')
refused({'op': 'rules', 'rules': {'nonsense': 1}}, 'an unknown rule')
refused('not a dict', 'a non-dict message')
refused({'op': 'add', 'type': 'Laser'}, 'an unknown optics type')
refused({'op': 'add', 'params': {'__class__': int}},
        'a dunder construction parameter')
refused({'op': 'add', 'params': {'name': 'sneaky'}},
        'setting the name through params')
refused({'op': 'add', 'name': 'M1'}, 'a duplicate name')
refused({'op': 'add', 'name': ''}, 'an empty name')
refused({'op': 'remove', 'target': 'nope'}, 'removing what is not there')
refused({'op': 'rename', 'target': 'M1', 'name': 'M2'},
        'renaming onto a name in use')
refused({'op': 'rename', 'target': 'M1', 'name': '  '},
        'renaming to blank')
refused({'op': 'rename', 'target': 'M1', 'name': None},
        'renaming to nothing')
refused({'op': 'rename', 'target': 'nope', 'name': 'x'},
        'renaming what is not there')
check('a refused rename left the name alone',
      layout.get_optics('M1').name == 'M1')
check('the name survived the refusal', M1.name == 'M1', M1.name)
check('a refused add left the layout alone',
      len(layout.optics) == n1 - 1,
      '(%d optics)' % len(layout.optics))

# The whitelist must not be a way to reach arbitrary attributes.
check('the whitelist has no dunder',
      not any(a.startswith('_') for a in EDITABLE_OPTIC_ATTRS))
check("'name' is not editable", 'name' not in EDITABLE_OPTIC_ATTRS)

print('--- save and load through the protocol ---')
layout, (M1, M2, M3) = make_layout()
layout.draw_options = {'width_mode': 'y', 'sigma_main': 3.0}
M2.max_stray_order = 1
n_original = len(layout.trace())
path = os.path.join(OUT, 'stage2b_roundtrip.json')

layout.apply_edit({'op': 'save', 'path': path})
check('the file was written', os.path.exists(path),
      '(%d bytes)' % os.path.getsize(path))
check('saving does not invalidate the trace', layout.beams is not None)

with open(path) as f:
    saved = json.load(f)
check('drawing options are persisted',
      saved.get('draw_options') == {'width_mode': 'y', 'sigma_main': 3.0},
      str(saved.get('draw_options')))
check('per-optics tracing settings are persisted',
      [o for o in saved['optics'] if o['name'] == 'M2'][0]['max_stray_order'] == 1)

# Wreck the layout, then load it back.
M1.HRcenter = [0.8, 0.3]
layout.apply_edit({'op': 'add', 'name': 'JUNK', 'params': {}})
layout.draw_options = {}
layout.beams = None
n_wrecked = len(layout.trace())

layout.apply_edit({'op': 'load', 'path': path})
check('the trace was invalidated', layout.beams is None)
check('the layout came back',
      len(layout.trace()) == n_original,
      '(%d -> %d -> %d beams)' % (n_original, n_wrecked, len(layout.beams)))
check('the added optics is gone',
      'JUNK' not in [o.name for o in layout.optics],
      str([o.name for o in layout.optics]))
check('the drawing options came back',
      layout.draw_options == {'width_mode': 'y', 'sigma_main': 3.0},
      str(layout.draw_options))
check('the per-optics setting came back',
      layout.get_optics('M2').max_stray_order == 1)

# The point of loading in place: the objects the user holds stay the
# ones the layout holds.
check('the registered object is the same one as before',
      layout.get_optics('M1') is M1)
check('and its position was restored',
      np.allclose(np.asarray(M1.HRcenter), [0.5, 0.0], atol=1e-12),
      str(list(np.asarray(M1.HRcenter))))
check('the source object survived too',
      layout.sources[0] is layout.get_source('b0'))

# A layout whose elements differ has to build new ones.
other = os.path.join(OUT, 'stage2b_other.json')
lay2, _ = make_layout()
lay2.apply_edit({'op': 'rename', 'target': 'M1', 'name': 'ZZ'})
lay2.save(other)
layout.apply_edit({'op': 'load', 'path': other})
check('an element the file does not name is dropped',
      'M1' not in [o.name for o in layout.optics],
      str([o.name for o in layout.optics]))
check('and the one it does name is built',
      layout.get_optics('ZZ') is not M1)
layout.apply_edit({'op': 'load', 'path': path})

print('--- save and load: what is refused ---')
refused({'op': 'save'}, 'a save with no path')
refused({'op': 'load', 'path': ''}, 'a load with a blank path')
refused({'op': 'load', 'path': os.path.join(OUT, 'no_such_file.json')},
        'loading a file that is not there')
bad = os.path.join(OUT, 'stage2b_bad.json')
with open(bad, 'w') as f:
    f.write('this is not json')
refused({'op': 'load', 'path': bad}, 'loading something that is not a layout')
check('a refused load left the layout alone',
      len(layout.trace()) == n_original,
      '(%d beams)' % len(layout.beams))

print('--- widget round trip ---')
layout, (M1, M2, M3) = make_layout()
w = layout.widget()
check('widget is editable by default', w.editable)
check('no error to start with', w.error == '')

n_before = len(w.scene['beams'])
c_before = [float(x) for x in np.asarray(M1.center)]
ok = w.apply_edit({'op': 'move', 'target': 'M1',
                   'center': [c_before[0], c_before[1] + 0.01]})
check('the edit was accepted', ok)
check('the scene was pushed',
      abs(w.scene['optics'][0]['center'][1] - (c_before[1] + 0.01)) < 1e-9,
      str(w.scene['optics'][0]['center']))
check('the beams were re-traced', len(w.scene['beams']) > 0,
      '(%d -> %d)' % (n_before, len(w.scene['beams'])))
check('the edit was logged', w.edits == [{'op': 'move', 'target': 'M1',
                                          'center': [c_before[0],
                                                     c_before[1] + 0.01]}])
check('the user object moved too',
      abs(float(np.asarray(M1.center)[1]) - (c_before[1] + 0.01)) < 1e-9)

print('--- through the real ipywidgets comm dispatch ---')
# apply_edit() is the shortcut the rest of these checks take. The front
# end does not call it: it calls model.send(), which ipywidgets turns
# into Widget._handle_msg -> _handle_custom_msg -> the on_msg callbacks.
# Drive that whole path, or a signature mismatch in the handler goes
# unnoticed until someone drags a mirror.
def comm_msg(content):
    return {'content': {'data': {'method': 'custom', 'content': content}},
            'buffers': []}

# ipywidgets' own dispatcher must not have been overridden.
import inspect
sig = inspect.signature(type(w)._handle_custom_msg)
check('Widget._handle_custom_msg is intact',
      list(sig.parameters) == ['self', 'content', 'buffers'],
      str(list(sig.parameters)))

c_now = [float(x) for x in np.asarray(M1.center)]
n_edits = len(w.edits)
try:
    w._handle_msg(comm_msg({'op': 'move', 'target': 'M1',
                            'center': [c_now[0], c_now[1] + 0.005]}))
    dispatched = True
    err = ''
except Exception as e:
    dispatched = False
    err = '%s: %s' % (type(e).__name__, e)
check('a custom message is dispatched without raising', dispatched, err)
check('and reaches the layout',
      abs(float(np.asarray(M1.center)[1]) - (c_now[1] + 0.005)) < 1e-9,
      str(list(np.asarray(M1.center))))
check('and is logged', len(w.edits) == n_edits + 1,
      '(%d -> %d)' % (n_edits, len(w.edits)))

# Messages that are not edits must be ignored, not crash the comm.
for junk in [{'method': 'update'}, {'no_op_here': 1}, 'a string', None]:
    try:
        w._handle_msg(comm_msg(junk))
        ok = True
        detail = ''
    except Exception as e:
        ok = False
        detail = '%s: %s' % (type(e).__name__, e)
    check('a non-edit message is ignored: %r' % (junk,), ok, detail)

w.apply_edit({'op': 'move', 'target': 'M1', 'center': c_now})

print('--- widget refuses bad edits without raising ---')
scene_before = w.scene
n_log = len(w.edits)
ok = w.apply_edit({'op': 'set', 'target': 'M1', 'attrs': {'name': 'x'}})
check('rejected', not ok)
check('reported through the error traitlet', w.error.startswith('EditError'),
      w.error)
check('the scene was left alone', w.scene is scene_before)
check('nothing was logged', len(w.edits) == n_log,
      '(%d -> %d)' % (n_log, len(w.edits)))

ok = w.apply_edit({'op': 'move', 'target': 'M1', 'center': [0.5, 0.0]})
check('a later good edit clears the error', ok and w.error == '', w.error)

print('--- read-only widget ---')
ro = layout.widget(editable=False)
check('editable is off', not ro.editable)
check('an edit is refused',
      ro.apply_edit({'op': 'move', 'target': 'M1', 'center': [1, 1]}) is False)
check('and reported', 'read-only' in ro.error, ro.error)

detached = wmod.LayoutViewer(scene=layout.scene_dict())
check('a layout-less widget is not editable', not detached.editable)

print('--- data for the browser check ---')
lay2, (m1, m2, m3) = make_layout()
with open(os.path.join(OUT, 'stage2b_scene.json'), 'w') as f:
    json.dump(lay2.scene_dict(), f)
print('  wrote stage2b_scene.json (%d optics)' % len(lay2.scene_dict()['optics']))

print()
print('%d passed, %d failed' % (npass, nfail))
sys.exit(1 if nfail else 0)
