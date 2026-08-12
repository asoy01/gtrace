Optical layouts
===============================

An :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` holds a whole optical system in one object: the optics, the source beams and the rules that govern the non-sequential trace. It is what you draw, what you save to disk, and what the interactive viewer edits.

Nothing forces you to use it. The optics and beams described in :doc:`basic_concepts` and :doc:`propagation` work on their own, and the :doc:`tutorial` gets as far as a traced mode cleaner without one. What the layout adds is a single place to keep a system that you intend to trace, redraw and adjust repeatedly.

The two phases
---------------

A layout is not a substitute for the code that builds an optical system. Placing mirrors so that a cavity closes on itself, or so that a beam arrives at a mirror at the intended angle of incidence, is done with ordinary Python: sequential tracing, ``scipy.optimize``, cavity eigenmode solving. That code is where the physics of your design lives.

The intended workflow therefore has two phases:

**Phase 1 — build and align, with ordinary Python code.**

.. code-block:: python

    import gtrace.optcomp as opt
    import gtrace.beam as beam
    from gtrace.unit import *

    PRM = opt.Mirror(HRcenter=[0, 0], normAngleHR=0.0,
                     diameter=25*cm, thickness=10*cm,
                     inv_ROC_HR=1./458.1, name='PRM')
    PR2 = opt.Mirror(HRcenter=[14.7, 0], normAngleHR=deg2rad(180-2.5),
                     diameter=25*cm, thickness=10*cm, name='PR2')

    src = beam.GaussianBeam(q0=1j*10.0, pos=[-1.0, 0.0],
                            dirAngle=0.0, wl=1064*nm, name='src')

**Phase 2 — register the result and work with it as a whole.**

.. code-block:: python

    from gtrace.layout import OpticalLayout, TraceRules

    layout = OpticalLayout(optics=[PRM, PR2], sources=[src],
                           rules=TraceRules(order=10, power_threshold=1e-3),
                           name='PRC')

    layout.trace()          # non-sequential trace from every source
    layout.show()           # open the viewer

Held by reference
------------------

The layout does not copy what you register. ``layout.get_optics('PRM')`` returns the very object bound to ``PRM`` in your code. Consequently:

.. code-block:: python

    PRM.HRcenter = [0.01, 0.0]
    layout.trace()          # the new result reflects the move

and, in the other direction, an edit made in the viewer changes the object that ``PRM`` names. This is what makes the viewer more than a picture: there is exactly one copy of the model, and both you and the GUI are looking at it.

The same reasoning drives :py:meth:`update_from_file<gtrace.layout.OpticalLayout.update_from_file>`, which loads a saved layout *in place*: optics whose name and type match are updated rather than replaced, so ``PRM`` does not silently become a stale orphan the moment you load a file.

Note that :py:meth:`trace<gtrace.layout.OpticalLayout.trace>` is the one exception: it traces a *copy* of each source beam, so the registered source is not consumed by tracing and stays where you put it.

Tracing rules
--------------

:py:class:`TraceRules<gtrace.layout.TraceRules>` collects what governs the non-sequential trace:

``order``
    How many internal reflections are followed when a beam hits an element. Defaults to 10.

``power_threshold``
    Beams weaker than this are not propagated further. Defaults to 0.1. This is the knob to turn when chasing ghost beams: lowering it makes the trace find fainter paths, and slower.

``open_beam_length``
    How long a beam that hits nothing is drawn. Defaults to 1.0.

``order`` is a property of the trace as a whole. How deep the ghosts of *one particular element* are worth chasing is a property of that element, so it lives there instead, as the ``max_stray_order`` attribute of the optics. What raises a beam's stray order in the first place, and the flags that say which face of an element is meant for what, are described in :ref:`stray-order`.

``open_beam_length`` applies to the beams the trace produces. A *source* that reaches nothing keeps its own ``length`` instead, which is the state a layout is in while it is being built.

All three can be changed from a front end; see :ref:`editing-a-source`.

Drawing options
----------------

How a layout is drawn is a display choice, not part of the model. Changing one redraws; it does not re-trace. The available options and their defaults are collected in ``gtrace.layout.DRAW_OPTIONS``:

============================ ========== =======================================
Option                       Default    Meaning
============================ ========== =======================================
``sigma_main``               2.7        Width of the drawn envelope of the main
                                        beams, in units of the 1/e² radius
``sigma_stray``              2.7        The same, for stray beams
``width_mode``               ``'x'``    Which transverse direction the envelope
                                        shows: ``'x'``, ``'y'`` or ``'avg'``
``drawMainWidth``            True       Whether to draw the main beam envelope
``drawStrayWidth``           True       Whether to draw the stray beam envelope
``drawBeamLabels``           False      Annotate each beam with its name and
                                        power
``drawOpticsNames``          True       Annotate each element with its name
``fontSize``                 False      Annotation size, or False for the
                                        gtrace default
============================ ========== =======================================

An option can be given per call, or set on the layout to apply to every subsequent drawing:

.. code-block:: python

    layout.render_html('trace.html', width_mode='y', sigma_main=1.0)

    layout.draw_options['width_mode'] = 'avg'   # applies from now on

A misspelt option raises ``TypeError`` rather than being ignored. A setting that is silently dropped looks exactly like a setting that had no effect, and there is no way to tell the two apart from the output.

.. _why-2.7-sigma:

Why 2.7 σ
^^^^^^^^^^

The envelope is drawn at 2.7 times the 1/e² radius because that is the aperture at which the diffraction loss of a Gaussian beam is 1 ppm. Drawing every envelope at the same σ means that every envelope in the picture carries the same meaning: "outside this line there is nothing that matters at the ppm level".

``width_mode`` exists because a beam is not round in general. After a beam passes through a wedged substrate at a non-normal incidence, its horizontal and vertical waists differ, and there is no single correct answer to "how wide is this beam". The drawing cannot make that choice for you.

Saving and loading
-------------------

A layout serialises to plain JSON:

.. code-block:: python

    layout.save('layout.json')

    layout2 = OpticalLayout.load('layout.json')     # a new layout
    layout.update_from_file('layout.json')          # in place, see above

The file holds the optics, the sources, the tracing rules and the drawing options. The trace result is not saved; it is regenerated by :py:meth:`trace<gtrace.layout.OpticalLayout.trace>`, and saving a derived quantity that can be recomputed only creates an opportunity for it to disagree with its inputs.

Editing from a front end
-------------------------

The viewer changes a layout by sending it messages, which you can also send yourself:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'PRM',
                       'HRcenter': [0.02, 0.0]})
    layout.apply_edit({'op': 'set', 'target': 'PRM',
                       'attrs': {'diameter': 0.15}})

The operations are ``move``, ``rotate``, ``set``, ``align``, ``slide``, ``add``, ``copy``, ``remove``, ``rename``, ``rules``, ``draw``, ``save``, ``load``, ``export``, ``undo`` and ``redo``. Every message is a plain dict, so the same protocol travels over a notebook widget's comm as over any other transport.

A ``set`` may carry several attributes at once, and they are not applied in the order the message happens to list them: the anchor goes on before the curvatures it governs, and the orientation before the position that is measured from it. A message is a JSON object, whose key order is not something to rest on.

``align`` puts an element square across a beam, which is where almost every element on a bench is meant to sit and what a drag cannot say precisely. It names the beam by its index in the last trace, with the name as a check, and gives a point; the element is turned to face the beam and slid onto its axis at the projection of that point. See :doc:`viewer` for the Ctrl-drag that sends it.

``slide`` is the degree of freedom aligning leaves: it moves an element along a beam's axis by a distance in metres, positive downstream, and touches nothing else. It names the beam the same way::

    layout.apply_edit({'op': 'slide', 'target': 'L1',
                       'beam': 'b0', 'beam_index': 0, 'distance': 0.05})

The set of attributes a message may touch is an explicit whitelist (``EDITABLE_OPTIC_ATTRS``), and some attributes are further restricted to a set of permitted values (``ATTR_CHOICES``). Messages arrive from a browser, so "anything ``setattr`` accepts" is not a safe rule. An operation, target or attribute outside those sets raises :py:class:`EditError<gtrace.layout.EditError>` and leaves the layout untouched.

An attribute on the whitelist may still be one the target does not have, or one that refuses the value it is given. ``f`` is both: only a :py:class:`Lens<gtrace.optcomp.Lens>` has a focal length, and assigning to it re-solves both curvatures, which not every blank can be ground to. Either refusal comes back as an ``EditError`` with the reason, and the optics is left as it was.

``add`` builds a :py:class:`Mirror<gtrace.optcomp.Mirror>`, a :py:class:`CyMirror<gtrace.optcomp.CyMirror>` or a :py:class:`Lens<gtrace.optcomp.Lens>` (``CREATABLE_OPTIC_TYPES``). A mirror fills the parameters it was not given from the optics already registered, so that an element added to a system of 10 cm optics is a 10 cm optics. A lens does not: its coatings, aperture and wedge are the lens's own, and it is built from catalogue defaults at ``DEFAULT_LENS_F`` instead. It also accepts the parameters only a lens has (``CREATABLE_LENS_PARAMS``: ``f``, ``shape`` and ``ROC_HR``):

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Lens', 'name': 'L1',
                       'params': {'f': 0.3, 'shape': 'plano-convex',
                                  'HRcenter': [0.4, 0.0]}})

Renaming has its own operation rather than being an editable attribute, because the name is the identity that edits are resolved by; changing it needs a uniqueness check.

.. _editing-a-source:

Editing a source
^^^^^^^^^^^^^^^^^

The same operations reach the source beams, and mean for a laser what they mean for an element: ``move`` says where it stands, ``rotate`` which way it fires, ``set`` what light it puts out.

.. code-block:: python

    layout.apply_edit({'op': 'move',   'target': 'b0', 'pos': [0.1, 0.0]})
    layout.apply_edit({'op': 'rotate', 'target': 'b0', 'dirAngle': 0.0})
    layout.apply_edit({'op': 'set',    'target': 'b0',
                       'attrs': {'waist_size_x': 0.35e-3,
                                 'waist_pos_x': 0.12}})
    layout.apply_edit({'op': 'add', 'type': 'Source', 'name': 'S1',
                       'params': {'pos': [0.0, 0.3],
                                  'waist_size': 0.2e-3}})

A source stands at a point and is aimed, so the attributes are different ones (``EDITABLE_SOURCE_ATTRS``) and ``move`` and ``rotate`` name ``pos`` and ``dirAngle`` rather than an element's centre and face. ``align`` and ``slide`` do not apply at all: there is nothing to square a laser onto — it is what the beams are square to.

**A laser is specified by its waist, not by a q-parameter.** ``waist_size_x``, ``waist_size_y``, ``waist_pos_x`` and ``waist_pos_y`` are not attributes a :py:class:`GaussianBeam<gtrace.beam.GaussianBeam>` has; each stands for one half of one q-parameter and is converted here. Setting a size does not move the waist and moving it does not resize it, and the two directions are independent. ``qx`` and ``qy`` may still be set directly, as ``[real, imag]``; :py:func:`q_from_waist<gtrace.layout.q_from_waist>` and :py:meth:`waist<gtrace.beam.GaussianBeam.waist>` convert between the two descriptions.

A waist position is the distance from the laser forward along the beam, positive downstream, which is how :py:meth:`waist<gtrace.beam.GaussianBeam.waist>` reports it.

**Through this protocol, changing the wavelength keeps the waist and changes the divergence.** A q-parameter says nothing on its own — what width it comes to depends on the wavelength — so changing one of the two has to keep the other, and the waist is what the laser is specified by. The model already behaves this way for the refractive index, whose handler holds the reduced q fixed. Assigning ``b.wl`` directly in a cell is untouched and keeps the q-parameter instead.

A new source inherits nothing from the sources already registered, unlike a new mirror. A laser is not cut to match the one beside it, and a q-parameter carried over would describe a waist measured from a point the new source does not stand at; ``DEFAULT_SOURCE_WAIST`` and ``DEFAULT_SOURCE_WL`` are used instead. ``waist_size`` and ``waist_pos`` given to ``add`` stand for both directions at once (``CREATABLE_SOURCE_PARAMS``).

**Optics, sources and dimensions share one namespace.** An edit message names its target and nothing else, so a name that meant one thing in one message and another in the next would be a trap. :py:meth:`add_source<gtrace.layout.OpticalLayout.add_source>` refuses a name an optics or a dimension has taken, as it always did for another source.

The tracing rules have their own operation, and each value is checked: ``order`` is a whole number no greater than ``MAX_RULE_ORDER``, since each order is another round of reflections at every element::

    layout.apply_edit({'op': 'rules', 'rules': {'order': 20,
                                                'power_threshold': 1e-9}})

Three operations deliberately do *not* invalidate the trace result: ``draw`` changes display settings, and ``save`` and ``export`` write a file. None of them changes the physics, so none causes a re-trace. Nor does anything done to a dimension, for the same reason.

``export`` writes the drawing rather than the model — today only ``{'op': 'export', 'format': 'dxf', 'path': ...}``, which is :py:meth:`export_dxf<gtrace.layout.OpticalLayout.export_dxf>`. See :ref:`dxf-export`.

.. _dimensions:

Dimensions
-----------

A :py:class:`Dimension<gtrace.layout.Dimension>` is a distance measured between two points of the layout, kept as a note on it:

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Dimension', 'name': 'D1',
                       'params': {'p1': list(M1.HRcenter),
                                  'p2': list(M1.ARcenter),
                                  'offset': 0.17}})
    layout.get_dimension('D1').length          # in metres

Dimensions are registered on the layout beside the optics, saved with it, and taken back by undo like anything else. They share one namespace with the optics, so ``remove``, ``rename`` and ``set`` resolve their target across both — a front end has a name under the cursor, not a class. ``move`` and ``rotate`` do not apply: a dimension is two points rather than a body, and either end moves on its own::

    layout.apply_edit({'op': 'set', 'target': 'D1',
                       'attrs': {'p2': [0.6, 0.0]}})

``offset`` is where the dimension *line* is drawn, in metres, positive to the left of the direction from ``p1`` to ``p2``; :py:meth:`line_ends<gtrace.layout.Dimension.line_ends>` gives the two ends it lands on. It is a choice about the drawing rather than about the measurement: what a bench wants measured usually runs along a beam or through an element, which is exactly where a line drawn on top of it cannot be read. **It changes nothing about what is measured** — the span is still between ``p1`` and ``p2``, and so is the length and the test below.

**A dimension does not hold on to the element a point was taken from.** An element that then moves leaves the measurement where it was made, which is what a note should do; measuring the same thing again after a change means drawing it again.

What a dimension comes to is worked out afresh every time the scene is built, never stored, so it cannot go stale:

.. code-block:: python

    layout.get_dimension('D1').measure(layout.optics)
    # {'length': 0.0935, 'optical': 0.1355, 'inside': 'M1', 'n': 1.45}

``optical`` is reported **only when the whole span runs inside one substrate**, where it is the physical distance times the refractive index — the optical thickness of a substrate is the measurement this answers. A span that crosses in and out of glass has an optical length too, but it is not a dimension of anything: it depends on where the ends happen to fall, so it is left out rather than written next to a number it would be mistaken for.

The question behind that is :py:meth:`contains_segment<gtrace.optcomp.Optics.contains_segment>`, which asks the optics itself rather than describing its faces a second time. :py:meth:`isHit<gtrace.optcomp.Mirror.isHit>` reports a surface only when it is approached from outside, so from inside a substrate it finds nothing at all — and that is the whole of the test. Ends lying exactly on a face count as inside, since that is where such a measurement is usually taken from.

.. _mechanics:

Assemblies
-----------

Two absorbing faces in a V are one beam dump, a pair of steering mirrors is one periscope, and a bench is built out of such assemblies rather than out of loose elements. :py:meth:`assemble<gtrace.layout.OpticalLayout.assemble>` says so — the follower keeps its place relative to the host, and moving or turning the host carries it along:

.. code-block:: python

    layout.assemble('D2', 'D1')                  # keeps where it stands
    layout.assemble('D2', 'D1', offset=[0.06, 0.0],
                    offset_angle=deg2rad(-40))   # or at a place you name
    layout.disassemble('D2')

The offset is in the host's frame — origin at its substrate centre, x along its HR normal, the same frame a mount attaches in — and what lands there is the follower's **anchor point**, the one it is already held by. ``fix_rotation`` decides who may change the relative angle: True, the default, is a face of a dump, built at its angle; False lets the angle be edited, and the element still turns *with* its host. The host may be another element or a body.

**A follower's pose is stored, not derived.** A body attached to an optics computes its pose on every read; an element cannot, because an ``Optics`` holds its pose in traits whose derived geometry — the face centres, the normals — is what the trace reads, and computing it on demand would mean rewriting that. So the joint is written into the follower **just before the layout is read**, by :py:meth:`trace<gtrace.layout.OpticalLayout.trace>`, :py:meth:`draw<gtrace.layout.OpticalLayout.draw>` and :py:meth:`snap_points<gtrace.layout.OpticalLayout.snap_points>`.

That comes to the same thing as deriving it, for the same reason: there is no notification to miss, because nothing is listening. Assigning ``M1.HRcenter`` in a cell and then tracing carries the assembly along. The one thing it cannot cover is reading a follower's pose in a cell **without** tracing or drawing first — that value is the one from before the host moved. A layout with no assemblies is not touched at all.

Placing a follower is refused, in the same words a held body already uses: it goes where its host goes, so ``move``, ``align``, ``slide`` and a typed pose are turned away. A pose written into one would be overwritten at the next trace, which is worse than a refusal because it would look as though it had worked. The exception is the turn of an element whose ``fix_rotation`` is False, which is read back into the joint so that the next settle keeps it. An element another one follows cannot be removed until it is let go of, and :py:meth:`copy_optics<gtrace.layout.OpticalLayout.copy_optics>` brings the followers along — the second face of a dump is part of the dump.

A circle is refused, and so is a file that describes one: a pose that comes from itself is not wrong so much as endless.

Mechanics
----------

A bench is not only light. What holds the optics takes up room, bumps into things and has to be bolted somewhere, so a layout carries that too. A :py:class:`Mechanics<gtrace.mechanics.Mechanics>` is a named body of drawing primitives that the trace never sees:

.. code-block:: python

    import gtrace.draw as draw
    from gtrace.mechanics import Mechanics

    clamp = Mechanics(shapes=[draw.Rectangle([-0.015, -0.015], 0.03, 0.03),
                              draw.Circle([0.0, 0.0], 0.003)],
                      center=[0.2, 0.1], name='C1')
    layout.add_mechanics(clamp)

The shapes are in the body's **own** coordinates and the pose carries them onto the bench: ``center`` is where the local origin lands, ``rotationAngle`` how far the body is turned about it. The shapes never change when the body moves, which is what lets one drawing serve every copy of a part; :py:meth:`world_shapes<gtrace.mechanics.Mechanics.world_shapes>` is where they are, and it builds new primitives rather than moving the ones you hold.

A body is drawn, picked, measured, saved and exported like anything else, and it is invisible to the beams — adding, moving or editing one does not invalidate a trace. It goes on the ``mechanics`` layer by default, so a DXF or the viewer's layer panel can take all of it out of the way at once. What a click lands on is decided by :py:meth:`contains<gtrace.mechanics.Mechanics.contains>`, a point-in-polygon test against the body's :py:meth:`outline<gtrace.mechanics.Mechanics.outline>`; of several bodies under one point the smallest wins, so a mount standing on a breadboard is not shadowed by it.

Standing on something
^^^^^^^^^^^^^^^^^^^^^^

A mirror mount is a body that stands where its mirror stands, so it has no pose of its own:

.. code-block:: python

    from gtrace.mechanics import mirror_mount

    layout.add_mechanics(mirror_mount(name='MT1', attached_to=M1))

``center`` and ``rotationAngle`` of an attached body are **derived on every read** from the host's pose and the attachment offset. There is no notification to miss and no stored copy to go stale: move the mirror in a cell, in the viewer or by loading a file, and the mount is already where it should be. The price is that an attached body cannot be moved on its own — which is what "attached" means. Writing to its pose, or dragging it, is refused.

Where a body sits on its host is a **coordinate convention** rather than a number: the local origin of a part is the point that comes to rest at the host's substrate centre. A mount is therefore drawn around the mirror it will hold, and needs no offset to land correctly. ``offset`` and ``offset_angle`` are there for the times you mean to sit slightly off it.

**A body may stand on another body.** A bench stacks: the mount is bolted to a pedestal, the pedestal is held down by a clamping fork. ``attached_to`` takes either, the chain follows the optics at the root of it, and a cycle is refused — a pose that derived from itself would not be wrong so much as endless. :py:func:`host_pose<gtrace.mechanics.host_pose>` is the one place that knows the difference between the two kinds of host: an optics is turned by its HR normal, a body by its own angle.

What differs when one body stands on another is where it is pinned. A mount is pinned by its origin, since that is the point drawn to coincide with its optic. A pedestal dropped into a hole is pinned by *that hole*, and a fork by the bore it closes on — so ``attach_point`` says which point of the body is held, in its own coordinates, and the pose derives from that:

.. code-block:: text

    angle  = host angle + offset_angle
    centre = host frame(offset) - R(angle) · attach_point

With the default attach point of ``[0, 0]`` the second term vanishes and this is the rule every mount was already drawn to. Attaching through the edit protocol picks it up from the drawing: the point of the body that already coincides with a point of the host is the one it is pinned by, so "drop it on the hole, then attach it" pins it by the hole.

``fix_rotation`` decides who may change the relative angle. True — the default — is a mount bolted to its mirror: it faces where the host faces and there is nothing to type. False is a clamping fork, which swings about the point it is pinned by; the ``Angle`` row and Shift-drag then set it. Either way the body turns **with** the host, because a stack that came apart when the mirror was aimed would not be a stack.

**Named points** are what all of this is stood on. ``points`` is a dict of local points a part names for itself — ``'post'`` for the hole under a mount, ``'axis'`` for a pedestal, ``'bore'`` and ``'screw'`` for a fork — and it travels with the model in the library. They join the snap points, so a drag settles on them, a measurement reaches them, and Align aims by them.

:py:meth:`detach<gtrace.mechanics.Mechanics.detach>` bakes the derived pose in and frees the body; :py:meth:`attach<gtrace.mechanics.Mechanics.attach>` seats it on a host, at the model's own place by default, or where it already stands with ``keep_pose=True``. A layout saves the host's **name** and the offset, never the derived pose, and re-links the two when it is loaded. An optics with something attached to it cannot be removed until it is let go of.

:py:meth:`copy_optics<gtrace.layout.OpticalLayout.copy_optics>` — ``{'op': 'copy', 'target': 'M1'}`` — adds a second one of an element **with the whole stack standing on it**: the mount bolted to it, the pedestal under the mount, the fork over the pedestal, each pinned to the copy exactly as its original is pinned to the original. One of a pair of steering mirrors is not one element; it is the element and everything built under it, and none of that is worth assembling twice.

The copies are made through the same dicts a saved layout is written with, so what is copied is what would have been saved — by value, sharing nothing. The poses of the bodies are the one thing not copied, because they were never stored: each derives its own from the copy it now stands on. Without a name the copy takes the original's without its trailing number and the first free one after it, so a copy of ``M1`` is ``M2``; without an offset it stands its own diameter away along both axes, far enough to clear what it was made from. Only an element is copied this way — a stack stands on an element at its root, and a body on its own is one call to the model library away.

The model library
^^^^^^^^^^^^^^^^^^

Parts repeat, so they are worth registering once:

.. code-block:: python

    from gtrace.mechanics import (register_model, models, from_model,
                                  save_models, load_models)

    register_model('CLAMP-30', clamp, 'a 30 mm clamp with one bolt hole')
    models()                                   # name -> description
    layout.add_mechanics(from_model('CLAMP-30', name='C2',
                                    center=[0.4, 0.1]))

    save_models('parts.json', names=['CLAMP-30'])
    load_models('parts.json')                  # merged by name, last wins

A model is a **value** - shapes, a layer, a description and the builder parameters - copied in when it is registered and copied out when it is used, so a body and the model it came from cannot drift into each other. Loading a file merges it into the registry name by name, which is how a library is assembled from several files; a file with one bad shape in it is refused whole, leaving the registry untouched.

What a body keeps is the shapes themselves, and the model name only as a label. **The saved layout is the truth**: a library that has moved on cannot change a drawing you already made. :py:meth:`relink_mechanics<gtrace.layout.OpticalLayout.relink_mechanics>` is how you deliberately ask for the newer definition, and it touches neither pose nor attachment.

The stock models are generic on purpose — ``BB3030``, ``MOUNT-25``, ``HOLDER-50`` and the like — because gtrace does not invent a vendor's dimensions. The builders behind them take the numbers you measured:

.. code-block:: python

    from gtrace.mechanics import (breadboard, round_breadboard,
                                  mirror_mount, lens_holder,
                                  pedestal, clamping_fork)

    breadboard(0.45, 0.30, pitch=0.025, hole_diameter=0.006)
    round_breadboard(0.30, pitch=0.025, hole_diameter=0.006)
    mirror_mount(scale=1.0, knobs=True)
    lens_holder(length=0.030, thickness=0.010)
    pedestal(post_diameter=0.0254, base_diameter=0.0318)
    clamping_fork(bore_diameter=0.0260, length=0.0738)

:py:func:`round_breadboard<gtrace.mechanics.round_breadboard>` is the board that goes in the bottom of a vacuum tank. The grid is the rectangular board's — symmetric about the centre, on the same pitch — and the rim decides which of its holes exist: a hole is drilled where it lies a margin in from the edge and left out where it does not, so the rows shorten towards the rim the way a real disc is drilled.

A body built by one of these keeps the parameters it was built from, in ``params``, which is what makes it resizable: :py:meth:`resize<gtrace.mechanics.Mechanics.resize>` re-drills a breadboard at the new size rather than scaling it, so the holes keep their diameter and their pitch. :py:attr:`resizable<gtrace.mechanics.Mechanics.resizable>` says how — ``'box'`` for a body with two sides, ``'round'`` for one with a single size, and ``None`` for a body drawn by hand, whose shapes are all anyone knows about it. **A round body has one size, not two:** either ``width`` or ``height`` sets its diameter, and two that disagree are refused rather than resolved by picking one.

Editing a body from a front end
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same operations reach a body, with its own whitelist (``EDITABLE_MECHANICS_ATTRS``: ``center``, ``rotationAngle``, ``attached_to``, ``offset``, ``offset_angle``, ``fix_rotation``, ``width`` and ``height``):

.. code-block:: python

    layout.apply_edit({'op': 'add', 'type': 'Mechanics', 'name': 'BB1',
                       'params': {'model': 'BB4530',
                                  'center': [0.3, 0.15]}})
    layout.apply_edit({'op': 'set', 'target': 'MT1',
                       'attrs': {'attached_to': 'M2'}})
    layout.apply_edit({'op': 'set', 'target': 'BB1',
                       'attrs': {'width': 0.6, 'height': 0.45}})

An ``add`` naming a ``model`` and no ``shapes`` is built from the library; one carrying ``shapes`` is built from them. ``attached_to`` takes the name of an optics **or of another body**, or ``None``. Seating a body on an *optics* puts it at the model's place, since where a mount belongs on a mirror is the library's business rather than the cursor's; seating it on a *body* keeps where it already is, since which hole of a mount a pedestal sits in is a choice made on the bench. Setting ``width`` or ``height`` goes through ``resize``, which says so when the body is not one that has a size. A ``rotate`` on an attached body is refused unless its turn is free.

Drawing a part
^^^^^^^^^^^^^^^

:py:meth:`Mechanics.edit<gtrace.mechanics.Mechanics.edit>` opens the shape editor on a body — the same viewer, handed a scene of nothing but the shapes, drawn in the local frame with the origin marked. See :ref:`the-shape-editor` for what it offers. The model behind it is :py:class:`ShapeEditor<gtrace.draw.viewer.editor.ShapeEditor>`, which is drivable without a browser and speaks a protocol of its own:

.. code-block:: python

    from gtrace.draw.viewer.editor import ShapeEditor

    ed = ShapeEditor(clamp)
    ed.apply_edit({'op': 'add_shape', 'type': 'circle'})
    ed.apply_edit({'op': 'set_shape', 'index': 2,
                   'attrs': {'radius': 0.004}})
    ed.apply_edit({'op': 'rotate_shape', 'index': 0, 'angle': 0.7854})
    ed.apply_edit({'op': 'set_points',
                   'points': [{'name': 'post', 'point': [-0.0135, 0.0]}]})
    ed.apply_edit({'op': 'undo'})

The operations are ``add_shape``, ``set_shape``, ``remove_shape``, ``duplicate_shape``, ``move_shape``, ``rotate_shape``, ``set_points``, ``save_model``, ``undo`` and ``redo``. A shape is edited by taking it apart into the dict :py:func:`shape_to_dict<gtrace.draw.serialize.shape_to_dict>` writes, changing what the message names and building it again, so the constructors are the only rule about what a shape is; what they do not catch — a size of none or less, a coordinate at infinity, an outline of one vertex — is refused on the way out. An index is a **place in the list**, which is also the order the shapes are drawn in, so removing one renumbers those after it.

A turn is the one edit that is not a set of attributes, because what turning means differs by kind: an arc's two angles move, a text turns with its own rotation, and a ``Rectangle`` — a corner, a width and a height, with its sides along the axes — has no turned form at all and comes back as the closed polyline of its four corners. That rule is :py:func:`turned_shape<gtrace.mechanics.turned_shape>`, which is also how a turned body's rectangles reach the bench. ``pivot`` defaults to :py:func:`shape_centre<gtrace.mechanics.shape_centre>`, the middle of the shape's bounding box.

``set_points`` carries the **whole list** of named points rather than one of them, because there is no index that survives a rename: a point is known by its name, and a name is the thing being edited. So renaming one, moving one, adding one and taking one away are all the same message — which also makes each of them one step of undo. Two points cannot share a name, and a point cannot go unnamed. The scene channel is ``points``, a list of ``{'name': str, 'point': [x, y], 'index': int}``.

The editor holds the ``Mechanics`` **by reference**, like everything else here, so a body already registered in a layout is redrawn there at the layout's next draw, with its attachment, pose and builder parameters untouched.

Scene channels for a front end
-------------------------------

:py:meth:`scene_dict<gtrace.layout.OpticalLayout.scene_dict>` adds eight entries to what :py:func:`scene_to_dict<gtrace.draw.serialize.scene_to_dict>` builds: ``can_undo`` and ``can_redo``, the ``dimensions`` above with their measurements, ``snap`` — the points of the optics a front end may snap a measurement to — ``sources`` and ``rules``, and ``mechanics`` and ``mechlib`` for the bodies.

``sources`` is what says which of the beams the user put there. Nothing else can: a source is traced from a *copy* of itself, so its own beam sits in ``beams`` looking exactly like the ones the trace made from it. Each entry carries where the laser stands, which way it fires, and the light it emits — including the waist, worked out on this side rather than stored, for the same reason a dimension's length is. ``rules`` carries the tracing rules, which are not a property of any element but decide how much of the picture there is.

Each dimension carries a ``line``, the two ends its line lands on once the offset is applied, so that only one place has an opinion about which side the offset goes.

``mechanics`` carries each body's pose, what it is attached to, and the outline a front end picks it by — worked out here, since it is the same polygon :py:meth:`contains<gtrace.mechanics.Mechanics.contains>` tests against and there is no reason for a browser to have a second opinion about it. ``mechlib`` is the model library as names and descriptions, which is what the ``+ Mechanics`` menu is; the shapes stay on this side until one is chosen.

``snap`` carries the four corners of each substrate, the apex of each face and the middle, the points a body names for itself, and the centre of every screw hole a body carries. The named points come before the holes, because two marks at one place are the same point as far as snapping goes and the first wins: a mount's post hole is both a circle in the drawing and the point it is stood on its pedestal by, and ``MT post`` says more than ``MT hole``. They come from Python because they are geometry: a corner is where the wedge and the sagitta of a curved face put it, and there is no reason for a second description of that to live in a browser. Beam ends are deliberately *not* in it — the scene already carries the ends of every beam literally, so a front end can offer those without anything being worked out twice.

Undo and redo
--------------

Every edit that goes through ``apply_edit`` keeps the state before it, so it can be taken back — and the state an undo steps out of is kept in turn, so it can be put back:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.8, 0.3]})
    layout.can_undo             # True
    layout.undo()               # or apply_edit({'op': 'undo'})
    layout.can_redo             # True
    layout.redo()               # or apply_edit({'op': 'redo'})

The history lives on the layout rather than in a front end, so undo means the same thing however the edit arrived. It goes back ``UNDO_DEPTH`` edits, and the redo side is bounded by the same number since only undoing fills it. A refused edit changes nothing and costs no step, and neither does ``save``, which only writes a file.

**An edit that goes through discards whatever is waiting to be redone.** Once the layout has taken a different turn, the branch that was stepped out of is no longer somewhere to return to: the states put aside describe elements the new edit may have renamed, removed or moved on from.

A step holds the registered elements themselves alongside their serialized values, and restoring one puts those values back onto those same objects — through a rename, and through a removal, since an element taken out of the layout is held in the history and put back as itself. See :py:meth:`undo<gtrace.layout.OpticalLayout.undo>` for what that does *not* cover.
