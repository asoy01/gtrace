Optical layouts
===============================

An :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` holds a whole
optical system in one object: the optics, the source beams and the rules
that govern the non-sequential trace. It is what you draw, what you save to
disk, and what the interactive viewer edits.

Use a layout when you intend to trace, redraw and adjust one system
repeatedly. The optics and beams described in :doc:`basic_concepts` and
:doc:`propagation` also work on their own, and the :doc:`tutorial` gets as
far as a traced mode cleaner without a layout.

Build it, then register it
---------------------------

The work falls into two phases. First, build and align the system with
ordinary Python:

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

Then register the result and work with it as a whole:

.. code-block:: python

    from gtrace.layout import OpticalLayout, TraceRules

    layout = OpticalLayout(optics=[PRM, PR2], sources=[src],
                           rules=TraceRules(order=10, power_threshold=1e-3),
                           name='PRC')

    layout.trace()          # non-sequential trace from every source
    layout.show()           # open the viewer

A layout does not replace the code of the first phase. You place the
mirrors with ordinary Python: sequential tracing, ``scipy.optimize``,
cavity eigenmode solving. That is how you close a cavity on itself, or make
a beam arrive at a mirror at the intended angle of incidence. The layout
takes the result and gives you one object to trace, draw, save and edit.

Held by reference
------------------

The layout does not copy what you register. ``layout.get_optics('PRM')``
returns the very object bound to ``PRM`` in your code. So:

.. code-block:: python

    PRM.HRcenter = [0.01, 0.0]
    layout.trace()          # the new result reflects the move

and, in the other direction, an edit made in the viewer changes the object
that ``PRM`` names. There is one copy of the model, and both you and the GUI
are looking at it.

:py:meth:`update_from_file<gtrace.layout.OpticalLayout.update_from_file>`
follows the same rule. It loads a saved layout *in place*: optics whose name
and type match are updated, not replaced, so ``PRM`` still names a
registered optics after you load a file.

:py:meth:`trace<gtrace.layout.OpticalLayout.trace>` is the one exception. It
traces a *copy* of each source beam, so the registered source is not
consumed by tracing and stays where you put it.

Tracing rules
--------------

:py:class:`TraceRules<gtrace.layout.TraceRules>` collects what governs the
non-sequential trace:

``order``
    How many ghost reflections a beam may go through before it stops being
    followed. Defaults to 10.

``power_threshold``
    Beams weaker than this are not propagated further. Defaults to 0.1. This
    is the value to change when you look for ghost beams. Lower it and the
    trace finds fainter paths, and takes longer.

``open_beam_length``
    How long a beam that hits nothing is drawn. Defaults to 1.0.

Each beam carries its own count, and the count is not reset when the beam
leaves one element for the next, so ``order`` limits the whole trace. How
many ghosts of *one particular element* should be followed is a property of
that element. It lives there instead, as the ``max_stray_order`` attribute
of the optics. What raises a beam's stray order, and the flags that say
which face of an element is meant for what, are described in
:ref:`stray-order`.

``open_beam_length`` applies to the beams the trace produces. A *source*
that reaches nothing keeps its own ``length``, which is the state a layout
is in while it is being built.

All three can also be changed from the viewer, or by the message in
:ref:`editing-a-source`.

Drawing options
----------------

How a layout is drawn is a display choice, not part of the model. Changing
one redraws; it does not re-trace. The available options and their defaults
are collected in ``gtrace.layout.DRAW_OPTIONS``:

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

An option can be given per call, or set on the layout to apply to every
subsequent drawing:

.. code-block:: python

    layout.render_html('trace.html', width_mode='y', sigma_main=1.0)

    layout.draw_options['width_mode'] = 'avg'   # applies from now on

A misspelt option raises ``TypeError`` instead of being ignored::

    >>> layout.render_html('trace.html', widthMode='avg')
    TypeError: Unknown drawing option 'widthMode'. Known options are
    drawBeamLabels, drawMainWidth, drawMechanicsNames, drawOpticsNames,
    drawStrayWidth, fontSize, sigma_main, sigma_stray, width_mode.

.. _why-2.7-sigma:

Why 2.7 σ
^^^^^^^^^^

The envelope is drawn at 2.7 times the 1/e² radius because that is the
aperture at which the diffraction loss of a Gaussian beam is 1 ppm. Drawing
every envelope at the same σ gives every envelope in the picture the same
meaning: outside this line there is nothing that matters at the ppm level.

``width_mode`` exists because a beam is not round in general. After a beam
passes through a wedged substrate at a non-normal incidence, its horizontal
and vertical waists differ. There is then no single correct answer to "how
wide is this beam", and the drawing cannot make the choice for you.

Saving and loading
-------------------

A layout serialises to plain JSON:

.. code-block:: python

    layout.save('layout.json')

    layout2 = OpticalLayout.load('layout.json')     # a new layout
    layout.update_from_file('layout.json')          # in place, see above

The file holds the optics, the sources, the tracing rules and the drawing
options. The trace result is not saved. Call
:py:meth:`trace<gtrace.layout.OpticalLayout.trace>` after loading a file.
Drawing a layout that has no trace result traces it first, so
:py:meth:`show<gtrace.layout.OpticalLayout.show>` works straight after a
load.

.. _dimensions:

Dimensions
-----------

A :py:class:`Dimension<gtrace.layout.Dimension>` is a distance measured
between two points of the layout, kept as a note on it:

.. code-block:: python

    from gtrace.layout import Dimension

    layout.add_dimension(Dimension(M1.HRcenter, M1.ARcenter,
                                   name='D1', offset=0.17))
    layout.get_dimension('D1').length          # in metres

Dimensions are registered on the layout beside the optics, saved with it,
and taken back by undo like anything else. They share one namespace with the
optics.

``offset`` is where the dimension *line* is drawn, in metres, positive to
the left of the direction from ``p1`` to ``p2``.
:py:meth:`line_ends<gtrace.layout.Dimension.line_ends>` gives the two ends it
lands on. This is a choice about the drawing, not about the measurement.
The distances you want to measure usually run along a beam or through an
element, and a line drawn on top of them cannot be read. **It changes
nothing about what is measured.** The span is still between ``p1`` and
``p2``, and so are the length and the test below.

**A dimension does not hold on to the element a point was taken from.** An
element that then moves leaves the measurement where it was made. To measure
the same thing again after a change, draw it again.

What a dimension comes to is worked out afresh every time the scene is
built, and never stored, so it cannot go stale:

.. code-block:: python

    layout.get_dimension('D1').measure(layout.optics)
    # {'length': 0.0999, 'optical': 0.1450, 'inside': 'PRM', 'n': 1.45}

``optical`` is reported **only when the whole span runs inside one
substrate**, where it is the physical distance times the refractive index.
The optical thickness of a substrate is the measurement this answers. A span
that crosses in and out of glass also has an optical length, but that length
is not a dimension of anything. It depends on where the ends happen to fall,
so gtrace leaves it out.

Whether a span runs inside one substrate is answered by
:py:meth:`contains_segment<gtrace.optcomp.Optics.contains_segment>`, which
asks the optics itself.
:py:meth:`isHit<gtrace.optcomp.Mirror.isHit>` reports a surface only when the
ray approaches it from outside, so from inside a substrate it finds nothing
at all. Ends that lie exactly on a face count as inside, because that is
where such a measurement is usually taken from.

Assemblies
-----------

Two absorbing faces in a V make one beam dump, and a pair of steering
mirrors makes one periscope.
:py:meth:`assemble<gtrace.layout.OpticalLayout.assemble>` says that one
element follows another. The follower keeps its place relative to the host,
and moving or turning the host carries it along:

.. code-block:: python

    layout.assemble('D2', 'D1')                  # keeps where it stands
    layout.assemble('D2', 'D1', offset=[0.06, 0.0],
                    offset_angle=deg2rad(-40))   # or at a place you name
    layout.disassemble('D2')

The offset is in the host's frame: origin at its substrate centre, x along
its HR normal, the same frame a mount attaches in. What lands there is the
follower's **anchor point**, the one it is already held by. ``fix_rotation``
decides who may change the relative angle. True, the default, suits a face
of a dump, which is built at its angle. False lets the angle be edited, and
the element still turns *with* its host. The host may be another element or
a body.

A follower cannot be placed by hand. ``move``, ``align``, ``slide`` and a
typed pose are refused, in the same words an attached body uses: the
follower goes where its host goes. There is one exception. If
``fix_rotation`` is False, the turn of the element is read back into the
joint, so that the next settle keeps it.
:py:meth:`copy_optics<gtrace.layout.OpticalLayout.copy_optics>` brings the
followers along, because the second face of a dump is part of that dump.

**A follower's pose is stored, not derived.** A body attached to an optics
computes its pose on every read. An element cannot. An ``Optics`` holds its
pose in traits, and the trace reads the geometry derived from them: the face
centres and the normals. The joint is therefore written into the follower
**just before the layout is read**, by
:py:meth:`trace<gtrace.layout.OpticalLayout.trace>`,
:py:meth:`draw<gtrace.layout.OpticalLayout.draw>` and
:py:meth:`snap_points<gtrace.layout.OpticalLayout.snap_points>`.

One case follows from that. Reading a follower's pose in a cell **without**
tracing or drawing first gives the value from before the host moved. Assign
``M1.HRcenter``, then call ``layout.trace()``, and the follower is where it
should be. A layout with no assemblies is not touched at all.

**Removing something takes what stands on it.**
:py:meth:`remove_optics<gtrace.layout.OpticalLayout.remove_optics>` and
:py:meth:`remove_mechanics<gtrace.layout.OpticalLayout.remove_mechanics>`
take the target and everything whose pose comes from it, however deep, and
return the list of what went. A mount whose mirror has gone would derive its
pose from something no longer in the layout. It is one edit, so it is one
step of undo. To keep something that is standing on it, detach or
disassemble it first.

A circle is refused, in a call and in a loaded file: a pose that comes from
itself has no value.

The beam dump
^^^^^^^^^^^^^^

:py:func:`beam_dump<gtrace.layout.beam_dump>` builds the assembly the
feature was written for: two absorbing faces in a V and the housing they sit
in, jointed so the three move as one.

.. code-block:: python

    faces, bodies = layout.add_beam_dump(center=[0.3, 0.0], angle=0.0)
    [o.name for o in faces], [b.name for b in bodies]
    # (['BD1a', 'BD1b'], ['BD1box'])

``angle`` is **the direction the light travels**, so a dump is aimed the way
the beam runs, not by where its mouth points. What comes back is split into
``(optics, bodies)``, which is what every builder here returns. Each list
has the hosts first, which is the order they are registered in. They are
named from the
dump: ``BD1a`` and ``BD1b`` are its two faces and ``BD1box`` its housing. A
dump is numbered and its pieces lettered, so ``BD2b`` is the far face of the
second dump. Without a name it is given the first free one.

The two faces stand in a V because a black face is not perfectly black.
What one face sends back, the other catches and sends back again, so the
light works its way into the wedge instead of coming out the way it came.
With the default 4% a beam is down to 0.16% after two bounces and to a part
in ten million after five. The faces are therefore elements, and not a shape
drawn on the housing.

The V comes from three numbers in the drawing: the apex 25 mm above the post
hole, faces 50 mm long, opening 28°. **The reflectivity is not one of them.**
What a black absorber returns depends on the glass and the polarisation, so
:py:data:`DUMP_REFLECTIVITY<gtrace.layout.DUMP_REFLECTIVITY>` is a
placeholder to be measured and passed.

Aim it into one side of the V, not at the apex. A ray sent exactly at the
apex hits neither face, since that is the point where the two of them end.
The same is true of a real dump.

.. _mechanics:

Mechanics
----------

A bench is not only light. What holds the optics takes up room, bumps into
things and has to be bolted somewhere, so a layout carries that too. A
:py:class:`Mechanics<gtrace.mechanics.Mechanics>` is a named body of drawing
primitives that the trace never sees:

.. code-block:: python

    import gtrace.draw as draw
    from gtrace.mechanics import Mechanics

    clamp = Mechanics(shapes=[draw.Rectangle([-0.015, -0.015], 0.03, 0.03),
                              draw.Circle([0.0, 0.0], 0.003)],
                      center=[0.2, 0.1], name='C1')
    layout.add_mechanics(clamp)

The shapes are in the body's **own** coordinates, and the pose carries them
onto the bench: ``center`` is where the local origin lands,
``rotationAngle`` how far the body is turned about it. The shapes do not
change when the body moves, so one drawing serves every copy of a part.
:py:meth:`world_shapes<gtrace.mechanics.Mechanics.world_shapes>` tells you
where they are. It builds new primitives, and does not move the ones you
hold.

A body is drawn, picked, measured, saved and exported like anything else,
and it is invisible to the beams: adding, moving or editing one does not
invalidate a trace. It goes on the ``mechanics`` layer by default, so a DXF
or the viewer's layer panel can take all of it out of the way at once. What
a click lands on is decided by
:py:meth:`contains<gtrace.mechanics.Mechanics.contains>`, a point-in-polygon
test against the body's
:py:meth:`outline<gtrace.mechanics.Mechanics.outline>`. Of several bodies
under one point the smallest wins, so a mount standing on a breadboard is
not shadowed by it.

Standing on something
^^^^^^^^^^^^^^^^^^^^^^

A mirror mount is a body that stands where its mirror stands, so it has no
pose of its own:

.. code-block:: python

    from gtrace.mechanics import mirror_mount

    layout.add_mechanics(mirror_mount(name='MT1', attached_to=M1))

``center`` and ``rotationAngle`` of an attached body are **derived on every
read** from the host's pose and the attachment offset. There is no stored
copy that could go stale. Move the mirror in a cell, in the viewer, or by
loading a file, and the mount is already where it should be. The cost is
that an attached body cannot be moved on its own, which is what "attached"
means. Writing to its pose, or dragging it, is refused.

Where a body sits on its host is a **coordinate convention**, not a number.
The local origin of a part is the point that comes to rest at the centre of
the host's substrate. A mount is therefore drawn around the mirror it holds,
and needs no offset to land correctly. ``offset`` and ``offset_angle`` are
there for when you mean to sit slightly off it.

**A body may stand on another body.** A bench stacks: the mount is bolted to
a pedestal, the pedestal is held down by a clamping fork. ``attached_to``
takes either one. The chain follows the optics at its root, and a cycle is
refused. :py:func:`host_pose<gtrace.mechanics.host_pose>` is the one place
that knows the difference between the two kinds of host: an optics is turned
by its HR normal, and a body by its own angle.

What differs when one body stands on another is where it is pinned. A mount
is pinned by its origin, since that is the point drawn to coincide with its
optic. A pedestal dropped into a hole is pinned by *that hole*, and a fork by
the bore it closes on. ``attach_point`` says which point of the body is held,
in its own coordinates, and the pose derives from that:

.. code-block:: text

    angle  = host angle + offset_angle
    centre = host frame(offset) - R(angle) · attach_point

With the default attach point of ``[0, 0]`` the second term vanishes, and
this is the rule every mount was already drawn to.

``fix_rotation`` decides who may change the relative angle. True, the
default, suits a mount bolted to its mirror: it faces where the host faces
and there is nothing to type. False suits a clamping fork, which swings
about the point it is pinned by; the ``Angle`` row and Shift-drag then set
it. Either way the body turns **with** the host.

**Named points** are what all of this stands on. ``points`` is a dict of
local points a part names for itself: ``'post'`` for the hole under a mount,
``'axis'`` for a pedestal, ``'bore'`` and ``'screw'`` for a fork. It travels
with the model in the library. Named points join the snap points, so a drag
settles on them, a measurement reaches them, and Align aims by them.

:py:meth:`detach<gtrace.mechanics.Mechanics.detach>` bakes the derived pose
in and frees the body. :py:meth:`attach<gtrace.mechanics.Mechanics.attach>`
seats it on a host, at the model's own place by default, or where it already
stands with ``keep_pose=True``. A layout saves the host's **name** and the
offset, never the derived pose, and re-links the two when it is loaded.
Removing an optics takes the bodies standing on it with it; detach one first
to keep it.

:py:meth:`copy_optics<gtrace.layout.OpticalLayout.copy_optics>` adds a second
one of an element **with the whole stack standing on it**: the mount bolted
to it, the pedestal under the mount, and the fork over the pedestal. Each
one is pinned to the copy the way its original is pinned to the original.
One of a pair of steering mirrors is the element and everything built under
it, and you do not want to assemble that twice.

The copies are made through the same dicts a saved layout is written with,
so what is copied is what would have been saved: by value, sharing nothing.
The poses of the bodies are the one thing not copied, because they were
never stored; each derives its own from the copy it now stands on. Without a
name, the copy takes the original's name without its trailing number and the
first free number after it, so a copy of ``M1`` is ``M2``. Without an offset
it stands its own diameter away along both axes, far enough to clear what it
was made from.

Only an element is copied this way, since a stack stands on an element at
its root. To get a second copy of a single body, build it from the model
library with :py:func:`from_model<gtrace.mechanics.from_model>`.

The model library
^^^^^^^^^^^^^^^^^^

What actually goes onto a bench is not a mirror but a mirror in a mount, on
a pedestal, held down by a fork. :py:func:`assembly<gtrace.layout.assembly>`
builds one:

.. code-block:: python

    from gtrace.layout import assembly_kinds

    [k['kind'] for k in assembly_kinds()]
    # ['MIRROR-1IN', 'MIRROR-2IN', 'LENS-1IN', 'LENS-2IN']

    optics, bodies = layout.add_assembly('MIRROR-2IN', center=[0.3, 0.1],
                                         angle=deg2rad(45))
    [o.name for o in optics], [b.name for b in bodies]
    # (['M1'], ['MT1', 'P1', 'FK1'])

Four objects come back: the element first, and then the parts that hold it.
Each part is attached to the one below it. The mount goes on the optic at
the position the model was designed for. The pedestal goes in the hole the
mount is bolted down through. The fork goes round the pedestal, with its
turn free. **The element is the thing to move.** Everything else derives its
pose from the element.

:py:meth:`add_assembly<gtrace.layout.OpticalLayout.add_assembly>` registers
all four and fills in the names. Each piece takes the first number that is
free for its own kind, so a second two-inch mirror comes down as ``M2``,
held by ``MT2`` on ``P2`` in ``FK2``.

:py:func:`mirror_assembly<gtrace.layout.mirror_assembly>` and
:py:func:`lens_assembly<gtrace.layout.lens_assembly>` are what the kinds are
made of. They take the same arguments, plus the models to build the parts
from. Pass ``None`` for a model to leave that piece out.
``mount_offset`` says where the mount is really bolted. It is in the optic's
own frame, with x along the face normal. The two-inch kind sits its mount
**5 mm further back** than the designed position in the drawing. That 5 mm
is a bench measurement, and the model does not know it. What the pedestal
stands in moves with the mount, because the post hole is a point of the
mount.

Parts repeat, so they are worth registering once:

.. code-block:: python

    from gtrace.mechanics import (register_model, models, from_model,
                                  save_models, load_models)

    register_model('CLAMP-30', clamp, 'a 30 mm clamp with one bolt hole',
                   prefix='CL')          # its parts are CL1, CL2
    models()                                   # name -> description
    layout.add_mechanics(from_model('CLAMP-30', name='C2',
                                    center=[0.4, 0.1]))

    save_models('parts.json', names=['CLAMP-30'])
    load_models('parts.json')                  # merged by name, last wins

A model is a **value**: the shapes, a layer, a description, what its parts
are called, and the builder parameters. gtrace copies them in when the model
is registered, and copies them out when it is used, so a body and the model
it came from stay independent.

``prefix`` is what the bodies built from the model are named, before the
number. It is part of the definition, and not part of any one layout. A part
is known by what it is: a mount is ``MT1``, whatever catalogue the footprint
came from. The stock models use ``MT`` for a mount, ``P`` for a pedestal,
``FK`` for a fork, ``HLD`` for a holder and ``BB`` for a breadboard. ``PD``
is deliberately unused, because it reads as photodetector. A model that
gives no prefix leaves the naming to whoever adds the body, which is ``H``
in the viewer. Loading a file merges it into the registry name by name,
which is how a library is assembled from several files. A file with one bad
shape in it is refused whole, and the registry is left untouched.

A body keeps the shapes themselves, and the model name only as a label.
**The saved layout is the truth**: a library that has moved on cannot change
a drawing you already made.
:py:meth:`relink_mechanics<gtrace.layout.OpticalLayout.relink_mechanics>` is
how you ask for the newer definition, and it touches neither pose nor
attachment.

The stock models are generic on purpose: ``BB3030``, ``MOUNT-25``,
``HOLDER-50`` and so on. gtrace does not invent the dimensions of a
vendor's part. The builders behind the models take the numbers you
measured:

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

:py:func:`round_breadboard<gtrace.mechanics.round_breadboard>` is the board
that goes in the bottom of a vacuum tank. The grid is the same as the
rectangular board's: symmetric about the centre, on the same pitch. The rim
decides which of the holes exist. A hole is drilled where it lies a margin
in from the edge, and left out where it does not, so the rows get shorter
towards the rim, as on a real disc.

A body built by one of these keeps the parameters it was built from, in
``params``. That is what lets you resize it.
:py:meth:`resize<gtrace.mechanics.Mechanics.resize>` re-drills a breadboard
at the new size instead of scaling it, so the holes keep their diameter and
their pitch. :py:attr:`resizable<gtrace.mechanics.Mechanics.resizable>` says
how: ``'box'`` for a body with two sides, ``'round'`` for one with a single
size, and ``None`` for a body drawn by hand, whose shapes are all anyone
knows about it. **A round body has one size, not two:** either ``width`` or
``height`` sets its diameter, and two that disagree are refused.

Drawing a part
^^^^^^^^^^^^^^^

:py:meth:`Mechanics.edit<gtrace.mechanics.Mechanics.edit>` opens the shape
editor on a body. It is the same viewer, given a scene of nothing but the
shapes, drawn in the local frame with the origin marked. See
:ref:`the-shape-editor` for what it offers, and :doc:`editing` for driving
it from code.

Undo and redo
--------------

Every edit that goes through
:py:meth:`apply_edit<gtrace.layout.OpticalLayout.apply_edit>` keeps the
state before it, so it can be taken back. The state an undo steps out of is
kept in turn, so it can be put back:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'M1', 'HRcenter': [0.8, 0.3]})
    layout.can_undo             # True
    layout.undo()               # or apply_edit({'op': 'undo'})
    layout.can_redo             # True
    layout.redo()               # or apply_edit({'op': 'redo'})

The history lives on the layout, not in a front end, so undo means the same
thing however the edit arrived. It goes back ``UNDO_DEPTH`` edits. The redo
side has the same limit, because only an undo fills it. A refused edit
changes nothing and costs no step, and neither does ``save``, which only
writes a file.

**An edit that goes through discards whatever is waiting to be redone.**
Once the layout has taken a different turn, you cannot return to the branch
you stepped out of. The states put aside describe elements that the new edit
may have renamed, removed, or changed.

A step holds the registered elements themselves alongside their serialized
values, and restoring one puts those values back onto those same objects.
That holds through a rename, and through a removal, since an element taken
out of the layout is held in the history and put back as itself. See
:py:meth:`undo<gtrace.layout.OpticalLayout.undo>` for what that does *not*
cover.

Editing from a front end
-------------------------

A layout is edited by the viewer, and by anything else that speaks the same
protocol, through
:py:meth:`apply_edit<gtrace.layout.OpticalLayout.apply_edit>`:

.. code-block:: python

    layout.apply_edit({'op': 'move', 'target': 'PRM',
                       'HRcenter': [0.02, 0.0]})

For what the viewer's buttons and drags do, see :doc:`viewer`. For the
messages themselves, see :doc:`editing`: which operations exist, what each
one may touch, and what the scene carries back.
