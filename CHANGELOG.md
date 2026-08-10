# Changelog

Notable changes to gtrace. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and gtrace aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries marked **Changed results** alter the numbers gtrace produces for
an unchanged input. They are corrections rather than regressions, but a
system traced with an earlier version will not reproduce bit for bit
across them.

## Unreleased

### Added

- **`Mechanics`: hardware on the layout** (`gtrace.mechanics`). A named
  body - a breadboard, a mirror mount, the housing of a beam dump -
  that is drawn, saved and edited like everything else and that the
  trace never sees. Anything that is to stop light is still an
  `Optics`; a `Mechanics` is only to be seen.
  - The geometry is a list of `gtrace.draw` primitives in local
    coordinates, placed on the bench by a pose (`center`,
    `rotationAngle`). The pose is the only statement of where the body
    is: the world shapes, the outline and the snap points are derived
    from it, so moving the body is a change of two numbers.
  - Registered with `OpticalLayout.add_mechanics()`, saved by value in
    the layout file (with an optional `model` name as a label), drawn
    on a `hardware` layer of its own - which the viewer and any CAD
    reading the DXF can switch off as one thing - and picked in the
    viewer by point-in-polygon on its outline, after everything else:
    a beam, an optics or a mount lying on a breadboard wins the click
    over it, and among mechanics the smallest wins.
  - In the viewer, clicking a mechanics opens a pose panel (centre,
    angle, rename, remove); dragging a *selected* mechanics moves it
    and Shift-dragging turns it. An unselected one is not grabbed - a
    breadboard can cover most of the bench, and a drag across it
    should pan the view. Hardware covered by an optics - a mount is
    covered completely by its own mirror - takes the last turn of the
    repeated-click walk that already steps from an element into the
    beams under it.
  - **`+ Hardware`** adds a model from the library at the centre of
    the view; the menu is filled from a `mechlib` scene channel, so
    it lists whatever the library holds when the scene was built.
  - **The attachment is edited from the panel.** The Attached to row
    is a choice of every optics in the layout, plus free: picking an
    optics *seats the mount on it at the model's designed position* -
    a mount is built around its optic, so where it belongs on the
    host is unique and the library's to say, carried by the
    convention that a model's local origin is the point that stands
    at the host's substrate centre. Picking free detaches it where it
    is. While attached, Offset rows adjust where it stands on the
    host - the one thing about an attached body's place that is its
    own to edit; the derived position stays derived, so nothing can
    fall out of step.
  - **A breadboard resizes by its corners.** A body a builder made
    carries its parameters (`Mechanics.params`), and a selected,
    resizable one shows four corner handles: dragging one cuts the
    board to a new size with the opposite corner standing still, and
    Python *re-drills* the grid from the parameters - same pitch,
    same holes - rather than scaling the drawing. Width and Height
    rows in the panel edit the same thing in numbers, and
    `Mechanics.resize()` from Python. The parameters travel through
    save, undo, copy and the model library; a hand-drawn body has
    none, and says so when asked to resize.
  - **The screw holes are snap points.** The measuring tool takes
    them like any marked point, and a dragged mirror or lens lands
    its anchor point on the nearest hole when it comes close (a small
    reach, well under the grid pitch; hold Alt to ride free). The
    holes are where a bench actually puts things.
  - Hardware names are drawn only when the new `drawMechanicsNames`
    option asks (off by default): the hardware is background, and a
    name across a breadboard labels what nobody needed named.
  - Editing a mechanics does not invalidate the trace: the picture
    changes, the beams did not move.
  - **A mechanics can be attached to an optics** (`attached_to`),
    which is what a mirror mount is. An attached body has no pose of
    its own: `center` and `rotationAngle` are derived on every read
    from the host's pose and an offset in the host's frame, so moving
    the mirror moves the mount with no callback to miss and no stored
    copy to go stale. The price is the meaning of the word: an
    attached body cannot be moved on its own - its pose rows go
    read-only, a drag on it pans, and a `move` through the protocol
    is refused with the reason. `detach()` (or
    `set attached_to: null`) bakes the derived pose in and frees it;
    `attach()` with no offset seats the body at its designed position
    (`keep_pose=True` pins it where it stands instead). A saved
    layout carries the host's *name* and the offset - no pose - and
    loading joins the two back up; removing an optics with hardware
    attached is refused until the hardware is detached or removed.
  - Edit operations: `add` with `type: 'Mechanics'` (shapes arrive
    serialized, as the layout file carries them), `move` (`center`),
    `rotate` (`rotationAngle`), `set`, `remove`, `rename`. The scene
    gains a `mechanics` channel; corners and centre join the snap
    points, so the measuring tool reaches the hardware.
- **A model library for hardware**, in `gtrace.mechanics`. The
  definitions are data - the same serialized shapes a saved layout
  carries - under one name each: `register_model(name, source)` puts a
  shape you settled on into the library by value, `models()` lists
  what there is, and `from_model(name, ...)` builds a `Mechanics`
  carrying the model name as its label. The label stays a label: a
  layout saves the shapes themselves, and a library that has moved on
  changes nothing until `layout.relink_mechanics()` is explicitly
  asked to redraw the labelled bodies from the current definitions
  (pose, attachment and layer stay put; unlabelled bodies and models
  the library does not know are left alone).
- Two parametric builders behind it: `breadboard(width, height, ...)`
  - a plate with the standard symmetric 25 mm hole grid - and
  `mirror_mount(scale=1, knobs=True)`, a one-inch kinematic mount
  drawn from a measured Polaris-style top view: front plate, the
  adjustment gap with the two adjuster tips showing, back plate, and
  the knobs on their stems. Its local origin is the substrate centre
  of the mounted optic, 3 mm behind the front face, so `attached_to`
  with no offset seats it with a 6 mm optic flush with that face.
  `mirror_mount_2in()` is its two-inch counterpart, measured from
  the Thorlabs KA2A drawing rather than scaled: its own plate widths
  and adjuster spacing, the same 3.2 gap, and the origin 3.95 behind
  the front face, where the drawing's optic pocket centres a
  standard 12.7 thick optic. The library is seeded with stock built
  from them (`BB3030`, `BB4530`, `BB6045`, `MOUNT-25`, `MOUNT-50`);
  further vendor models are yours to register from measured
  footprints.
- `shape_from_dict` in `gtrace.draw.serialize`: the inverse of
  `shape_to_dict`, which loading a mechanics needs.
- `verify_mechanics.py` (258 checks) and `verify_mech_browser.py` (72).
  The suite is 21 files and 4168 checks.

### Fixed

- `gtrace.draw.serialize.UnknownShapeError` derived from
  `BaseException`, so it sailed through every `except Exception`
  between the serializer and the user - including the one that shows a
  failure in the widget. The copies of this class in `renderer.py` and
  `draw.py` were fixed in 0.4.0; this one had been missed.

- **Source beams are editable from the viewer.** Each registered source
  is drawn as a small laser at the point its light comes from; clicking
  it opens a properties panel, dragging it moves it, and Shift-dragging
  turns it about that point. `+ Source` adds one. Until now the layout
  could be edited in every part except the one the light came from.
  - The box is what makes a source visible at all. A source is traced
    from a *copy* of itself, so its own beam is in the drawing looking
    exactly like the beams the trace made from it, and nothing said
    which was which. It is sized in screen pixels and stays that size
    as the view is zoomed: a layout runs from a bench to a kilometre,
    so a body sized in metres would be a dot on one and would cover the
    other. It is not exported to DXF.
  - Once the drawn beam is wider than the aperture it comes out of,
    the box grows with the view instead - a fixed-size nose past that
    point would be a picture of something that cannot happen. The
    crossing is exactly where the two meet, so nothing jumps.
  - The panel edits the beam as a **waist** - its size and where it
    sits - rather than as a q-parameter, which is what a laser is
    specified by and what mode matching is done in terms of. The
    conversion is Python's, next to `GaussianBeam.waist()`.
- `q_from_waist`, `rayleigh_range` and `source_waist` in
  `gtrace.layout`, which convert between the two descriptions.
- Scene channels `sources` and `rules`. The first is what tells a
  source beam from a traced one; the second carries the tracing rules,
  which decide how much of the picture there is.
- Edit operations reaching a source: `move` (`pos`), `rotate`
  (`dirAngle` / `dirVect`), `set` (`EDITABLE_SOURCE_ATTRS`, including
  the derived `waist_size_x` / `waist_pos_x` and their y counterparts),
  `add` with `type: 'Source'`, `remove` and `rename`.
- A **Tracing rules** panel for `order`, `power_threshold` and
  `open_beam_length`. These were editable through the protocol from the
  start and had nothing to reach them.
- `verify_source.py` (267 checks) and `verify_source_browser.py` (99).
  The suite is 19 files and 3804 checks.
- Japanese documentation. The handwritten pages and the prose cells
  of both tutorial notebooks carry gettext translations under
  `docs/source/locale/ja/`, built with `-D language=ja` and meant to
  be served by Read the Docs as a linked translation project; a
  paragraph without a translation falls back to the English original,
  so the two languages cannot drift apart silently. Code cells, their
  outputs and the API reference stay English.

### Changed

- The notebook widget starts with its **drawing as tall as it is wide**
  rather than at a fixed 520 pixels, capped to 70% of the window and
  never below 520. The side panel is a fixed strip of the width and is
  a column of numbers rather than part of the picture, so the height
  comes from the width less that strip - or from the whole width below
  the breakpoint where the panel stacks underneath instead. A cell
  output is a letterbox, and the first thing anyone did with the old
  default was drag it taller. `height` still takes a number, and the
  traitlet now defaults to 0, meaning "work one out"; dragging the grip
  writes a height back, which settles it.
- The **Diameter** and **Thickness** rows of the optics panel are in
  millimetres, as the focal length already was. That is how a blank is
  ordered and spoken of, and a 1 inch mirror reading `0.0254` is
  arithmetic rather than a specification. Where an element *stands*
  stays in metres — a distance across the bench rather than a dimension
  of the part — and the edit messages are in metres as before.
- The **Anchor** choices read `HR center` and `substrate center`,
  rather than `HR apex` and `substrate centre`.
- One add button per kind of thing — `+ Mirror`, `+ Lens`, `+ Source` —
  with the spherical and cylindrical variants behind the first two,
  instead of a button apiece. A cylindrical mirror is a mirror; five
  buttons read as five unrelated things, and wrapped in a side bar that
  narrow.
- The tutorial page introduces each of its two notebooks in a
  subsection of its own — what it covers and where its reference
  pages are — instead of listing the second one without a word.
- Running the tutorial no longer means cloning the repository: both
  notebooks are self-contained, so the tutorial page now links
  straight to the files on GitHub, and `pip install
  "gtrace[notebook]"` plus the download is the whole setup.

## 0.4.0 — 2026-08-09

### Added

- `Lens`, ordered by focal length rather than by two radii. `shape` is
  spelt as a catalogue spells it (`'plano-convex'`, `'meniscus'`, ...),
  the curvatures are solved for as a *thick* lens, and assigning to `f`
  re-solves both of them, keeping the shape and leaving the lens where
  it is. `LensGeometryError` distinguishes seventeen ways a blank cannot
  be ground to what was asked for, and says what would be needed.
- `CyLens`, a cylindrical lens: ordered by focal length exactly like a
  `Lens` — same shapes, same thick-lens solve, same refusals — and
  shaped like a `CyMirror`, so the focal length lands in the plane
  `curve_direction` names and the other plane is a plain window. A
  `+ CyLens` button in the viewer goes with it.
- `Optics.anchor_point` names the point an optics is held by: the apex
  of the front face, or the middle of the substrate. It is what stays
  put when a curvature changes, and what the optics turns about when
  `normAngleHR` or `normVectHR` is assigned. `Mirror` and `CyMirror`
  default to the apex, so that sweeping a telescope's radii does not
  walk the beam spot off it and steering it pivots the reflection
  point; `Lens` and `CyLens` default to the middle, since the beam
  goes through. `rotate()` pivots the anchor point by default - for
  every mirror that is the HR apex, exactly what it has always done -
  with `center=True` for the middle of the substrate and an array for
  a given point.
- `Optics.get_corners()`, `Optics.contains_segment()` and
  `Optics.contains_point()` answer where the substrate is, rather than
  what a beam meeting it does.
- **Dimensions.** A distance measured between two points is registered
  on the layout beside the optics, saved with it and taken back by undo.
  Where the whole span runs inside one substrate the optical distance is
  reported too. See `OpticalLayout.dimensions` and `Dimension`.
- `inch` in `gtrace.unit`.
- A worked mode-matching exercise in the tutorial
  (`docs/source/tutorial/modematching.ipynb`): coupling a laser into a
  Fabry-Perot cavity with two catalogue lenses, searched as thin
  lenses and then verified and refined on the traced bench.
- The tracing docs now say how the stray order is counted — what
  raises it, that the counter resets when a beam crosses to the next
  element, and the four attributes that govern it (`HRtransmissive`,
  `HRreflective`, `max_stray_order`, `term_on_HR`) — in a subsection
  of *Non-sequential trace*, referenced from the lens and layout
  pages.
- `OpticalLayout.export_dxf()`, the companion of `render_html()`. It
  draws the dimensions on a layer of their own, which CAD can switch
  off; `dimensions=False` leaves them out. `draw_dimensions()` does the
  drawing and is callable on its own.
- `Mirror.HRreflective` says whether the HR face is meant to reflect,
  as `HRtransmissive` says whether it is meant to transmit. With it
  False every reflection at the HR face — from outside or from inside
  the substrate — counts one order of stray, as reflections at the AR
  face always have. `Lens` and `CyLens` default it False: their faces
  are meant to pass, so what reflects off them is a ghost, and giving
  a lens a real reflectivity to chase its ghosts now yields ghosts
  carrying the order they deserve rather than order 0. Mirrors, beam
  splitters and input test masses keep the default True and trace as
  before. Shown in the viewer's Tracing group and saved with the
  layout.
- A **Drawing (DXF)** panel in the viewer, with a file name of its own
  beside the **Optical layout (JSON)** one, and an `export` operation
  in the edit protocol. The two are kept apart because the layout is
  the model and the DXF is a picture of it: pressing Load on a
  drawing could only be a mistake.

### Added — viewer

- **Clicking through an element to the beams under it.** A beam ends
  on the surface of the element it hits, inside the element's grab
  circle, and the element always won the click — so the beam
  parameters at a surface could not be read at all. Clicking the same
  spot again now steps from the element into the bundle of beams under
  the click and back around: the same gesture that already walks a
  bundle of overlapping beams.
- **Measure.** Two clicks say what is being measured and a third says
  where the dimension line is drawn; extension lines carry the ends out
  to it. Ends snap to the corners and faces of the elements, to the
  middle of a substrate and to the ends of the beams. Works in the
  written HTML file too, which measures without a kernel — with no
  optical distance there, since that is a question about surfaces.
- **Redo**, beside undo, on the button, Ctrl+Shift+Z and Ctrl+Y.
- **Lenses**: a `+ Lens` button, a focal-length row in millimetres, and
  an Anchor row naming the point the element is held by.
- **Placing an element against a beam**: Ctrl+drag drops it square
  across a beam and centred on it; an *Along beam* / *Move by* pair
  slides it along that beam by a typed distance; Ctrl+click picks which
  beam, stepping through an overlapping bundle.
- **Room to work**: the viewer can be dragged taller by its bottom edge,
  and the side panel folds away to give the drawing the whole width.

### Fixed

- The notebook widget opened with the whole drawing shrunk to a dot.
  anywidget calls `render()` before the output area has been laid out,
  so the element measured zero and the scale was set from that.
- **Changed results.** A curved AR surface sat one sagitta behind where
  the sides of the substrate end, so the body did not close. Invisible
  while the AR is flat, which it is for nearly every mirror, and first
  order as soon as it is not.
- **Changed results.** Changing a curvature did not carry into the rest
  of the substrate: `ARcenter` and `center` were left with the old
  sagitta. On the KAGRA layout this moves the beams inside the substrate
  of `MMT1` by up to 14.68 µm; the main beams move by less than 1e-10
  mm.
- **Changed results.** `CyMirror` was cylindrical in shape only. Its two
  hit methods kept the cross-section the trace sees and the optical
  power of the surface in a single variable, and with the curvature out
  of the plane of the trace that variable has to be zero — the section
  really is a straight line — so the power went with it. What came out
  was a mirror that focused in *both* planes when `curve_direction` was
  `'h'`, and in *neither* when it was `'v'`. The one function that knew
  the difference, `cyl_refl_defl_angle`, was never called by anything.
  A `CyMirror` now focuses in the plane it is curved in and leaves the
  other alone, per Siegman Table 15.1. Nothing else in gtrace uses
  `CyMirror`, so no other result moves; the KAGRA layout is unaffected.
- **Changed results.** The transmission matrices in
  `cyl_refl_defl_angle` were a copy of the spherical ones and had never
  been told which plane was which, so both planes were given the
  curvature. Only in *reflection* is the uncurved plane the identity: a
  tilted flat interface still scales the beam in the plane of incidence
  and still carries the index change, and it now keeps both while losing
  the power. This was dead code before the previous entry.
- The `meniscus` example in the `Lens` docstring raised rather than
  building a lens.
- `renderer.UnknownShapeError` and `draw.NumberOfElementError` derived
  from `BaseException`, so they walked straight through every
  `except Exception` between the renderer and the top — including the
  one the notebook widget uses to turn a failure into a message the
  user can see. Both also spelt their constructor `__initi__`, so the
  message never reached the exception.
- The widget said nothing the second time it was told to say the same
  thing: a traitlet notifies on a *change* of value, so saving to the
  same path twice confirmed it once, and the same refusal twice was
  reported once.
- **Changed results.** A ghost reflecting off the HR from inside the
  substrate lost `Refl_HR` twice: once before the power was checked
  against the threshold, and once more after it passed. Every ghost
  entering through the HR and making more than one round trip came
  out weak by one factor of `Refl_HR` per internal HR bounce — about
  1% per bounce on a mirror, a factor of 2 on a 50/50 beam splitter —
  or was cut by the power threshold and never came out at all. Beams
  entering through the AR were counted correctly. `Mirror` and
  `CyMirror` both, whose loop this is.

## 0.3.1 — 2026-08-03

### Fixed

- `setup.py` read `README.md` without an encoding, so on a Japanese
  Windows it was decoded as cp932 and installing from source failed with
  `UnicodeDecodeError`. The old README was pure ASCII, so this only
  appeared once the README gained a non-ASCII character — at which point
  neither `pip install .` nor a build from the sdist worked at all on
  such a machine. This release exists for that fix.

### Changed

- The tutorial is now one notebook, `docs/source/tutorial/`, covering
  the viewer as well as the tracing. The duplicate under `Manuals/` is
  gone, and the DXF files it produces were regenerated.
- Installation is documented in both the README and the introduction,
  including what the viewer needs.

## 0.3.0 — 2026-08-03

The release that added the browser viewer.

### Added

- **A self-contained HTML viewer.** `render_html()` writes one file with
  the scene, the viewer code and the styling inlined — no server and
  nothing to install. Unlike a DXF it carries the physics: clicking
  anywhere along a beam, not only at a vertex, reports the radius, the
  wavefront ROC, the complex q, the waist and its distance, the Rayleigh
  range, the Gouy phase and the accumulated optical path at that point,
  separately in x and y.
- **A notebook widget.** `widget()` shows the same viewer as a cell
  output. It needs `anywidget`, which is an extra: `pip install
  gtrace[notebook]`.
- **`OpticalLayout`**, a container for the optics, the sources and the
  tracing rules. It holds the registered objects *by reference*, so an
  element moved in the viewer is the object the user's own variable
  names.
- **Editing from the viewer.** Elements can be dragged, rotated,
  added, removed, renamed and edited field by field; each interaction is
  one plain-dict message applied by `OpticalLayout.apply_edit`, so
  anything the GUI can do can be done from a cell.
- Save and load the layout as JSON, from the viewer or from Python.
  Loading in place keeps the identity of the elements it recognises.
- `show()`, which picks the widget inside a Jupyter kernel with
  anywidget installed and writes an HTML file otherwise.

### Fixed

- **Changed results.** The centre of a substrate was computed by two
  different formulas depending on which handler ran, one of which
  averaged a chord centre with a point on the far arc. The centre is the
  midpoint of the two chord planes.
- **Changed results.** Changing `diameter` did not recompute the
  sagittae, which depend on the aperture as well as on the radius. A
  curved mirror widened from 10 cm to 20 cm at R = 2 m kept a sagitta of
  0.625 mm where 2.50 mm was right — about 1.9 mm of error in the
  position of the HR surface.
- `GaussianBeam.R` used the wrong sign convention for the propagation
  distance.
- A recursion error in the non-sequential trace, where the stray order
  was not reset for a new beam.
- `drawOptSys` referred to a name that did not exist.

### Changed

- `python_requires` is now `>=3.9`, which is what numpy 2.x needs and
  what the code was already assuming.
- The licence is declared as `BSD-2-Clause`. The `LICENSE` is two
  clauses and carries no advertising clause, so it was never 3-Clause.

## 0.2.4 and earlier

Not recorded here. See the git history.
