# Changelog

Notable changes to gtrace. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and gtrace aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries marked **Changed results** alter the numbers gtrace produces for
an unchanged input. They are corrections rather than regressions, but a
system traced with an earlier version will not reproduce bit for bit
across them.

## 0.7.0 — 2026-08-21

### Changed

- **Tracing a layout takes about a sixteenth of the time it did.**
  Tracing the KAGRA interferometer with ghost beams - 45 elements, 482
  beams - went from 1.58 s to 0.099 s, and the tutorial layout from
  0.076 s to 0.005 s. Nothing about how gtrace is used has changed.
  Four things were done, in the order they matter:

  - **`isHit` asks whether a beam comes near an element before testing
    its faces.** It used to intersect all four faces of every element
    for every beam. On the KAGRA layout it is called 12,825 times and
    271 of those are hits; a single test against the circle that holds
    the substrate rejects 95% of the rest. On its own this is a
    four-fold speed-up, and it is the one that matters more the more
    elements a layout has, since the work grows as beams x elements x
    four faces.

  - **`get_side_info` is worked out once per element rather than once
    per beam per element.** It takes no arguments and follows from the
    shape and the pose, and a KAGRA trace asked for the same answer
    12,842 times. It is now kept until the shape or the pose changes.
    **Do not write into what it returns**: the same list and the same
    arrays are handed to every caller.

  - **`GaussianBeam.copy()` copies the beam's dictionary** instead of
    running `copy.deepcopy` over a traits object. Tracing a layout
    copies a beam at every surface it meets, and each copy took 121
    microseconds.
    This also removes the garbage the deepcopy left behind: the same
    trace run three times used to take 1.09, 1.09 and 3.81 seconds, and
    now takes the same time every run.

  - **The geometry is worked out on plain floats** rather than on
    two-element numpy arrays. The intersection routines do about twenty
    floating point operations, and were spending their time in call
    overhead: a two-element `np.linalg.norm` costs a microsecond and a
    2x2 `np.linalg.solve` calls into LAPACK. gtrace no longer calls
    LAPACK while tracing.

- **Changed results.** The last digits move. Two of the four changes
  above are exact - a trace of the KAGRA layout comes out bit for bit
  identical after the culling and the caching - and two of them reorder
  floating point arithmetic:

  `copy()` no longer re-derives the beam it is copying. The old one
  went through the traits machinery, which recomputed `q` from `qx` and
  `qy`, re-normalized `dirVect` and rebuilt it from `dirAngle` on the
  way. The copy now carries the values the beam actually has.

  The geometry no longer solves a 2x2 system with a solver, but with
  Cramer's rule, which rounds differently. Held against the old
  routines over 60,000 random cases the two agree to within one part in
  1e13 of the larger coordinate, and the worst cases are all at an
  inverse ROC of 1/7000 - the KAGRA arm mirrors - where the centre of
  the arc is 7000 m away and the intersection is the difference of two
  quantities that size.

  Across the whole KAGRA layout every optic position, every beam name,
  every beam count and every entity count is unchanged, and eight of
  the nine exported drawings move by at most 2.2e-10 mm on coordinates
  of 20 m. The ninth, the input optics with its ghost beams, moves by
  up to 3.2e-4 mm - and moves by 0.0275 mm, eighty-five times further,
  under the two exact changes alone. That drawing turns a change in the
  last digit into a tenth of a micron whichever direction the last
  digit goes, so what it measures is its own conditioning rather than
  the size of anything done here.

- **`GaussianBeam.copy()` was never a deep copy**, whatever its
  docstring said. It duplicated the traits and left plain attributes
  pointing at the originals, so a list attached to a beam was the same
  list on the copy. That is still what happens - arrays duplicated,
  everything else shared - and the docstring now says so.

### Added

- **`tests/bench_trace.py`**, which measures how long a trace takes and
  prints the beam count and the sum of the optical path lengths, so the
  same command
  that reports the time also reports whether the physics moved. It
  takes the tutorial layout or a pickled one.

- **`tests/gui/verify_geometry.py`** (439 checks), covering the
  geometry kernel now that it is written on floats: the two
  intersection routines against a circle and a line worked out in the
  suite rather than taken from them, `vector_rotation_2D` over both of
  its paths including the array shapes the drawing code sends, and
  `_surface_matrices` giving nan transmission past the critical angle,
  which is how a caller tells total internal reflection from an
  ordinary refraction.

### Documentation

- **The manual was rewritten.** Every page now puts the usage first and
  the internals after it, so a first reader can follow how gtrace is
  used before meeting the model behind it. Introduction, Basic
  concepts, Propagation, Optical layouts, The viewer and the Tutorial
  were all reworked, and the tutorial notebook was reordered along the
  same line: build a bench, work on it in the viewer, save and export
  it, and only then read about the coordinates, the objects, the ghost
  beams and the edit messages.

- **The edit protocol has a page of its own**, `editing.rst`, split out
  of Optical layouts. It documents every message `apply_edit` accepts,
  what each one may touch, and the ten channels `scene_dict` adds to a
  scene.

- **The prose was shortened for readers whose first language is not
  English.** Long sentences were split, dash parentheticals were
  removed, and vague words were replaced with the names of the things
  they stood for.

- **The Japanese translation was redone** for the rewritten pages.

- Three statements that did not match the code were corrected: the
  scene channel list said nine channels and omitted `assemblies`; the
  `+ Shape` and library sections did not mention the pedestal and the
  clamping fork that 0.6.0 added; and the model library listing in the
  tutorial notebook was still the seven models of 0.5.0.

- Three section headings in the mode matching notebook were written at
  level 1, which produced extra top-level sections on the rendered
  page. They are level 2 now.

- **`order` was described wrongly on three pages.** Introduction,
  Optical layouts and the tutorial notebook all called it the number of
  internal reflections followed when a beam enters a substrate. It is
  the number of ghost reflections a beam may go through in total: every
  beam carries the count, and the count is not reset when the beam
  leaves one element for the next. The three now say so and point at
  the stray order section of Propagation.

- **The tutorial named the wrong beams.** It said `s` was a reflection
  and `t` a transmission, and did not mention `r` at all, which
  contradicted both the figure beside it and the chapter further down.
  `r` leaves back on the side the beam came from, `s` travels inside
  the substrate, and `t` comes out of the far side. The same section
  gave `b0:M1t1` as a beam name; a beam is named `M1:t1`, after the
  element that produced it and the beam it left as.

- **A printed label in the tutorial said the opposite of its value.**
  `print('HR counts as stray :', M2.HRreflective)` printed `True` for a
  mirror whose HR reflection is the main beam and is *not* counted as
  stray. The label reads `HR meant to reflect` now.

- **The mode matching notebook disagreed with itself about the laser.**
  The opening table quoted a waist diameter of 0.2 mm while the code
  used 0.2 mm as the radius, which is what the printed output reports.
  The table says radius.

- **The prerequisites of the notebooks are written down.** The mode
  matching notebook needs Matplotlib, joblib and SciPy on top of
  gtrace, and said so only at the import lines. The Tutorial page and
  the notebook now give the `pip install` line, and the parallel scan
  says it can be skipped.

- **Code examples that could not be run were completed.** The
  `hitFromHR` and `non_seq_trace` examples in Propagation used a mirror
  and a source that were never defined; they now build both and show
  the output they produce. Basic concepts uses `GaussianBeam` and
  `q_from_waist` without importing them, so the imports were added to
  the first block of the page. The edit protocol shows what setting one
  waist does to the other three.

- The landing page carried the `sphinx-quickstart` template comment and
  nothing but a table of contents. It says what gtrace does, how to
  install it and where to start.

- **The docstrings said the same wrong thing about `order`.**
  `TraceRules`, `non_seq_trace` and the ten `hit` / `hitFromHR` /
  `hitFromAR` docstrings described it as the number of internal
  reflections computed at an element. It is the largest `stray_order` a
  produced beam may have, and the count is carried over from the
  incident beam. These docstrings are published as the API reference,
  so the manual repeated the error in two places at once.

- **The focal length of a `'v'` cylindrical mirror was written
  ambiguously.** `R/2\cos\theta` renders as *R*/2·cos θ, which is not
  what is meant. It is now `R/(2\cos\theta)`.

- **The Japanese translation was brought back in line with the English**
  and reviewed as whole pages rather than message by message. Wording
  that was a word-for-word transposition of the English was rewritten:
  metaphors that had been carried over literally (「予算」「関門」
  「焼き付ける」「ページを閉じるまでの命」), 無生物主語, and phrases
  whose meaning could not be recovered without the English beside them.
  Three terms were unified across the manual: 光源ビーム for a source
  beam, ウェスト for a waist, and ウェスト半径 for a waist size, which
  one page had as ウェスト径 although the value is a radius.

## 0.6.0 — 2026-08-13

### Added

- **A cavity mirror can be looked through.** `term_on_HR` stops a beam
  at the surface so that two facing high reflectors do not pass it back
  and forth for ever, and until now it stopped everything: nothing was
  computed at that element at all, so the beam that goes *through* an
  input coupler - which is what a detector behind one sees - was lost
  with the reflection.

  `term_on_HR_transmits` says what stopping means. False, the default
  and what `term_on_HR` has always done, ends the beam at the surface.
  True drops only the external reflection, the one beam that can come
  back at the power a cavity needs, and lets the element be hit as
  usual otherwise: the substrate is crossed and drawn, the beam leaves
  through the far face, and the ghosts inside are unfolded. All of that
  is counted and capped by `order`, `max_stray_order` and the power
  threshold exactly as anywhere else, because it goes through `hit()`
  by the ordinary route.

  A ghost that leaves through the HR from inside the substrate is not
  dropped. It is a round trip weaker, it costs a stray order, and the
  budget is what ends it - the same treatment every other ghost gets.
  The gate is unchanged too: a beam arriving above `term_on_HR_order`
  is not terminated and reflects as it always did.

  The suppression lives in `non_seq_trace`, not in `hitFromHR`.
  `hitFromHR` is the sequential interface, and code that calls it
  directly - as the KAGRA layouts do - asks for `r1` by name; taking it
  away there would break those callers, which is how `OMC-Layout-O4`
  came to fail in 0.4.0. Nothing about the flag changes a trace that
  leaves it alone: the KAGRA layouts are byte for byte identical across
  it.

  `verify_stray.py` grows 16 checks on what is dropped and what is not,
  the gate, `order` and `max_stray_order` still bounding what survives,
  and the flag surviving a save and load. A file written before this
  loads with it off.

- **An element and the parts that hold it, in one call.** What is
  bolted to a bench is not a mirror but a mirror in a mount, on a
  pedestal, held down by a fork - four objects and three joints, which
  was four steps of undo in the viewer and three offsets to look up.

  ```python
  from gtrace.layout import assembly, assembly_kinds

  assembly_kinds()          # MIRROR-1IN, MIRROR-2IN, LENS-1IN, LENS-2IN
  layout.add_assembly('MIRROR-2IN', center=[0.3, 0.1], angle=deg2rad(45))
  layout.add_assembly('LENS-1IN', center=[0.6, 0.1], f=150*mm)
  ```

  What comes back is `(optics, bodies)` - the elements and the parts
  that hold them, split - each attached to the one below - the mount at the model's
  designed position, the pedestal in the hole the mount is bolted down
  through, the fork round the pedestal with its turn free. **The
  element is the thing to move**: everything else derives its pose from
  it. `add_assembly` registers the lot and fills the names in, each
  piece taking the first number free for its own kind, so a second
  two-inch mirror is `M2` held by `MT2` on `P2` in `FK2`.

  It is split because that is the division everything downstream
  makes: a layout registers the two by different doors, and the trace
  sees only the first. Handing back one flat list left every caller
  sorting it out again by class.

  `mirror_assembly` and `lens_assembly` are what the kinds are made of
  and take the models to build the parts from; `None` leaves a piece
  out. `mount_offset` says where the mount is really bolted, in the
  optic's own frame with x along the face normal - the two-inch kind
  sits its mount **5 mm further back** than the drawing's designed
  position, a bench measurement rather than something the model knows. `+ Assembly` in the viewer adds one, as **one message and so one
  step of undo**.

  An assembly is a builder rather than a model on the library shelf,
  and cannot be one: a model holds shapes, and the first piece of every
  assembly is an element. A saved layout carries what was built - the
  optics, the bodies and the attachments - like a beam dump's.

  `verify_assembly_parts.py` arrives with 66 checks: the stack and what
  holds what, the pedestal landing on the point the mount's model names
  for its hole, moving the element carrying everything, the names, one
  message and one undo, saving and loading, a piece left out, and what
  the protocol refuses. `verify_mech_browser.py` grows 24 driving the
  real menu.

- **A part added from the viewer is named for what it is.** Everything
  the `+ Mechanics` menu put down was called `H1`, `H2`, whichever
  model it came from - a name that says nothing about what is standing
  there. A model now says what its parts are called, and the stock
  says it:

  | model | parts |
  |---|---|
  | `MOUNT-25`, `MOUNT-50` | `MT1`, `MT2` |
  | `PEDESTAL-25` | `P1`, `P2` |
  | `FORK-125` | `FK1`, `FK2` |
  | `HOLDER-25`, `HOLDER-50` | `HLD1`, `HLD2` |
  | `BB3030`, `BBR30`, … | `BB1`, `BB2` |

  `PD` is deliberately nobody's prefix: a photodetector is what that
  reads as, which is why a pedestal is `P`.

  It is `register_model(..., prefix='MT')`, so a part of your own says
  it in the same place the stock does - through the argument, or
  through the **Save to library** panel, which grows a field for it.
  It travels with the model through `save_models` and `load_models`; a
  library written before this loads with none, and a model that says
  nothing still gives `H1`. `model_prefix(name)` reads it back, and the
  `mechlib` channel carries it so a front end has the name to hand.

  A shape put down with `+ Shape` is named for the shape it is:
  `CIRC1`, `RECT1`, `LINE1`, `POLY1`, `ARC1`, `TEXT1`.

  Nothing already in a layout is renamed - a name is a name - and
  nothing about how names are checked has changed: they share the one
  namespace, and the first free number is what is offered.

- **A body that is one shape drawn by hand is edited by that shape,
  from the layout.** `+ Shape` put one down and then the layout could
  only move it and turn it: a radius, a width, the ends of a line were
  reachable through `Mechanics.edit()` alone, which is another viewer
  and a cell in a notebook. A wall that can be put down and not resized
  is not worth putting down.

  The panel now shows the shape's own rows under the pose rows - the
  same rows the shape editor shows, in the frame the shape is written
  in - and its grips stand on the drawing. A drag on one is worked out
  in the body's own frame, so a corner lands where it was let go
  however the body is turned, and it is `shapeHandleAttrs` that works
  it out: the same function answering the same question in the other
  viewer.

  The body carries its shape in the `mechanics` channel, and
  `{'op': 'set', 'target': ..., 'attrs': {'shape': {...}}}` sets it.
  What has no one shape refuses it rather than guessing: a part off the
  library shelf is cut to size with `width` and `height`, and one of
  several shapes is edited where there is a list to pick from.

  The rules about what can be drawn moved to
  `gtrace.draw.serialize.build_shape`, which both the shape editor and
  the layout now come through: a rectangle of no width was refused in
  one of them and would have been let through the other.

- **A body drawn as a straight line can be clicked.** Its outline has
  no area, and a point is never inside one of those, so a body whose
  one shape was a horizontal or vertical line could not be selected at
  all. A click that passes near the outline now counts, at the same
  reach a shape is taken hold of by in the editor - for a body with no
  inside only, so a click beside a breadboard still misses it.

  `verify_mech_shape_browser.py` arrives with 46 checks driving real
  drags: the rows a circle, a rectangle and a line offer, where the
  grips stand on a body that is turned, every message fed to a real
  layout, and what offers nothing - a breadboard, a body of several
  shapes, an attached body.

- **A shape can be put down from the viewer.** `+ Shape` sits next to
  `+ Mechanics` and opens on the six drawing primitives - rectangle,
  circle, line, polyline, arc and text - putting one down at the centre
  of the view as a body of that one shape. A tank wall, an aperture,
  the edge of a table, a note on the drawing: things that were a cell
  of `Mechanics(shapes=[draw.Circle(...)])` until now. What comes down
  is moved, turned, dimensioned and edited like any other body, and
  `Mechanics.edit()` opens it for a second shape.

  **What it puts down is sized to the view.** The sizes are a bench's -
  a 20 mm plate, a 5 mm hole - which is right over a part and invisible
  over three kilometres of interferometer, so they are scaled by how
  wide the view is: `+ Circle` leaves something that can be seen and
  taken hold of at any zoom, and the number wanted is typed in
  afterwards.

  The shapes themselves are Python's. `NEW_SHAPES` - what a shape of
  each kind looks like when it is first put down - moved from
  `gtrace.draw.viewer.editor` to `gtrace.draw.serialize`, where the
  rest of what a shape dict is lives, and rides in the scene as the new
  `newshapes` channel; the page scales what it is given rather than
  holding its own answer to what a new circle is. It still reads from
  `editor` under the name it had. No message is new: the body arrives
  as the `add` a layout already took, with its shapes spelled out.

  `verify_mech_browser.py` grows 32 checks driving the real menu: every
  kind sends a body of that one shape at the centre of the view, each
  message is fed to `apply_edit`, what is sent is the scene's own new
  shape scaled by the view, and the same button at half the zoom puts
  down twice as much.

- **A rectangle can be turned.** `draw.Rectangle` takes an `angle` and
  the `pivot` that angle is taken about, so a plate set on a bench at
  30 degrees is a rectangle rather than a polyline that used to be one
  - it keeps its width, its height and the four numbers a panel edits:

  ```python
  draw.Rectangle([0, 0], 0.2, 0.1, angle=deg2rad(30))
  draw.Rectangle([0, 0], 0.2, 0.1, angle=deg2rad(30), pivot=[0, 0])
  ```

  The pivot is kept rather than folded into the corner, so setting the
  angle again turns the rectangle about the same point; it travels
  with the shape when it is carried. A pivot of `None` - the default -
  is the middle of the rectangle, worked out when it is asked for
  rather than written down, which is what lets a rectangle that is
  moved keep turning about itself. `angle_in_rad=False` takes degrees,
  as on `Arc` and `Text`.

  `corners()` is where the shape says where it is, and everything that
  draws, bounds, picks, exports or drags one now asks it: the bounding
  box of a body, the DXF (an LWPOLYLINE through the corners it really
  has, as it always was), the snap points, the shape editor, and the
  page - where a turned rectangle is an SVG rect with a rotate about
  its pivot, and its corners, its grips and what a click falls inside
  all follow the turn.

  Nothing square to the axes moved: `angle` defaults to 0, every part
  gtrace ships is drawn exactly as it was, and a layout written before
  this loads with no angle and no pivot. A rectangle carried by a body
  that is *itself* turned still comes out as the closed polyline of its
  corners, as before - the body's turn is not written into the shape.

  Three suites arrive with it: `verify_rect.py` (78 checks, the class
  and everything downstream of it), `verify_rect.js` (441, the page
  and gtrace agreeing about where the corners are, and what a drag on
  one means) and `verify_rect_browser.py` (11, where the browser
  actually paints it, read back through `getScreenCTM`).

- **`term_on_HR_order` and `term_on_HR_transmits` are constructor
  arguments**, on `Mirror`, `CyMirror`, `Lens` and `CyLens`, next to
  `term_on_HR` which always was one. A cavity mirror can be built as
  one:

  ```python
  ETM = Mirror(name='ETM', term_on_HR=True, term_on_HR_order=3,
               term_on_HR_transmits=True)
  ```

  Setting them afterwards still works and means the same thing. The
  defaults are what the attributes always were, 0 and False.

### Changed

- **The two-inch mount's adjuster knobs are 2.1 inch apart.** They were
  drawn 35.6 mm apart, which is what the KA2A drawing dimensions, and
  that is closer together than they are on the mount. `MOUNT-50` and
  everything built from it - `mirror_mount_2in()`, the `MIRROR-2IN`
  assembly - are wider across the back for it. Every other dimension
  is still the drawing's.

  A layout saved before this keeps the shapes it was saved with, since
  a layout carries them by value; `relink_mechanics('MOUNT-50')` is how
  it picks the new drawing up.

- **`beam_dump()` returns `(optics, bodies)`** rather than one flat
  list of three, so that every builder here hands back the same shape.
  `[face1, face2, housing]` becomes `([face1, face2], [housing])`:

  ```python
  (f1, f2), (box,) = beam_dump(name='BD1')
  faces, bodies = layout.add_beam_dump(name='BD1')
  ```

  The split is the division everything downstream makes - a layout
  registers elements and bodies by different doors, and the trace sees
  only the elements - so it is made once, where the pieces are built,
  instead of by every caller with an `isinstance`. `add_beam_dump()`
  returns the same pair.

### Fixed

- **The Fix rotation checkbox on a body did nothing.** A clamping fork
  is held by the post it clamps and may be swung about it, and that
  row is what says so - but the panel read it the way it reads every
  other row, as a number out of the input's value. A checkbox has none
  ('on' is not a number), so the panel quietly put itself back and no
  message was ever sent: the row could be clicked, changed nothing, and
  came back as it was when the body was selected again. It is now read
  the way a checkbox is read, which is what the optics panel has always
  done with its own. Both ends of the plumbing were already there - the
  attribute is editable, and the message builder knew the row - so
  nothing else changes.

  `verify_mech_browser.py` grows 6 checks: the row on a body whose turn
  is free, the message ticking it sends, Python freezing the turn and
  undo letting it go, and a tick that changes nothing sending nothing.

- **`copy()` carries `term_on_HR_order` and `term_on_HR_transmits`.**
  **Changed results.** It rebuilt the element from `term_on_HR` alone,
  so the copy of a mirror gated at order 2 was gated at 0 and the copy
  of one kept transmitting stopped transmitting. Copying through the
  layout - the viewer's copy button, `copy_optics`, save and load - was
  never affected, since that route goes through the attribute
  whitelist. The KAGRA layouts are byte for byte identical across it:
  they copy their elements before setting these.

  `verify_stray.py` grows 13 checks: the three settings from the
  constructor and through `copy()` on all four classes, and a mirror
  set up at construction tracing what one set up afterwards does.

## 0.5.0 — 2026-08-13

### Added

- **A shape editor for the bodies**, behind `Mechanics.edit()`. A
  part's geometry is a list of drawing primitives, and until now the
  only way to lay one out was to write the numbers in a cell.
  - It is not a second viewer: it is the same one, handed a scene of
    nothing but the shapes being edited, drawn in the local frame
    **with the origin marked** - the origin being the point that
    comes to sit at the host's substrate centre when the body is
    attached, so seeing it is most of what makes a part right. Zoom,
    pan, undo, measuring and the layer panel come along because they
    were never about optics in the first place.
  - The side bar swaps: buttons that put a rectangle, circle, line,
    polyline, arc or text down at the origin; a list of the shapes in
    the order they are drawn, which is where one is picked, copied,
    moved earlier or later and taken away; the numbers of whichever
    is picked, in millimetres and degrees; and one button that
    registers the part in the model library under a name.
  - **A part is drawn by hand as well as by number.** A click picks a
    shape out of the drawing - by its outline, or by what it encloses,
    the smallest winning - and the same place clicked again steps down
    through whatever overlaps there. The shape on show is carried by
    dragging it, and stands on grips: the four corners of a rectangle
    (the opposite one staying put), a point on the rim of a circle for
    its radius, the two ends of a line, where an arc starts, stops and
    how far out it runs, and one grip per vertex of a polyline. A drag
    settles on the marked points - the origin, and the corners,
    centres, vertices and **edge midpoints** of the other shapes -
    unless Alt says to take the cursor at its word. The measuring tool
    reaches the same points, midpoints included. Each gesture commits as one `set_shape`,
    so it is one step of undo and goes through the same constructor a
    typed row does.
  - **A shape is turned** by Shift-dragging it, or with `[` and `]`
    for 45° at a time - the same gesture and the same keys that turn
    an element out on the bench. The turn is about the middle of the
    shape's bounding box, which is the box already drawn around it.
    A `Rectangle` is a corner, a width and a height with its sides
    along the axes, so **a turned rectangle comes back as the closed
    polyline of its four corners** - the rule gtrace has always drawn
    a turned body's rectangles by, now stated once in
    `mechanics.turned_shape` and used by both. One undo puts the
    rectangle back.
  - **A polyline is edited vertex by vertex.** The rows work on the
    vertex the grips pick out and say which of how many it is;
    `+ Vertex` puts a corner in halfway along to the next one and
    `- Vertex` takes the one in hand out. Fewer than two vertices
    draws nothing, and is refused.
  - **The points a part names for itself are edited here too**, in a
    panel of their own: they belong to the part rather than to any
    one shape, and one of them is often nowhere near the drawing of
    the feature it stands for - a mount is bolted to its pedestal
    from underneath, so its post hole is in no top view at all. Each
    is a ring with its name beside it, in the amber of the origin
    cross, the origin being the one point every part already has.
    Pick one from the list or click its ring; the rows give its name
    and its place in millimetres; drag the ring to carry it, settling
    on the same marked points a shape does, Alt to take the cursor at
    its word. A ring is picked ahead of the shapes under it and
    behind the grips of the shape on show. `+ Point` names one at the
    origin and `- Point` takes the picked one away. They join the
    marked points, so a shape settles on them as well.
  - It edits the `Mechanics` **by reference**, like everything else
    in gtrace, so a body already registered in a layout is redrawn
    there at the layout's next draw - attachment, pose and builder
    parameters all untouched.
  - `mechanics.turned_shape(shape, angle, offset)`,
    `mechanics.rotate_shape(shape, angle, pivot)` and
    `mechanics.shape_centre(shape)`: how a drawing primitive turns,
    which `Mechanics.world_shapes()` was already doing privately and
    the editor now needs too.
  - `gtrace.draw.viewer.editor.ShapeEditor` is the model behind it,
    drivable without a browser. A shape is edited by taking it apart
    into the dict `shape_to_dict` writes, changing what the message
    names and building it again, so a value that describes no shape
    is refused by the constructor and leaves the shape that was there
    untouched. What the constructors do not catch - a size of none or
    less, a coordinate at infinity - is refused on the way out. The
    named points arrive as one `set_points` carrying the whole list,
    since there is no index that survives a rename: a point is known
    by its name, and the name is the thing being edited. Two points
    cannot share one, and a point cannot go unnamed.

- **Aiming an optics by places** - an `Align` menu, and the keys
  behind it. A drag puts an element approximately anywhere and
  Ctrl-drag squares it onto a beam that already exists; what was
  missing is the angles a bench is laid out by *before* there is a
  beam to point at.
  - **Line 2 points** (`a`): the face ends up square across
    the line between two places, looking from the first towards the
    second. A line has two normals, and the click order says which -
    so clicking the two places the other way about turns the element
    right round, which is how a face is flipped.
  - **Bisect 3 points** (`b`): from, at, to. The face takes the
    bisector of the corner, which is where a mirror folding light
    from the first place to the last has to look.
  - **Turn ±45°** (`]` and `[`): the quarter turn a steering mirror
    is specified by, from wherever it faces now.
  - The places snap to the same marks a measurement takes - faces,
    corners, screw holes, beam ends - which is what makes this exact
    rather than a steadier drag. The arm to the cursor and the
    outline the optics would take are previewed as it is taken.
  - Aiming turns and does not move: which way an element faces and
    where it stands are two questions, and the second already has
    answers (Ctrl-drag, the Center rows, Along beam / Move by).

- **The element panel says what an element follows.** `Assembled to`
  picks an element or a body for it to follow, and `(free)` lets it go
  where it stands; neither the element itself nor anything that
  already follows it is offered. While it follows something the pose
  rows are the host's doing - they show where the host put it and
  refuse the keyboard rather than taking a value only for the next
  trace to write over it - and it cannot be dragged. The exception is
  its turn when `Fix rotation` is off, which is how the opening of a
  V is set. `Joint x`, `Joint y` and `Joint angle` nudge it without
  letting it go.

- **Changed: removing something takes what stands on it.** The mount
  on the mirror, the pedestal under the mount, the far face of a beam
  dump and its housing. It used to be refused, which was safe and
  wrong: those are not things that happen to be near the element,
  they are things whose place is *its* place, and asking for each of
  them separately was asking twice for one thing. `remove_optics()`
  and `remove_mechanics()` return what went, it is one step of undo,
  and letting something go first - `disassemble()`, `detach()`, or
  `(free)` in the panel - is what keeps it.

- **`+ Dump` puts one on the bench from the viewer**, facing a beam
  that runs along +x, at the centre of the view. It is a kind of its
  own in the add row rather than a variant of something else, and it
  cannot be on the model library shelf `+ Mechanics` opens: a model
  holds shapes, and two of the three pieces of a dump are elements.
  On the protocol it is one `add` of type `BeamDump`, taking
  `center`, `angle` and `reflectivity` and nothing else - the rest of
  the dimensions are the drawing's.
  - **A dump is numbered and its pieces lettered**: `BD1a` and `BD1b`
    are the two faces of the first dump and `BD1box` its housing, so
    `BD2b` reads as the far face of the second. `unique_dump_name()`
    finds a number that leaves all three free, since a dump has no
    object of its own to hold a name.

- **`beam_dump()` and `OpticalLayout.add_beam_dump()`** - two
  absorbing faces in a V and the housing they sit in, jointed so the
  three move as one. `angle` is the direction the light travels, so a
  dump is aimed the way the beam runs rather than by where its mouth
  points.
  - The point of the V is that a black face is not perfectly black:
    what one face sends back the other catches and sends back again,
    so the light works its way into the wedge instead of coming out
    the way it came. With the default 4% a beam is down to 0.16%
    after two bounces and to a part in ten million after five. That
    is worth tracing, which is why the faces are elements rather than
    a shape drawn on the housing.
  - The dimensions are a real drawing's, and three of them settle the
    whole V - the apex 25 mm above the post hole, 50 mm faces, a 28
    degree opening - so the face centres come out where the drawing
    dimensions them. **The reflectivity is not from the drawing** and
    `DUMP_REFLECTIVITY` says so: what a black absorber returns
    depends on the glass and on the polarisation, so it is a
    placeholder to be measured and passed.
  - Aim it into one side of the V rather than at the apex, which is
    where the two faces end: a ray sent exactly there hits neither,
    on a bench as well as here.

- **An element may follow another: `assemble()` and
  `disassemble()`.** Two absorbing faces in a V are one beam dump, a
  pair of steering mirrors is one periscope, and a bench is built out
  of such assemblies rather than out of loose elements. The follower
  keeps its place relative to the host, and moving or turning the
  host carries it along; the host may be another element or a body.
  The joint is the one a body already uses - an offset in the host's
  frame, a relative angle, and `fix_rotation` to say who may change
  it - and what lands at the offset is the follower's own anchor
  point.
  - **A follower's pose is stored, not derived, and settled just
    before the layout is read** - by `trace()`, `draw()` and
    `snap_points()`. An `Optics` holds its pose in traits whose
    derived geometry is what the trace reads, so a pose computed on
    demand would have meant rewriting that. Settling comes to the
    same thing for the same reason: there is no notification to miss,
    because nothing is listening. Assigning `M1.HRcenter` in a cell
    and then tracing carries the assembly along. What it cannot cover
    is reading a follower's pose without tracing or drawing first.
    A layout with no assemblies is not touched at all.
  - Placing a follower is refused in the same words a held body
    already uses, since a pose typed into one would be written over
    at the next trace - which is worse than a refusal, because it
    would look as though it had worked. `move`, `align`, `slide` and
    a typed pose are turned away; the turn of an element whose
    `fix_rotation` is false goes through and is read back into the
    joint. An element another follows cannot be removed until it is
    let go of, `copy_optics()` brings the followers along, and a
    circle is refused - in a call and in a loaded file.
  - `verify_assembly.py` (58 checks). The KAGRA path is untouched:
    with nothing to settle, nothing is written.

- **`round_breadboard(diameter)`**, and `BBR30` and `BBR45` on the
  library shelf. A vacuum tank is round and so is the board in the
  bottom of it. The grid is the rectangular board's - symmetric about
  the centre, on the same 25 mm pitch - and the rim decides which of
  its holes exist: drilled where it lies a margin in from the edge,
  left out where it does not, so the rows shorten towards the rim the
  way a real disc is drilled.
  - **A round body has one size, not two.** `Mechanics.resizable` now
    says *how* a body resizes - `'box'`, `'round'` or `None` - where
    it used to answer only whether. `resize()` takes either name as
    the diameter and **refuses two that disagree** rather than
    resolving them by picking one: a round board asked to be 300 by
    400 is a misunderstanding, not a size.
  - In the viewer the panel offers a single **Diameter** row instead
    of Width and Height, and a dragged corner sets that one number.
    The centre stays where it is, since a disc has no opposite corner
    to hold still, and the handles stand on the square it is
    inscribed in.

- **A numeric panel row is a calculator, and knows its own unit.** A
  bench measurement is usually arrived at rather than known, so a row
  takes the sum as well as the answer: `2*25.4` is 50.8, `300/4` is
  75. Brackets, the four operations and a leading minus, parsed
  character by character - nothing is evaluated as code, so a row is
  a calculator and not a way into the page.
  - **A value may carry a unit of its own in square brackets**, and
    it converts into the unit the row is labelled with: `1[in]` in a
    millimetre row is 25.4, in a metre row 0.0254. Lengths `m`, `cm`,
    `mm`, `um`, `nm`, `in`, `mil`, `ft`; angles `rad`, `mrad`, `deg`;
    power `W`, `mW`, `uW`, `kW`. The unit converts the number it
    follows and the rest is ordinary arithmetic in the row's unit, so
    `1[in]+2` in a millimetre row is 27.4.
  - A unit of the wrong kind is refused rather than quietly taken as
    a bare number - a length typed into an angle converts to nothing
    - and so is any unit in a row that has none of its own, such as
    an order or a refractive index. A refused entry sends nothing and
    the row goes back to what the model holds, as it always did.
  - `verify_input.js` (119 checks) hammers the parser on its own
    under Node; the browser suites check that each row hands it the
    unit it is labelled with, which is the half that could be wrong
    by a factor of a thousand without saying anything.

- **A measurement reports its two components**, `Δx` and `Δy`, beside
  the distance - in the dimension panel and in the status bar while
  the measurement is being taken. A bench is built on axes: a mount
  goes 300 along and 75 across, and that pair is as often the number
  wanted as the straight line between the points. Signed from the
  first point to the second, the way the direction already read.

- **`copy_optics()`, and a Copy button in the element panel** - a
  second one of an element, with the whole stack standing on it: the
  mount bolted to it, the pedestal under the mount, the fork over the
  pedestal, each pinned to the copy exactly as its original is pinned
  to the original. One of a pair of steering mirrors is not one
  element, it is the element and everything built under it, and none
  of that is worth assembling twice. `{'op': 'copy', 'target': 'M1'}`
  on the protocol, one step of undo.
  - The copies are made through the same dicts a saved layout is
    written with, so what is copied is what would have been saved -
    by value, sharing nothing. The poses of the bodies are the one
    thing not copied, because they were never stored: each derives
    its own from the copy it now stands on.
  - Without a name the copy takes the original's without its trailing
    number and the first free one after it, so a copy of `M1` is
    `M2`; without an offset it stands its own diameter away along
    both axes, far enough to clear what it was made from and near
    enough to be plainly the same thing moved.
  - Only an element is copied this way. A stack stands on an element
    at its root, and a body on its own is one call to the model
    library away.

- **A dragged laser settles on the screw holes**, landing the point
  its light leaves from on the hole, as a dragged element already
  landed its anchor. A laser is bolted to the bench like anything
  else, and that point is the one the model keeps fixed when the beam
  is turned, so it is the one worth landing. Alt rides free, and the
  status bar names the hole - both as they already were for an
  element.

- **A bench stacks: a body may stand on another body.** A mount is
  bolted to a pedestal and the pedestal is held down by a clamping
  fork, and `attached_to` now takes either an optics or another
  `Mechanics`. The chain follows the optics at the root of it - move
  or turn the mirror and the whole stack comes along - and a cycle is
  refused, since a pose deriving from itself is not wrong so much as
  endless. `mechanics.host_pose()` is the one place that knows an
  optics is turned by its HR normal and a body by its own angle.
  - **Parts name their own points.** `Mechanics.points` is a dict of
    local points a part names for itself - `'post'` for the hole under
    a mount, `'axis'` for a pedestal, `'bore'` and `'screw'` for a
    fork - and they travel with the model in the library. They join
    the snap points, so a drag settles on them, a measurement reaches
    them and Align aims by them.
  - **A dragged body settles on them.** Everything a measurement can
    snap to is somewhere a part can be placed: drop a pedestal near
    the hole under a mount and it lands on it exactly. Alt takes the
    cursor at its word, as it already did for an optics over a screw
    hole.
  - **`attach_point` says which point of the body is pinned**, in its
    own coordinates. The default is the local origin, which is the
    rule every mount was already drawn to, so nothing that existed
    moves by a bit. Attaching through the protocol picks it up from
    the drawing - the point already coinciding with one of the host's
    - which is what makes "drop it on the hole, then attach it" pin it
    by that hole, and what makes a fork swing about its post.
  - **`fix_rotation` decides who may change the relative angle.**
    True, the default, is a mount bolted to its mirror: it faces where
    the mirror faces. False is a clamping fork: the **Fix rotation**
    row and Shift-drag swing it about the point it is pinned by. Either
    way it turns *with* the host - a stack that came apart when the
    mirror was aimed would not be a stack.
  - **Attached to** in the panel lists the other bodies as well as the
    optics. Seating a body on an optics puts it at the model's own
    place, as before; seating it on another body keeps where it is,
    since which hole of a mount a pedestal sits in is a choice made on
    the bench and not by the library.
  - Removing a body something stands on is refused, as removing a held
    optics already was. A saved layout writes the host's name, the
    offset, the attach point and `fix_rotation`, and loading joins the
    chain back up whatever order the file lists it in.

- **`pedestal()` and `clamping_fork()`**, and `PEDESTAL-25` and
  `FORK-125` in the library. Drawn from the published drawings of a
  1 inch pedestal post and the small clamping fork that holds it - a
  25.4 mm post on a 31.8 mm base, and a 73.8 mm fork with a 26 mm
  bore, its prong tips standing 3.8 mm ahead of the bore centre -
  with the fork's waisted outline approximated by straight tapers and
  arcs. Both mounts gain a `'post'` point, the hole they are bolted
  down through, measured 13.5 mm behind the substrate centre, and a
  4 mm circle drawn there: the hole is bored from underneath and is
  in no top view, but where it is belongs in one, since a top view of
  a bench is read for what is bolted down where. `points=` overrides
  the point. Where a named point and a circle land on one place, the
  named one is offered first and so is the one a snap takes - the
  same rule, for the same reason, as an optics winning over the beam
  that ends on it.

- **The word is "mechanics", not "hardware".** Mounts and holders are
  optomechanics, but what a layout carries is not always optical - the
  wall of a vacuum tank, a bench, a beam dump housing - so the general
  word is the right one, and one word is better than two. The class was
  always `Mechanics`; the button is now **`+ Mechanics`**, the layer is
  `mechanics`, and an unnamed body is `P1`, `P2`, ... for part. Nothing
  of this has been released, so nothing is carried forward: a layout
  saved before the change keeps whatever layer name it was written
  with.

- **`Mechanics`: the bodies on a bench** (`gtrace.mechanics`). A named
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
    on a `mechanics` layer of its own - which the viewer and any CAD
    reading the DXF can switch off as one thing - and picked in the
    viewer by point-in-polygon on its outline, after everything else:
    a beam, an optics or a mount lying on a breadboard wins the click
    over it, and among mechanics the smallest wins.
  - In the viewer, clicking a mechanics opens a pose panel (centre,
    angle, rename, remove); dragging a *selected* mechanics moves it
    and Shift-dragging turns it. An unselected one is not grabbed - a
    breadboard can cover most of the bench, and a drag across it
    should pan the view. A body covered by an optics - a mount is
    covered completely by its own mirror - takes the last turn of the
    repeated-click walk that already steps from an element into the
    beams under it.
  - **`+ Mechanics`** adds a model from the library at the centre of
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
  - Body names are drawn only when the new `drawMechanicsNames`
    option asks (off by default): a body is background, and a
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
    loading joins the two back up; removing an optics with a body
    attached is refused until the body is detached or removed.
  - Edit operations: `add` with `type: 'Mechanics'` (shapes arrive
    serialized, as the layout file carries them), `move` (`center`),
    `rotate` (`rotationAngle`), `set`, `remove`, `rename`. The scene
    gains a `mechanics` channel; corners and centre join the snap
    points, so the measuring tool reaches the bodies.
- **A model library for the parts**, in `gtrace.mechanics`. The
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
  standard 12.7 thick optic. `lens_holder(length, thickness)` is a
  plain rectangle centred on the substrate centre of the optic it
  holds. The library is seeded with stock built from them (`BB3030`,
  `BB4530`, `BB6045`, `MOUNT-25`, `MOUNT-50`, `HOLDER-25` at 30 x 10
  mm, `HOLDER-50` at 56 x 12.7 mm); further vendor models are yours
  to register from measured footprints.
- **The library saves and loads.** `save_models(path, names=None)`
  writes the shelf - or the named part of it - to a JSON file
  carrying exactly what the registry holds, and `load_models(path)`
  merges a file back in, name by name with the file winning: the
  same rule `register_model` already has, so a library built out of
  several files is one call per file. A load checks everything
  before merging anything - a file with one unreadable shape in it
  changes nothing.
- `shape_from_dict` in `gtrace.draw.serialize`: the inverse of
  `shape_to_dict`, which loading a mechanics needs.
- `verify_mechanics.py` (315 checks), `verify_mech_browser.py` (81),
  `verify_align_browser.py` (34), `verify_editor.py` (143),
  `verify_editor_browser.py` (47) and
  `verify_editor_drag_browser.py` (60). With `verify_input.js`,
  `verify_assembly.py` and `verify_beam.py`, the suite is 29 files and
  4996 checks.

### Fixed

- **A beam given exactly `q0=1j` came out of its first propagation with
  a real q.** **Changed results**, at the last bits; see the end of this
  entry. A `GaussianBeam` keeps its q-parameter twice: `qx` and
  `qy`, and the reduced `qrx` and `qry`, which are what an ABCD
  transform is applied to. The reduced pair, the width and the best
  matching circular q are all derived by trait handlers when `qx` or
  `qy` is assigned - and traits does not notify when an assignment
  matches the value already there. `qx` defaults to `1j`, so a beam
  constructed with that exact value kept every derived default: a width
  of zero, and a reduced q of *zero*. The first propagation transformed
  that zero, giving the beam a real q - infinite Rayleigh range, no
  waist - and the next thing to ask its width divided by zero. A layout
  whose source was written that way could not be traced at all.

  The constructor now sets both q-parameters before deriving anything
  from them, and derives explicitly rather than relying on a
  notification that may not come. `copy()` does the same, which also
  settles the circular q of a copied beam in glass: `deepcopy` assigns
  the traits in its own order, so a handler could leave that value
  describing a half-built beam.

  Nothing in gtrace writes `1j` - every source it builds goes through
  `q_from_waist` - which is how the crash survived from before 0.3.0
  without anyone meeting it.

  **What does change for an existing layout is the last bit or two.**
  The old `copy()` carried a beam's q across as `(q/n)*n`, because it
  copied the reduced q and let the handler rebuild the q from it. That
  round trip is not exact for a beam in glass, and it is gone: a copy
  now carries the q it was made from. On the KAGRA layouts the effect
  is 26 of 51 lines of `MainBeamList.csv` and 24 of 59 of
  `StrayBeamList.csv` differing in their q-parameters, by at most
  1.11e-16 absolutely and 8.3e-15 relatively - some 37 units in the
  last place. Every DXF is identical within the comparison tolerance,
  the largest coordinate difference being 1.14e-13, and
  `OpticsList.csv`, `bKAGRA_Mirror_Coordinates.txt` and `bKAGRA_log.txt`
  are byte for byte the same. Nothing moves by a distance anything
  could measure, but a run will not reproduce 0.4.0 bit for bit.

  `verify_beam.py` is new: 129 checks asking that a beam's derived
  values agree with the q it was given, over q-parameters that include
  the trait's own default, after propagating, in glass, through
  `copy()`, and over every beam a trace produces.

- **A ghost stopped being a ghost as soon as it left the element that
  made it.** **Changed results.** `non_seq_trace` set a beam's
  `stray_order` back to zero every time it travelled from one element
  to the next, so a ghost arrived at the next mirror looking like the
  main beam: drawn in the main beam's colour, and handed a fresh
  allowance of `order` ghosts of its own.
  - `order` therefore bounded nothing. The gate inside `hit` is
    `stray_order <= order`, and zeroing the arriving beam only makes
    it easier to pass, so every element multiplied the branches and
    the recursion ended on `power_threshold` alone. On two mirrors
    with `Refl_AR=0.5` and `order=3`, tracing produced 17763 beams
    with the reset and 630 without it.
  - The line came in as a one-line commit in 2025 titled "fixing a
    recursion error", which it cannot have done: zeroing a counter
    compared with `<=` lets more beams through, never fewer. The
    recursion error a cavity causes is what `term_on_HR` is for.
  - On the KAGRA layouts the whole effect is one of classification.
    Both notebooks trace identically - same beams, same positions,
    same powers, and `MainBeamList.csv`, `StrayBeamList.csv` and
    `OpticsList.csv` byte for byte - and **176 beams that were being
    drawn as main beams are now drawn as stray**, which is what they
    are. Their envelopes follow, at the stray sigma.
  - See #6.

- **A mirror would not reflect a beam that was already stray.**
  **Changed results.** `hitFromHR` capped its first, external
  reflection with `stray_order <= order`, so a ghost arriving at a
  face *meant* to reflect produced nothing unless the caller asked
  for an order at least as high as the order the beam already
  carried. At the default `order=0` a mirror simply did not reflect
  a ghost.
  - That cap came in with `HRreflective` in 0.4.0, alongside the
    increment that makes a reflection off a lens face count as stray.
    The increment is right; the cap was not. `order` is a budget for
    the ghosts a call may *make*, not a test the arriving beam has to
    pass, and a face meant to reflect makes none - the beam leaves at
    the order it arrived at. A face not meant to reflect does make
    one, and that is still counted and still capped.
  - It was invisible in a trace, which resets the stray order as a
    beam travels from one element to the next, so every check passed.
    It was fatal to code calling `hitFromHR` directly: **both KAGRA
    layout notebooks stopped running** - `OMC-Layout-O4` in four
    cells and `KAGRA-OptLayout-Main` in fifteen, all
    `KeyError: 'r1'`. Both run again, and against the tree from
    before `HRreflective` the only differences left are the ghost
    powers that release deliberately corrected.
  - `verify_stray.py` (33 checks) is new and is about this: what a
    stray order is, what `order` caps, and the difference between a
    ghost made and a ghost met. Nothing covered it before, which is
    how the cap got in.

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
- **The tutorial is rewritten around the viewer.** It used to build
  mirrors and beams, pass them around by hand and write the results to
  DXF, reaching an `OpticalLayout` and the viewer only in a chapter
  appended at the end. It now puts a system into a layout in its first
  code cell and opens it, because that is the loop the work actually
  happens in: place something, look at what the beams do, move it.
  Reading a beam off the drawing, aiming an element by places,
  measuring across a substrate, standing the optics in mounts on a
  breadboard and drawing a part in the shape editor each have a section
  of their own, and every gesture is shown as the message it sends. The
  KAGRA input mode cleaner stays as the worked example of the other
  half of the workflow - build and align in ordinary Python, then
  register and look - and DXF closes the notebook rather than carrying
  it. The version that led with DXF is in the repository's history.
- **The reference pages cover the mechanics.** `layout.rst` gains
  Mechanics: the bodies themselves, attaching one to an optics, the
  model library, the messages that reach them and the shape editor's
  own protocol. `viewer.rst` gains how a body is picked and dragged,
  the corner handles, screw-hole snapping, aiming by places, and the
  shape editor. Two figures are new - a bench with its mounts, and a
  part open in the editor - and `docs/make_viewer_figures.py` now
  photographs the three scenes they come from.
- The tutorial page introduces each of its two notebooks in a
  subsection of its own — what it covers and where its reference
  pages are — instead of listing the second one without a word.
- Running the tutorial no longer means cloning the repository: both
  notebooks are self-contained, so the tutorial page now links
  straight to the files on GitHub, and `pip install
  "gtrace[notebook]"` plus the download is the whole setup.
- **The English documentation is rewritten to read straight.** All six
  handwritten pages and the prose cells of both tutorial notebooks. The
  old text made its points by paradox, leaned on "which is what X is
  for", used "X rather than Y" for cadence where nothing was being
  contrasted, and carried its argument in dashed asides: 124 of those
  across the pages, now 30. It also argued for design decisions the
  reader had not asked about; why a decision was taken belongs in the
  commit that took it, and the documentation keeps the "why" a reader
  needs in order to predict what the code will do. Where a rule was
  buried in an aside it is now a paragraph of its own. No behaviour,
  number, code sample, class reference or link changed, and the
  Japanese translation follows in the entry below.
- **The Japanese documentation is retranslated, page by page.** All 637
  messages across the nine `.po` files, not only the ones the rewrite
  above made fuzzy. The old text had been produced one message at a
  time, which is how a `.po` file presents itself, and it read that
  way: paragraphs that were individually defensible and collectively
  incoherent, with the English em dashes carried across into a
  language that does not use them that way. Each page is now read
  whole before any of its messages are written. Terminology is
  consistent across the nine files, and the GUI's own labels stay in
  English so that a reader can find the row. Both languages build with
  no warnings, and the rendered pages are checked for the inline
  markup that Japanese reST silently fails to recognise.

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
