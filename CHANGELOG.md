# Changelog

Notable changes to gtrace. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and gtrace aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries marked **Changed results** alter the numbers gtrace produces for
an unchanged input. They are corrections rather than regressions, but a
system traced with an earlier version will not reproduce bit for bit
across them.

## Unreleased

Slated for 0.4.0. Not yet on PyPI.

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
- `OpticalLayout.export_dxf()`, the companion of `render_html()`. It
  draws the dimensions on a layer of their own, which CAD can switch
  off; `dimensions=False` leaves them out. `draw_dimensions()` does the
  drawing and is callable on its own.
- A **Drawing (DXF)** panel in the viewer, with a file name of its own
  beside the **Optical layout (JSON)** one, and an `export` operation
  in the edit protocol. The two are kept apart because the layout is
  the model and the DXF is a picture of it: pressing Load on a
  drawing could only be a mistake.

### Added — viewer

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
