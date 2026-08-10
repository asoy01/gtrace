# GUI verification suites

Checks for the browser-based viewer in `gtrace/draw/viewer/` and the
`OpticalLayout` model behind it.

```
pixi run python tests/gui/run_all.py
```

That runs everything in dependency order and prints a summary. A single
suite can be run on its own once an earlier run has left its inputs in
`_work/`.

Everything the suites generate goes to `tests/gui/_work/`, which is
ignored by git and safe to delete.

## Requirements

- **Node.js** for `verify_stage1.js`, which loads `viewer.js` directly.
- **A Chrome-like browser** for the browser suites, found automatically
  or named through `GTRACE_CHROME`.

Both are optional: a suite that cannot run says so and `run_all.py`
counts it as skipped rather than passed.

## What each covers

| Suite | Checks |
|---|---|
| `verify_surfaces.py` | Where the two faces of a substrate are, against an arc derived independently of the class: both curvature signs, both faces, at four heights across the aperture, plus `isHit` agreeing with what `hitFrom*` then traces, and the drawn arc meeting the drawn sides. Not a GUI check, but it wants counted assertions and this is where those live |
| `verify_lens.py` | `Lens`: for seventeen lenses, the focal length **measured by tracing a ray through them** and reading the angle it leaves at, which owes nothing to the formula the constructor solved. Also what the constructor refuses and why, the defaults a lens needs, copy and save/load, and that a lens does not turn the beam through it into a ghost |
| `verify_cylindrical.py` | `CyMirror`: the reflection and transmission matrices **against Siegman Table 15.1 written down from the book**, for both curve directions across angles and curvatures, with the spherical results pinned bit-for-bit to the pre-fix implementation |
| `verify_cylens.py` | `CyLens`: the focal length ordered lands **in the plane `curve_direction` names and nowhere else** - the in-plane value traced like a Lens, the out-of-plane one read off the ABCD matrix extracted from how `qy` transforms, and the flat plane held to being a window (no power, no magnification, the right length of glass). Plus what is new over Lens: the direction through copy, save/load and the edit protocol |
| `verify_dimensions.py` | Substrate corners against `get_side_info`, what a span runs inside (`contains_segment`, including the hollow of a concave face), the snap points, and the `Dimension` model through the edit protocol |
| `verify_stage1.py` | `renderHTML`: self-containment (no fetched URL, no surviving placeholder), the embedded scene's channels, every entry point, the optics channel the properties panel needs |
| `verify_stage1.js` | **`viewer.js` physics against gtrace**: `beamParamsAt` and `projectOnBeam` over every beam of a traced system at five points each, plus the waist relations. Runs in Node against the real file |
| `verify_browser.py` | Headless browser: the DOM the viewer builds, empty-layer marking, and **the HTML file `renderHTML` actually writes** |
| `verify_interact.py` | Headless browser with real `MouseEvent`s: hover readout against gtrace, pinning, cycling through overlapping beams, the direction arrow, layer visibility, zoom about the cursor |
| `verify_stage2.py` | The widget: ESM assembly, traitlets, `update()`, backend selection |
| `verify_stage2_browser.py` | The ESM imported from a blob URL **the way anywidget imports it**, driven with a stand-in model |
| `verify_stage2b.py` | The edit protocol end to end, what it refuses, save/load, and the model invariants underneath (sagitta, substrate centre, `max_stray_order`). Also lenses through the protocol: what a lens inherits and what it must not, setting `f`, the anchor, and that a scene carrying one is still strict JSON |
| `verify_stage2b_browser.py` | Headless browser: dragging an optics, and feeding the resulting messages back through `apply_edit` |
| `verify_props_browser.py` | Headless browser: the properties panel, its unit conversions, add and remove, rename, the display controls, the layout file buttons, and the rows only some classes have (curve direction, focal length, ROC anchor) |
| `verify_measure_browser.py` | Headless browser: the measuring tool - snapping to the points Python put in the scene, the three clicks, the offset, the dimension panel, and that a static page can still measure |
| `verify_source.py` | Source beams: the waist a laser is specified by **against the q-parameter it converts to**, in both directions over four decades of size and wavelength; that setting one half of the pair leaves the other and the other axis alone; the `sources` channel that tells a source from a traced beam; the protocol reaching a source at last (move, rotate, set, add, remove, rename) and what it refuses; the one namespace optics, sources and dimensions now share; and the tracing rules |
| `verify_mechanics.py` | `Mechanics`: the hardware the trace never sees. That the pose is the only statement of where a body is (the world shapes and the outline are derived, and a turned rectangle survives as the closed polyline of its corners), `point_in_polygon` on its own and through rotated bodies, drawing onto the hardware layer, shape serialization both ways, the edit protocol reaching a mechanics (add, move, rotate, set, rename, remove) and what it refuses, **that moving hardware does not invalidate the trace**, the one namespace all four kinds share, and undo/redo keeping identity through removal and rename. And **attachment**: a mount whose pose is derived from its host on every read - following a host moved in Python or through the protocol with no notification anywhere, refusing its own pose while attached, detaching where it stands, saving by host name and no pose, relinking on load onto the same objects, and a host with hardware attached refusing removal. And the **model library**: builders drilling the standard symmetric grid and standing a mount behind its origin, registration by value, and `relink_mechanics` as the one deliberate, explicit exception to "the saved values are the truth" |
| `verify_mech_browser.py` | Headless browser: hardware picked **by its area and last** - a beam, an optics and a mount each win the click over the breadboard under them, and among mechanics the smallest wins - the pose panel and its units, that dragging an unselected board pans rather than moves, dragging and Shift-dragging the selection with every message fed back through `apply_edit`, a hidden hardware layer being unpickable, and the read-only page. An attached clamp: the panel naming its host, its pose rows refusing the keyboard, and a drag on it panning rather than moving |
| `verify_source_browser.py` | Headless browser: the laser drawn for each source - that it is at the point the light comes from, that it **keeps its size and stays clickable zoomed in forty times**, that it wins the pick against an element on the same spot - the source panel and its units, dragging and Shift-dragging with the preview compared against what Python then does, add and remove, the tracing rules panel, and the read-only page |

`tests/test_gtrace.py` (the DXF renderer) and
`tests/test_beam_propagation.py` (the propagation convention) are run
too, since the viewer rests on both.

## Two things these suites learned the hard way

**Drive the entrance the framework uses, not the internals.** The Stage
2b checks called `apply_edit` directly. Ninety-one of them passed
against a widget that could not receive a single message, because the
handler had been given a name that shadowed ipywidgets' own dispatcher.
`verify_stage2b.py` now sends a real comm message through
`Widget._handle_msg`.

**Drive every output path separately.** The read-only viewer was
exercised through the widget's ESM with a scene from `scene_dict`. That
reproduces "no Python behind the page" but never the scene the static
file actually carries - so the static HTML shipped without its optics
channel, and clicking a mirror in it did nothing. `verify_browser.py`
now opens the file `renderHTML` wrote and clicks a mirror in it.

## The tutorial notebook

```
pixi run python tests/gui/run_notebook.py
```

Executes `docs/source/tutorial/gtrace-tutorial.ipynb` in place. This is
the only check that runs the documented code end to end, so an example
that has drifted away from the library turns up here rather than in a
reader's session.

It then removes three things that would otherwise change on every run
for reasons that have nothing to do with the notebook: the saved widget
state, which embeds a copy of `viewer.js`; the widget outputs, whose
text is an object repr carrying a memory address; and nbconvert's
per-cell wall-clock timings. Consecutive stream outputs are joined, since
where the flushes fell is not information either.

The result is reproducible: run it twice without changing anything and
the file does not change, so a diff against it means something real did.
