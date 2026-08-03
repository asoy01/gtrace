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
| `verify_stage1.py` | `renderHTML`: self-containment (no fetched URL, no surviving placeholder), the embedded scene's channels, every entry point, the optics channel the properties panel needs |
| `verify_stage1.js` | **`viewer.js` physics against gtrace**: `beamParamsAt` and `projectOnBeam` over every beam of a traced system at five points each, plus the waist relations. Runs in Node against the real file |
| `verify_browser.py` | Headless browser: the DOM the viewer builds, empty-layer marking, and **the HTML file `renderHTML` actually writes** |
| `verify_interact.py` | Headless browser with real `MouseEvent`s: hover readout against gtrace, pinning, cycling through overlapping beams, the direction arrow, layer visibility, zoom about the cursor |
| `verify_stage2.py` | The widget: ESM assembly, traitlets, `update()`, backend selection |
| `verify_stage2_browser.py` | The ESM imported from a blob URL **the way anywidget imports it**, driven with a stand-in model |
| `verify_stage2b.py` | The edit protocol end to end, what it refuses, save/load, and the model invariants underneath (sagitta, substrate centre, `max_stray_order`) |
| `verify_stage2b_browser.py` | Headless browser: dragging an optics, and feeding the resulting messages back through `apply_edit` |
| `verify_props_browser.py` | Headless browser: the properties panel, its unit conversions, add and remove, rename, the display controls, the layout file buttons |

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
