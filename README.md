# gtrace — a Gaussian beam ray-tracing package in Python

gtrace traces Gaussian beams through a two-dimensional arrangement of
optics. gtrace follows the q-parameter, not a geometric ray. The result
therefore gives the beam radius, the wavefront curvature and the Gouy
phase everywhere along the path, not just the geometry. gtrace also
follows the beams that are transmitted, reflected and internally
reflected inside a wedged substrate, so you can find the ghost beams in
a real interferometer layout.

The elements you can place are mirrors and lenses: `Mirror`, `Lens`, and
their cylindrical versions `CyMirror` and `CyLens`.

gtrace was written for KAGRA and is used to lay out the KAGRA optical
benches.

## Installation

### From PyPI

```sh
pip install gtrace              # the library and the HTML viewer
pip install "gtrace[notebook]"  # ... and the viewer as a Jupyter widget
```

Python 3.9 or newer. gtrace itself needs only numpy, scipy and traits.

`pip install "gtrace[notebook]"` does not install Jupyter itself. If you
do not have Jupyter:

```sh
pip install jupyterlab
```

### From a clone

```sh
git clone https://github.com/asoy01/gtrace.git
cd gtrace
pip install ".[notebook]"
```

Use `pip install -e ".[notebook]"` instead if you plan to change gtrace
itself. The quotes are needed in zsh. Without them, zsh tries to expand
the brackets as a glob.

### What the viewers need

gtrace has two viewers, and they need different packages:

| viewer | needs |
|---|---|
| self-contained HTML page — `render_html()`, `show(backend='html')` | nothing beyond gtrace and a web browser |
| Jupyter widget — `widget()`, or `show()` inside a notebook | `anywidget` (≥ 0.9), which installs `ipywidgets`, and a Jupyter front end such as JupyterLab |

`show()` uses the widget when it runs in a Jupyter kernel that has
anywidget installed. Otherwise `show()` writes the HTML file, so the same
code works in both cases. Without anywidget, `widget()` raises
`WidgetNotAvailable`. The message says how to install anywidget, and how
to open the HTML viewer instead.

## Running the tutorial

The tutorial is a Jupyter notebook in the source tree. The tutorial
needs the clone above and a Jupyter front end:

```sh
pip install jupyterlab
jupyter lab docs/source/tutorial/gtrace-tutorial.ipynb
```

The notebook editor of VS Code also works: open the file and select the
interpreter you installed gtrace into.

Run the cells from the top. The viewer appears in the cell output. Only
the viewer needs `anywidget`; the rest of the notebook runs without it.
The notebook writes its result files into its own directory:
`tutorial_viewer.html`, `tutorial_layout.json` and `tutorial_layout.dxf`
from the last chapter, and `bench_parts.json` from the section on
mechanics.

To read the tutorial without running it:
<https://gtrace.readthedocs.io/en/latest/tutorial.html>.

## Usage

Build the optics with ordinary Python, collect them into a layout, and
look at the result:

```python
import gtrace.beam as beam
import gtrace.optcomp as opt
import gtrace.optics.gaussian as gauss
from gtrace.layout import OpticalLayout, TraceRules
from gtrace.unit import *
import numpy as np

src = beam.GaussianBeam(q0=gauss.Rw2q(ROC=np.inf, w=1*mm), wl=1064*nm,
                        pos=[0, 0], dirAngle=0, name='src')

M1 = opt.Mirror(HRcenter=[0.5, 0.0], normAngleHR=deg2rad(180-45),
                diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                inv_ROC_HR=0.0, n=1.45, name='M1')
M2 = opt.Mirror(HRcenter=[0.5, 0.4], normAngleHR=deg2rad(-45),
                diameter=10*cm, thickness=5*cm, wedgeAngle=deg2rad(0.25),
                inv_ROC_HR=1.0/2.0, n=1.45, name='M2')

layout = OpticalLayout(optics=[M1, M2], sources=[src],
                       rules=TraceRules(order=10, power_threshold=1e-3))

layout.show()
```

`TraceRules` sets how far the trace follows the ghost beams. `order` is
the number of ghost reflections a beam may go through before gtrace
stops following it. gtrace does not reset the count when a beam leaves
one element for the next, so `order` limits the whole path and not one
element. `power_threshold` is the smallest beam power gtrace still
follows, in watts. A source beam carries 1 W unless you set `P`, so
`1e-3` follows the ghosts down to a thousandth of the source power.

`show()` opens an interactive viewer. Click anywhere along a beam, not
only at a vertex, and the viewer reports the beam radius, the wavefront
ROC, the complex q, the waist and its distance, the Rayleigh range, the
Gouy phase and the accumulated optical path length at that point,
separately in x and y.

In a Jupyter notebook the viewer appears in the cell output, and you can
edit the layout there: drag an element to move it, shift-drag to rotate
it, click it to edit its properties, and undo a change you did not want.
The layout holds your objects by reference, so a mirror you move in the
browser is the object created in your own code, and the trace and the
drawing are updated with it.

`Measure` is a button in the viewer. It measures a distance on the
drawing: click the two points, then place the line. The ends snap to the
corners and faces of the elements, and to the ends of the beams. When the
whole span runs inside a substrate, the viewer writes the optical distance
next to the physical one. Dimensions are stored in the layout and saved
with it.

Outside a notebook `show()` writes one self-contained HTML file. You can
send the file to a collaborator, who opens it in a web browser, reads the
beam parameters and measures distances on it:

```python
layout.render_html('trace.html')
```

Use DXF output to pass a layout to other engineering tools:

```python
import gtrace.draw.renderer as renderer
renderer.renderDXF(layout.draw(), 'trace.dxf')
```

## Documentation

Full documentation, including the tutorial, is at
<https://gtrace.readthedocs.io/>.

- [Tutorial](https://gtrace.readthedocs.io/en/latest/tutorial.html) —
  conventions, beams, mirrors, sequential and non-sequential tracing,
  the KAGRA input mode cleaner, and the viewer. Runnable as
  [`docs/source/tutorial/gtrace-tutorial.ipynb`](docs/source/tutorial/gtrace-tutorial.ipynb).
- [Basic concepts](https://gtrace.readthedocs.io/en/latest/basic_concepts.html)
  and [Beam propagation](https://gtrace.readthedocs.io/en/latest/propagation.html)
  — the conventions that the whole package uses.
- [Optical layouts](https://gtrace.readthedocs.io/en/latest/layout.html)
  and [The viewer](https://gtrace.readthedocs.io/en/latest/viewer.html).

The [`Manuals`](Manuals) directory holds the slides from the first
presentation of gtrace. The slides cover ABCD matrices, the q-parameter
and the basic concepts.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
