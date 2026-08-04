# gtrace — a Gaussian beam ray-tracing package in Python

gtrace traces Gaussian beams through a two-dimensional arrangement of
optics. It follows the q-parameter rather than a geometric ray, so a
result carries the beam radius, the wavefront curvature and the Gouy
phase everywhere along the path, not just the geometry. Beams that are
transmitted, reflected and internally reflected inside a wedged substrate
are all followed, which is what makes it useful for chasing ghost beams
in a real interferometer layout.

It was written for KAGRA and is used to lay out its optical benches.

## Installation

### From PyPI

```sh
pip install gtrace              # the library and the HTML viewer
pip install "gtrace[notebook]"  # ... and the viewer as a Jupyter widget
```

Python 3.9 or newer. gtrace itself needs only numpy, scipy and traits.

### What the viewer needs

The viewer has two front ends, and they do not cost the same:

| front end | needs |
|---|---|
| self-contained HTML — `render_html()`, `show(backend='html')` | nothing beyond gtrace and a web browser |
| Jupyter widget — `widget()`, or `show()` inside a notebook | `anywidget` (≥ 0.9), which brings `ipywidgets` with it, and a Jupyter front end |

`show()` picks the widget when it finds itself in a Jupyter kernel with
anywidget installed, and writes the HTML file otherwise, so the same code
works both ways. Without anywidget the widget raises `WidgetNotAvailable`
and says so.

### From a clone

```sh
git clone https://github.com/asoy01/gtrace.git
cd gtrace
pip install ".[notebook]"
```

Use `pip install -e ".[notebook]"` instead if you intend to change gtrace
itself. The quotes matter in zsh, which would otherwise try to expand the
brackets as a glob.

## Running the tutorial

The tutorial is a Jupyter notebook, and it lives in the source tree, so
it needs the clone above and a Jupyter front end:

```sh
pip install jupyterlab
jupyter lab docs/source/tutorial/gtrace-tutorial.ipynb
```

VS Code's notebook editor works just as well: open the file and select
the interpreter you installed gtrace into.

Run the cells from the top. The last chapter opens the viewer in the cell
output, which is the part that wants `anywidget`; everything before it
works without. The notebook writes its results next to itself —
`SeqTrace.dxf`, `NonSeq.dxf` and `MC.dxf` from the tracing sections, and
`tutorial_viewer.html`, `tutorial_layout.json` and `tutorial_layout.dxf`
from the layout section.

To read it rather than run it:
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

`show()` opens an interactive viewer: click anywhere along a beam — not
only at a vertex — and it reports the beam radius, the wavefront ROC,
the complex q, the waist and its distance, the Rayleigh range, the Gouy
phase and the accumulated optical path length at that point, separately
in x and y.

In a Jupyter notebook it renders in the cell output and can be edited:
drag an element to move it, shift-drag to rotate it, click it to edit its
properties, and take back what you did not mean. The layout holds your
objects by reference, so a mirror you move in the browser is the object
your own code named, and the trace and the drawing follow.

`Measure` takes a dimension off the drawing: click the two points, then
place the line. The ends snap to the corners and faces of the elements
and to the ends of the beams, and where the whole span runs inside a
substrate the optical distance is written alongside the physical one.
Dimensions are kept with the layout and saved with it.

Outside a notebook it writes one self-contained HTML file — no server,
nothing to install — which you can send to a collaborator, who can read
the beam parameters off it and take dimensions on it:

```python
layout.render_html('trace.html')
```

DXF output is unchanged and still the way to hand a layout to the rest
of an engineering workflow:

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
  — the conventions the whole package rests on.
- [Optical layouts](https://gtrace.readthedocs.io/en/latest/layout.html)
  and [The viewer](https://gtrace.readthedocs.io/en/latest/viewer.html).

The [`Manuals`](Manuals) directory holds the slides the package was
first presented with, on ABCD matrices, the q-parameter and the basic
concepts.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
