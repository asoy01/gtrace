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

```sh
pip install gtrace
```

Python 3.9 or newer, with numpy, scipy and traits. The notebook widget
additionally needs `anywidget`:

```sh
pip install gtrace[notebook]
```

To install from a checkout instead:

```sh
pip install .
```

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
properties. The layout holds your objects by reference, so a mirror you
move in the browser is the object your own code named, and the trace and
the drawing follow.

Outside a notebook it writes one self-contained HTML file — no server,
no CDN, nothing to install — which you can send to a collaborator as it
stands:

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
