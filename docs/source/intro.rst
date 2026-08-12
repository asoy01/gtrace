Introduction
=============

gtrace is a python package to trace the propagation of Gaussian beams among optical components such as mirrors and lenses. The features of gtrace include:

 * Automatically track the Gaussian beam propagation, i.e. q-parameter change, using the ABCD matrix method.
 * Reflection and refraction at interface surfaces are properly treated.
 * Automatically track the optical distance traveled by a beam through dielectric media.
 * Sequential or non-sequential trace modes are available.
 * Exporting the results to DXF files.
 * A built-in viewer that needs no CAD software: zoom, pan, and click anywhere along a beam to read out its parameters at that point. It runs as a self-contained HTML file or as a cell output in a Jupyter notebook.
 * An :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>` container that holds a whole optical system, so that a layout can be traced, drawn, saved to JSON and edited from the viewer.

The main motivation behind the development of gtrace was to help the design of the optical layout for the `KAGRA <http://gwcenter.icrr.u-tokyo.ac.jp/en/>`_ interferometer. That layout has to satisfy many constraints at once, and placing mirrors on a CAD by hand and adjusting distances and orientations one at a time is too slow. We needed to adjust the layout by computer, and for that we needed a representation a program can work with. gtrace represents mirrors and beams as python objects, so propagation becomes interactions between beams and mirrors, and the optimisation can be automated.

The main ingredients of gtrace are mirrors and beams. A mirror is an instance of the :py:class:`Mirror <gtrace.optcomp.Mirror>` class. You place mirrors in a 2D plane and set their size, curvature, reflectivities, wedge angle and so on. You then launch a beam, an instance of :py:class:`GaussianBeam<gtrace.beam.GaussianBeam>`, from a point in that plane. When the beam hits a mirror it splits into sub-beams: reflections, refractions, and the beams that come back out of the substrate. Those beam objects are yours to propagate further. In non-sequential mode gtrace propagates them for you until a termination condition is met — the beam hits nothing, or its power falls below a threshold, and so on. At the end you have a collection of beam objects, which you can export to a DXF file.

Two ways of working
--------------------

The description above is the low level interface: you hold the mirrors and the beams yourself and pass them around. Everything in gtrace can be done this way, and the :doc:`tutorial` starts there.

On top of it there is a container, :py:class:`OpticalLayout<gtrace.layout.OpticalLayout>`, which holds the optics, the source beams and the tracing rules as one object. Register your optics into a layout and you get tracing, drawing, JSON persistence and the interactive viewer without carrying the lists around by hand. The optics are held *by reference*: the mirror you registered and the mirror in your own variable are the same object, so an edit made in the viewer changes the object in your code. The last part of the :doc:`tutorial` goes through it; :doc:`layout` and :doc:`viewer` describe it in reference detail.

Installation
-------------

::

    pip install gtrace              # the library and the HTML viewer
    pip install "gtrace[notebook]"  # ... and the viewer as a Jupyter widget

Python 3.9 or newer. gtrace itself needs only numpy, scipy and traits.

What the viewer needs
^^^^^^^^^^^^^^^^^^^^^^

The viewer has two front ends, with different requirements:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Front end
     - Needs
   * - Self-contained HTML — ``render_html()``,
       ``show(backend='html')``
     - Nothing beyond gtrace and a web browser
   * - Jupyter widget — ``widget()``, or ``show()`` inside a notebook
     - ``anywidget`` (>= 0.9), which brings ``ipywidgets`` with it,
       and a Jupyter front end

Neither route needs Node.js, a build step, a CDN or network access. The viewer's JavaScript and CSS ship inside the package and are inlined into the page it writes. No CAD software is needed either.

:py:meth:`show<gtrace.layout.OpticalLayout.show>` picks the widget when it is running in a Jupyter kernel with anywidget installed, and writes the HTML file otherwise, so the same code works both ways. Without anywidget the widget raises ``WidgetNotAvailable`` and says so.

From a clone
^^^^^^^^^^^^^

To work from the source tree instead::

    git clone https://github.com/asoy01/gtrace.git
    cd gtrace
    pip install ".[notebook]"

Use ``pip install -e ".[notebook]"`` if you intend to change gtrace itself. The quotes matter in zsh, which would otherwise expand the brackets as a glob.

Running the tutorial
^^^^^^^^^^^^^^^^^^^^^

The :doc:`tutorial` is a pair of Jupyter notebooks. Both are self-contained: they need gtrace and nothing else from the repository, so downloading the ``.ipynb`` files is enough, and the :doc:`tutorial` page links to them on GitHub. A Jupyter front end runs them::

    pip install jupyterlab
    jupyter lab gtrace-tutorial.ipynb

VS Code's notebook editor works too: open the file and select the interpreter you installed gtrace into. Run the cells from the top. The last chapter opens the viewer in the cell output, which is the part that needs anywidget; everything before it works without.

Limitations
------------

gtrace is at this moment limited to 2D optical layouts. This limitation might be lifted in the future.

.. .. math:: \int^{\infty}_{-1} e^{i\omega t} dt


