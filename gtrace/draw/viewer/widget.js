/*
 * anywidget front end for the gtrace viewer (Stage 2).
 *
 * This file is not standalone: widget.py concatenates viewer.js in front
 * of it and serves the result as the widget's ESM module. viewer.js is
 * an IIFE that publishes GTraceViewer on globalThis, so the core is
 * already in scope by the time render() runs. Everything specific to
 * the notebook transport lives here, so that the same core can be
 * driven by a static HTML page (Stage 1) or a websocket (Stage 3)
 * without change.
 *
 * Copyright (c) 2011-2026, Yoichi Aso. BSD license.
 */

/*
 * How tall the viewer makes itself when nothing has said.
 *
 * A cell output is a letterbox, and 520 pixels of one is not enough of a
 * bench drawing to work in: the first thing anyone did with it was drag
 * it taller. So the height is taken from the width, which is the cell's
 * own and therefore already the right scale for the window.
 *
 * From the width of the *drawing*, though, not of the widget: the side
 * panel is a fixed 380 pixels of it and is a column of numbers, not part
 * of the picture. Squaring the whole widget makes the drawing itself
 * taller than it is wide, which is the wrong way round for a bench.
 *
 * Capped, because a maximized JupyterLab cell is wider than the window
 * is tall and a viewer whose bottom edge is below the fold is unusable
 * in its own way. The cap never falls below FALLBACK_HEIGHT: an
 * embedder that reports a nonsense viewport - an output frame measuring
 * only what it has so far been given, say - should leave the widget at
 * the size it used to be rather than at nothing.
 */
const SIDE_PANEL_WIDTH = 380;
const NARROW_BREAKPOINT = 700;
const VIEWPORT_FRACTION = 0.70;
const FALLBACK_HEIGHT = 520;

function autoHeight(el) {
    const width = el.getBoundingClientRect().width;
    // Nothing to go on yet. anywidget calls render() before the output
    // area is laid out, so this is the ordinary case on the first pass,
    // not an error: the observer below runs again when a width arrives.
    if (!width) { return 0; }
    // Narrow enough and the side panel stops standing beside the
    // drawing and stacks under it - the same breakpoint the stylesheet
    // uses - so there the drawing has the whole width.
    const drawing = width > NARROW_BREAKPOINT
        ? width - SIDE_PANEL_WIDTH : width;
    const cap = Math.max((globalThis.innerHeight || 0) * VIEWPORT_FRACTION,
                         FALLBACK_HEIGHT);
    const floor = globalThis.GTraceViewer.MIN_HEIGHT;
    return Math.max(floor, Math.round(Math.min(drawing, cap)));
}

function render({ model, el }) {
    const host = document.createElement('div');
    host.className = 'gt-widget';

    // A height of zero means "work it out": see autoHeight. Anything
    // else is a height someone chose, from Python or by dragging the
    // grip, and is used as it stands.
    //
    // Until a width is known there is nothing to work it out from, and
    // the fixed height the widget used to have stands in. That is a
    // provisional answer rather than a fallback to live with: whichever
    // of the three chances below sees a real width first replaces it,
    // and refits, since the drawing was framed against the wrong box.
    let resolved = false;
    const applyHeight = () => {
        const set = model.get('height');
        if (set) {
            host.style.height = set + 'px';
            return true;
        }
        const auto = autoHeight(el);
        host.style.height = (auto || FALLBACK_HEIGHT) + 'px';
        if (!auto || resolved) { return !!auto; }
        // The first time a real width is known. Anything drawn before
        // now was framed against the provisional height.
        resolved = true;
        if (el.gtraceViewer) { el.gtraceViewer.fit(); }
        return true;
    };
    applyHeight();
    el.appendChild(host);

    // Editing is enabled by handing the core somewhere to send edits.
    // Here that is the widget's comm; the live server will hand it a
    // websocket instead, with the same message format.
    const onEdit = model.get('editable')
        ? (msg) => model.send(msg)
        : null;

    // Set from Python, a new height reframes the drawing to suit it.
    // Set by a drag of the bottom edge, it must not: the view is already
    // where the user put it, and refitting would throw away the zoom
    // they were working at.
    let dragged = false;

    // A cell output is a letterbox and a bench drawing is not, so the
    // viewer can be dragged taller by its bottom edge. The height is
    // this element's, so it is written back to the traitlet that set it:
    // that is what makes the new height survive a re-render, and lets
    // Python see what it was dragged to.
    const viewer = globalThis.GTraceViewer.mount(host, model.get('scene'), {
        layoutPath: model.get('layout_path'),
        dxfPath: model.get('dxf_path'),
        onEdit: onEdit,
        resizable: true,
        onResize: (h) => {
            dragged = true;
            model.set('height', h);
            model.save_changes();
        }
    });

    // Python pushes a new scene whenever the layout is re-traced. Keep
    // the current zoom, pan and layer visibility so that a re-trace does
    // not throw away where the user was looking.
    const onScene = () => viewer.setScene(model.get('scene'));
    const onHeight = () => {
        applyHeight();
        if (dragged) { dragged = false; return; }
        viewer.fit();
    };

    // While the height is being worked out, it follows the width: split
    // the notebook pane and the viewer squares itself up again. It stops
    // following the moment anything sets a height - the grip writes one
    // back, so a drag settles it for good.
    //
    // This is also what resolves the first pass. anywidget calls
    // render() before the output area has been laid out, so the width
    // there is zero; the run that matters is the one this fires when a
    // real width arrives. Getting a fit wrong at that moment is what
    // once left the whole drawing scaled down to a point, so the height
    // is only ever applied here - the viewer refits itself off its own
    // observer once it has a size.
    let observer = null;
    if (globalThis.ResizeObserver) {
        observer = new ResizeObserver(() => {
            if (!model.get('height')) { applyHeight(); }
        });
        observer.observe(el);
    }

    // Two more chances, because the observer is not to be relied on for
    // the one run that matters. A resize callback is delivered on a
    // rendering step, and an embedder that has already laid the output
    // area out before calling render() never resizes it again - there
    // is nothing for the observer to report. A headless browser stops
    // delivering them altogether once the page has settled, which is
    // how this was found. Both of these are cheap and idempotent.
    const retry = () => { if (!model.get('height')) { applyHeight(); } };
    if (globalThis.requestAnimationFrame) { requestAnimationFrame(retry); }
    const retryTimer = setTimeout(retry, 0);

    // An edit Python refused: say so rather than leaving the drawing
    // silently disagreeing with what the user just did.
    const banner = document.createElement('div');
    banner.className = 'gt-error';
    banner.style.display = 'none';
    el.appendChild(banner);
    const onError = () => {
        const msg = model.get('error');
        banner.textContent = msg;
        banner.style.display = msg ? '' : 'none';
        // A refused edit sends no new scene, so anything the viewer
        // assumed about it - a rename it followed ahead of Python -
        // has to be put back.
        if (msg) { viewer.revertSelection(); }
    };

    // Confirmation of something that leaves no mark on the drawing.
    const notice = document.createElement('div');
    notice.className = 'gt-notice';
    notice.style.display = 'none';
    el.appendChild(notice);
    const onNotice = () => {
        const msg = model.get('notice');
        notice.textContent = msg;
        notice.style.display = msg ? '' : 'none';
    };

    model.on('change:scene', onScene);
    model.on('change:height', onHeight);
    model.on('change:error', onError);
    model.on('change:notice', onNotice);
    onError();
    onNotice();

    // Expose the viewer for debugging and for the tests. The height
    // resolver goes with it: what it comes to depends on the width of
    // an element and on the size of the window, neither of which a
    // check can arrange and then wait on reliably in a headless run.
    el.gtraceViewer = viewer;
    el.gtraceApplyHeight = applyHeight;

    return () => {
        model.off('change:scene', onScene);
        model.off('change:height', onHeight);
        model.off('change:error', onError);
        model.off('change:notice', onNotice);
        if (observer) { observer.disconnect(); observer = null; }
        clearTimeout(retryTimer);
        viewer.destroy();
    };
}

export default { render };
