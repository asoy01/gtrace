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

function render({ model, el }) {
    const host = document.createElement('div');
    host.className = 'gt-widget';
    host.style.height = (model.get('height') || 520) + 'px';
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
        host.style.height = (model.get('height') || 520) + 'px';
        if (dragged) { dragged = false; return; }
        viewer.fit();
    };

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

    // Expose the viewer for debugging and for the tests.
    el.gtraceViewer = viewer;

    return () => {
        model.off('change:scene', onScene);
        model.off('change:height', onHeight);
        model.off('change:error', onError);
        model.off('change:notice', onNotice);
        viewer.destroy();
    };
}

export default { render };
