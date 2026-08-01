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

    const viewer = globalThis.GTraceViewer.mount(host, model.get('scene'), {
        title: model.get('title'),
        onEdit: onEdit
    });

    // Python pushes a new scene whenever the layout is re-traced. Keep
    // the current zoom, pan and layer visibility so that a re-trace does
    // not throw away where the user was looking.
    const onScene = () => viewer.setScene(model.get('scene'));
    const onTitle = () => {
        const t = el.querySelector('.gt-title');
        if (t) { t.textContent = model.get('title'); }
    };
    const onHeight = () => {
        host.style.height = (model.get('height') || 520) + 'px';
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
    };

    model.on('change:scene', onScene);
    model.on('change:title', onTitle);
    model.on('change:height', onHeight);
    model.on('change:error', onError);
    onError();

    // Expose the viewer for debugging and for the tests.
    el.gtraceViewer = viewer;

    return () => {
        model.off('change:scene', onScene);
        model.off('change:title', onTitle);
        model.off('change:height', onHeight);
        model.off('change:error', onError);
        viewer.destroy();
    };
}

export default { render };
