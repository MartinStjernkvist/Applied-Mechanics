"""
Finite Element Method utilities for course MHA021
=================================================

This module provides plotting utilities for discrete elements, element-level
stiffness/force routines for springs, bars, and 2D frames, as well as helpers
for assembly, solving, and post-processing.

- Author: Jim Brouzoulis (adapted & cleaned by Copilot)
- Version: 2025-11-10

Conventions
-----------
- **Indices** in assembly/connectivity are 1-based in user-facing functions
  (matching many FEM textbooks); internal arrays use 0-based indexing.
- **Frames (2D)** use DOFs per node: ``[ux, uy, rz]`` → element DOFs
  ``[ux1, uy1, rz1, ux2, uy2, rz2]`` in *global* coordinates unless otherwise stated.
- NumPy-style docstrings and examples (copy/paste ready for students).

Dependencies
------------
- numpy
- plotly
- sympy (only for nice LaTeX display via `displayvar`)

"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']=120
import plotly.graph_objects as go
from IPython.display import Math, display, Latex
from typing import  Union


# Add parent directory to path and import functions from mha021.py
"""
# Import necessary libraries
import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
        
from mha021 import *
"""

# ---------------------------------------------------------------------------
# Plotting: discrete mesh and deformed shapes
# ---------------------------------------------------------------------------

def draw_discrete_elements(
    ex: np.ndarray,
    ey: np.ndarray,
    ez: Optional[np.ndarray] = None,
    **kw,
) -> go.Figure:
    """Plot a wireframe-only mesh (2D or 3D) using Plotly.

    Elements are drawn as polylines (with optional closure). Nodes can be shown
    as markers. Optional annotations can label unique nodes (offset outward)
    and element centroids. Hover information for nodes/elements is preserved
    through invisible marker traces.

    Parameters
    ----------
    ex, ey : array_like, shape (nel, nen)
        Element nodal coordinates in the global X and Y directions.
    ez : array_like or None, shape (nel, nen), optional
        Global Z coordinates (provide to enable 3D plotting). Default ``None``.
    **kw : Any
        Optional keyword arguments controlling appearance:
        - ``title`` (str), ``xlabel`` (str), ``ylabel`` (str), ``zlabel`` (str)
        - ``color`` (str): line color; default ``"black"``
        - ``line_style`` (str): ``"solid"|"dash"|"dot"|"dashdot"``
        - ``line_width`` (float): default 2
        - ``closed`` (bool): close each polyline; default True
        - ``equal`` (bool): axis equal; default True
        - ``show_nodes`` (bool): draw node markers; default True
        - ``marker`` (str): Plotly symbol; default ``"circle-open"``
        - ``marker_size`` (float): default 7
        - ``marker_line_width`` (float): default 1.6
        - ``marker_face_color`` (str|None): default None
        - ``annotate`` (bool|{"nodes","elements","both"}): add labels
        - ``node_ids`` (Sequence): labels for unique nodes (else 1..N)
        - ``element_ids`` (Sequence): labels for elements (else 1..nel)
        - ``node_precision`` (int): rounding used for node dedup; default 6

    Returns
    -------
    plotly.graph_objects.Figure
        The configured figure. Call ``fig.show()`` to display.

    Examples
    --------
    >>> import numpy as np
    >>> ex = np.array([[0.0, 1.0], [1.0, 1.0]])
    >>> ey = np.array([[0.0, 0.0], [0.0, 1.0]])
    >>> fig = draw_discrete_elements(ex, ey, annotate='both')
    >>> # fig.show()
    """
    ex = np.atleast_2d(np.asarray(ex, float))
    ey = np.atleast_2d(np.asarray(ey, float))
    if ex.shape != ey.shape:
        raise ValueError("ex and ey must have same shape")

    is3d = ez is not None
    if is3d:
        ez = np.atleast_2d(np.asarray(ez, float))
        if ez.shape != ex.shape:
            raise ValueError("ez must match ex/ey shape")

    nel, nn = ex.shape

    title = kw.get("title", "")
    xlabel = kw.get("xlabel", "X")
    ylabel = kw.get("ylabel", "Y")
    zlabel = kw.get("zlabel", "Z")
    color = kw.get("color", "black")
    dash = {
        "solid": "solid",
        "dash": "dash",
        "dashed": "dash",
        "dot": "dot",
        "dotted": "dot",
        "dashdot": "dashdot",
    }.get(kw.get("line_style", "solid"), "solid")
    lw = float(kw.get("line_width", 2))
    closed = bool(kw.get("closed", True))
    equal = bool(kw.get("equal", True))
    show_nodes = bool(kw.get("show_nodes", True))

    symbol = kw.get("marker", "circle-open")
    open_marker = isinstance(symbol, str) and symbol.endswith("-open")
    base_symbol = symbol[:-5] if open_marker else symbol
    msize = float(kw.get("marker_size", 7))
    mlinew = float(kw.get("marker_line_width", 1.6))
    mface = kw.get("marker_face_color", None)
    if mface is None and open_marker:
        mface = "rgba(0,0,0,0)"  # simulate open marker

    annotate = kw.get("annotate", False)
    node_ids = kw.get("node_ids", None)
    elem_ids = kw.get("element_ids", None)
    node_prec = int(kw.get("node_precision", 6))

    # Characteristic length for label offset
    all_x, all_y = ex.flatten(), ey.flatten()
    if is3d:
        all_z = ez.flatten()
        bbox = np.array([
            all_x.max()-all_x.min(),
            all_y.max()-all_y.min(),
            all_z.max()-all_z.min(),
        ], float)
    else:
        bbox = np.array([
            all_x.max()-all_x.min(),
            all_y.max()-all_y.min(),
        ], float)
    char_len = float(np.linalg.norm(bbox)) or 1.0
    node_off_len = 0.02 * char_len

    # Build polylines with None separators
    X, Y, Z = [], [], []
    for i in range(nel):
        xi, yi = ex[i].tolist(), ey[i].tolist()
        if closed and nn > 1:
            xi += [xi[0]]; yi += [yi[0]]
        X += xi + [None]; Y += yi + [None]
        if is3d:
            zi = ez[i].tolist()
            if closed and nn > 1:
                zi += [zi[0]]
            Z += zi + [None]

    fig = go.Figure()
    if not is3d:
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines",
                                 line=dict(color=color, width=lw, dash=dash),
                                 hoverinfo="skip", name="elements"))
    else:
        fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines",
                                   line=dict(color=color, width=lw),
                                   hoverinfo="skip", name="elements"))

    # Nodes
    if show_nodes:
        node_xy = dict(x=ex.ravel(), y=ey.ravel())
        node_marker = dict(symbol=base_symbol, size=msize,
                           color=(mface or color),
                           line=dict(color=color, width=mlinew))
        if not is3d:
            fig.add_trace(go.Scatter(**node_xy, mode="markers",
                                     marker=node_marker, name="nodes",
                                     hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter3d(**(node_xy | {"z": ez.ravel()}),
                                       mode="markers", marker=node_marker,
                                       name="nodes",
                                       hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"))

    # Helpers: annotations
    def _add_annotations_2d(xs, ys, texts, bg="rgba(255,255,255,0.9)", col="#333", size=12):
        for x, y, t in zip(xs, ys, texts):
            fig.add_annotation(x=x, y=y, xref="x", yref="y", text=str(t),
                               showarrow=False, font=dict(color=col, size=size),
                               bgcolor=bg, bordercolor="#888", borderwidth=1, borderpad=3)

    def _add_annotations_3d(xs, ys, zs, texts, bg="rgba(255,255,255,0.9)", col="#333", size=12):
        existing = []
        if getattr(fig.layout, "scene", None) and getattr(fig.layout.scene, "annotations", None):
            existing = list(fig.layout.scene.annotations)
        new = existing + [dict(x=float(x), y=float(y), z=float(z), text=str(t),
                               showarrow=False, font=dict(color=col, size=size),
                               bgcolor=bg, bordercolor="#888", borderwidth=1, borderpad=3)
                          for x, y, z, t in zip(xs, ys, zs, texts)]
        fig.update_layout(scene=dict(annotations=new))

    def _edges_for_element():
        idxs = list(range(nn))
        pairs = [(idxs[j], idxs[j+1]) for j in range(nn-1)]
        if closed and nn > 1:
            pairs.append((nn-1, 0))
        return pairs

    # Node/element labels
    if annotate:
        want_nodes = annotate in ("nodes", "both", True)
        want_elements = annotate in ("elements", "both")
        if not is3d:
            pts = np.c_[ex.ravel(), ey.ravel()]
        else:
            pts = np.c_[ex.ravel(), ey.ravel(), ez.ravel()]
        rpts = np.round(pts, node_prec)
        _, uniq_idx = np.unique(rpts, axis=0, return_index=True)
        un = pts[np.sort(uniq_idx)]

        # NODES
        if want_nodes:
            if not is3d:
                dir_accum = {i: np.zeros(2) for i in range(un.shape[0])}
                pairs = _edges_for_element()
                for e in range(nel):
                    for i0, i1 in pairs:
                        p0 = np.array([ex[e, i0], ey[e, i0]])
                        p1 = np.array([ex[e, i1], ey[e, i1]])
                        t = p1 - p0
                        nrm = np.linalg.norm(t)
                        if nrm == 0: 
                            continue
                        t /= nrm
                        perp = np.array([-t[1], t[0]])
                        for k, p in enumerate(un):
                            if np.allclose(p, p0, atol=10**(-node_prec)):
                                dir_accum[k] += perp
                            if np.allclose(p, p1, atol=10**(-node_prec)):
                                dir_accum[k] += perp
                xn, yn, ntexts, hxn, hyn, htxt = [], [], [], [], [], []
                for i, p in enumerate(un):
                    d = dir_accum[i]
                    d = d/np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.array([1, 1])/np.sqrt(2)
                    off = node_off_len * d
                    xi, yi = p[0]+off[0], p[1]+off[1]
                    lab = node_ids[i] if node_ids is not None else (i+1)
                    xn.append(xi); yn.append(yi); ntexts.append(lab)
                    hxn.append(xi); hyn.append(yi); htxt.append(f"Node {lab}")
                _add_annotations_2d(xn, yn, ntexts, bg="rgba(255,255,255,0.9)", col="#1f77b4", size=12)
                fig.add_trace(go.Scatter(x=hxn, y=hyn, mode="markers",
                                         marker=dict(size=1, color="rgba(0,0,0,0)"),
                                         hovertext=htxt, hoverinfo="text", showlegend=False))
            else:
                dir_accum = {i: np.zeros(3) for i in range(un.shape[0])}
                pairs = _edges_for_element()
                for e in range(nel):
                    for i0, i1 in pairs:
                        p0 = np.array([ex[e, i0], ey[e, i0], ez[e, i0]])
                        p1 = np.array([ex[e, i1], ey[e, i1], ez[e, i1]])
                        t = p1 - p0
                        nrm = np.linalg.norm(t)
                        if nrm == 0:
                            continue
                        t /= nrm
                        ref = np.array([0, 0, 1]) if abs(np.dot(t, [0, 0, 1])) < 0.95 else np.array([0, 1, 0])
                        perp = np.cross(ref, t)
                        perp = perp/np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else np.array([1, 0, 0])
                        for k, p in enumerate(un):
                            if np.allclose(p, p0, atol=10**(-node_prec)):
                                dir_accum[k] += perp
                            if np.allclose(p, p1, atol=10**(-node_prec)):
                                dir_accum[k] += perp
                xn, yn, zn, ntexts = [], [], [], []
                hxn, hyn, hzn, htxt = [], [], [], []
                for i, p in enumerate(un):
                    d = dir_accum[i]
                    d = d/np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.array([1, 1, 1])/np.sqrt(3)
                    off = node_off_len * d
                    xi, yi, zi = p[0]+off[0], p[1]+off[1], p[2]+off[2]
                    lab = node_ids[i] if node_ids is not None else (i+1)
                    xn.append(xi); yn.append(yi); zn.append(zi); ntexts.append(lab)
                    hxn.append(xi); hyn.append(yi); hzn.append(zi); htxt.append(f"Node {lab}")
                _add_annotations_3d(xn, yn, zn, ntexts, bg="rgba(255,255,255,0.9)", col="#1f77b4", size=12)
                fig.add_trace(go.Scatter3d(x=hxn, y=hyn, z=hzn, mode="markers",
                                           marker=dict(size=2, color="rgba(0,0,0,0)"),
                                           hovertext=htxt, hoverinfo="text", showlegend=False))
        # ELEMENTS
        if want_elements:
            cx, cy = ex.mean(axis=1), ey.mean(axis=1)
            if not is3d:
                ex_hov, ey_hov, etext, htxt = [], [], [], []
                for i in range(nel):
                    lab = elem_ids[i] if elem_ids is not None else (i+1)
                    _add_annotations_2d([cx[i]], [cy[i]], [lab], bg="lightgrey", col="#000", size=12)
                    ex_hov.append(cx[i]); ey_hov.append(cy[i]); etext.append(lab); htxt.append(f"Element {lab}")
                fig.add_trace(go.Scatter(x=ex_hov, y=ey_hov, mode="markers",
                                         marker=dict(size=1, color="rgba(0,0,0,0)"),
                                         hovertext=htxt, hoverinfo="text", showlegend=False))
            else:
                cz = ez.mean(axis=1)
                ex_hov, ey_hov, ez_hov, etext, htxt = [], [], [], [], []
                for i in range(nel):
                    lab = elem_ids[i] if elem_ids is not None else (i+1)
                    _add_annotations_3d([cx[i]], [cy[i]], [cz[i]], [lab], bg="lightgrey", col="#000", size=12)
                    ex_hov.append(cx[i]); ey_hov.append(cy[i]); ez_hov.append(cz[i]); etext.append(lab); htxt.append(f"Element {lab}")
                fig.add_trace(go.Scatter3d(x=ex_hov, y=ey_hov, z=ez_hov, mode="markers",
                                           marker=dict(size=2, color="rgba(0,0,0,0)"),
                                           hovertext=htxt, hoverinfo="text", showlegend=False))

    # Layout
    if not is3d:
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                          template="plotly_white", showlegend=False,
                          margin=dict(l=40, r=20, t=60, b=40))
        if equal:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_layout(title=title, template="plotly_white", showlegend=False,
                          margin=dict(l=0, r=0, t=60, b=0),
                          scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel,
                                     aspectmode="data" if equal else "auto"))
    return fig

# Alias kept for backwards compatibility (as in legacy scripts)
eldraw = draw_discrete_elements


def plot_deformed_bars(
    fig: go.Figure,
    ex: np.ndarray,
    ey: np.ndarray,
    ed: np.ndarray,
    scale: Optional[float] = None,
    color: str = "blue",
    width: int = 3,
    name: str = "deformed",
) -> go.Figure:
    """Plot deformed shape of 2-node bar elements in 2D.

    Displacements are assumed given as element arrays ``[u1x, u1y, u2x, u2y]``
    per element in *global* coordinates. If ``scale`` is ``None``, an automatic
    visual scaling is computed from average element length and displacement.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing figure to add traces to.
    ex, ey : array_like, shape (nel, 2)
        Node coordinates for each element.
    ed : array_like, shape (nel, 4)
        Element nodal displacements in global coords: ``[ux1, uy1, ux2, uy2]``.
    scale : float or None
        Visual scale factor for displacements. If ``None``, computed.
    color : str
        Line color for deformed shape; default ``"blue"``.
    width : int
        Line width; default 3.
    name : str
        Trace name; default ``"deformed"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The updated figure.

    Examples
    --------
    >>> import numpy as np, plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> ex = np.array([[0.0, 1.0]])
    >>> ey = np.array([[0.0, 0.0]])
    >>> ed = np.array([[0.0, 0.0, 0.01, 0.0]])
    >>> fig = plot_deformed_bars(fig, ex, ey, ed, scale=None)
    >>> # fig.show()
    """
    ex = np.asarray(ex, float); ey = np.asarray(ey, float); ed = np.asarray(ed, float)
    if scale is None:
        lengths = np.sqrt((ex[:, 1] - ex[:, 0])**2 + (ey[:, 1] - ey[:, 0])**2)
        avg_length = np.mean(lengths)
        deformation_magnitudes = np.linalg.norm(ed[:, 0:4].reshape(-1, 2), axis=1)
        avg_deformation = np.mean(deformation_magnitudes)
        scale = (avg_length / (10 * avg_deformation)) if avg_deformation > 0 else 1.0

    for el in range(ed.shape[0]):
        p1 = np.array([ex[el, 0], ey[el, 0]], float) + scale * np.array(ed[el, 0:2], float)
        p2 = np.array([ex[el, 1], ey[el, 1]], float) + scale * np.array(ed[el, 2:4], float)
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                                 mode="lines", line=dict(color=color, width=width),
                                 name=name, hoverinfo="skip"))

    fig.add_annotation(text=f"Scale factor: {scale:.3f}", xref="paper", yref="paper",
                       x=0.99, y=0.01, showarrow=False,
                       font=dict(size=12, color="black"), align="right",
                       bgcolor="rgba(255,255,255,0.7)")
    return fig


# ---------------------------------------------------------------------------
# Element matrices and section results (springs, bars, 2D frames)
# ---------------------------------------------------------------------------

def spring1e(k: float) -> np.ndarray:
    """Two-node axial spring element stiffness (local coordinates).

    Parameters
    ----------
    k : float
        Spring stiffness.

    Returns
    -------
    ndarray, shape (2, 2)
        Local element stiffness matrix.

    Examples
    --------
    >>> spring1e(100.0)
    array([[ 100., -100.],
           [-100.,  100.]])
    """
    return k * np.array([[1.0, -1.0], [-1.0, 1.0]])


def spring1s(k: float, ed: Sequence[float]) -> float:
    """Axial force in a two-node spring.

    Parameters
    ----------
    k : float
        Spring stiffness.
    ed : sequence of float, length 2
        End displacements in the spring's local axis ``[u1, u2]``.

    Returns
    -------
    float
        Axial force (positive in tension).

    Examples
    --------
    >>> spring1s(100.0, [0.0, 0.01])
    1.0
    """
    ed = np.asarray(ed, float).reshape(2)
    return float(k * (ed[1] - ed[0]))


def bar2e(ex: Sequence[float], ey: Sequence[float], E: float, A: float, qx: Optional[float] = None):
    """2D bar (truss) element: global stiffness and optional consistent load.

    Parameters
    ----------
    ex, ey : sequence of float, length 2
        Element node coordinates ``[x1, x2]`` and ``[y1, y2]``.
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.
    qx : float, optional
        Distributed axial load along the bar (local x), positive when tensile.

    Returns
    -------
    Ke : ndarray, shape (4, 4)
        Global element stiffness matrix.
    fe : ndarray, shape (4, 1)
        Global consistent load vector, only if ``qx`` is provided.

    Examples
    --------
    >>> Ke = bar2e([0, 1], [0, 0], E=210e9, A=1e-4)
    >>> Ke.shape
    (4, 4)
    >>> Ke_q, fe = bar2e([0, 1], [0, 0], 210e9, 1e-4, qx=1000.0)
    >>> fe.ravel().sum() != 0
    True
    """
    DEA = E * A
    qx = qx or 0.0
    x1, x2 = float(ex[0]), float(ex[1])
    y1, y2 = float(ey[0]), float(ey[1])
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    if L <= 0:
        raise ValueError("Zero length element")

    Kle = (DEA / L) * np.array([[1, -1], [-1, 1]], float)
    fle = qx * L * np.array([[0.5], [0.5]], float)

    nx, ny = dx / L, dy / L
    G = np.array([[nx, ny, 0, 0], [0, 0, nx, ny]], float)
    Ke = G.T @ Kle @ G
    fe = G.T @ fle
    return (Ke, fe) if qx else Ke


def bar2s(ex: Sequence[float], ey: Sequence[float], E: float, A: float, ed: Sequence[float]) -> float:
    """Axial force in a 2D bar (truss) element from global displacements.

    Parameters
    ----------
    ex, ey : sequence of float, length 2
        Element node coordinates ``[x1, x2]`` and ``[y1, y2]``.
    E : float
        Young's modulus.
    A : float
        Cross-sectional area.
    ed : sequence of float, length 4
        Element nodal displacements in *global* coordinates:
        ``[ux1, uy1, ux2, uy2]``.

    Returns
    -------
    float
        Axial force (positive in tension).

    Examples
    --------
    >>> ex, ey = [0, 1], [0, 0]
    >>> ed = [0.0, 0.0, 0.001, 0.0]
    >>> round(bar2s(ex, ey, 210e9, 1e-4, ed)) > 0
    True
    """
    x1, x2 = float(ex[0]), float(ex[1])
    y1, y2 = float(ey[0]), float(ey[1])
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    if L <= 0:
        raise ValueError("Zero length element")
    nx, ny = dx / L, dy / L
    G = np.array([[nx, ny, 0, 0], [0, 0, nx, ny]], float)
    a = G @ np.reshape(np.asarray(ed, float), (4, 1))
    delta = a[1] - a[0]  # local elongation
    N = E * A * (delta / L)
    return float(N)


# Geometry helpers

def _L_dir(ex: Sequence[float], ey: Sequence[float], ez: Optional[Sequence[float]] = None):
    x1, x2 = float(ex[0]), float(ex[1])
    y1, y2 = float(ey[0]), float(ey[1])
    if ez is None:
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L <= 0:
            raise ValueError("Zero length element")
        l, m = dx / L, dy / L
        return L, (l, m)
    else:
        z1, z2 = float(ez[0]), float(ez[1])
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        L = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        if L <= 0:
            raise ValueError("Zero length element")
        return L, (dx / L, dy / L, dz / L)


def beam1e(L, E, I, qY=None):
    """
    1D Euler–Bernoulli beam element (local) with DOFs/node [w, theta] -> 4x4.

    Parameters
    ----------
    L : float
        Length. 
    E : float
        Young's modulus.
    I : float
        Second moment of area about z (out of the x–y plane of the line).
    qY : float, optional
        Uniform transverse load (local), positive in +w. If provided, the
        consistent load vector is returned together with Ke.

    Returns
    -------
    Ke : (4,4) ndarray
        Local element stiffness (Euler–Bernoulli).
    fe : (4,1) ndarray, optional
        Local consistent nodal load for uniform qY, if qY is not None.

    Notes
    -----
    - Local DOF ordering is [w1, th1, w2, th2].
    - No coordinate transformation is applied (1D local operator).
    """

    EI = E * I
    L2 = L * L
    L3 = L * L * L

    Ke = (EI / L3) * np.array([
        [ 12.0,    6.0*L,  -12.0,    6.0*L],
        [  6.0*L,  4.0*L2,  -6.0*L,  2.0*L2],
        [ -12.0,  -6.0*L,   12.0,   -6.0*L],
        [  6.0*L,  2.0*L2,  -6.0*L,  4.0*L2],
    ], float)

    if qY is None:
        return Ke

    fe = np.array([
        [ qY*L/2.0],
        [ qY*L**2/12.0],
        [ qY*L/2.0],
        [-qY*L**2/12.0],
    ], float)

    return Ke, fe


def beam1s(L, E, I, ed, nep=21):
    """
    Section fields along a 1D Euler–Bernoulli beam element from FE DOFs only.

    Parameters
    ----------
    L : float
        Length.
    E, I : float
        Material and section property.
    ed : sequence of float, length 4
        Local element DOFs [w1, th1, w2, th2].
    nep : int, default 21
        Number of evaluation points on [0, L].

    Returns
    -------
    dict
        {
          "X": X,         # (nep,1) local positions in [0,L]
          "w": w,         # (nep,1) deflection
          "theta": th,    # (nep,1) rotation
          "V": V,         # (nep,1) shear force  (V = EI * w''' )
          "M": M,         # (nep,1) bending moment (M = EI * w'')
        }

    Notes
    -----
    - Sign convention: M = -EI * d2w/dx2 (hogging positive).
      Flip sign in post-processing if you prefer hogging-positive moment.
    - This routine is purely local (no global transform).
    """
    EI = E * I

    ed = np.asarray(ed, float).reshape(4)
    X = np.linspace(0.0, L, int(nep))
    xi = X / L

    w_vals = []
    th_vals = []
    M_vals = []
    V_vals = []

    for t in xi:
        # your Hermite operator: (N, dNdx, d2Ndx2, d3Ndx3)
        Nw, dNdx, d2Ndx2, d3Ndx3 = _hermite_ops(L, float(t))

        w  = Nw @ ed
        th = dNdx @ ed
        M  = -EI * (d2Ndx2 @ ed)
        V  = -EI * (d3Ndx3 @ ed)

        w_vals.append(w)
        th_vals.append(th)
        M_vals.append(M)
        V_vals.append(V)

    w = np.array(w_vals, float)
    th = np.array(th_vals, float)
    M  = np.array(M_vals, float)
    V  = np.array(V_vals, float)

    return {"X": X, "w": w, "theta": th, "V": V, "M": M}





# ---  2D frame element (Euler–Bernoulli) ---
def beam2e(
    ex: Sequence[float],
    ey: Sequence[float],
    E: float,
    A: float,
    I: float,
    qXY: Optional[Sequence[float]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    2D frame (Euler–Bernoulli) element: returns global stiffness (6x6)
    and optional consistent global load vector (6x1) for uniform local loads.

    DOFs/node (2D frame): [ux, uy, rz]  -> element DOFs: [ux1, uy1, rz1, ux2, uy2, rz2]

    Args
    ----
    ex, ey : [x1,x2], [y1,y2] nodal coordinates
    E, A, I: material and section properties
    qXY    : optional uniform distributed loads in LOCAL axes [qX, qY]
             qX > 0 acts along local +x (axial), qY > 0 acts along local +y (transverse)

    Returns
    -------
    Ke           : (6x6) element stiffness in GLOBAL axes
    (Ke, fe_global) if qXY is provided: also returns (6x1) consistent global load.
    """
    # --- geometry & local direction cosines via original helper ---
    L, (c, s) = _L_dir(ex, ey)
    
    EA, EI = E * A, E * I

    # FULL local 6x6 stiffness assembled
    # DOF order: [ux1, uy1, rz1, ux2, uy2, rz2]
    L2, L3 = L * L, L * L * L

    # axial part on [ux1, ux2]
    k_ax = (EA / L) * np.array([[ 1.0, -1.0],
                                [-1.0,  1.0]], dtype=float)

    # bending (Euler–Bernoulli) on [uy1, rz1, uy2, rz2]
    k_b = np.array([
        [ 12*EI/L3,   6*EI/L2,  -12*EI/L3,   6*EI/L2],
        [  6*EI/L2,   4*EI/L,    -6*EI/L2,   2*EI/L ],
        [-12*EI/L3,  -6*EI/L2,   12*EI/L3,  -6*EI/L2],
        [  6*EI/L2,   2*EI/L,    -6*EI/L2,   4*EI/L ],
    ], dtype=float)

    # place submatrices into local 6x6
    k_loc = np.zeros((6, 6), dtype=float)
    k_loc[np.ix_([0, 3], [0, 3])] = k_ax               # axial -> indices [0, 3]
    k_loc[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = k_b    # bending -> indices [1, 2, 4, 5]

    # --- local consistent load vector (6x1), if requested ---
    fe_loc = None
    if qXY is not None:
        qX = float(qXY[0])
        qY = float(qXY[1])
        fe_loc = np.zeros((6, 1), dtype=float)

        # axial uniform load qX -> consistent nodal forces on ux DOFs
        if qX != 0.0:
            fe_loc[[0, 3], 0] += (qX * L / 2.0) * np.array([1.0, 1.0])

        # transverse uniform load qY -> standard EB consistent loads on [uy, rz]
        if qY != 0.0:
            fe_b = np.array([  qY * L / 2.0,
                               qY * L2 / 12.0,
                               qY * L / 2.0,
                              -qY * L2 / 12.0 ], dtype=float).reshape(4, 1)
            fe_loc[[1, 2, 4, 5], 0:1] += fe_b

    # --- transform to GLOBAL ---
    # 3x3 nodal rotation (local -> global for [ux, uy, rz] at a node)
    Tn = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    # geometric transformation matrix (6x6 block-diagonal)
    Lmat = np.zeros((6, 6), dtype=float)
    Lmat[:3, :3] = Tn
    Lmat[3:, 3:] = Tn

    Ke = Lmat.T @ k_loc @ Lmat
    if fe_loc is not None:
        feg = Lmat.T @ fe_loc
        return Ke, feg
    return Ke



import numpy as np
from typing import Sequence, Optional, Dict, Any

def beam2s(
    ex: Sequence[float],
    ey: Sequence[float],
    ed: Sequence[float],
    E: float,
    A: float,
    I: float,
) -> Dict[str, Any]:
    """
    Boundary sectional forces for a 2D frame (Euler–Bernoulli) element.

    Computes forces only at the two boundaries (x=0 and x=L) using
        f^e = K^e * a^e,
    then maps to the sectional force vector in LOCAL axes as
        [-N(0), -V(0), M(0), N(L), V(L), -M(L)].

    Args
    ----
    ex, ey : nodal coordinates [x1,x2], [y1,y2]
    ed     : element displacement vector in GLOBAL coords
             DOF order: [ux1, uy1, rz1, ux2, uy2, rz2]
    E, A, I: material and section properties
    nep    : (unused; kept for signature compatibility)

    Returns
    -------
    dict with:
      - "X"      : np.array([0.0, L])   (positions along the element)
      - "forces" : (6,1) array with [-N(0), -V(0), M(0), N(L), V(L), -M(L)]
    """
    # --- geometry and local frame ---
    L, dircos = _L_dir(ex, ey)         # expects (l, m) for 2D
    l, m = float(dircos[0]), float(dircos[1])

    # 3x3 rotation for a node's [ux, uy, rz] (local <- global for forces)
    Tn = np.array([[ l,  m, 0.0],
                   [-m,  l, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)

    # Block-diagonal 6x6 geometric transformation (rename G -> L)
    Lmat = np.zeros((6, 6), dtype=float)
    Lmat[:3, :3] = Tn
    Lmat[3:, 3:] = Tn

    # --- element stiffness from beam2e (GLOBAL coordinates) ---
    Ke = beam2e(ex, ey, E, A, I)  # call without loads -> returns 6x6 GLOBAL K
    # If your beam2e might return (Ke, fe), extract Ke:
    if isinstance(Ke, tuple):
        Ke = Ke[0]

    # --- nodal forces in GLOBAL and then map to LOCAL ---
    ed = np.asarray(ed, dtype=float).reshape(6)
    f_glob = Ke @ ed                 # [Fx1, Fy1, Mz1, Fx2, Fy2, Mz2] in GLOBAL
    f_loc  = Lmat @ f_glob           # rotate to LOCAL: f_loc = L * f_glob

    # Map to sectional forces at x=0 and x=L with requested signs
    N0, V0, M0 = f_loc[0], f_loc[1], f_loc[2]
    NL, VL, ML = f_loc[3], f_loc[4], f_loc[5]

    return {
        "N": [-N0, NL],
        "V": [-V0, VL],
        "M": [M0, -ML]
    }

# Shape functions (Hermite for beam; linear for bar)

def _hermite_ops(L: float, xi: float):
    # Hermite shapes in xi in [0,1] for [w1, th1, w2, th2]
    N = np.array([1 - 3*xi**2 + 2*xi**3,
                  L*(xi - 2*xi**2 + xi**3),
                  3*xi**2 - 2*xi**3,
                  L*(-xi**2 + xi**3)], float)
    dNdx = np.array([(-6*xi + 6*xi**2)/L,
                     (1 - 4*xi + 3*xi**2),
                     ( 6*xi - 6*xi**2)/L,
                     (-2*xi + 3*xi**2)], float)
    d2Ndx2 = np.array([(-6 + 12*xi)/L**2,
                       (-4 + 6*xi)/L,
                       ( 6 - 12*xi)/L**2,
                       (-2 + 6*xi)/L], float)
    d3Ndx3 = np.array([12/L**3, 6/L**2, -12/L**3, 6/L**2], float)
    return N, dNdx, d2Ndx2, d3Ndx3


def _plot_deformed_beam(fig: go.Figure, ex, ey, ed, scale=1.0, color="blue", width=3, name="deformed") -> go.Figure:
    """Plot deformed shape for a single 2D frame element using Hermite v(x).

    Parameters
    ----------
    fig : Figure
        Figure to append to.
    ex, ey : sequence of float, length 2
        Element node coordinates.
    ed : sequence of float, length 6
        Global element DOFs ``[ux1, uy1, rz1, ux2, uy2, rz2]``.
    scale : float
        Visual scale for deformations.
    color : str
        Line color.
    width : int
        Line width.
    name : str
        Trace base name.

    Returns
    -------
    Figure
        Updated figure.
    """
    L, (c, s) = _L_dir(ex, ey)
    xhat = np.array([c, s]); yhat = np.array([-s, c])
    p0 = np.array([ex[0], ey[0]], float)

    Tn = np.array([[ c, s, 0], [-s, c, 0], [0, 0, 1]], float)
    T = np.zeros((6, 6), float); T[:3, :3] = Tn; T[3:, 3:] = Tn
    a = (T @ np.reshape(np.asarray(ed, float), (6, 1))).ravel()

    u1, u2 = a[0], a[3]
    v1, rz1, v2, rz2 = a[1], a[2], a[4], a[5]

    Xs = np.linspace(0.0, L, 101); xi = Xs / L
    pts = []
    for t, x in zip(xi, Xs):
        N, _, _, _ = _hermite_ops(L, float(t))
        v = float(N @ np.array([v1, rz1, v2, rz2]))
        u = (1 - t) * u1 + t * u2
        loc = x * xhat + scale * (u * xhat + v * yhat)
        pts.append(p0 + loc)
    pts = np.vstack(pts)
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines",
                             line=dict(color=color, width=width),
                             name=name, hoverinfo="skip"))
    return fig


def plot_deformed_beams(fig: go.Figure, ex_list, ey_list, ed_list, scale=1.0, color="blue", width=3, include_axial=False, name="deformed") -> go.Figure:
    """Plot deformed shape for multiple 2D frame elements.

    Parameters
    ----------
    fig : Figure
        Figure to append to.
    ex_list, ey_list : sequence of array_like
        Lists (or tuples) of element coordinates.
    ed_list : sequence of array_like
        List of global element displacement vectors ``[ux1, uy1, rz1, ux2, uy2, rz2]``.
    scale : float
        Visual scale factor.
    color : str
        Line color.
    width : int
        Line width.
    include_axial : bool
        Reserved (axial included in `_plot_deformed_beam`).
    name : str
        Base trace name; element index is appended if multiple.

    Returns
    -------
    Figure
        Updated figure.

    Examples
    --------
    >>> import numpy as np, plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> ex_list = [np.array([0.0, 1.0])]
    >>> ey_list = [np.array([0.0, 0.0])]
    >>> ed_list = [np.array([0, 0, 0, 0.0, 0.01, 0])]
    >>> fig = plot_deformed_beams(fig, ex_list, ey_list, ed_list, scale=10)
    >>> # fig.show()
    """
    for i, (ex, ey, ed) in enumerate(zip(ex_list, ey_list, ed_list)):
        elem_name = f"{name}_{i+1}" if len(ex_list) > 1 else name
        fig = _plot_deformed_beam(fig, ex, ey, ed, scale=scale, color=color, width=width, name=elem_name)
    return fig


def _lagrange_bar_ops(L: float):
    # Linear bar in xi in [0,1]: u = [1-xi, xi]·[u1,u2]
    N = lambda xi: np.array([1 - xi, xi], float)
    dNdx = np.array([-1.0 / L, 1.0 / L], float)  # constant
    return N, dNdx


# ---------------------------------------------------------------------------
# System-level helpers: pretty display, solver, DOF extraction, assembly
# ---------------------------------------------------------------------------

def displayvar(name: str, var, post: Optional[str] = None, accuracy: Optional[int] = None) -> None:
    """Display a variable as LaTeX: ``name = value``.

    Uses SymPy's LaTeX printer. If ``var`` is a NumPy array, it is converted to
    a SymPy Matrix for crisp typesetting. If ``accuracy`` is given, the value is
    shown approximately with that many significant digits.

    Parameters
    ----------
    name : str
        Symbolic name to display.
    var : Any
        Value to display (number, array, sympy expression, ...).
    accuracy : int, optional
        Number of significant digits for approximate print. If ``None``, exact
        expressions are printed when possible.

    Examples
    --------
    >>> displayvar("P", 1)
    >>> import numpy as np; displayvar("K", np.eye(2))
    >>> displayvar("pi", sp.pi, accuracy=5)
    """
    if isinstance(var, np.ndarray):
        var = sp.Matrix(var)
    if accuracy is None:
        display(Math(f"{name} = {sp.latex(var)}") )
    else:
        display(Math(f"{name} \\approx {sp.latex(sp.sympify(var).evalf(accuracy))}"))



def solve_eq(K, f, bc_dofs=None, bc_vals=None):
    """
    Solve the linear system K a = f with optional essential boundary conditions.

    This generalized solver accepts omitted/None/empty boundary lists and will
    solve the unconstrained system in those cases. It uses the course’s 1-based
    DOF numbering convention for `bc_dofs`.

    Parameters
    ----------
    K : (n, n) array_like
        Global stiffness matrix (assumed symmetric positive definite on the
        free-DOF subspace when constraints are applied).
    f : (n,) or (n, 1) array_like
        Global right-hand side (load vector). Column vectors are accepted and
        flattened internally; the return arrays are 1-D.
    bc_dofs : sequence of int, optional
        1-based indices of essential degrees of freedom. If None or empty,
        the problem is treated as unconstrained.
    bc_vals : sequence of float, optional
        Values to prescribe at the essential DOFs; must be the same length as
        `bc_dofs` when provided.

    Returns
    -------
    a : (n,) ndarray
        Solution vector of unknowns.
    r : (n,) ndarray
        Reaction vector defined as r = K a - f (nonzero primarily at
        constrained DOFs when constraints are applied).

    Notes
    -----
    - DOF numbering for constraints is **1-based** to match the course
      assembly/solver conventions elsewhere in the module.
    - If no constraints are provided (or `bc_dofs`/`bc_vals` are empty),
      the full system is solved directly.
    - For constrained solves, the method performs the usual partitioning:
        K_ff a_f = f_f - K_fc a_c
      where the constrained values a_c are taken from `bc_vals`.

    Examples
    --------
    Unconstrained solve (no BCs):
    >>> K = np.array([[2., -1.],
    ...               [-1., 2.]])
    >>> f = np.array([1., 0.])
    >>> a, r = solve_eq(K, f)
    >>> np.allclose(K @ a, f)
    True
    >>> r.shape
    (2,)

    Constrained solve with 1-based DOFs:
    >>> K = np.array([[2., -1.],
    ...               [-1., 2.]])
    >>> f = np.array([1., 0.])
    >>> # Constrain DOF 1 (1-based) to 0.0
    >>> a, r = solve_eq(K, f, bc_dofs=[1], bc_vals=[0.0])
    >>> a[0]  # constrained
    0.0
    >>> np.allclose(K @ a - f, r)  # reactions
    True

    Column-vector f is accepted:
    >>> f_col = np.array([[1.0], [0.0]])
    >>> a, r = solve_eq(K, f_col, bc_dofs=[1], bc_vals=[0.0])
    >>> a.shape, r.shape
    ((2,), (2,))
    """
    # Convert to arrays and flatten f for consistent algebra
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).reshape(-1)

    n = K.shape[0]
    if K.ndim != 2 or K.shape[1] != n or f.shape[0] != n:
        raise ValueError("Incompatible shapes: K must be (n,n) and f must be (n,) or (n,1).")

    # If constraints are absent (None) or empty, solve directly
    if bc_dofs is None or bc_vals is None:
        a = np.linalg.solve(K, f)
        r = K @ a - f
        return a, r

    # Validate constraints
    bc_dofs = np.asarray(bc_dofs, dtype=int)
    bc_vals = np.asarray(bc_vals, dtype=float)
    if bc_dofs.shape[0] != bc_vals.shape[0]:
        raise ValueError("bc_dofs and bc_vals must have the same length.")
    if np.any(bc_dofs <= 0) or np.any(bc_dofs > n):
        raise ValueError("bc_dofs must be valid 1-based indices in [1, n].")

    # Convert to 0-based for indexing; build constrained/free sets
    c = bc_dofs - 1
    c = np.unique(c)  # guard against duplicates
    a = np.zeros(n, dtype=float)
    a[c] = bc_vals

    all_dofs = np.arange(n, dtype=int)
    fset = np.setdiff1d(all_dofs, c, assume_unique=False)

    # Partitioned solve on free DOFs
    if fset.size == 0:
        # All DOFs constrained: just compute reactions r = K a - f
        r = K @ a - f
        return a, r

    Kff = K[np.ix_(fset, fset)]
    Kfc = K[np.ix_(fset, c)]
    ff  = f[fset]

    rhs = ff - Kfc @ a[c]
    a[fset] = np.linalg.solve(Kff, rhs)

    # Reactions everywhere (mostly nonzero at constrained DOFs)
    r = K @ a - f
    return a, r


def extract_dofs(a: Sequence[float], Edofs: np.ndarray) -> np.ndarray:
    """Extract element DOFs from the global displacement vector.

    Parameters
    ----------
    a : sequence of float, shape (n,)
        Global displacement vector.
    Edofs : ndarray, shape (nel, nen_dofs)
        Element DOF connectivity matrix with **1-based** global DOF numbers.

    Returns
    -------
    ndarray, shape (nel, nen_dofs)
        For each element, the corresponding set of DOFs extracted from ``a``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1, 7)
    >>> Edofs = np.array([[1, 2, 3], [4, 5, 6]])
    >>> extract_dofs(a, Edofs)
    array([[1., 2., 3.],
           [4., 5., 6.]])
    """
    a = np.asarray(a, float).reshape(-1)
    Edofs = np.asarray(Edofs, int)
    nel, nen_dofs = Edofs.shape
    Aelem = np.zeros((nel, nen_dofs), float)
    for i in range(nel):
        Aelem[i, :] = a[Edofs[i, :] - 1]
    return Aelem


# def assem(K: np.ndarray, Ke: np.ndarray, dofs: Sequence[int]) -> np.ndarray:
#     """Assemble an element contribution into a global vector or matrix.

#     ``dofs`` are **1-based** indices. If ``K`` is square, ``Ke`` is assumed to
#     be a square element matrix of compatible size and is added into ``K`` at the
#     corresponding DOF positions. If ``K`` is a column vector, ``Ke`` is assumed
#     to be an element right-hand-side vector (column) and is added accordingly.

#     Parameters
#     ----------
#     K : ndarray
#         Global target (square matrix or column vector).
#     Ke : ndarray
#         Element matrix (if ``K`` is square) or element vector (if ``K`` is a
#         column vector).
#     dofs : sequence of int
#         **1-based** global DOF numbers for this element.

#     Returns
#     -------
#     ndarray
#         Updated global array ``K`` (mutated and returned for convenience).

#     Examples
#     --------
#     >>> import numpy as np
#     >>> K = np.zeros((4, 4))
#     >>> Ke = np.ones((2, 2))
#     >>> assem(K, Ke, [1, 3])
#     array([[1., 0., 1., 0.],
#            [0., 0., 0., 0.],
#            [1., 0., 1., 0.],
#            [0., 0., 0., 0.]])
#     """
#     K = np.asarray(K)
#     Ke = np.asarray(Ke)
#     nrows, ncols = K.shape

#     dofs = np.asarray(dofs, int)
#     if np.max(dofs) > nrows:
#         raise AssertionError("Attempting to assemble into DOFs beyond global size")
#     if np.min(dofs) <= 0:
#         raise AssertionError(f"All DOF numbers must be > 0. dofs = {dofs.tolist()}")

#     if nrows == ncols:  # square matrix
#         for row, dof_i in enumerate(dofs):
#             for col, dof_j in enumerate(dofs):
#                 K[dof_i - 1, dof_j - 1] += Ke[row, col]
#     elif ncols == 1:  # column vector
#         for row, dof_i in enumerate(dofs):
#             K[dof_i - 1, 0] += Ke[row]
#     else:
#         raise AssertionError("K must be square or a column vector")
#     return K
import numpy as np

def assem(K, Ke, dofs):
    """
    Assemble element contribution Ke into global container K at 1-based DOFs.

    Supports:
      - Matrix assembly:   K: (n, n),           Ke: (m, m)
      - Vector assembly:   K: (n,) or (n, 1),   Ke: (m,), (m, 1), or (1, m)

    Parameters
    ----------
    K : array_like
        Global stiffness/matrix (n x n) or global RHS vector (n,) or (n,1).
        NOTE: If you intended a zeros column vector of length 15, use np.zeros((15,1)),
        not np.array((15,1)) (the latter is an array of two scalars [15, 1]).
    Ke : array_like
        Element matrix (m x m) or element RHS vector (m,), (m,1) or (1,m).
    dofs : array_like of int
        1-based DOF indices, shape (m,), (m,1), or (1,m).

    Returns
    -------
    K : ndarray
        Assembled global container (modified in-place and also returned).

    Raises
    ------
    AssertionError
        If shapes or DOF ranges are incompatible.
    """
    # Convert to NumPy arrays (no copy unless needed)
    K = np.asarray(K)
    Ke = np.asarray(Ke)

    # ---- Friendly guard for the common mistake: np.array((n,1)) ----
    # This produces a 1D array of two numbers [n, 1], not a (n,1) array.
    if K.ndim == 1 and K.size == 2 and np.issubdtype(K.dtype, np.integer):
        raise AssertionError(
            "It looks like you passed np.array((n,1)) for K, which creates a 1D array [n,1].\n"
            "To create a column vector, use np.zeros((n,1)). To create a square matrix, use np.zeros((n,n))."
        )

    # ---- Normalize DOFs: accept (m,), (m,1) or (1,m) ----
    dofs = np.asarray(dofs, dtype=int).ravel()
    if dofs.size == 0:
        raise AssertionError("'dofs' is empty.")
    if np.min(dofs) <= 0:
        raise AssertionError(f"All DOF numbers must be > 0. dofs = {dofs.tolist()}")

    m = dofs.size  # element size
    idx = dofs - 1  # 1-based -> 0-based

    # ---- Decide assembly mode based on K ----
    if K.ndim == 2 and K.shape[0] == K.shape[1]:
        # =========================
        #        MATRIX MODE
        # =========================
        n = K.shape[0]
        if np.max(dofs) > n:
            raise AssertionError(
                f"Attempting to assemble into DOFs beyond global size: max(dofs)={np.max(dofs)} > {n}"
            )

        # Ke must be (m, m) after squeezing singleton dims
        Ke2 = np.asarray(Ke)
        if Ke2.ndim == 1:
            # A 1D Ke cannot be a matrix; give a clear error
            raise AssertionError(f"For matrix assembly, Ke must be 2D (m,m). Got 1D Ke with shape {Ke2.shape}.")
        # Squeeze to remove any (m,1) or (1,m) cases that are not square
        Ke2 = np.squeeze(Ke2)
        if Ke2.ndim != 2 or Ke2.shape != (m, m):
            raise AssertionError(f"Ke shape {Ke.shape} incompatible with len(dofs)={m} (expected {(m, m)}).")

        # Assemble
        K[np.ix_(idx, idx)] += Ke2
        return K

    elif (K.ndim == 1) or (K.ndim == 2 and K.shape[1] == 1):
        # =========================
        #        VECTOR MODE
        # =========================
        # Normalize K to 1D view
        if K.ndim == 2:
            if K.shape[1] != 1:
                raise AssertionError(f"For vector assembly, K must be (n,) or (n,1). Got {K.shape}.")
            K_vec = K[:, 0]
        else:
            K_vec = K

        n = K_vec.shape[0]
        if np.max(dofs) > n:
            raise AssertionError(
                f"Attempting to assemble into DOFs beyond global size: max(dofs)={np.max(dofs)} > {n}"
            )

        # Normalize Ke to 1D length m: accept (m,), (m,1), or (1,m)
        Ke2 = np.asarray(Ke)
        if Ke2.ndim == 2:
            if 1 in Ke2.shape and max(Ke2.shape) == m:
                Ke_vec = Ke2.reshape(-1)  # flatten row/col vector
            else:
                raise AssertionError(
                    f"For vector assembly, Ke must be (m,), (m,1) or (1,m). Got Ke.shape={Ke2.shape}, expected length {m}."
                )
        else:
            Ke_vec = Ke2

        Ke_vec = np.asarray(Ke_vec).ravel()
        if Ke_vec.size != m:
            raise AssertionError(
                f"Ke length {Ke_vec.size} incompatible with len(dofs)={m} (vector assembly)."
            )

        # Accumulate (handles potential repeated DOFs robustly)
        np.add.at(K_vec, idx, Ke_vec)

        # If original K was (n,1), reflect changes back
        if K.ndim == 2:
            K[:, 0] = K_vec
            return K
        else:
            return K_vec

    else:
        raise AssertionError(
            f"K must be a square matrix (n,n) or a vector (n,) / (n,1). Got K.shape={K.shape}, K.ndim={K.ndim}."
        )