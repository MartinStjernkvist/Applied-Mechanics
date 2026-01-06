"""
Finite Element Method utilities for course MHA021
=================================================

This module provides plotting utilities for discrete elements, element-level
stiffness/force routines for springs, bars, and 2D frames, as well as helpers
for assembly, solving, and post-processing.

- Author: Jim Brouzoulis 
- Version: 2025-12-22

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

from typing import List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']=120
import plotly.graph_objects as go
from IPython.display import Math, display
from typing import Union, Dict
from typing import Literal 
ElementType = Literal["quad", "tri"]
import math
from scipy.linalg import eigh
from plotly.express.colors import sample_colorscale


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


def bar2m(
    ex: Sequence[float],
    ey: Sequence[float],
    rho: float,
    A: float
) -> np.ndarray:
    """
    Consistent global mass matrix for a 2D bar (truss) element (4x4).

    DOFs: [u1x, u1y, u2x, u2y] (two translational DOFs per node)

    Args
    ----
    ex, ey : [x1, x2], [y1, y2] nodal coordinates
    rho    : material density
    A      : cross-sectional area

    Returns
    -------
    M : (4,4) ndarray
        Global consistent mass matrix.
    """
    L, (c, s) = _L_dir(ex, ey)
    # Local axial 2x2 consistent mass (linear axial shape functions)
    m_loc = (rho * A * L / 6.0) * np.array([[2.0, 1.0],
                                            [1.0, 2.0]])

    # Map global translational DOFs -> local axial DOFs
    # [u1_axial, u2_axial]^T = S @ [u1x, u1y, u2x, u2y]^T
    S = np.array([[c, s, 0.0, 0.0],
                  [0.0, 0.0, c, s]])

    # Global consistent mass
    M = S.T @ m_loc @ S
    return M


def beam2m(
    ex: Sequence[float],
    ey: Sequence[float],
    rho: float,
    A: float
) -> np.ndarray:
    """
    Consistent global mass matrix for a 2D Euler–Bernoulli frame (beam) element (6x6).

    Local DOFs per node: [u (axial), v (transverse), rz (rotation)]
    Global DOFs: [u1x, u1y, rz1, u2x, u2y, rz2]

    Args
    ----
    ex, ey : [x1, x2], [y1, y2] nodal coordinates
    rho    : material density
    A      : cross-sectional area

    Returns
    -------
    M : (6,6) ndarray
        Global consistent mass matrix (axial + bending) in GLOBAL axes.
    """
    L, (c, s) = _L_dir(ex, ey)

    # ---- Local consistent mass (frame EB) with DOF order [u1, v1, rz1, u2, v2, rz2] ----
    # Axial 2x2 block
    m_axial = (rho * A * L / 6.0) * np.array([[2.0, 1.0],
                                              [1.0, 2.0]])

    # Bending 4x4 block (Euler–Bernoulli)
    m_bend = (rho * A * L / 420.0) * np.array([
        [156.0,     22.0*L,   54.0,     -13.0*L],
        [22.0*L,   4.0*L**2,  13.0*L,   -3.0*L**2],
        [54.0,      13.0*L,  156.0,     -22.0*L],
        [-13.0*L, -3.0*L**2, -22.0*L,   4.0*L**2]
    ])

    M_loc = np.zeros((6, 6))
    # place axial
    M_loc[np.ix_([0, 3], [0, 3])] = m_axial
    # place bending
    M_loc[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = m_bend

    # ---- Transform local [u, v, rz] at each node to global [ux, uy, rz] ----
    R = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]])

    T = np.zeros((6, 6))
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R

    M = T @ M_loc @ T.T
    return M


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


    # Normalize inputs
    K = np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float).reshape(-1)
    n = K.shape[0]

    if K.ndim != 2 or K.shape[1] != n or f.shape[0] != n:
        raise ValueError("Incompatible shapes: K must be (n,n) and f must be (n,) or (n,1).")

    # No essential BCs -> direct solve
    if bc_dofs is None or bc_vals is None or len(bc_dofs) == 0:
        a = np.linalg.solve(K, f)
        r = K @ a - f
        return a, r

    # Convert to arrays
    bc_dofs = np.asarray(bc_dofs, dtype=int)
    bc_vals = np.asarray(bc_vals, dtype=float)

    if bc_dofs.shape[0] != bc_vals.shape[0]:
        raise ValueError("bc_dofs and bc_vals must have the same length.")
    if np.any(bc_dofs <= 0) or np.any(bc_dofs > n):
        raise ValueError("bc_dofs must be valid 1-based indices in [1, n].")

    # Convert to 0-based
    c0 = bc_dofs - 1
    v0 = bc_vals

    # ---- Stable unique on DOFs, preserving the *first* occurrence (and its value) ----
    # np.unique(..., return_index=True) gives first indices; sorting those preserves input order.
    uniq_c, first_idx = np.unique(c0, return_index=True)
    order = np.sort(first_idx)                    # stable keep-first
    c = c0[order]
    vals = v0[order]

    # ---- Check for conflicting duplicates (same DOF with different values) ----
    # If there were duplicates, ensure all values for each DOF match the kept one.
    if len(c) != len(c0):
        # map dof -> kept value
        kept = {d: val for d, val in zip(c, vals)}
        for d_in, v_in in zip(c0, v0):
            if not np.isclose(v_in, kept[d_in]):
                raise ValueError(f"Conflicting duplicate BC for DOF {d_in+1}: "
                                 f"{v_in} vs {kept[d_in]}.")

    # Build free set
    all_dofs = np.arange(n, dtype=int)
    fset = np.setdiff1d(all_dofs, c, assume_unique=False)

    # Initialize solution with zeros, set constrained values
    a = np.zeros(n, dtype=float)
    a[c] = vals

    # If everything is constrained, reactions only
    if fset.size == 0:
        r = K @ a - f
        return a, r

    # Partitioned system
    Kff = K[np.ix_(fset, fset)]
    Kfc = K[np.ix_(fset, c)]
    ff  = f[fset]

    rhs = ff - Kfc @ a[c]
    a[fset] = np.linalg.solve(Kff, rhs)

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

def plot_mesh(
    nodes: np.ndarray,
    elements: np.ndarray,
    edges: Dict[str, np.ndarray],
    node_size: int = 6,
    show_node_ids: bool = False,
    show_element_ids: bool = False,
    title: str = "Mesh",
) -> go.Figure:
    """
    Interactive Plotly visualization of a 2D mesh.

    Parameters
    ----------
    nodes : np.ndarray
        Array of nodal coordinates, shape (N, 2).
    elements : np.ndarray
        Connectivity array, shape (M, k) where k = 3 (triangles) or 4 (quads).
    edges : dict
        Dictionary with boundary node indices for any named node sets.
    node_size : int
        Size of node markers.
    show_node_ids : bool
        If True, display node IDs.
    show_element_ids : bool
        If True, display element IDs.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    # Convert to 0-based indexing
    e0 = elements - 1
    k = elements.shape[1]

    seg_x, seg_y = [], []
    if k == 4:  # quad
        cyc = [0, 1, 2, 3, 0]
    elif k == 3:  # triangle
        cyc = [0, 1, 2, 0]
    else:
        raise ValueError("Elements must have 3 (tri) or 4 (quad) nodes.")

    for conn in e0:
        pts = nodes[conn]
        for a, b in zip(cyc[:-1], cyc[1:]):
            seg_x += [pts[a, 0], pts[b, 0], None]
            seg_y += [pts[a, 1], pts[b, 1], None]

    fig = go.Figure()

    # Element edges
    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y, mode="lines",
        line=dict(color="black", width=1),
        name="Element edges", hoverinfo="skip"
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=nodes[:, 0], y=nodes[:, 1],
        mode="markers+text" if show_node_ids else "markers",
        text=[str(i + 1) for i in range(nodes.shape[0])] if show_node_ids else None,
        textposition="top center",
        marker=dict(size=node_size, color="rgba(120,120,120,0.9)"),
        name="Nodes",
        hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>"
    ))

    # Boundary node sets (dynamic)
    palette = [
        "royalblue", "darkorange", "seagreen", "crimson", "purple", "goldenrod",
        "teal", "magenta", "darkcyan", "olive", "chocolate", "indigo"
    ]
    color_cycle = iter(palette)

    for name, idx in edges.items():
        if len(idx) > 0:
            idx0 = idx - 1  # convert to 0-based
            color = next(color_cycle, "gray")
            fig.add_trace(go.Scatter(
                x=nodes[idx0, 0], y=nodes[idx0, 1],
                mode="markers",
                marker=dict(size=node_size + 2, color=color),
                name=name.capitalize(),
                hovertemplate=f"{name} node<extra></extra>"
            ))

    # Element IDs
    if show_element_ids:
        centers = np.array([nodes[conn].mean(axis=0) for conn in e0])
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode="text",
            text=[str(i + 1) for i in range(e0.shape[0])],
            textfont=dict(color="midnightblue"),
            name="Element ID"
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="x", scaleanchor="y", scaleratio=1, zeroline=False),
        yaxis=dict(title="y", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="left", x=0),
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan"
    )
    return fig


def plot_deformed_mesh(
    nodes: np.ndarray,
    elements: np.ndarray,
    Ed: np.ndarray,
    scale: float = 1.0,
    field: str = "utotal",
    colorscale: str = "Jet",
    showscale: bool = True,
    colorbar_title: str | None = None,
    show_original: bool = True,
    original_color: str = "lightgray",
    title: str = "Deformed Mesh Animation",
    cycles: int = 3,
    duration: float = 2.0
) -> go.Figure:
    """
    Plot an animated deformed mesh using Plotly (optimized for performance).

    - Supports triangular (3-node) and quadrilateral (4-node) elements.
    - Maintains fixed axis limits and orthographic camera (stable view).
    - Optionally overlays the original (undeformed) mesh for reference.
    - Deformed mesh edges are a separate togglable trace in the legend.
    - Animation starts at maximum deformation and runs `cycles` oscillation cycles in `duration` seconds.

    FIXES:
    - Quads are consistently triangulated as (0,1,2) and (0,2,3) after enforcing CCW ordering.
    - Triangles are enforced CCW as well, preventing alternating normals and checkered visuals.
    - Auto-detect 0-based vs 1-based element indexing.
    """

    # Ensure numpy arrays and validate dimensions
    nodes = np.asarray(nodes, float)
    elements = np.asarray(elements, int)
    Ed = np.asarray(Ed, float)
    M, k = elements.shape
    if k not in (3, 4):
        raise ValueError("Only 3-node triangles or 4-node quadrilaterals are supported.")
    if Ed.shape != (M, 2*k):
        raise ValueError(f"Ed must have shape (M, {2*k}) for k={k} elements.")

    # ---- Robust zero/one-based connectivity ---------------------------------
    if elements.min() == 0:
        e0 = elements.copy()
    else:
        e0 = elements - 1  # assume 1-based input

    N = nodes.shape[0]

    # ---- Aggregate per-node global displacements (average of contributing elements)
    global_disp = np.zeros((N, 2))
    count = np.zeros(N)
    node_indices = e0.ravel()
    disp_pairs = Ed.reshape(-1, 2)
    np.add.at(global_disp, node_indices, disp_pairs)
    np.add.at(count, node_indices, 1)
    mask = count > 0
    global_disp[mask] /= count[mask][:, None]

    # ---- Helper: enforce CCW for triangles and quads in XY-plane ------------
    def ccw_tri(tri_idx: np.ndarray) -> np.ndarray:
        """Ensure each triangle (i,j,k) is CCW in XY by swapping j<->k when needed."""
        p = nodes[tri_idx]  # (T, 3, 2)
        # Signed area (twice): cross((p1-p0), (p2-p0)) -> z-component
        area2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
                (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        tri_ccw = tri_idx.copy()
        flip = area2 < 0
        if np.any(flip):
            tri_ccw[flip, 1], tri_ccw[flip, 2] = tri_ccw[flip, 2], tri_ccw[flip, 1]
        return tri_ccw

    def ccw_quad(quad_idx: np.ndarray) -> np.ndarray:
        """Ensure each quad (0..3) is CCW in XY using the shoelace formula."""
        q = nodes[quad_idx]  # (Q, 4, 2)
        x, y = q[..., 0], q[..., 1]
        area2 = np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        quad_ccw = quad_idx.copy()
        flip = area2 < 0
        if np.any(flip):
            quad_ccw[flip] = quad_ccw[flip, ::-1]
        return quad_ccw

    # ---- Build triangulation -------------------------------------------------
    if k == 3:
        tri_conn = ccw_tri(e0.copy())  # (M,3) all CCW
    else:
        # Enforce CCW for quads, then split consistently as (0,1,2) & (0,2,3)
        q_ccw = ccw_quad(e0.copy())  # (M,4)
        tri1 = q_ccw[:, [0, 1, 2]]
        tri2 = q_ccw[:, [0, 2, 3]]
        tri_conn = np.vstack([tri1, tri2])  # (2M,3)
        tri_conn = ccw_tri(tri_conn)        # ensure triangles CCW

    tri_i, tri_j, tri_k = tri_conn[:, 0], tri_conn[:, 1], tri_conn[:, 2]

    # ---- Edges (for overlay lines) ------------------------------------------
    edge_pattern = [0, 1, 2, 0] if k == 3 else [0, 1, 2, 3, 0]
    edge_idx = []
    for conn in e0:
        for a, b in zip(edge_pattern[:-1], edge_pattern[1:]):
            edge_idx.extend([conn[a], conn[b], -1])  # -1 as break indicator
    edge_idx = np.array(edge_idx, dtype=int)
    safe_idx = edge_idx.copy()
    safe_idx[safe_idx < 0] = 0
    Z_edge_base = np.zeros_like(edge_idx, dtype=float)
    Z_edge_base[edge_idx < 0] = np.nan

    # ---- Axis ranges (constant over animation) ------------------------------
    nodes_max = nodes + scale * global_disp
    nodes_min = nodes - scale * global_disp
    xmin, ymin = np.minimum(nodes_max, nodes_min).min(axis=0)
    xmax, ymax = np.maximum(nodes_max, nodes_min).max(axis=0)

    # ---- Field selection + color range --------------------------------------
    ux = global_disp[:, 0]
    uy = global_disp[:, 1]
    u_mag = np.linalg.norm(global_disp, axis=1)
    if field == "ux":
        field_vals = ux
        field_label = "u_x"
    elif field == "uy":
        field_vals = uy
        field_label = "u_y"
    else:
        field_vals = u_mag
        field_label = "|u|"
    max_val = np.max(np.abs(field_vals)) if field in ("ux", "uy") else np.max(field_vals)
    cmin_global = -max_val if field in ("ux", "uy") else 0.0
    cmax_global = max_val if field in ("ux", "uy") else max_val
    if np.isclose(cmin_global, cmax_global):
        cmax_global = cmin_global + 1e-6

    # ---- Initial (max deformation) state ------------------------------------
    nodes_def_max = nodes + scale * global_disp
    intensity_init = field_vals
    text_init = [f"{field_label}: {val:.4g}" for val in intensity_init]
    zeros = np.zeros(N)

    # ---- Traces --------------------------------------------------------------
    traces = []
    if show_original:
        traces.append(go.Mesh3d(
            x=nodes[:, 0], y=nodes[:, 1], z=zeros,
            i=tri_i, j=tri_j, k=tri_k,
            color=original_color, opacity=0.4,
            name="Original", hoverinfo="skip",
            # flatshading=True,  # optionally enforce flat shading
        ))

    traces.append(go.Mesh3d(
        x=nodes_def_max[:, 0], y=nodes_def_max[:, 1], z=zeros,
        i=tri_i, j=tri_j, k=tri_k,
        intensity=intensity_init, intensitymode="vertex",
        colorscale=colorscale,
        cmin=cmin_global, cmax=cmax_global,
        showscale=showscale,
        colorbar=dict(title=colorbar_title or field_label),
        text=text_init,
        hoverinfo="text",
        name="Deformed",
        # flatshading=True,  # optionally enforce flat shading
    ))

    X_edge_init = nodes_def_max[safe_idx, 0]
    Y_edge_init = nodes_def_max[safe_idx, 1]
    X_edge_init[edge_idx < 0] = np.nan
    Y_edge_init[edge_idx < 0] = np.nan
    traces.append(go.Scatter3d(
        x=X_edge_init.tolist(), y=Y_edge_init.tolist(), z=Z_edge_base.tolist(),
        mode="lines", line=dict(color="black", width=2),
        name="Deformed Edges", showlegend=True,
        hoverinfo="x+y+z+name"
    ))

    mesh_index = 1 if show_original else 0
    edge_index = 2 if show_original else 1

    # ---- Animation frames ----------------------------------------------------
    frames = []
    frames_per_cycle = 30
    total_frames = cycles * frames_per_cycle
    t_vals = np.linspace(0, 2 * math.pi * cycles, total_frames + 1)[1:]
    sin_factors = np.sin(t_vals + 0.5 * math.pi)

    for fac in sin_factors:
        nodes_def = nodes + (fac * scale) * global_disp
        if field == "ux":
            frame_vals = fac * ux
        elif field == "uy":
            frame_vals = fac * uy
        else:
            frame_vals = abs(fac) * u_mag
        frame_text = [f"{field_label}: {val:.4g}" for val in frame_vals]

        X_edge = nodes_def[safe_idx, 0]
        Y_edge = nodes_def[safe_idx, 1]
        X_edge[edge_idx < 0] = np.nan
        Y_edge[edge_idx < 0] = np.nan

        frames.append(go.Frame(
            data=[
                go.Mesh3d(
                    x=nodes_def[:, 0], y=nodes_def[:, 1], z=zeros,
                    intensity=frame_vals, text=frame_text, hoverinfo="text"
                ),
                go.Scatter3d(
                    x=X_edge.tolist(), y=Y_edge.tolist(), z=Z_edge_base.tolist(),
                    mode="lines", line=dict(color="black", width=2),
                    hoverinfo="x+y+z+name"
                )
            ],
            traces=[mesh_index, edge_index]
        ))

    # ---- Figure + layout -----------------------------------------------------
    xrange, yrange = float(xmax - xmin), float(ymax - ymin)
    max_range = max(xrange, yrange) or 1e-6
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=f"{title} (Scale={scale})",
        scene=dict(
            xaxis=dict(title="X", range=[xmin, xmax], autorange=False, showgrid=False, zeroline=False),
            yaxis=dict(title="Y", range=[ymin, ymax], autorange=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=xrange/max_range, y=yrange/max_range, z=0.001),
            camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0), projection=dict(type="orthographic"))
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=40, b=70),
        template="plotly_white",
        dragmode="pan",
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": int(round((duration*1000)/total_frames)), "redraw": True}, "fromcurrent": True}]},
                {"label": "❚❚ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
            ]
        }]
    )
    return fig

def plot_element_values(
    nodes: np.ndarray,
    elements: np.ndarray,
    Ed: np.ndarray,
    element_values: np.ndarray,      # (M,), one scalar per element (e.g., stress)
    title: str,
    scale: float = 1.0,
    colorscale: str = "jet",
    showscale: bool = True,
    colorbar_title: Optional[str] = "",
    show_original: bool = True,
    original_color: str = "lightgray",
    original_opacity: float = 0.35,
    edges_visible: bool = True,      
    edges_color: str = "black",
    edges_width: int = 2,
) -> go.Figure:
    """
    Plot the deformed mesh with a constant color per element. Use this to vizualise scalar fields like a stress component. 
    Works for triangles and quads.

    Parameters
    ----------
    nodes : (N, 2) float
        Nodal coordinates.
    elements : (M, k) int
        1-based connectivity with k=3 (tri) or k=4 (quad).
    Ed : (M, 2*k) float
        Element nodal displacements [ux0, uy0, ..., ux(k-1), uy(k-1)] per element.
    element_values : (M,) float
        Scalar per element (e.g., stress, plastic strain) used for coloring.
    scale : float
        Displacement scale factor for visualization.
    colorscale : str
        Plotly colorscale name.
    showscale : bool
        Whether to show the colorbar.
    colorbar_title : str or None
        Colorbar title.
    show_original : bool
        Show undeformed mesh (semi-transparent background). Not added to legend.
    original_color : str
        Color for original fill.
    original_opacity : float
        Opacity for original fill.
    edges_visible : bool
        Initial visibility of the *deformed* element edges (legend toggles them on/off).
    edges_color : str
        Color for deformed edges.
    edges_width : int
        Line width for deformed edges.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    Ed = np.asarray(Ed, dtype=float)
    element_values = np.asarray(element_values, dtype=float)

    M, k = elements.shape
    if k not in (3, 4):
        raise ValueError("Only triangles (k=3) or quads (k=4) are supported.")
    if Ed.shape != (M, 2 * k):
        raise ValueError(f"Ed must be (M, {2*k}) for k={k}. Got {Ed.shape}.")
    if element_values.shape != (M,):
        raise ValueError(f"element_values must be shape (M,), got {element_values.shape}.")

    # 0-based connectivity
    e0 = elements - 1
    cycle = [0, 1, 2, 0] if k == 3 else [0, 1, 2, 3, 0]

    # Deformed mesh triangles (faces)
    X, Y, Z = [], [], []
    I, J, K = [], [], []
    intensity = []
    hover_text = []

    # Original background (optional)
    orig_X, orig_Y, orig_Z = [], [], []
    orig_I, orig_J, orig_K = [], [], []

    # Deformed edges for legend
    edge_x, edge_y, edge_z = [], [], []

    for el_idx, conn in enumerate(e0):
        pts_org = nodes[conn]              # (k, 2)
        disp = Ed[el_idx].reshape(k, 2)    # (k, 2)
        pts_def = pts_org + scale * disp   # (k, 2)
        val = float(element_values[el_idx])

        # --- Original background (fill only; no legend) ---
        if show_original:
            if k == 3:
                tris_o = [(0, 1, 2)]
            else:
                d02 = np.linalg.norm(pts_org[0] - pts_org[2])
                d13 = np.linalg.norm(pts_org[1] - pts_org[3])
                tris_o = [(0, 1, 2), (0, 2, 3)] if d02 <= d13 else [(0, 1, 3), (1, 2, 3)]
            for a, b, c in tris_o:
                base = len(orig_X)
                for idx in (a, b, c):
                    orig_X.append(pts_org[idx, 0])
                    orig_Y.append(pts_org[idx, 1])
                    orig_Z.append(0.0)
                orig_I.append(base)
                orig_J.append(base + 1)
                orig_K.append(base + 2)

        # --- Deformed faces (constant per-element color) ---
        if k == 3:
            tris_d = [(0, 1, 2)]
        else:
            d02 = np.linalg.norm(pts_def[0] - pts_def[2])
            d13 = np.linalg.norm(pts_def[1] - pts_def[3])
            tris_d = [(0, 1, 2), (0, 2, 3)] if d02 <= d13 else [(0, 1, 3), (1, 2, 3)]

        for a, b, c in tris_d:
            base = len(X)
            for idx in (a, b, c):
                X.append(pts_def[idx, 0])
                Y.append(pts_def[idx, 1])
                Z.append(0.0)
                # same value on all three vertices of element triangles -> flat color
                intensity.append(val)
                hover_text.append(f"Element {el_idx+1}, value = {val:.4g}")
            I.append(base)
            J.append(base + 1)
            K.append(base + 2)

        # --- Deformed edges (single trace, togglable in legend) ---
        for a, b in zip(cycle[:-1], cycle[1:]):
            edge_x += [pts_def[a, 0], pts_def[b, 0], None]
            edge_y += [pts_def[a, 1], pts_def[b, 1], None]
            edge_z += [0.0, 0.0, None]

    # Arrays for faces
    X = np.array(X); Y = np.array(Y); Z = np.array(Z)
    intensity = np.array(intensity)
    cmin, cmax = intensity.min(), intensity.max()
    if np.isclose(cmin, cmax):
        cmax = cmin + 1e-12  # avoid zero color range

    fig = go.Figure()

    # --- Original background fill (always visible if enabled; not in legend) ---
    if show_original:
        fig.add_trace(go.Mesh3d(
            x=orig_X, y=orig_Y, z=orig_Z,
            i=orig_I, j=orig_J, k=orig_K,
            color=original_color,
            opacity=original_opacity,
            flatshading=True,
            hoverinfo="skip",
            name="Original (background)",
            showlegend=False,
        ))

    # --- Deformed faces (elements) ---
    fig.add_trace(go.Mesh3d(
        x=X, y=Y, z=Z,
        i=I, j=J, k=K,
        intensity=intensity,
        intensitymode="vertex",
        colorscale=colorscale,
        cmin=cmin, cmax=cmax,
        showscale=showscale,
        colorbar=dict(title=colorbar_title or "Value"),
        flatshading=True,      # keep faces visually uniform
        text=hover_text,
        hoverinfo="text",
        name="Elements",
        showlegend=False,      # elements always visible; not a legend item
    ))

    # --- Deformed element edges — legend item for toggling ---
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color=edges_color, width=edges_width),
        hoverinfo="skip",
        name="Element edges",       # <-- legend label
        showlegend=True,            # <-- appears in legend
        visible=True if edges_visible else "legendonly",
    ))

    # --- View / layout ---
    fig.update_layout(
        title=f"{title} (Scale factor = {scale:g})",
        scene=dict(
            xaxis=dict(title="X", showgrid=False, zeroline=False, tickangle=0, showspikes=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False, tickangle=0, showspikes=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=0, y=0, z=2),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                projection=dict(type="orthographic"),
            ),
        ),
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    return fig



def plot_scalar_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    Ed: np.ndarray,
    colorscale: str = "jet",
    showscale: bool = True,
    colorbar_title: str = "Value",
    show_edges: bool = True,
    edge_color: str = "black",
    edge_width: float = 1.0,
    title: str = "Scalar Field"
) -> go.Figure:
    """
    Plot a static 2D finite element mesh with scalar node values using Plotly.
    
    Each node is colored by a scalar value (e.g., temperature). 
    Hovering over a node shows its scalar value.
    
    Parameters:
        nodes (ndarray): shape (N, 2), XY coordinates of N nodes.
        elements (ndarray): shape (M, 3) or (M, 4), 1-indexed node indices for M elements (triangles or quads).
        Ed (ndarray): shape (M, k), scalar values at each element's nodes (k=3 for triangles, k=4 for quads).
                     Ed[i,j] is the value at the j-th node of element i.
        colorscale (str): Plotly colorscale name for the scalar field.
        showscale (bool): Show the colorbar legend for the scalar values.
        colorbar_title (str): Title for the colorbar (e.g., "Temperature (°C)").
        show_edges (bool): Overlay element edges as lines for clarity.
        edge_color (str): Color for the edge lines.
        edge_width (float): Line width for edges.
        title (str): Plot title.
    
    Returns:
        go.Figure: Plotly figure object with the mesh and scalar field.
    """
    # Prepare data arrays
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    Ed = np.asarray(Ed, dtype=float)
    N = nodes.shape[0]
    M, k = elements.shape
    
    if k not in (3, 4):
        raise ValueError("Elements must have 3 or 4 nodes (triangles or quads).")
    if Ed.shape != (M, k):
        raise ValueError(f"Ed must have shape (M, {k}) corresponding to the elements.")
    
    # Convert to 0-index for internal use
    elem = elements - 1
    
    # Compute averaged scalar value per node (average of all element values at that node)
    node_values = np.zeros(N)
    counts = np.zeros(N, dtype=int)
    all_idx = elem.ravel()
    all_vals = Ed.ravel()
    np.add.at(node_values, all_idx, all_vals)
    np.add.at(counts, all_idx, 1)
    mask = counts > 0
    node_values[mask] /= counts[mask]
    
    # Build triangular faces (split quads into triangles if needed)
    if k == 3:
        tri_conn = elem.copy()
    else:
        # Split each quad into two triangles along the shorter diagonal
        diag02 = np.linalg.norm(nodes[elem[:, 0]] - nodes[elem[:, 2]], axis=1)
        diag13 = np.linalg.norm(nodes[elem[:, 1]] - nodes[elem[:, 3]], axis=1)
        use_diag02 = diag02 <= diag13
        tri_conn = np.empty((2*M, 3), dtype=int)
        tri_conn[0::2, :] = np.column_stack([
            elem[:, 0],
            elem[:, 1],
            np.where(use_diag02, elem[:, 2], elem[:, 3])
        ])
        tri_conn[1::2, :] = np.column_stack([
            elem[:, 0],
            np.where(use_diag02, elem[:, 2], elem[:, 1]),
            elem[:, 3]
        ])
    tri_i, tri_j, tri_k = tri_conn.T  # unpack columns
    
    # Setup hover text for each node's value (to display on hover)
    hover_text = [f"{val:.4g}" for val in node_values]
    
    # Create the mesh trace with vertex intensity
    mesh_trace = go.Mesh3d(
        x=nodes[:, 0], y=nodes[:, 1], z=np.zeros(N),  # flat mesh in XY plane
        i=tri_i, j=tri_j, k=tri_k,
        intensity=node_values,
        intensitymode="vertex",  # interpolate color from vertex values
        colorscale=colorscale,
        cmin=node_values.min(), cmax=node_values.max(),
        flatshading=True,  # flat color per triangle (no Gouraud lighting on faces)
        lighting=dict(ambient=1, diffuse=0, specular=0, roughness=0, fresnel=0),
        text=hover_text,
        hoverinfo="text",
        showscale=showscale,
        colorbar=dict(title=colorbar_title),
        name="Scalar Field"
    )
    
    # Create edge trace if needed
    edge_traces = []
    if show_edges:
        edge_x, edge_y, edge_z = [], [], []
        for conn in elem:
            loop = list(conn) + [conn[0]]
            for a, b in zip(loop[:-1], loop[1:]):
                edge_x += [nodes[a, 0], nodes[b, 0], None]
                edge_y += [nodes[a, 1], nodes[b, 1], None]
                edge_z += [0, 0, None]
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color=edge_color, width=edge_width),
            name="Mesh Edges",
            showlegend=True,
            hoverinfo="skip"
        )
        edge_traces.append(edge_trace)
    
    # Build figure with the specified camera and legend layout
    fig = go.Figure(data=[mesh_trace] + edge_traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", 
            # aspectratio=dict(x=1, y=1, z=0.001),
            camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0),
                        projection=dict(type="orthographic"))
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=40, b=70)
    )
    return fig



def plot_vector_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    vectors: np.ndarray,
    colorscale: str = "Jet",
    arrow_width: float = 2.0,
    arrowhead_length: float = 0.2,
    arrowhead_angle: float = np.pi / 6,  # 30 degrees
    show_edges: bool = True,
    edge_color: str = "black",
    edge_width: float = 1.0,
    title: str = "Vector Field"
) -> go.Figure:
    """
    Visualize a 2D vector field over a mesh with arrows (quiver plot) using Plotly.

    Parameters:
        nodes (ndarray): Array of node coordinates, shape (N, 2).
        elements (ndarray): Array of elements (triangles/quads) with 1-based node indices, shape (M, k).
        vectors (ndarray): Array of vectors (u,v) at each element center, shape (M, 2).
        colorscale (str): Name of Plotly colorscale for arrow magnitudes (e.g., 'Viridis').
        arrow_width (float): Line width for arrow shafts and heads.
        arrowhead_length (float): Fraction of arrow length to use for the arrowhead lines (e.g., 0.2 = 20%).
        arrowhead_angle (float): Angle in radians between the arrow shaft and each arrowhead line.
        show_edges (bool): If True, overlay mesh edges.
        edge_color (str): Color for mesh edge lines.
        edge_width (float): Line width for mesh edge lines.
        title (str): Plot title.

    Returns:
        go.Figure: Plotly Figure containing the quiver plot.
    """
    # Convert inputs to numpy array for safety
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    vectors = np.asarray(vectors, dtype=float)
    M = elements.shape[0]

    # Validate dimensions
    if vectors.shape != (M, 2):
        raise ValueError(f"vectors must have shape (M, 2), got {vectors.shape}")

    # Compute element centers (average of node coordinates for each element)
    elem_indices = elements - 1  # convert to 0-based indices
    centers = nodes[elem_indices].mean(axis=1)  # shape (M, 2)

    # Compute vector magnitudes for coloring
    magnitudes = np.linalg.norm(vectors, axis=1)
    min_mag, max_mag = magnitudes.min(), magnitudes.max()

    # Determine scale: aim for the longest arrow to be ~10% of the plot size (adjustable)
    # bbox = nodes.ptp(axis=0)  # range (max-min) in x and y
    bbox = np.ptp(nodes, axis=0)
    diag = np.linalg.norm(bbox)  # diagonal length of mesh bounding box
    scale = 0.1 * diag / (max_mag + 1e-9)  # avoid division by zero if max_mag is 0

    # Prepare figure
    fig = go.Figure()

    # Create scatter traces for each arrow (shaft + arrowhead)
    # We'll sample the colorscale for each arrow's normalized magnitude[3](https://stackoverflow.com/questions/62710057/access-color-from-plotly-color-scale)
    norm_mags = (magnitudes - min_mag) / (max_mag - min_mag + 1e-9)
    colors = sample_colorscale(colorscale, norm_mags.tolist())
    for (x0, y0), (u, v), mag, color in zip(centers, vectors, magnitudes, colors):
        # Arrow end (tip)
        xf = x0 + scale * u
        yf = y0 + scale * v

        # Compute arrowhead base points (one on each side of the tip)
        # Arrow length in plot units:
        L = np.hypot(xf - x0, yf - y0)
        if L == 0:
            continue  # zero-length vector, nothing to draw
        # Arrowhead line length (fraction of L)
        h = arrowhead_length * L
        # Unit vector along arrow (pointing from base to tip)
        ux, uy = (xf - x0) / L, (yf - y0) / L
        # Rotate unit vector by ±arrowhead_angle to get arrowhead directions[1](https://www.bomberbot.com/python/quiver-plots-using-plotly-in-python-visualizing-vector-fields/)
        angle = arrowhead_angle
        # Left side arrowhead (rotate by +angle)
        lx = np.cos(angle) * ux - np.sin(angle) * uy
        ly = np.sin(angle) * ux + np.cos(angle) * uy
        # Right side arrowhead (rotate by -angle)
        rx = np.cos(-angle) * ux - np.sin(-angle) * uy
        ry = np.sin(-angle) * ux + np.cos(-angle) * uy
        # Arrowhead base coordinates
        hx1, hy1 = xf - h * lx, yf - h * ly
        hx2, hy2 = xf - h * rx, yf - h * ry

        # Combine segments: shaft (x0->xf) and two head lines (xf->hx1 and xf->hx2)
        arrow_x = [x0, xf, None, xf, hx1, None, xf, hx2, None]
        arrow_y = [y0, yf, None, yf, hy1, None, yf, hy2, None]

        fig.add_trace(go.Scatter(
            x=arrow_x, y=arrow_y,
            mode='lines',
            line=dict(color=color, width=arrow_width),
            hoverinfo='text',
            text=[f"|v| = {mag:.3g}"] * 3,  # hover text on these points (same magnitude on all segments)
            showlegend=False
        ))

    # Overlay mesh edges if requested
    if show_edges:
        edge_x, edge_y = [], []
        for conn in elem_indices:
            # Close the loop back to the first node
            loop = np.append(conn, conn[0])
            for i in range(len(loop) - 1):
                n1, n2 = loop[i], loop[i+1]
                edge_x += [nodes[n1, 0], nodes[n2, 0], None]
                edge_y += [nodes[n1, 1], nodes[n2, 1], None]
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(color=edge_color, width=edge_width),
            name='Mesh edges',
            hoverinfo='skip'
        ))

    # Add a dummy scatter for the colorbar (two invisible points for min and max)[3](https://stackoverflow.com/questions/62710057/access-color-from-plotly-color-scale)
    fig.add_trace(go.Scatter(
        x=[nodes[:,0].min(), nodes[:,0].min()],
        y=[nodes[:,1].min(), nodes[:,1].min()],
        mode='markers',
        marker=dict(
            color=[min_mag, max_mag],
            colorscale=colorscale,
            cmin=min_mag, cmax=max_mag,
            opacity=0  # invisible points
        ),
        showlegend=False,
        hoverinfo='skip',
        marker_colorbar=dict(title="|v|")  # colorbar showing magnitude
    ))

    # Set equal aspect ratio for x and y axes to preserve arrow directions
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    
    fig.update_layout(
        title=title,
        height=600,  # Increase figure height
        xaxis=dict(title="x", scaleanchor="y", scaleratio=1, zeroline=False, showgrid=False),
        yaxis=dict(title="y", zeroline=False, showgrid=False),
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",  # Set default interaction to pan
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,  # Move legend up slightly
            xanchor="center",
            x=0.5
        )
    )

    return fig


# ------------------------------------------------------------------
# Edof builder
# ------------------------------------------------------------------
def build_edof(elements: np.ndarray, dofs_per_node: int = 2) -> np.ndarray:
    """
    Build Edof matrix (all DoFs per element row-wise), node-major order.

    For 1-based node indices and d=2:
        DoFs for node n = [2n-1, 2n].

    Parameters
    ----------
    elements : (M, k) int
        Element connectivity (1-based).
    dofs_per_node : int, default 2
        Number of DoFs per node.

    Returns
    -------
    Edof : (M, k*d) int
        DoF connectivity row-wise per element.
    """
    if dofs_per_node <= 0:
        raise ValueError("dofs_per_node must be positive.")

    elems = elements.astype(np.int64, copy=False)
    M, k = elems.shape
    d = dofs_per_node
    Edof = np.empty((M, k * d), dtype=np.int64)

    for a in range(d):
        Edof[:, a::d] = (elems - 1) * d + (a + 1)

    return Edof



# ------------------------------------------------------------------
# Mesh and Mesh generator
# ------------------------------------------------------------------

class Mesh:
    """
    A simple container for finite element mesh data.

    Attributes
    ----------
    nodes : ndarray
        Array of node coordinates, shape (n_nodes, 2) or (n_nodes, 3).
    elements : ndarray
        Array of element connectivity, shape (n_elements, n_nodes_per_element).
    edges : dict
        Dictionary of edge node indices, e.g., {"left": [...], "right": [...], ...}.
    """

    def __init__(self, nodes, elements, edges, dofs_per_node):
        self.nodes = nodes
        self.elements = elements
        self.edges = edges
        self.edofs = build_edof(elements, dofs_per_node=dofs_per_node)
        self.num_dofs = nodes.shape[0] * dofs_per_node

    def __repr__(self):
        return (
            f"<Mesh>\n"
            f"  Nodes:    {self.nodes.shape[0]} (dim = {self.nodes.shape[1]})\n"
            f"  # Dofs:   {self.num_dofs}\n"
            f"  Elements: {self.elements.shape[0]} (nodes/elem = {self.elements.shape[1]})\n"
            f"  Edges:    {self.edges}\n"
        )
    
    def plot(self, title="Mesh", show_node_ids=False, show_element_ids=False):
        fig = plot_mesh(
            nodes=self.nodes, 
            elements=self.elements, 
            edges=self.edges, 
            title=title,
            show_node_ids=show_node_ids, 
            show_element_ids=show_element_ids
        )
        return fig


class MeshGenerator:

    @staticmethod
    def semistructured_rectangle_mesh_quads(
        width: float,
        height: float,
        nx: int,
        ny: int,
        element_type: ElementType = "quad",
        origin: Tuple[float, float] = (0.0, 0.0),
        # tri_pattern: Literal["\\", "/", "alternating"] = "\\",  # accepted for signature parity; ignored here
        dofs_per_node: int = 2,
        skew_frac_default = 0.05
    ) -> Mesh:
        """
        Semi-structured rectangular Q4 mesh:
        - Straight horizontal grid lines (y = const).
        - 'Vertical' grid lines smoothly inclined via a parabolic top x-shift.
        - Perfect rectangles when CA2_SEMISTRUCT_SKEW_FRAC == 0.0 (no cutoff; shift term becomes zero).

        Parameters
        ----------
        width, height : float
            Rectangle size in x and y directions.
        nx, ny : int
            Number of elements along x and y. Nodes: (nx+1) by (ny+1).
        element_type : {"quad", "tri"}, default "quad"
            Only "quad" is supported in this generator; "tri" raises ValueError.
        origin : (float, float), default (0.0, 0.0)
            Lower-left corner coordinates (x0, y0).
        tri_pattern : {"\\", "/", "alternating"}, default "\\"
            Accepted for API compatibility; ignored for quads.
        dofs_per_node : int, default 2
            Degrees of freedom per node.

        Returns
        -------
        Mesh
            - nodes   : (N, 2) float array [x, y], N = (nx+1)*(ny+1).
            - elements: (Ne, 4) int array, 1-based Q4 connectivity (LL, LR, UR, UL).
            - edges   : dict of 1-based int arrays {"bottom","top","left","right"}.
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers.")
        if element_type != "quad":
            raise ValueError("This semi-structured generator currently supports only 'quad' elements.")

        x0, y0 = origin
        npx, npy = nx + 1, ny + 1
        N = npx * npy
        nodes = np.empty((N, 2), dtype=float)

        # --- Helper: enforce strictly increasing sequence to avoid crossings ---
        def _enforce_monotone_ascending(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            x_mono = x.copy()
            for i in range(1, x_mono.size):
                if x_mono[i] <= x_mono[i - 1]:
                    x_mono[i] = x_mono[i - 1] + eps
            # Ensure exact right corner if any drift occurred
            if x_mono[-1] != x[-1]:
                drift = x_mono[-1] - x[-1]
                if drift != 0.0:
                    n = x_mono.size
                    corr = np.linspace(0.0, drift, n)
                    x_mono -= corr
                    for i in range(1, n):
                        if x_mono[i] <= x_mono[i - 1]:
                            x_mono[i] = x_mono[i - 1] + eps
                    x_mono[-1] = x[-1]
            return x_mono

        # --- Bottom x (uniform) ---
        xb = np.linspace(x0, x0 + width, npx)

        # --- Top x with smooth skew (zero amplitude -> structured automatically) ---
        t = np.linspace(0.0, 1.0, npx)
        skew = float(skew_frac_default)
        # Smooth, zero-at-ends parabola: peak = skew * width at mid-span
        shift = (4.0 * skew * width) * t * (1.0 - t)
        xt = (x0 + width * t) + shift

        # Enforce exact corners
        xt[0], xt[-1] = x0, x0 + width

        # Only enforce monotonicity when there is actual skew (to avoid micro-tilt at zero)
        if not np.allclose(shift, 0.0):
            xt = _enforce_monotone_ascending(xt)

        # --- Straight horizontal layers; linear x interpolation between xb and xt ---
        ylev = np.linspace(y0, y0 + height, npy)
        for j, y in enumerate(ylev):
            # Lambda varies smoothly with y; when skew==0, xt==xb -> xcol==xb -> perfect verticals
            lam = (y - y0) / height if height > 0.0 else 0.0
            xcol = (1.0 - lam) * xb + lam * xt
            base = j * npx
            nodes[base : base + npx, 0] = xcol
            nodes[base : base + npx, 1] = y

        # Node index (0-based) at column i and row j (j=0 at bottom)
        def nid(i: int, j: int) -> int:
            return j * npx + i

        # --- Edge nodes (0-based initially) ---
        bottom = np.array([nid(i, 0) for i in range(npx)], dtype=np.int64)
        top    = np.array([nid(i, ny) for i in range(npx)], dtype=np.int64)
        left   = np.array([nid(0, j) for j in range(npy)], dtype=np.int64)
        right  = np.array([nid(nx, j) for j in range(npy)], dtype=np.int64)

        # --- Q4 connectivity (0-based) ---
        Ne = nx * ny
        elements0 = np.empty((Ne, 4), dtype=np.int64)
        e = 0
        for j in range(ny):
            for i in range(nx):
                n00 = nid(i, j)         # lower-left
                n10 = nid(i + 1, j)     # lower-right
                n11 = nid(i + 1, j + 1) # upper-right
                n01 = nid(i, j + 1)     # upper-left
                elements0[e, :] = (n00, n10, n11, n01)
                e += 1

        # --- Convert to 1-based indexing to match structured_rectangle_mesh ---
        elements = elements0 + 1
        bottom  += 1
        top     += 1
        left    += 1
        right   += 1

        edges = {
            "bottom": bottom,
            "top": top,
            "left": left,
            "right": right,
        }
        return Mesh(nodes, elements, edges, dofs_per_node)

 
    @staticmethod
    def structured_rectangle_mesh(
        width: float,
        height: float,
        nx: int,
        ny: int,
        element_type: ElementType = "tri",
        origin: Tuple[float, float] = (0.0, 0.0),
        tri_pattern: Literal["\\", "/", "alternating"] = "\\",
        dofs_per_node: int =2
    ) -> Dict[str, np.ndarray]:
        """
        Create a structured rectangular mesh using either triangles or quads.

        Parameters
        ----------
        width : float
            Rectangle width in x-direction.
        height : float
            Rectangle height in y-direction.
        nx : int
            Number of elements along x.
        ny : int
            Number of elements along y.
        element_type : {"quad", "tri"}, default "quad"
            Element type. "tri" creates two triangles per cell.
        origin : (float, float), default (0, 0)
            Lower-left corner coordinates.
        tri_pattern : {"\\", "/", "alternating"}, default "alternating"
            Diagonal pattern for triangulation.

        Returns
        -------
        dict with keys:
            - "nodes": (N, 2) float64, node coordinates [x, y].
            - "elements": (M, k) int64, connectivity matrix:
                * k=4 for quads (order: lower-left, lower-right, upper-right, upper-left).
                * k=3 for triangles (counter-clockwise).
            - "edges": dict of arrays:
                * "bottom", "top", "left", "right" (1-based indices).
            - "spacing": (dx, dy) cell sizes.
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive integers.")

        x0, y0 = origin
        dx = width / nx
        dy = height / ny

        # --- Nodes ---
        xs = np.linspace(x0, x0 + width, nx + 1)
        ys = np.linspace(y0, y0 + height, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape: (ny+1, nx+1)
        nodes = np.column_stack([X.ravel(order="C"), Y.ravel(order="C")])  # (N, 2)

        def nid(i: int, j: int) -> int:
            """Node index (0-based) for column i and row j (j=0 at bottom)."""
            return j * (nx + 1) + i

        # --- Edge nodes (0-based initially) ---
        bottom = np.array([nid(i, 0) for i in range(nx + 1)], dtype=np.int64)
        top = np.array([nid(i, ny) for i in range(nx + 1)], dtype=np.int64)
        left = np.array([nid(0, j) for j in range(ny + 1)], dtype=np.int64)
        right = np.array([nid(nx, j) for j in range(ny + 1)], dtype=np.int64)

        # --- Elements (0-based) ---
        quads: List[List[int]] = []
        for j in range(ny):
            for i in range(nx):
                n00 = nid(i, j)         # lower-left
                n10 = nid(i + 1, j)     # lower-right
                n11 = nid(i + 1, j + 1) # upper-right
                n01 = nid(i, j + 1)     # upper-left
                quads.append([n00, n10, n11, n01])

        if element_type == "quad":
            elements = np.array(quads, dtype=np.int64)
        elif element_type == "tri":
            tris: List[List[int]] = []
            for j in range(ny):
                for i in range(nx):
                    idx = j * nx + i
                    n00, n10, n11, n01 = quads[idx]
                    if tri_pattern == "alternating":
                        if (i + j) % 2 == 0:
                            tris.append([n00, n10, n11])
                            tris.append([n00, n11, n01])
                        else:
                            tris.append([n00, n10, n01])
                            tris.append([n10, n11, n01])
                    elif tri_pattern == "\\":
                        tris.append([n00, n10, n11])
                        tris.append([n00, n11, n01])
                    elif tri_pattern == "/":
                        tris.append([n00, n10, n01])
                        tris.append([n10, n11, n01])
                    else:
                        raise ValueError("Invalid tri_pattern.")
            elements = np.array(tris, dtype=np.int64)
        else:
            raise ValueError("element_type must be 'quad' or 'tri'.")

        # --- Convert to 1-based indexing ---
        elements += 1
        bottom += 1; top += 1; left += 1; right += 1

        edges = {
            "bottom": bottom,
            "top": top,
            "left": left,
            "right": right,
        }
        return Mesh(nodes, elements, edges, dofs_per_node)


def spy(matrix: np.ndarray, marker_size: int = 8, title: str = "Spy Plot") -> go.Figure:
    """
    Create an interactive spy plot for a sparse matrix using Plotly.

    Nonzero entries are shown as markers, similar to Matplotlib's `plt.spy()`.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix (2D array) to visualize.
    marker_size : int, optional
        Size of the markers representing nonzero entries. Default is 8.
    title : str, optional
        Title of the plot. Default is "Spy Plot".

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure showing nonzero entries.

    Notes
    -----
    - Hover info displays the matrix indices and values.
    - The y-axis is reversed to match matrix orientation (row 0 at top).
    - Maintains equal aspect ratio for rows and columns.
    - Tight margins for a clean look.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 0, 0, 2],
    ...               [0, 3, 0, 0],
    ...               [4, 0, 5, 0],
    ...               [0, 0, 0, 6]])
    >>> fig = plot_spy(A, marker_size=10, title="Sparse Matrix Spy Plot")
    >>> fig.show()
    """
    # Find nonzero entries
    rows, cols = np.nonzero(matrix)

    # Prepare hover text
    hover_text = [f"A[{r},{c}] = {matrix[r, c]}" for r, c in zip(rows, cols)]

    # Create scatter plot for nonzero entries
    fig = go.Figure(data=go.Scatter(
        x=cols, y=rows,
        mode='markers',
        marker=dict(size=marker_size, color='black'),
        text=hover_text,
        hoverinfo="text"
    ))

    # Layout adjustments with tight margins
    fig.update_layout(
        title=title,
        xaxis=dict(title="Column", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Row", autorange="reversed"),  # row 0 at top
        template="plotly_white",
        dragmode="pan",
        margin=dict(l=5, r=5, t=30, b=5)  # tight margins
    )

    return fig

# Continuum elements:
def cst_element(nodes: np.ndarray, D: np.ndarray, t: float, body_load=None):
    """
    Compute stiffness matrix and optional body load vector for a CST element
    (3-node constant strain triangle).

    Parameters
    ----------
    nodes : (3, 2) ndarray
        Node coordinates [[x1,y1],[x2,y2],[x3,y3]].
    D : (3, 3) ndarray
        Constitutive matrix.
    t : float
        Thickness.
    body_load : array-like of length 2, optional
        Body force components [bx, by].

    Returns
    -------
    Ke : (6, 6) ndarray
        Element stiffness matrix.
    fe : (6,) ndarray
        Element load vector.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape != (3, 2):
        raise ValueError("nodes must be (3,2).")
    D = np.asarray(D, dtype=float)
    if D.shape != (3, 3):
        raise ValueError("D must be (3,3).")
    if t <= 0:
        raise ValueError("Thickness must be positive.")

    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    # Area
    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    if A <= 0:
        raise ValueError("Triangle area must be positive.")

    # Derivatives of shape functions
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    dN_dx = b / (2*A)
    dN_dy = c / (2*A)

    # B-matrix (3x6)
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i]     = dN_dx[i]
        B[1, 2*i+1]   = dN_dy[i]
        B[2, 2*i]     = dN_dy[i]
        B[2, 2*i+1]   = dN_dx[i]

    # Stiffness matrix
    Ke = B.T @ D @ B * A * t

    # Body load vector
    fe = np.zeros(6)
    if body_load is not None:
        bx, by = body_load
        for i in range(3):
            fe[2*i]   = bx * A * t / 3
            fe[2*i+1] = by * A * t / 3

    return Ke, fe


def cst_element_M(nodes: np.ndarray, rho: float, t: float):
    """
    Compute mass matrix for a CST (3-node constant strain triangle) element.
    """
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    # Area
    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    return rho*A*t/12*np.array([
        [2, 0, 1, 0, 1, 0],
        [0, 2, 0, 1, 0, 1],
        [1, 0, 2, 0, 1, 0],
        [0, 1, 0, 2, 0, 1],
        [1, 0, 1, 0, 2, 0],
        [0, 1, 0, 1, 0, 2]
    ])

def cst_element_stress_strain(nodes: np.ndarray, D: np.ndarray, ae: np.ndarray):
    """
    Compute stress and strain for a CST (3-node constant strain triangle) element.

    Parameters
    ----------
    nodes : (3, 2) ndarray
        Node coordinates [[x1,y1],[x2,y2],[x3,y3]].
    D : (3, 3) ndarray
        Constitutive matrix.
    ae : (6,) ndarray
        Nodal displacement vector [u1,v1,u2,v2,u3,v3].

    Returns
    -------
    stress : (3,) ndarray
        Stress vector [σ_xx, σ_yy, σ_xy].
    strain : (3,) ndarray
        Strain vector [ε_xx, ε_yy, γ_xy].
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape != (3, 2):
        raise ValueError("nodes must be (3,2).")
    D = np.asarray(D, dtype=float)
    if D.shape != (3, 3):
        raise ValueError("D must be (3,3).")
    ae = np.asarray(ae, dtype=float)
    if ae.shape != (6,):
        raise ValueError("displacements must be length 6.")

    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    # Area
    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    if A <= 0:
        raise ValueError("Triangle area must be positive.")

    # Derivatives of shape functions
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    dN_dx = b / (2*A)
    dN_dy = c / (2*A)

    # B-matrix (3x6)
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i]     = dN_dx[i]
        B[1, 2*i+1]   = dN_dy[i]
        B[2, 2*i]     = dN_dy[i]
        B[2, 2*i+1]   = dN_dx[i]

    # Compute strain and stress
    strain = B @ ae
    stress = D @ strain

    return stress, strain


def hooke_2d_plane_stress(E: float, nu: float) -> np.ndarray:
    """
    Isotropic Hooke matrix for 2D plane stress (Voigt: [ϵ_xx, ϵ_yy, γ_xy]).
    """
    f = E / (1.0 - nu**2)
    return f * np.array([
        [1.0,  nu,   0.0],
        [ nu, 1.0,   0.0],
        [0.0, 0.0, (1.0 - nu)/2.0]
    ], dtype=float)


def hooke_2d_plane_strain(E: float, nu: float) -> np.ndarray:
    """
    Isotropic Hooke matrix for 2D plane strain (Voigt: [ϵ_xx, ϵ_yy, γ_xy]).
    """
    f = E / ((1.0 + nu) * (1.0 - 2.0*nu))
    return f * np.array([
        [1.0 - nu,     nu,            0.0],
        [    nu,   1.0 - nu,          0.0],
        [   0.0,       0.0,  (1.0 - 2.0*nu)/2.0]
    ], dtype=float)

def hooke_3d(E: float, nu: float) -> np.ndarray:
    """
    Isotropic Hooke matrix for full 3D elasticity (Voigt form),
    using engineering shear strains:
      stress  = [σx,  σy,  σz,  τyz,  τxz,  τxy]^T
      strain  = [εx,  εy,  εz,  γyz,  γxz,  γxy]^T

    D is 6x6 and satisfies:  [stress] = D [strain]
    """
    f = E / ((1.0 + nu) * (1.0 - 2.0*nu))       # common factor
    a = (1.0 - nu)                              # normal-diagonal block (before scaling)
    b = nu                                      # normal off-diagonals
    s = (1.0 - 2.0*nu) / 2.0                    # shear diagonal (=> G after scaling)

    return f * np.array([
        [a,  b,  b,  0.0, 0.0, 0.0],
        [b,  a,  b,  0.0, 0.0, 0.0],
        [b,  b,  a,  0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0,  s,   0.0, 0.0],
        [0.0, 0.0, 0.0,  0.0, s,   0.0],
        [0.0, 0.0, 0.0,  0.0, 0.0, s  ]
    ], dtype=float)



def flow2t_Ke_fe(nodes, t, D, Q=0.0):
    """
    Compute the element stiffness matrix (Ke) and load vector (fe) for a 2D triangular element 
    in steady-state heat conduction problems.

    Parameters
    ----------
    nodes : ndarray of shape (3, 2)
        Coordinates of the triangle's three nodes, each row is [x, y].
    t : float
        Thickness of the element.
    D : ndarray of shape (2, 2)
        Thermal conductivity matrix.
    Q : float, optional
        Uniform heat source per unit volume (default is 0.0).

    Returns
    -------
    Ke : ndarray of shape (3, 3)
        Element stiffness matrix.
    fe : ndarray of shape (3, 1)
        Element load vector due to the heat source.
    """
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    if A <= 0:
        raise ValueError("Triangle area must be positive.")

    B = np.array([
        [y2 - y3, y3 - y1, y1 - y2],
        [x3 - x2, x1 - x3, x2 - x1]
    ]) / (2*A)

    Ke = B.T @ D @ B * t * A
    fe = Q * A * t / 3 * np.array([[1.], [1.], [1.]])

    return Ke, fe


def flow2t_qe(nodes: np.ndarray, D: np.ndarray, ae: np.ndarray):
    """
    Compute the heat flux vector (qe) for a 2D triangular element in steady-state heat conduction.

    Parameters
    ----------
    nodes : ndarray of shape (3, 2)
        Coordinates of the triangle's three nodes, each row is [x, y].
    D : ndarray of shape (2, 2)
        Thermal conductivity matrix.
    ae : ndarray of shape (3,)
        Nodal temperatures for the element.

    Returns
    -------
    qe : ndarray of shape (2,)
        Heat flux vector [qx, qy] at the element level.
    """
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    if A <= 0:
        raise ValueError("Triangle area must be positive.")

    B = np.array([
        [y2 - y3, y3 - y1, y1 - y2],
        [x3 - x2, x1 - x3, x2 - x1]
    ]) / (2*A)

    qe = -D @ B @ ae

    return qe


def convection_Ke_fe(nodes, alpha, t, Tamb):
    """
    Compute the element stiffness matrix (Kce) and load vector (fce) 
    due to convection for a linear boundary segment in 2D heat transfer.

    Parameters
    ----------
    nodes : ndarray of shape (2, 2)
        Coordinates of the two boundary nodes, each row is [x, y].
    alpha : float
        Convection heat transfer coefficient [W/(m^2·°C)].
    t : float
        Out-of-plane thickness of the domain [m].
    Tamb : float
        Ambient temperature at the boundary segment [°C].

    Returns
    -------
    Kce : ndarray of shape (2, 2)
        Boundary element stiffness matrix due to convection.
    fce : ndarray of shape (2, 1)
        Boundary element load vector due to convection.
    """
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    Kce = (alpha * t * L / 3) * np.array([[1, 0.5],
                                         [0.5, 1]])

    fce = (alpha * t * Tamb * L / 2) * np.array([[1],
                                                [1]])

    return Kce, fce

def gauss_integration_rule(ngp: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss–Legendre integration points and weights for [-1, 1].

    Parameters
    ----------
    ngp : int
        Number of Gauss points per direction (supported: 1, 2, or 3).

    Returns
    -------
    coords : np.ndarray
        Array of Gauss point coordinates in [-1, 1].
    weights : np.ndarray
        Array of corresponding Gauss weights.

    Raises
    ------
    NotImplementedError
        If ngp is not 1, 2, or 3.

    Examples
    --------
    >>> gauss_integration_rule(2)
    (array([-0.57735027,  0.57735027]), array([1., 1.]))
    """
    return  np.polynomial.legendre.leggauss(ngp)


def extract_block(K: np.ndarray, free_dofs: List[int]) -> np.ndarray:
    """
    Extract a square submatrix (block) from a symmetric matrix `K` using the specified degrees of freedom.

    Parameters
    ----------
    K : np.ndarray
        The full symmetric matrix (e.g., stiffness matrix).
    free_dofs : List[int]
        A list of degrees of freedom (1-based indexing) to extract. These will be used for both rows and columns.

    Returns
    -------
    np.ndarray
        The extracted square submatrix corresponding to the specified degrees of freedom.

    Example
    -------
    >>> K = np.array([[10, 2, 3],
    ...               [2, 20, 5],
    ...               [3, 5, 30]])
    >>> free_dofs = [1, 3]
    >>> extract_block(K, free_dofs)
    array([[10,  3],
           [ 3, 30]])
    """
    idx = np.array(free_dofs, dtype=int) - 1  # Convert to 0-based indexing
    return K[np.ix_(idx, idx)]
