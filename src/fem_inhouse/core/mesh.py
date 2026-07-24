"""Structured rectangular CPS4 mesh used by the case study."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class StructuredMesh:
    """Regular mesh with both displacement components prescribed on its edges."""

    x_size: float
    y_size: float
    base_element_size: float
    scale_factor: float
    nx: int = field(init=False)
    ny: int = field(init=False)
    n_nodes: int = field(init=False)
    n_elems: int = field(init=False)
    n_dof: int = field(init=False)
    node_ids: NDArray = field(init=False)
    coords: NDArray = field(init=False)
    elem_ids: NDArray = field(init=False)
    conn: NDArray = field(init=False)
    dofs_bc: NDArray = field(init=False)
    dofs_free: NDArray = field(init=False)

    def __post_init__(self) -> None:
        if self.x_size <= 0 or self.y_size <= 0:
            raise ValueError("mesh extents must be positive")
        if self.base_element_size <= 0 or self.scale_factor <= 0:
            raise ValueError("element size and scale factor must be positive")
        self.nx = round(self.x_size / self.base_element_size)
        self.ny = round(self.y_size / self.base_element_size)
        if not np.isclose(self.nx * self.base_element_size, self.x_size) or not np.isclose(
            self.ny * self.base_element_size,
            self.y_size,
        ):
            raise ValueError("mesh extents must be integer multiples of base element size")

        nodes_x, nodes_y = self.nx + 1, self.ny + 1
        self.n_nodes = nodes_x * nodes_y
        self.n_elems = self.nx * self.ny
        self.n_dof = 2 * self.n_nodes
        self.node_ids = np.arange(self.n_nodes).reshape((nodes_x, nodes_y), order="F")

        x = np.linspace(0, self.nx * self.base_element_size, nodes_x) * self.scale_factor
        y = np.linspace(0, self.ny * self.base_element_size, nodes_y) * self.scale_factor
        ii, jj = np.meshgrid(np.arange(nodes_x), np.arange(nodes_y), indexing="ij")
        node_grid = self.node_ids[ii, jj]
        self.coords = np.zeros((self.n_nodes, 2))
        self.coords[node_grid.ravel(), 0] = np.repeat(x, nodes_y)
        self.coords[node_grid.ravel(), 1] = np.tile(y, nodes_x)

        self.elem_ids = np.arange(self.n_elems).reshape((self.nx, self.ny), order="F")
        ie, je = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        flat_elements = self.elem_ids[ie, je].ravel()
        self.conn = np.zeros((self.n_elems, 4), dtype=int)
        self.conn[flat_elements, 0] = self.node_ids[ie.ravel(), je.ravel()]
        self.conn[flat_elements, 1] = self.node_ids[ie.ravel() + 1, je.ravel()]
        self.conn[flat_elements, 2] = self.node_ids[ie.ravel() + 1, je.ravel() + 1]
        self.conn[flat_elements, 3] = self.node_ids[ie.ravel(), je.ravel() + 1]

        boundary_nodes = np.concatenate(
            (
                self.node_ids[:, 0],
                self.node_ids[:, -1],
                self.node_ids[0, 1:-1],
                self.node_ids[-1, 1:-1],
            )
        )
        boundary_dofs = np.union1d(2 * boundary_nodes, 2 * boundary_nodes + 1)
        self.dofs_bc = boundary_dofs.astype(int)
        self.dofs_free = np.setdiff1d(
            np.arange(self.n_dof),
            boundary_dofs,
        ).astype(int)

    @property
    def element_size(self) -> float:
        """Physical square-element size in mm."""

        return self.base_element_size * self.scale_factor

    def location_matrix(self) -> NDArray:
        """Return the eight global displacement DOFs of every element."""

        locations = np.empty((self.n_elems, 8), dtype=int)
        locations[:, 0::2] = 2 * self.conn
        locations[:, 1::2] = 2 * self.conn + 1
        return locations
