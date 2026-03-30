"""Tensor network primitives.

Tensor networks are the mathematical language of quantum many-body physics
and the most powerful classical method for simulating quantum systems.

Provides: tensor contraction, SVD decomposition, Matrix Product States (MPS),
and Matrix Product Operators (MPO).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Tensor:
    """An n-dimensional tensor with named indices.

    In the category-theoretic view, tensors are morphisms between
    tensor product spaces. Contraction is composition.
    """

    data: NDArray[np.complex128]
    index_labels: tuple[str, ...] = ()

    @property
    def rank(self) -> int:
        return len(self.data.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def contract(self, other: Tensor, indices: tuple[str, str]) -> Tensor:
        """Contract two tensors over specified index pair.

        This is the fundamental operation: it generalizes matrix multiplication
        to arbitrary tensor networks.
        """
        idx_self = self.index_labels.index(indices[0]) if indices[0] in self.index_labels else -1
        idx_other = other.index_labels.index(indices[1]) if indices[1] in other.index_labels else -1

        if idx_self == -1 or idx_other == -1:
            raise ValueError(f"Index labels {indices} not found in tensors")

        result_data = np.tensordot(self.data, other.data, axes=([idx_self], [idx_other]))
        new_labels = (
            tuple(l for i, l in enumerate(self.index_labels) if i != idx_self)
            + tuple(l for i, l in enumerate(other.index_labels) if i != idx_other)
        )
        return Tensor(data=result_data, index_labels=new_labels)

    def svd(
        self, left_indices: list[str], max_bond: int | None = None, cutoff: float = 1e-12
    ) -> tuple[Tensor, NDArray[np.float64], Tensor]:
        """SVD decomposition with optional truncation.

        Splits this tensor into U @ diag(S) @ V† along the specified partition.
        This is the core operation for building MPS representations.
        """
        # Reshape into matrix
        left_axes = [self.index_labels.index(l) for l in left_indices]
        right_axes = [i for i in range(self.rank) if i not in left_axes]

        left_shape = tuple(self.data.shape[i] for i in left_axes)
        right_shape = tuple(self.data.shape[i] for i in right_axes)

        perm = left_axes + right_axes
        reshaped = np.transpose(self.data, perm).reshape(
            int(np.prod(left_shape)), int(np.prod(right_shape))
        )

        u, s, vh = np.linalg.svd(reshaped, full_matrices=False)

        # Truncate
        if cutoff > 0:
            mask = s > cutoff
            u, s, vh = u[:, mask], s[mask], vh[mask, :]
        if max_bond is not None and len(s) > max_bond:
            u, s, vh = u[:, :max_bond], s[:max_bond], vh[:max_bond, :]

        bond_label = f"bond_{id(self)}"
        u_labels = tuple(left_indices) + (bond_label,)
        v_labels = (bond_label,) + tuple(self.index_labels[i] for i in right_axes)

        u_tensor = Tensor(
            data=u.reshape(left_shape + (len(s),)),
            index_labels=u_labels,
        )
        v_tensor = Tensor(
            data=vh.reshape((len(s),) + right_shape),
            index_labels=v_labels,
        )
        return u_tensor, s, v_tensor


@dataclass
class MPS:
    """Matrix Product State — efficient representation of 1D quantum states.

    An n-qubit state is decomposed into a chain of rank-3 tensors:
    |ψ⟩ = Σ A[1]^{s1} A[2]^{s2} ... A[n]^{sn} |s1 s2 ... sn⟩

    Bond dimension χ controls the expressiveness:
    - χ = 1: product states only
    - χ = 2^(n/2): exact representation of any state
    """

    tensors: list[NDArray[np.complex128]]
    bond_dims: list[int] = field(default_factory=list)

    @property
    def num_sites(self) -> int:
        return len(self.tensors)

    @classmethod
    def from_statevector(
        cls, sv: NDArray[np.complex128], max_bond: int | None = None
    ) -> MPS:
        """Convert a full state vector to MPS via successive SVD."""
        n = int(np.log2(len(sv)))
        remaining = sv.reshape([2] * n)

        tensors = []
        bond_dims = []

        for i in range(n - 1):
            shape = remaining.shape
            shape[0] if i == 0 else shape[0] * shape[1] // shape[0]
            mat = remaining.reshape(shape[0] * (1 if i == 0 else bond_dims[-1]), -1)

            # Actually reshape properly
            phys_and_left = shape[0]
            right_size = int(np.prod(shape[1:]))
            mat = remaining.reshape(phys_and_left, right_size)

            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            if max_bond and len(s) > max_bond:
                u, s, vh = u[:, :max_bond], s[:max_bond], vh[:max_bond, :]

            chi = len(s)
            bond_dims.append(chi)
            tensors.append(u.reshape(shape[0], chi))
            remaining = (np.diag(s) @ vh).reshape((chi,) + shape[1:])

        tensors.append(remaining)
        return cls(tensors=tensors, bond_dims=bond_dims)

    def to_statevector(self) -> NDArray[np.complex128]:
        """Contract MPS back to full state vector."""
        result = self.tensors[0]
        for t in self.tensors[1:]:
            result = np.tensordot(result, t, axes=([-1], [0]))
        return result.flatten()

    def total_bond_dimension(self) -> int:
        return max(self.bond_dims) if self.bond_dims else 1
