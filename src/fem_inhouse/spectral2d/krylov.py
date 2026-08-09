"""Non-symmetric Krylov solver dispatch for matrix-free spectral Newton steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import LinearOperator, gcrotmk, gmres, lgmres
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
KrylovMethod = Literal["gmres", "lgmres", "gcrotmk"]


@dataclass(slots=True)
class KrylovRecycleState:
    """Reusable subspace, scoped to one DIC increment."""

    lgmres_outer_v: list[tuple[FloatArray, FloatArray]] = field(default_factory=list)
    gcrotmk_cu: list[tuple[FloatArray, FloatArray]] = field(default_factory=list)

    def reset(self) -> None:
        self.lgmres_outer_v.clear()
        self.gcrotmk_cu.clear()


def solve_nonsymmetric_krylov(
    operator: LinearOperator,
    rhs: ArrayLike,
    *,
    preconditioner: LinearOperator,
    method: KrylovMethod,
    rtol: float,
    maximum_iterations: int,
    restart: int,
    recycle: KrylovRecycleState | None = None,
    lgmres_inner_m: int = 30,
    lgmres_outer_k: int = 3,
    gcrotmk_m: int = 20,
    gcrotmk_k: int = 10,
    callback: object | None = None,
    blas_threads: int | None = 1,
) -> tuple[FloatArray, int, int]:
    """Solve a non-symmetric linear system and return ``(x, info, calls)``."""

    if method not in {"gmres", "lgmres", "gcrotmk"}:
        raise ValueError(f"unsupported Krylov method: {method}")
    values = np.asarray(rhs, dtype=np.float64)
    calls = 0

    def counted(value: object) -> None:
        nonlocal calls
        calls += 1
        if callback is not None:
            callback(value)  # type: ignore[operator]

    with threadpool_limits(limits=blas_threads, user_api="blas"):
        if method == "gmres":
            solution, info = gmres(
                operator,
                values,
                M=preconditioner,
                rtol=rtol,
                atol=0.0,
                restart=restart,
                maxiter=maximum_iterations,
                callback=counted,
                callback_type="pr_norm",
            )
        elif method == "lgmres":
            solution, info = lgmres(
                operator,
                values,
                M=preconditioner,
                rtol=rtol,
                atol=0.0,
                maxiter=maximum_iterations,
                callback=counted,
                inner_m=lgmres_inner_m,
                outer_k=lgmres_outer_k,
                outer_v=recycle.lgmres_outer_v if recycle is not None else None,
                store_outer_Av=True,
            )
        else:
            solution, info = gcrotmk(
                operator,
                values,
                M=preconditioner,
                rtol=rtol,
                atol=0.0,
                maxiter=maximum_iterations,
                callback=counted,
                m=gcrotmk_m,
                k=gcrotmk_k,
                CU=recycle.gcrotmk_cu if recycle is not None else None,
            )
    if info != 0 and recycle is not None:
        # Recycled vectors can become stale after a strongly changing Newton
        # tangent. Retry once without them; the caller still owns the policy
        # for clearing the state before the next Newton solve.
        recycle.reset()
        retry_solution, retry_info, retry_calls = solve_nonsymmetric_krylov(
            operator,
            values,
            preconditioner=preconditioner,
            method=method,
            rtol=rtol,
            maximum_iterations=maximum_iterations,
            restart=restart,
            recycle=None,
            lgmres_inner_m=lgmres_inner_m,
            lgmres_outer_k=lgmres_outer_k,
            gcrotmk_m=gcrotmk_m,
            gcrotmk_k=gcrotmk_k,
            callback=callback,
            blas_threads=blas_threads,
        )
        return retry_solution, retry_info, calls + retry_calls
    return np.asarray(solution, dtype=np.float64), int(info), calls
