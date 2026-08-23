"""Causal TANN-FCC constitutive material, T0.

The first architecture of the forward path, per
`validation/tann_fcc_preregistration.md` and
`docs/adr/0011-tann-fcc-causal-identification.md`: a local, causal
constitutive law on the twelve FCC systems with latent internal variables,
thermodynamics by construction (GENERIC structure, `D >= 0` identically),
one permutation-invariant network shared by the twelve systems, RK4 at
fixed substeps, transaction semantics, and an algorithmic tangent by
automatic differentiation. No spatial convolution, no coordinates, no
frame index, no interior DIC.

State per point and system: `q = [gamma; z]`, `z in R^d`. Free energy:

    Psi = Psi_el(eps - eps_p) + 1/2 sum_alpha ||z_alpha||^2,
    eps_p = sum_alpha gamma_alpha P_alpha.

Generalised forces `A = -dPsi/dq`, and the evolution

    dq_alpha/ds = ||Delta eps|| M_alpha (A_alpha / sigma_ref),
    M_alpha = L_alpha L_alpha^T,   sigma_ref = 2 mu,

with `L` the lower-triangular output of the shared network, so that

    D = sum_alpha A_alpha^T M_alpha (A_alpha / sigma_ref) >= 0

holds identically, and part of the work may be stored in the latent
variables instead of being forced through the slip channel. The
`sigma_ref` normalisation is the difference that makes the dynamics
integrable: without it the elastic feedback rate is
`||d eps|| M 2 mu ~ 1e2` per substep, far beyond the RK4 stability limit
(`c h <= 2.785`), and the first material gate diverged at four substeps.
With the force scaled by `2 mu` the rate is `O(||d eps|| M)` and the
preregistered integrator is stable.

Plane stress is the analytic closure: `sigma_zz = sigma_xz = sigma_yz = 0`
determines the unobserved out-of-plane total strains.  The elastic transverse
shears vanish while the total transverse shears follow the plastic slips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS

# Measured on the RK4 graph (many small batched ops over ~1e3-2e4 points):
# 8 threads take 67 s for one algorithmic tangent where a single thread
# takes 8.7 s -- thread synchronisation dominates. Pin to one thread.
torch.set_num_threads(1)

FloatArray = np.ndarray


def _slip_tensors() -> torch.Tensor:
    """The twelve Schmid tensors (material frame), `(12, 3, 3)`."""

    tensors = np.empty((12, 3, 3), dtype=np.float64)
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return torch.from_numpy(tensors)


@dataclass(frozen=True)
class TannFCCConfig:
    """Frozen T0 configuration."""

    latent_dim: int = 2
    context_dim: int = 0  # T0: no spatial context; T1+ widens the embedding
    hidden_width: int = 32
    n_layers: int = 2
    n_substeps: int = 4
    integrator: str = "rk4"  # "rk4" (registered) or "implicit_euler" (stiff-capable)
    implicit_newton_iterations: int = 10
    implicit_newton_tolerance: float = 1e-12
    young_modulus_mpa: float = 205_000.0
    poisson_ratio: float = 0.30
    sigma_ref_mpa: float | None = None  # force scale; None -> 2 mu (E/(1+nu))
    seed: int = 20260817


@dataclass(frozen=True, slots=True)
class TannFCCTrial:
    """One transaction-safe material evaluation."""

    stress_in_plane_mpa: FloatArray  # (points, 3) Kelvin [xx, yy, sqrt2 xy]
    # (points, 3, 3) in-plane Kelvin, None when not requested
    consistent_tangent_mpa: FloatArray | None
    plastic_slip: FloatArray  # (points, 12) signed slip
    latent_state: FloatArray  # (points, 12, d)
    generalised_dissipation: FloatArray  # (points,) per-increment D
    slip_work: FloatArray  # (points,) the gamma channel of the work
    trial_state: FloatArray  # (points, 12, 1 + d)


class TannFCCNetwork(nn.Module):
    """Shared, permutation-invariant mobility network.

    Per system: `phi(A_alpha, z_alpha)`; the pool is the mean embedding over
    the twelve systems; the head produces the entries of a lower-triangular
    `(1+d) x (1+d)` matrix with a positive diagonal, so
    `M = L L^T` is symmetric positive definite. The system index never
    enters — the twelve systems are twelve equivalent realisations of one
    mechanism.
    """

    def __init__(self, config: TannFCCConfig):
        super().__init__()
        d = config.latent_dim
        self.d = d
        self.context_dim = config.context_dim
        self.matrix_size = d + 1
        self.n_entries = self.matrix_size * (self.matrix_size + 1) // 2
        generator = torch.Generator().manual_seed(config.seed)
        layers: list[nn.Module] = []
        width = config.hidden_width
        sizes = [1 + d + config.context_dim] + [width] * config.n_layers
        for index in range(config.n_layers):
            layers.append(nn.Linear(sizes[index], sizes[index + 1], dtype=torch.float64))
            layers.append(nn.SiLU())
        self.embedding = nn.Sequential(*layers)
        self.head = nn.Linear(width * 2, self.n_entries, dtype=torch.float64)
        # Tiny weights: the untrained law is near-elastic. The dedicated
        # generator makes the initialisation reproducible per config seed,
        # independent of the process-wide torch RNG state -- two batches
        # built in the same process must start from identical weights.
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=1e-2, generator=generator)

    def forward(
        self,
        force: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(points, 12, 1)` force and `(points, 12, d)` z -> `(points, 12, n_entries)`.

        `context` is the future spatial conditioning, `(..., 12, context_dim)`.
        T0 passes zeros; a zero context contributes exactly nothing to the
        linear embedding, so the law is unchanged by construction.
        """

        local_input = torch.cat([force, z], dim=-1)
        if context is not None:
            local_input = torch.cat([local_input, context], dim=-1)
        embedding = self.embedding(local_input)
        pooled = embedding.mean(dim=1, keepdim=True).expand_as(embedding)
        entries = self.head(torch.cat([embedding, pooled], dim=-1))
        # Positive diagonal so M is full rank; off-diagonal free.
        diagonal = nn.functional.softplus(entries[..., : self.matrix_size]) + 1e-6
        off = entries[..., self.matrix_size :]
        # Out-of-place row assembly: in-place writes into `lower` are not
        # vmap-safe, and the algorithmic tangent vmaps this forward.
        index = 0
        rows = []
        for row in range(self.matrix_size):
            parts = []
            for column in range(row + 1):
                if row == column:
                    parts.append(diagonal[..., row : row + 1])
                else:
                    parts.append(off[..., index : index + 1])
                    index += 1
            zero_pad = torch.zeros(
                (*entries.shape[:-1], self.matrix_size - row - 1),
                dtype=torch.float64,
                device=entries.device,
            )
            rows.append(torch.cat([*parts, zero_pad], dim=-1))
        lower = torch.stack(rows, dim=-2)
        return lower


class TannFCCBatch:
    """Transaction-safe TANN-FCC material batch on material points.

    Strains are in-plane Kelvin triples; the out-of-plane response is the
    analytic plane-stress closure. `evaluate` never mutates the committed
    state; only `commit` may advance it, and `revert` restores exactly.
    """

    def __init__(
        self,
        config: TannFCCConfig,
        *,
        point_count: int,
        systems_global: FloatArray,
    ):
        self.config = config
        self.point_count = point_count
        systems = np.asarray(systems_global, dtype=np.float64)
        if systems.shape != (point_count, 12, 3, 3):
            raise ValueError(f"systems_global must have shape {(point_count, 12, 3, 3)}")
        self._systems = torch.from_numpy(systems)
        lam = (
            config.young_modulus_mpa
            * config.poisson_ratio
            / ((1.0 + config.poisson_ratio) * (1.0 - 2.0 * config.poisson_ratio))
        )
        mu = config.young_modulus_mpa / (2.0 * (1.0 + config.poisson_ratio))
        self._lam = lam
        self._lambda_prime = lam * 2.0 * mu / (lam + 2.0 * mu)
        self._two_mu = 2.0 * mu
        if config.sigma_ref_mpa is None:
            self._sigma_ref = 2.0 * mu
        else:
            if config.sigma_ref_mpa <= 0.0:
                raise ValueError("sigma_ref_mpa must be positive")
            self._sigma_ref = float(config.sigma_ref_mpa)
        self._inplane_diag = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        self._network = TannFCCNetwork(config)
        self._committed_state = torch.zeros(
            (point_count, 12, 1 + config.latent_dim), dtype=torch.float64
        )
        self._committed_strain = torch.zeros((point_count, 3), dtype=torch.float64)
        self._trial_state: torch.Tensor | None = None
        self._trial_strain: torch.Tensor | None = None
        self._latest_trial: TannFCCTrial | None = None
        self._last_committed_tangent: FloatArray | None = None
        self._last_committed_dissipation: FloatArray | None = None
        self._last_committed_slip_work: FloatArray | None = None

    # -- geometry helpers ---------------------------------------------------

    def _inplane_plastic(self, gamma: torch.Tensor, systems: torch.Tensor) -> torch.Tensor:
        """In-plane plastic Kelvin strain `sum_alpha gamma_alpha P_in`.

        Batch-shape agnostic: `gamma` is `(..., 12)`, `systems` `(..., 12, 3, 3)`.
        """

        if gamma.ndim >= 3:
            gamma = gamma[..., 0]
        p_xx = systems[..., 0, 0]
        p_yy = systems[..., 1, 1]
        p_xy = systems[..., 0, 1]
        xx = torch.sum(gamma * p_xx, dim=-1)
        yy = torch.sum(gamma * p_yy, dim=-1)
        xy = torch.sqrt(torch.tensor(2.0, dtype=torch.float64)) * torch.sum(gamma * p_xy, dim=-1)
        return torch.stack([xx, yy, xy], dim=-1)

    def _out_of_plane_plastic(
        self, gamma: torch.Tensor, systems: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the out-of-plane plastic shears ``(xz, yz)``."""

        if gamma.ndim >= 3:
            gamma = gamma[..., 0]
        p_xz = systems[..., 0, 2]
        p_yz = systems[..., 1, 2]
        return torch.sum(gamma * p_xz, dim=-1), torch.sum(gamma * p_yz, dim=-1)

    def _stress_from(
        self, strain_in: torch.Tensor, gamma: torch.Tensor, systems: torch.Tensor
    ) -> torch.Tensor:
        """In-plane Kelvin stress for the given total in-plane strain and slips."""

        eps_p_in = self._inplane_plastic(gamma, systems)
        eps_e_in = strain_in - eps_p_in  # (..., 3) Kelvin
        trace = eps_e_in[..., 0] + eps_e_in[..., 1]
        # sigma = lambda' tr_in(eps_e) delta + 2 mu eps_e_in; out-of-place, vmap-safe.
        return self._two_mu * eps_e_in + self._lambda_prime * trace[..., None] * self._inplane_diag

    def _forces(
        self,
        strain_in: torch.Tensor,
        gamma: torch.Tensor,
        z: torch.Tensor,
        systems: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised forces of the plane-stress-condensed free energy.

        The unobserved transverse total strains are eliminated by imposing
        ``sigma_zz = sigma_xz = sigma_yz = 0``.  Consequently the reduced
        elastic energy is independent of the plastic ``xz`` and ``yz``
        components: their total-strain counterparts follow the slip and the
        elastic transverse shears remain zero.  The slip force is therefore
        exactly the in-plane resolved shear ``sigma:P``; adding a transverse
        compensation would instead describe a plane-*strain* constraint while
        reporting plane stress.
        """

        stress = self._stress_from(strain_in, gamma, systems)  # (..., 3)
        # tau_alpha = sigma_in : P_alpha, in-plane only (sigma_zz/xz/yz = 0);
        # the Kelvin shear contributes sigma_xy_K * P_xy_tensor * sqrt(2).
        p_xx = systems[..., 0, 0]
        p_yy = systems[..., 1, 1]
        p_xy = systems[..., 0, 1]
        tau = (
            stress[..., 0:1] * p_xx
            + stress[..., 1:2] * p_yy
            + stress[..., 2:3] * (p_xy * torch.sqrt(torch.tensor(2.0, dtype=torch.float64)))
        )  # (..., 12)
        force_gamma = tau[..., None]  # (..., 12, 1)
        force_z = -z
        return force_gamma, force_z

    def _mobility_flow(
        self,
        force_gamma: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(lower, flow)` -- the unlimited GENERIC flow `M (A / sigma_ref)`."""

        force_z = -z
        force_norm_gamma = force_gamma / self._sigma_ref
        force_norm = torch.cat([force_norm_gamma, force_z / self._sigma_ref], dim=-1)
        lower = self._network(force_norm_gamma, z, context)
        mobility = lower @ lower.transpose(-1, -2)
        flow = torch.einsum("...ab,...b->...a", mobility, force_norm)
        return lower, flow

    def _rhs(
        self,
        strain_at_s: torch.Tensor,
        q: torch.Tensor,
        rate_scale: torch.Tensor,
        systems: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma = q[..., 0:1]
        z = q[..., 1:]
        force_gamma, _ = self._forces(strain_at_s, gamma[..., 0], z, systems)
        _, flow = self._mobility_flow(force_gamma, z, context)
        flow_scaled = flow * rate_scale[..., None]
        # Smooth per-point slope limiter: the per-substep flow is bounded
        # by kappa * ||d eps||. At the operating point the flow is
        # ||d eps|| * O(1) against kappa = 128, so the law is untouched
        # to better than 1e-3 there; on Newton excursions far from
        # equilibrium the stiff elastic descent (rate ~ (2 mu / sigma_ref)
        # * M * gram) is what overflows RK4, and the limiter holds the
        # substep bounded instead. A positive scalar on M preserves the
        # GENERIC structure, so D >= 0 is untouched. The limiter is an
        # RK4-only guard: the implicit integrator does not need it.
        kappa = 128.0
        flow_power = torch.sum(flow_scaled**2, dim=(-2, -1))
        # the clamp keeps the zero-increment limit exact: flow_power = 0
        # at Delta eps = 0, and 0/1e-30 = 0 gives scale 1 and flow 0.
        denominator = (kappa * rate_scale[..., 0]) ** 2
        scale = torch.rsqrt(1.0 + flow_power / torch.clamp(denominator, min=1e-30))
        return flow_scaled * scale[..., None, None]

    def _integrate(
        self,
        strain_n: torch.Tensor,
        strain_trial: torch.Tensor,
        q_n: torch.Tensor,
        *,
        systems: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        include_work: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One increment over `s in [0, 1]` along the linear strain path.

        Returns the trial state, the integrated generalised dissipation and
        the integrated slip-channel work (the `tau dgamma` channel).
        `include_work=False` skips the dissipation/slip-work quadrature (the
        algorithmic tangent does not need it). The integrator is the
        registered RK4 unless `config.integrator == "implicit_euler"` --
        the L-stable one-step alternative for the stiff operating point
        (Amendment 3), where the numerical memory stays exactly the
        physical state `q`.
        """

        if systems is None:
            systems = self._systems
        if self.config.integrator == "implicit_euler":
            return self._integrate_implicit_euler(
                strain_n,
                strain_trial,
                q_n,
                systems=systems,
                context=context,
                include_work=include_work,
            )
        delta = strain_trial - strain_n
        # `sqrt` has no derivative at zero, and the solver evaluates the
        # reference tangent at a zero increment. The clamp keeps the
        # sqrt-gradient finite (a bare where would multiply cond=0 by an
        # infinite sqrt-gradient, 0 * inf = NaN), while the where keeps
        # Delta q = 0 exact at Delta eps = 0 and the tangent elastic there.
        norm_sq = torch.sum(delta**2, dim=-1, keepdim=True)
        rate_scale = torch.where(
            norm_sq > 0.0,
            torch.sqrt(torch.clamp(norm_sq, min=1e-30)),
            torch.zeros_like(norm_sq),
        )
        steps = self.config.n_substeps
        h = 1.0 / steps
        q = q_n
        point_count = q_n.shape[0]
        dissipation = torch.zeros((point_count,), dtype=torch.float64, device=strain_n.device)
        slip_work = torch.zeros_like(dissipation)
        for step in range(steps):
            s0 = step / steps
            strain_0 = strain_n + s0 * delta
            k1 = self._rhs(strain_0, q, rate_scale, systems, context)
            k2 = self._rhs(
                strain_0 + 0.5 * h * delta,
                q + 0.5 * h * k1,
                rate_scale,
                systems,
                context,
            )
            k3 = self._rhs(
                strain_0 + 0.5 * h * delta,
                q + 0.5 * h * k2,
                rate_scale,
                systems,
                context,
            )
            k4 = self._rhs(strain_n + (s0 + h) * delta, q + h * k3, rate_scale, systems, context)
            step_q = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if include_work:
                # path integrals: D = ||deps|| A^T M A, slip channel = tau dgamma.
                force_gamma, force_z = self._forces(
                    strain_n + (s0 + 0.5 * h) * delta,
                    step_q[..., 0],
                    step_q[..., 1:],
                    systems,
                )
                force = torch.cat([force_gamma, force_z], dim=-1)
                force_norm = force / self._sigma_ref
                lower = self._network(force_gamma / self._sigma_ref, step_q[..., 1:], context)
                mobility = lower @ lower.transpose(-1, -2)
                # D = A^T dq = sigma_ref ||d eps|| A_norm^T M A_norm (the
                # quadrature of the true dissipation, not a rescaled one).
                dissipation = dissipation + (
                    h
                    * rate_scale[..., 0]
                    * self._sigma_ref
                    * torch.einsum("...sa,...sab,...sb->...", force_norm, mobility, force_norm)
                )
                slip_work = slip_work + (
                    rate_scale[..., 0]
                    * torch.sum(force_gamma[..., 0] * (step_q[..., 0] - q[..., 0]), dim=-1)
                )
            q = step_q
        return q, dissipation, slip_work

    def _integrate_implicit_euler(
        self,
        strain_n: torch.Tensor,
        strain_trial: torch.Tensor,
        q_n: torch.Tensor,
        *,
        systems: torch.Tensor,
        context: torch.Tensor | None,
        include_work: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One implicit-Euler step: `q_new = q_n + rate M(q_new) A(q_new)/sigma_ref`.

        Batched per-point Newton with the exact local Jacobian via
        `vmap(jacrev)`. The forces and the mobility are evaluated at the
        NEW state -- for convex gradient flows this is where the
        unconditional-stability property of the scheme lives; the scheme
        is one-step, so the integrator carries no memory of its own and
        the latent state keeps its physical interpretation.
        """

        from torch.func import jacrev, vmap

        delta = strain_trial - strain_n
        norm_sq = torch.sum(delta**2, dim=-1, keepdim=True)
        rate_scale = torch.where(
            norm_sq > 0.0,
            torch.sqrt(torch.clamp(norm_sq, min=1e-30)),
            torch.zeros_like(norm_sq),
        )

        def residual_point(
            q_pt: torch.Tensor,
            q0_pt: torch.Tensor,
            rate_pt: torch.Tensor,
            strain_pt: torch.Tensor,
            systems_pt: torch.Tensor,
        ) -> torch.Tensor:
            gamma = q_pt[..., 0:1]
            z = q_pt[..., 1:]
            force_gamma, _ = self._forces(strain_pt, gamma[..., 0], z, systems_pt)
            _, flow = self._mobility_flow(force_gamma, z, None)
            return q_pt - q0_pt - rate_pt[..., None] * flow

        q = q_n.clone()
        point_count, n_systems, state_dim = q.shape
        flat_dim = n_systems * state_dim
        for _ in range(self.config.implicit_newton_iterations):
            gamma = q[..., 0:1]
            z = q[..., 1:]
            force_gamma, _ = self._forces(strain_trial, gamma[..., 0], z, systems)
            _, flow = self._mobility_flow(force_gamma, z, context)
            residual = q - q_n - rate_scale[..., None] * flow
            jac = vmap(jacrev(residual_point))(q, q_n, rate_scale, strain_trial, systems)
            jac = jac.reshape(point_count, flat_dim, flat_dim)
            step = torch.linalg.solve(jac, residual.reshape(point_count, flat_dim, 1))[..., 0]
            q = q - step.reshape(point_count, n_systems, state_dim)
            tolerance = self.config.implicit_newton_tolerance
            if float(torch.max(torch.abs(step))) <= tolerance * float(
                torch.max(torch.abs(q)) + 1.0
            ):
                break

        dissipation = torch.zeros((point_count,), dtype=torch.float64, device=q.device)
        slip_work = torch.zeros_like(dissipation)
        if include_work:
            gamma = q[..., 0:1]
            z = q[..., 1:]
            force_gamma, force_z = self._forces(strain_trial, gamma[..., 0], z, systems)
            _, flow = self._mobility_flow(force_gamma, z, context)
            force_norm = torch.cat(
                [force_gamma / self._sigma_ref, force_z / self._sigma_ref], dim=-1
            )
            # D = sigma_ref ||d eps|| A_norm^T M A_norm at the converged state.
            dissipation = (
                rate_scale[..., 0] * self._sigma_ref * torch.sum(force_norm * flow, dim=(-2, -1))
            )
            slip_work = rate_scale[..., 0] * torch.sum(
                force_gamma[..., 0] * (gamma[..., 0] - q_n[..., 0]), dim=-1
            )
        return q, dissipation, slip_work

    # -- the transactional contract -----------------------------------------

    def copy_weights_from(self, other: TannFCCBatch) -> None:
        """Copy the network parameters from another batch (same architecture)."""

        with torch.no_grad():
            for target, source in zip(
                self._network.parameters(), other._network.parameters(), strict=True
            ):
                target.copy_(source)

    def _algorithmic_tangent(
        self, strain: torch.Tensor, context: torch.Tensor | None = None
    ) -> FloatArray:
        """Per-point algorithmic tangent `d sigma / d eps_trial`, `(P, 3, 3)`.

        The integration has no cross-point coupling, so the Jacobian of the
        batched forward is block-diagonal: three vector-Jacobian products
        (one per strain component) recover the pointwise tangent exactly,
        at native autograd speed. The full `(P, 3, P, 3)` Jacobian is never
        formed -- it is the OOM that killed a previous session (`P^2 * 9 * 8`
        bytes, 28.8 GB at the 20 000 points of P43). Chunked so the autograd
        graph of one chunk is freed before the next is built.
        """

        tangent = np.empty((self.point_count, 3, 3), dtype=np.float64)
        eye = torch.eye(3, dtype=torch.float64)
        chunk = 2048
        for start in range(0, self.point_count, chunk):
            stop = min(start + chunk, self.point_count)
            strain_chunk = strain[start:stop].clone().requires_grad_(True)
            context_chunk = None if context is None else context[start:stop]
            q_trial, _, _ = self._integrate(
                self._committed_strain[start:stop],
                strain_chunk,
                self._committed_state[start:stop],
                systems=self._systems[start:stop],
                context=context_chunk,
                include_work=False,
            )
            stress = self._stress_from(strain_chunk, q_trial[..., 0], self._systems[start:stop])
            # `row[i]` is `d sigma_i / d eps` (the VJP with the i-th unit
            # cotangent), so stacking over the last axis gives
            # `stack[p, j, i] = d sigma_i / d eps_j`; the transpose turns it
            # into `tangent[p, i, j] = d sigma_i / d eps_j`, the layout the
            # solver contracts. The learned Jacobian is not assumed
            # symmetric, so the transpose is a real term, not a detail.
            rows = [
                torch.autograd.grad(
                    stress,
                    strain_chunk,
                    grad_outputs=eye[component].expand_as(stress),
                    retain_graph=(component < 2),
                )[0]
                for component in range(3)
            ]
            tangent[start:stop] = torch.stack(rows, dim=-1).transpose(-1, -2).detach().numpy()
        return tangent

    def evaluate(
        self,
        total_strain_kelvin: FloatArray,
        *,
        time_increment: float = 1.0,
        compute_tangent: bool = True,
        context: FloatArray | None = None,
    ) -> TannFCCTrial:
        strain = torch.from_numpy(np.asarray(total_strain_kelvin, dtype=np.float64))
        if strain.shape != (self.point_count, 3):
            raise ValueError(f"total_strain_kelvin must have shape {(self.point_count, 3)}")
        context_tensor = None
        if context is not None:
            context_tensor = torch.from_numpy(np.asarray(context, dtype=np.float64))
            expected = (self.point_count, 12, self.config.context_dim)
            if context_tensor.shape != expected:
                raise ValueError(f"context must have shape {expected}, got {context_tensor.shape}")
        elif self.config.context_dim > 0:
            # No context means zero context: the network input is widened
            # by context_dim regardless, so an explicit zero tensor keeps
            # the embedding shape contract.
            context_tensor = torch.zeros(
                (self.point_count, 12, self.config.context_dim), dtype=torch.float64
            )
        with torch.no_grad():
            q_trial, dissipation, slip_work = self._integrate(
                self._committed_strain,
                strain,
                self._committed_state,
                context=context_tensor,
            )
        gamma = q_trial[..., 0:1]
        z = q_trial[..., 1:]
        stress = self._stress_from(strain, gamma[..., 0], self._systems)
        tangent = self._algorithmic_tangent(strain, context_tensor) if compute_tangent else None
        self._trial_state = q_trial.detach()
        self._trial_strain = strain.detach()
        if tangent is None:
            tangent_np: FloatArray | None = None
        elif isinstance(tangent, np.ndarray):
            tangent_np = tangent
        else:
            tangent_np = tangent.detach().numpy()
        trial = TannFCCTrial(
            stress_in_plane_mpa=stress.detach().numpy(),
            consistent_tangent_mpa=tangent_np,
            plastic_slip=gamma[..., 0].detach().numpy(),
            latent_state=z.detach().numpy(),
            generalised_dissipation=dissipation.detach().numpy(),
            slip_work=slip_work.detach().numpy(),
            trial_state=self._trial_state.numpy(),
        )
        self._latest_trial = trial
        return trial

    # -- the PlaneStressMaterialBatch protocol -------------------------------

    @property
    def backend_name(self) -> str:
        return "tann-fcc-causal-t0"

    @property
    def completion_strategy(self) -> str:
        return "tann_fcc_analytic_plane_stress"

    @property
    def statistics(self):
        from fem_inhouse.core.plane_stress_material import PlaneStressBatchStatistics

        return PlaneStressBatchStatistics()

    @property
    def linear_system_matrix_type(self) -> str:
        """The learned algorithmic tangent is not assumed symmetric."""

        return "nonsymmetric"

    def evaluate_in_plane(
        self,
        in_plane_strain: FloatArray,
        *,
        time_increment: float,
        consistent_tangent: bool = True,
    ):
        """Light non-committed response required by the global Newton loop.

        `time_increment` is accepted for protocol compatibility and ignored:
        the TANN evolves along the strain path `s in [0, 1]`, not in time.
        """

        from fem_inhouse.core.kelvin import (
            stiffness_to_engineering,
            strain_from_engineering,
            stress_to_voigt,
        )
        from fem_inhouse.core.plane_stress_material import InPlaneConstitutiveTrial

        trial = self.evaluate(
            strain_from_engineering(in_plane_strain),
            compute_tangent=consistent_tangent,
        )
        return InPlaneConstitutiveTrial(
            stress_in_plane_mpa=stress_to_voigt(trial.stress_in_plane_mpa),
            tangent_in_plane_mpa=(
                None
                if trial.consistent_tangent_mpa is None
                else stiffness_to_engineering(trial.consistent_tangent_mpa)
            ),
            observables={
                "generalised_dissipation": trial.generalised_dissipation,
                "slip_work": trial.slip_work,
                "plastic_slip": trial.plastic_slip,
                "latent_state": trial.latent_state,
            },
        )

    def complete_trial(self, trial):
        """Reconstruct the complete three-dimensional trial state.

        The out-of-plane part is the analytic plane-stress closure: ``eps_e_zz``
        follows from the in-plane elastic trace and the unobserved total
        transverse shears follow the plastic shears, leaving
        ``eps_e_xz = eps_e_yz = 0``.  Re-evaluating isotropic Hooke elasticity
        then gives ``sigma_zz = sigma_xz = sigma_yz = 0``.
        """

        from fem_inhouse.core.plane_stress_material import ConstitutiveTrial

        latest = self._latest_trial
        if latest is None or self._trial_strain is None:
            raise TypeError("no TANN-FCC plane-stress trial available")
        strain_kelvin = self._trial_strain.numpy()
        systems = self._systems.numpy()
        gamma = latest.plastic_slip  # (P, 12)
        plastic = np.einsum("pa,paij->pij", gamma, systems)
        elastic = np.zeros_like(plastic)
        sqrt_two = np.sqrt(2.0)
        elastic[..., 0, 0] = strain_kelvin[..., 0] - plastic[..., 0, 0]
        elastic[..., 1, 1] = strain_kelvin[..., 1] - plastic[..., 1, 1]
        elastic[..., 0, 1] = strain_kelvin[..., 2] / sqrt_two - plastic[..., 0, 1]
        elastic[..., 1, 0] = elastic[..., 0, 1]
        elastic[..., 2, 2] = -(self._lam / (self._lam + 2.0 * self._two_mu / 2.0)) * (
            elastic[..., 0, 0] + elastic[..., 1, 1]
        )
        # Plane stress eliminates the unobserved transverse *elastic* shears.
        # Their total-strain values are therefore the plastic values, not zero.
        elastic[..., 0, 2] = 0.0
        elastic[..., 2, 0] = 0.0
        elastic[..., 1, 2] = 0.0
        elastic[..., 2, 1] = 0.0
        total = elastic + plastic
        stress_tensor = np.zeros_like(plastic)
        stress_tensor[..., 0, 0] = latest.stress_in_plane_mpa[..., 0]
        stress_tensor[..., 1, 1] = latest.stress_in_plane_mpa[..., 1]
        stress_tensor[..., 0, 1] = latest.stress_in_plane_mpa[..., 2] / sqrt_two
        stress_tensor[..., 1, 0] = stress_tensor[..., 0, 1]
        from fem_inhouse.core.kelvin import stiffness_to_engineering, stress_to_voigt

        return ConstitutiveTrial(
            stress_in_plane_mpa=stress_to_voigt(latest.stress_in_plane_mpa),
            tangent_in_plane_mpa=(
                None
                if latest.consistent_tangent_mpa is None
                else stiffness_to_engineering(latest.consistent_tangent_mpa)
            ),
            observables={
                "generalised_dissipation": latest.generalised_dissipation,
                "slip_work": latest.slip_work,
                "plastic_slip": latest.plastic_slip,
                "latent_state": latest.latent_state,
            },
            full_stress_tensor_mpa=stress_tensor,
            full_strain_tensor=total,
            elastic_strain_tensor=elastic,
            plastic_strain_tensor=plastic,
            plane_stress_residual_mpa=np.zeros((self.point_count, 3), dtype=np.float64),
        )

    def commit(self) -> None:
        if self._trial_state is None:
            raise RuntimeError("no trial state to commit")
        self._committed_state = self._trial_state.clone()
        assert self._trial_strain is not None
        self._committed_strain = self._trial_strain.clone()
        # Keep the accepted tangent for the discrete trajectory adjoint:
        # the mechanical transpose action uses the converged C_alg. The
        # accepted dissipation and slip work are kept for the diagnostics
        # the run artifact reports per state.
        if self._latest_trial is not None and self._latest_trial.consistent_tangent_mpa is not None:
            from fem_inhouse.core.kelvin import stiffness_to_engineering

            self._last_committed_tangent = np.array(
                stiffness_to_engineering(self._latest_trial.consistent_tangent_mpa),
                copy=True,
            )
        else:
            self._last_committed_tangent = None
        if self._latest_trial is not None:
            self._last_committed_dissipation = np.array(
                self._latest_trial.generalised_dissipation, copy=True
            )
            self._last_committed_slip_work = np.array(self._latest_trial.slip_work, copy=True)
        else:
            self._last_committed_dissipation = None
            self._last_committed_slip_work = None
        self._trial_state = None
        self._trial_strain = None
        self._latest_trial = None

    def revert(self) -> None:
        self._trial_state = None
        self._trial_strain = None
        self._latest_trial = None

    @property
    def last_committed_tangent(self) -> FloatArray | None:
        return self._last_committed_tangent

    @property
    def last_committed_dissipation(self) -> FloatArray | None:
        return self._last_committed_dissipation

    @property
    def last_committed_slip_work(self) -> FloatArray | None:
        return self._last_committed_slip_work

    # -- trajectory-adjoint VJPs ---------------------------------------------

    def increment_vjp(
        self,
        strain_prev: FloatArray,
        state_prev: FloatArray,
        strain_trial: FloatArray,
        cotangent_state: FloatArray,
        cotangent_stress: FloatArray,
    ) -> tuple[FloatArray, FloatArray, list[FloatArray]]:
        """VJPs of one material increment `(strain_prev, state_prev) -> strain_trial`.

        With `q_trial = Q(strain_trial; state_prev, theta)` and
        `sigma = S(strain_trial, q_trial)`, returns

            v_strain = (dQ/deps)^T c_state + (dS/deps)^T c_stress
            v_qprev  = (dQ/dq_prev)^T c_state + (dS/dq_prev)^T c_stress
            dtheta   = (dQ/dtheta)^T c_state + (dS/dtheta)^T c_stress

        chunked like the tangent so the autograd graph of one chunk is freed
        before the next is built.
        """

        chunk = 2048
        v_strain = np.zeros((self.point_count, 3), dtype=np.float64)
        v_qprev = np.zeros((self.point_count, 12, 1 + self.config.latent_dim), dtype=np.float64)
        dtheta: list[FloatArray] | None = None
        for start in range(0, self.point_count, chunk):
            stop = min(start + chunk, self.point_count)
            s_prev = torch.from_numpy(np.ascontiguousarray(strain_prev[start:stop]))
            q_prev = torch.from_numpy(np.ascontiguousarray(state_prev[start:stop])).requires_grad_(
                True
            )
            s_trial = torch.from_numpy(
                np.ascontiguousarray(strain_trial[start:stop])
            ).requires_grad_(True)
            systems = self._systems[start:stop]
            q_trial, _, _ = self._integrate(
                s_prev, s_trial, q_prev, systems=systems, include_work=False
            )
            stress = self._stress_from(s_trial, q_trial[..., 0], systems)
            c_state = torch.from_numpy(np.ascontiguousarray(cotangent_state[start:stop]))
            c_stress = torch.from_numpy(np.ascontiguousarray(cotangent_stress[start:stop]))
            parameters = list(self._network.parameters())
            grads_state = torch.autograd.grad(
                q_trial,
                (s_trial, q_prev, *parameters),
                grad_outputs=c_state,
                retain_graph=True,
            )
            grads_stress = torch.autograd.grad(
                stress,
                (s_trial, q_prev, *parameters),
                grad_outputs=c_stress,
                retain_graph=False,
            )
            v_strain[start:stop] = (grads_state[0] + grads_stress[0]).detach().numpy()
            v_qprev[start:stop] = (grads_state[1] + grads_stress[1]).detach().numpy()
            if dtheta is None:
                dtheta = [
                    (g_state + g_stress).detach().numpy()
                    for g_state, g_stress in zip(grads_state[2:], grads_stress[2:], strict=True)
                ]
            else:
                for target, g_state, g_stress in zip(
                    dtheta, grads_state[2:], grads_stress[2:], strict=True
                ):
                    target += (g_state + g_stress).detach().numpy()
        assert dtheta is not None
        return v_strain, v_qprev, dtheta

    def reset_committed(self, state: FloatArray, strain: FloatArray) -> None:
        """Replace the committed state and strain (benchmarks, adjoint)."""

        committed = np.asarray(state, dtype=np.float64)
        committed_strain = np.asarray(strain, dtype=np.float64)
        if committed.shape != (self.point_count, 12, 1 + self.config.latent_dim):
            expected = (self.point_count, 12, 1 + self.config.latent_dim)
            raise ValueError(f"state must have shape {expected}")
        if committed_strain.shape != (self.point_count, 3):
            raise ValueError(f"strain must have shape {(self.point_count, 3)}")
        self._committed_state = torch.from_numpy(committed.copy())
        self._committed_strain = torch.from_numpy(committed_strain.copy())
        self._trial_state = None
        self._trial_strain = None
        self._latest_trial = None
        self._last_committed_tangent = None
        self._last_committed_dissipation = None
        self._last_committed_slip_work = None

    @property
    def committed_state(self) -> FloatArray:
        return self._committed_state.numpy()

    @property
    def committed_strain(self) -> FloatArray:
        return self._committed_strain.numpy()
