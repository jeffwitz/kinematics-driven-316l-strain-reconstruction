#!/usr/bin/env python3
"""Learn the plastic flow direction itself, inside a reduced model.

The first rung built the directions by hand and both hand-built answers failed.
A fixed residual-driven basis fits where it is fitted and does not transfer;
isotropic normality `N(sigma)`, banded by equivalent stress, never gets
meaningfully below the elastic prediction. The measurement pointed at the
*direction* rather than at the amplitude parameterisation, so the direction is
what has to be learned. Substituting slip-system tensors would be one more
hand-built guess.

```text
S_n(x)  ->  F_theta  ->  Phi_n = [phi_1 ... phi_r]     r full-field directions
                             |
                    equilibrium picks a_n in R^r
                             |
              d eps_p_n = Phi_n a_n,  loss = |eps_sim - eps_DIC|^2
```

The network never sees the interior DIC. Its input is the *predictor* state --
stress, strain, accumulated plastic strain and plastic path, all Kelvin, all
from the previous plastic state pushed through equilibrium with the measured
boundary data. It never sees a coordinate either, so it cannot learn a map of
the specimen; what it can learn is a rule from local mechanical state to
plastic direction, shared over every point by the convolution.

Differentiating through the mechanics is exact rather than approximated. `A` is
linear and its adjoint is qualified at 1.5e-15 by a dot-product test, so it
wraps into an autograd function whose backward pass is `A^T` and nothing else.
The reduced solve is a ridge normal equation in `r` unknowns, which torch
differentiates directly; no implicit-function machinery is needed while the
coefficients are unconstrained.

Two properties are reported rather than imposed, because imposing either would
constrain the direction and that is the object under test. Dissipation enters as
a penalty on the negative part and is *measured* at the midpoint after the
rollout. Mode collapse is penalised through `|G - I|` on the Gram matrix and is
reported as the effective rank of `A Phi`.

The holdout is temporal, not spatial. The first rung established that a spatial
holdout is unreachable in principle here: the free plastic field, fitted exactly
everywhere else, still leaves 0.60 of the elastic defect inside a 30-pixel
square, because equilibrium and the surrounding data do not determine what
happens in a hole. Held-out *increments* have no such defect -- the basis is
regenerated from the state at each one, so carrying an increment it was never
fitted on is exactly the claim being made. Those increments contribute no
gradient; the rollout passes through them on the network's own predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear, nnls
from torch import nn

from fem_inhouse.core.kelvin import KELVIN_SCALE_2D, PLANE_STRESS_PLASTIC_GAUGE
from fem_inhouse.identification.tensor_plastic_observability import (
    TensorPlasticObservabilityOperator,
)
from fem_inhouse.spectral2d.grid import StructuredGrid2D
from fem_inhouse.spectral2d.newton_ebi import pack_interior, unpack_interior

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
    / "repaired_history_mm.npy"
)
PIXEL_SIZE_MM = 0.00184
YOUNG_MPA = 205_000.0
POISSON = 0.30


class _Identity:
    def apply(self, values):
        return np.asarray(values, dtype=np.float64)

    def adjoint(self, values):
        return np.asarray(values, dtype=np.float64)


class Generator(nn.Module):
    """Local state to `r` plastic directions, fully convolutional and dilated.

    No pooling and no coordinate channel. The dilations give three receptive
    ranges without ever collapsing position, which is what the pooled predictor
    of the inpainting attempt got wrong. Output modes are normalised to unit
    root-mean-square so the amplitude lives entirely in the coefficients and the
    Gram penalty compares shapes rather than scales.
    """

    def __init__(self, *, channels: int, rank: int, width: int = 48) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = channels
        for dilation in (1, 2, 4, 8, 1):
            layers += [
                nn.Conv2d(previous, width, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, width),
                nn.GELU(),
            ]
            previous = width
        layers.append(nn.Conv2d(width, 6 * rank, 1))
        self.body = nn.Sequential(*layers)
        self.rank = rank

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.body(state)[0]
        nx, ny = raw.shape[1], raw.shape[2]
        modes = raw.reshape(self.rank, 2, 3, nx, ny).permute(3, 4, 1, 2, 0)
        modes = modes.reshape(-1, 3, self.rank)
        scale = torch.einsum("pir,ij,pjr->r", modes, GAUGE, modes).clamp_min(1e-24).sqrt()
        return modes / scale


#: `eps_p_zz` is fixed by incompressibility, so the plane-stress plastic triple
#: is not an orthonormal subspace of deviatoric space and its norm is not the
#: Euclidean one. Every plastic magnitude in this script -- the accumulated
#: path fed to the network, the mode normalisation, the diversity Gram -- goes
#: through this gauge. An earlier revision used `|v|_2` for the path, which
#: silently corrupted one of the network's own input channels.
GAUGE = torch.from_numpy(PLANE_STRESS_PLASTIC_GAUGE)
#: The same object without the 2/3, i.e. the plain three-dimensional Frobenius
#: norm of the completed tensor, which is what a cosine against the stress
#: needs.
FROBENIUS = torch.from_numpy(1.5 * PLANE_STRESS_PLASTIC_GAUGE)


def gauge_norm(values: torch.Tensor) -> torch.Tensor:
    """`sqrt(v^T G_p v)` per material point: the equivalent plastic strain."""

    return torch.einsum("pi,ij,pj->p", values, GAUGE, values).clamp_min(0.0).sqrt()


def project_dissipative(modes: torch.Tensor, stress: torch.Tensor) -> torch.Tensor:
    """Each learned direction, pushed into the half-space `sigma . v >= 0`.

    ```text
    P(v) = v + relu(-sigma . v) / |sigma|^2 * sigma      so   sigma . P(v) = max(sigma . v, 0)
    ```

    This is emphatically not the J2 arm. Nothing tells the network that the
    plastic increment should be parallel to `N(sigma)`; it may learn any tensor
    direction whatsoever inside the whole half-space. The only thing forbidden
    is climbing the thermodynamic slope, which is the freedom the previous run
    showed the network exploiting -- 37 to 43 % of points anti-dissipative,
    because the gradients make those directions extremely profitable against
    the DIC loss.

    Incompressibility needs no projection here and imposing one would be wrong.
    Plastic incompressibility is `tr_3(eps_p) = 0`, which under plane stress
    fixes `eps_p_zz = -(eps_p_xx + eps_p_yy)` and leaves the in-plane triple
    entirely free; demanding a vanishing *in-plane* trace would force
    `eps_p_zz = 0`, a plane-strain plasticity this specimen does not have. With
    `sigma_zz = 0` the in-plane Kelvin dot product also equals the full
    three-dimensional `sigma : eps_p` exactly, so the half-space above is the
    complete thermodynamic condition rather than an in-plane shadow of it.

    Modes are renormalised afterwards: the projection changes their length, and
    the Gram penalty and the ridge scaling both compare shapes, not sizes.
    """

    overlap = torch.einsum("pir,pi->pr", modes, stress)
    square = stress.pow(2).sum(dim=1, keepdim=True)
    # Where the stress vanishes the half-space is the whole space and the
    # correction is a division by nothing; leave those directions untouched
    # rather than let a near-zero denominator manufacture an enormous mode.
    floor = 1e-12 * square.mean().clamp_min(1e-300)
    correction = (torch.relu(-overlap) / square.clamp_min(floor)).unsqueeze(1)
    pushed = modes + torch.where(
        (square < floor).unsqueeze(2), torch.zeros_like(overlap).unsqueeze(1), correction
    ) * stress.unsqueeze(2)
    scale = torch.einsum("pir,ij,pjr->r", pushed, GAUGE, pushed).clamp_min(1e-24).sqrt()
    return pushed / scale


def non_negative_solve(gram: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """`argmin_a >= 0  a^T G a - 2 a^T b`, differentiated through its active set.

    Cholesky alone solves the *unconstrained* system and settles nothing about
    the sign; the algorithm is an **active-set NNLS followed by a Cholesky
    solve on the free variables**, in that order. Cholesky only compresses the
    problem first: `G = L L^T` turns `min_{a>=0} a^T G a - 2 a^T b` into an
    ordinary `r x r` NNLS on `(L^T, L^-1 b)`, so finding the active set costs
    nothing beside one application of the mechanics. Only then are the free
    coefficients re-solved.

    Differentiating through it is exact almost everywhere by KKT: the active
    constraints are locally constant, so the gradient of the solution is the
    gradient of the reduced linear system. It is undefined exactly at the
    changes of active set, which is the same measure-zero caveat a ReLU carries.

    `check_non_negative_solve` validates this against `lsq_linear`, a
    trust-region method sharing no code path with the active-set one.
    """

    factor = torch.linalg.cholesky(gram)
    reduced = torch.linalg.solve_triangular(factor, right[:, None], upper=False)[:, 0]
    guess, _ = nnls(factor.T.detach().numpy(), reduced.detach().numpy())
    free = torch.from_numpy(np.nonzero(guess > 0.0)[0])
    answer = torch.zeros(gram.shape[0], dtype=gram.dtype)
    if free.numel() == 0:
        return answer
    solved = torch.linalg.solve(gram[free][:, free], right[free])
    return answer.index_put((free,), solved)


def check_non_negative_solve(trials: int = 12, rank: int = 9) -> float:
    """Agreement with an independent bounded solver, on random problems."""

    generator = np.random.default_rng(11)
    worst = 0.0
    for _ in range(trials):
        raw = generator.standard_normal((3 * rank, rank))
        gram = torch.from_numpy(raw.T @ raw + 1e-6 * np.eye(rank))
        right = torch.from_numpy(generator.standard_normal(rank) * 5.0)
        mine = non_negative_solve(gram, right).numpy()
        factor = np.linalg.cholesky(gram.numpy())
        reference = lsq_linear(
            factor.T,
            np.linalg.solve(factor, right.numpy()),
            bounds=(0.0, np.inf),
            tol=1e-14,
        ).x
        scale = max(np.linalg.norm(reference), 1e-12)
        worst = max(worst, float(np.linalg.norm(mine - reference) / scale))
        assert mine.min() > -1e-10, mine.min()
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", nargs=2, type=int, default=(1580, 1030))
    parser.add_argument("--pixels", type=int, default=100)
    parser.add_argument("--reference-state", type=int, default=20)
    parser.add_argument("--states", nargs="+", type=int, default=list(range(21, 41)))
    parser.add_argument("--holdout", nargs="+", type=int, default=[24, 28, 32, 36, 40])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--orthogonality", type=float, default=1e-2)
    parser.add_argument("--dissipation", type=float, default=1e-2)
    parser.add_argument("--project-dissipative", action="store_true")
    parser.add_argument("--load-weights", type=Path, default=None,
                        help="score a finished run's weights, adding no training")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(20260816)
    pixels = arguments.pixels
    x0, y0 = arguments.origin
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )
    points = operator.kinematics.material_point_count

    def kelvin_strain(field) -> np.ndarray:
        return operator.kelvin_strain(field).reshape(-1, 3)

    def stress_of(strain: np.ndarray) -> np.ndarray:
        return np.einsum("pi,pij->pj", strain.reshape(-1, 3), operator.elasticity)

    def divergence(stress_kelvin: np.ndarray) -> np.ndarray:
        voigt = stress_kelvin.reshape(-1, 3) / KELVIN_SCALE_2D
        return pack_interior(
            operator.kinematics.divergence_from_sample_stress(
                voigt.reshape((pixels, pixels, 2, 3))
            )
        )

    def elastic_lift(field: np.ndarray) -> np.ndarray:
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        return lifted

    def green(values: np.ndarray) -> np.ndarray:
        displacement = unpack_interior(
            operator.solve_stiffness(operator._strain_transpose(values.reshape(-1))), grid
        )
        return kelvin_strain(displacement)

    def apply_numpy(plastic: np.ndarray) -> np.ndarray:
        flat = plastic.reshape(points, 3, -1)
        return np.stack(
            [green(stress_of(flat[:, :, k])) for k in range(flat.shape[2])], axis=2
        ).reshape(plastic.shape)

    def transpose_numpy(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(points, 3, -1)
        return np.stack(
            [stress_of(green(flat[:, :, k])) for k in range(flat.shape[2])], axis=2
        ).reshape(values.shape)

    # A is linear and its adjoint is qualified, so this is exact.
    class Mechanics(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return torch.from_numpy(apply_numpy(tensor.detach().numpy().astype(np.float64)))

        @staticmethod
        def backward(ctx, gradient):
            return torch.from_numpy(
                transpose_numpy(gradient.numpy().astype(np.float64))
            )

    generator_rng = np.random.default_rng(20260816)
    left = generator_rng.standard_normal((points, 3))
    right = generator_rng.standard_normal((points, 3))
    discrepancy = abs(
        float((apply_numpy(left) * right).sum()) - float((left * transpose_numpy(right)).sum())
    ) / abs(float((apply_numpy(left) * right).sum()))
    print(f"adjoint dot-product test: {discrepancy:.3e}", flush=True)
    assert discrepancy < 1e-8
    if arguments.project_dissipative:
        agreement = check_non_negative_solve()
        print(f"non-negative solve against lsq_linear: {agreement:.3e}", flush=True)
        assert agreement < 1e-6

    report = json.loads((HISTORY.with_name("report.json")).read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    source = np.load(HISTORY, mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    reference = history[arguments.reference_state]
    residual = np.linalg.norm(divergence(stress_of(kelvin_strain(elastic_lift(
        history[40] - reference
    )))))
    print(f"elastic lifting residual: {residual:.3e}", flush=True)

    states = arguments.states
    holdout = set(arguments.holdout)
    training = [s for s in states if s not in holdout]
    measured = {s: kelvin_strain(history[s] - reference) for s in states}
    elastic = {s: kelvin_strain(elastic_lift(history[s] - reference)) for s in states}
    defect = {s: float(np.linalg.norm(measured[s] - elastic[s])) for s in states}
    reference_stress = stress_of(kelvin_strain(reference))
    print(f"{len(training)} training increments, {len(holdout)} held out: "
          f"{sorted(holdout)}", flush=True)

    def to_channels(*fields: np.ndarray) -> torch.Tensor:
        """Material-point fields to a `(1, C, nx, ny)` image, subcells as channels."""

        stacked = np.concatenate(
            [f.reshape(pixels, pixels, 2, -1) for f in fields], axis=3
        )
        image = stacked.reshape(pixels, pixels, -1).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image))[None]

    def rollout(network: nn.Module, rank: int, grad: bool):
        """One pass over every increment, gradients only on training ones."""

        plastic = torch.zeros((points, 3), dtype=torch.float64)
        path = torch.zeros((points, 1), dtype=torch.float64)
        previous_stress = torch.from_numpy(reference_stress)
        loss = torch.zeros((), dtype=torch.float64)
        scores, negative, singular = {}, [], []
        for state in states:
            plastic_np = plastic.detach().numpy()
            predictor = torch.from_numpy(elastic[state] + apply_numpy(plastic_np))
            stress = torch.from_numpy(reference_stress) + torch.from_numpy(
                stress_of(predictor.numpy() - plastic_np)
            )
            target = torch.from_numpy(measured[state]) - predictor
            state_image = to_channels(
                stress.numpy() / max(np.abs(stress.numpy()).std(), 1e-30),
                predictor.numpy() / max(np.abs(predictor.numpy()).std(), 1e-30),
                plastic_np / max(np.abs(plastic_np).std(), 1e-30),
                path.numpy() / max(float(path.numpy().std()), 1e-30),
            )
            wants = grad and state in training
            with torch.set_grad_enabled(wants):
                modes = network(state_image)
                if arguments.project_dissipative:
                    modes = project_dissipative(modes, stress)
                responses = Mechanics.apply(modes)
                flat = responses.reshape(-1, rank)
                gram = flat.T @ flat
                right_hand = flat.T @ target.reshape(-1)
                ridge = arguments.ridge * torch.diagonal(gram).mean().clamp_min(1e-300)
                regular = gram + ridge * torch.eye(rank, dtype=torch.float64)
                coefficients = (
                    non_negative_solve(regular, right_hand)
                    if arguments.project_dissipative
                    else torch.linalg.solve(regular, right_hand)
                )
                increment = modes @ coefficients
                simulated = predictor + responses @ coefficients
                remaining = torch.from_numpy(measured[state]) - simulated
                if wants:
                    overlap = torch.einsum("pir,ij,pjs->rs", modes, GAUGE, modes)
                    new_stress = torch.from_numpy(reference_stress) + torch.from_numpy(
                        stress_of(simulated.detach().numpy() - (plastic + increment)
                                  .detach().numpy())
                    )
                    power = (0.5 * (previous_stress + new_stress) * increment).sum(dim=1)
                    loss = (
                        loss
                        + remaining.pow(2).sum() / defect[state] ** 2
                        + arguments.orthogonality
                        * (overlap - torch.eye(rank, dtype=torch.float64)).pow(2).sum()
                        + arguments.dissipation * torch.relu(-power).mean()
                        / max(abs(float(power.detach().abs().mean())), 1e-30)
                    )
            with torch.no_grad():
                new_stress = torch.from_numpy(reference_stress) + torch.from_numpy(
                    stress_of(simulated.detach().numpy()
                              - (plastic + increment).detach().numpy())
                )
                power = (0.5 * (previous_stress + new_stress) * increment.detach()).sum(dim=1)
                # The count of offending points says nothing about how much
                # power flows the wrong way. For the fixed basis the two turn
                # out to agree closely -- 47 % of points and 47 % of |power| --
                # so the inadmissibility is not a small-amplitude edge effect.
                # The half-space forbids sigma . v < 0 but permits sigma . v = 0
                # with an enormous v: the network could trade anti-dissipative
                # work for a huge increment almost orthogonal to the stress,
                # which is admissible and not plausible. chi is that cosine on
                # the plastically active points, and the amplitude is reported
                # beside it so the trade is visible if it happens.
                amplitude = gauge_norm(increment.detach())
                frobenius = torch.einsum(
                    "pi,ij,pj->p", increment.detach(), FROBENIUS, increment.detach()
                ).clamp_min(0.0).sqrt()
                stress_norm = new_stress.pow(2).sum(dim=1).sqrt()
                active = amplitude > 0.1 * amplitude.mean().clamp_min(1e-300)
                cosine = power / (stress_norm * frobenius).clamp_min(1e-300)
                # chi is a genuine cosine: the denominator carries the
                # three-dimensional Frobenius norm `sqrt(z^T M_p z)` with
                # `M_p = 3/2 G_p`, not `p_eq`, which would inflate it by
                # sqrt(3/2) and let it exceed one. The pointwise mean is
                # dominated by the crowd of barely active points, so the
                # work-weighted global ratio is reported beside it.
                negative.append((
                    float((power < 0).double().mean()),
                    float(power.clamp_max(0).abs().sum()
                          / power.abs().sum().clamp_min(1e-300)),
                    float(power.sum()),
                    float(amplitude.sum()),
                    float(cosine[active].mean()) if bool(active.any()) else 0.0,
                    float(torch.quantile(power, 0.01)),
                    float(power.sum()),
                    float((stress_norm * frobenius).sum()),
                    float(cosine[active].median()) if bool(active.any()) else 0.0,
                    float(torch.quantile(cosine[active], 0.1))
                    if bool(active.any()) else 0.0,
                ))
                singular.append(
                    float(torch.linalg.svdvals(flat.detach())[-1]
                          / torch.linalg.svdvals(flat.detach())[0])
                )
                scores[state] = float(
                    torch.linalg.vector_norm(remaining.detach()) / defect[state]
                )
            plastic = (plastic + increment).detach()
            path = path + gauge_norm(increment.detach())[:, None]
            previous_stress = new_stress
        counts = np.asarray(negative)
        return (
            loss,
            scores,
            {
                "negative_point_fraction": float(counts[:, 0].mean()),
                "negative_power_share": float(counts[:, 1].mean()),
                "net_dissipation": float(counts[:, 2].sum()),
                "accumulated_equivalent_plastic_strain": float(counts[:, 3].sum() / points),
                "mean_stress_alignment_where_active": float(counts[:, 4].mean()),
                "first_percentile_of_power": float(counts[:, 5].min()),
                "global_stress_alignment": float(
                    counts[:, 6].sum() / max(counts[:, 7].sum(), 1e-300)
                ),
                "median_stress_alignment_where_active": float(counts[:, 8].mean()),
                "tenth_percentile_stress_alignment": float(counts[:, 9].mean()),
            },
            float(np.mean(singular)),
        )

    def krylov_basis(size: int) -> np.ndarray:
        """Fixed residual-driven basis, seeded from training increments only."""

        seeds = [transpose_numpy(measured[s] - elastic[s]).reshape(-1) for s in training]
        basis, _ = np.linalg.qr(np.asarray(seeds).T)
        columns, total = [basis], basis.shape[1]
        while total < size:
            grown = np.asarray(
                [transpose_numpy(apply_numpy(columns[-1][:, k].reshape(points, 3))).reshape(-1)
                 for k in range(columns[-1].shape[1])]
            ).T
            orthonormal, _ = np.linalg.qr(np.concatenate([*columns, grown], axis=1))
            addition = orthonormal[:, total:]
            if addition.shape[1] == 0:
                break
            columns.append(addition)
            total += addition.shape[1]
        return np.concatenate(columns, axis=1)[:, :size]

    def baseline(kind: str, rank: int) -> dict:
        """The same rollout and the same scoring, with modes nobody learned."""

        fixed = krylov_basis(rank) if kind == "krylov" else None
        plastic = np.zeros((points, 3))
        previous_stress = reference_stress
        scores, negative = {}, []
        for state in states:
            predictor = elastic[state] + apply_numpy(plastic)
            stress = reference_stress + stress_of(predictor - plastic)
            target = measured[state] - predictor
            if kind == "krylov":
                modes = fixed.reshape(points, 3, rank)
            else:
                xx, yy = stress[:, 0], stress[:, 1]
                shear = stress[:, 2] / np.sqrt(2.0)
                pressure = (xx + yy) / 3.0
                deviator = np.stack(
                    [xx - pressure, yy - pressure, np.sqrt(2.0) * shear], axis=1
                )
                norm = np.sqrt(np.maximum(
                    (xx - pressure) ** 2 + (yy - pressure) ** 2 + pressure**2
                    + 2.0 * shear**2, 1e-300))
                equivalent = np.sqrt(np.maximum(
                    xx**2 - xx * yy + yy**2 + 3.0 * shear**2, 0.0))
                centres = np.quantile(equivalent, np.linspace(0, 1, rank + 2)[1:-1])
                width = max(float(np.diff(centres).mean()) if rank > 1 else 1.0, 1e-12)
                bumps = np.exp(-0.5 * ((equivalent[:, None] - centres[None, :]) / width) ** 2)
                modes = (deviator / norm[:, None])[:, :, None] * bumps[:, None, :]
            responses = apply_numpy(modes).reshape(-1, rank)
            matrix = responses.T @ responses
            matrix += arguments.ridge * np.trace(matrix) / rank * np.eye(rank)
            coefficients = np.linalg.solve(matrix, responses.T @ target.reshape(-1))
            increment = modes @ coefficients
            plastic = plastic + increment
            simulated = elastic[state] + apply_numpy(plastic)
            new_stress = reference_stress + stress_of(simulated - plastic)
            power = (0.5 * (previous_stress + new_stress) * increment).sum(axis=1)
            gauge = PLANE_STRESS_PLASTIC_GAUGE
            amplitude = np.sqrt(np.maximum(
                np.einsum("pi,ij,pj->p", increment, gauge, increment), 0.0))
            frobenius = np.sqrt(np.maximum(
                np.einsum("pi,ij,pj->p", increment, 1.5 * gauge, increment), 0.0))
            stress_norm = np.sqrt((new_stress**2).sum(axis=1))
            active = amplitude > 0.1 * max(amplitude.mean(), 1e-300)
            cosine = power / np.maximum(stress_norm * frobenius, 1e-300)
            negative.append((
                float((power < 0).mean()),
                float(np.abs(np.minimum(power, 0.0)).sum()
                      / max(np.abs(power).sum(), 1e-300)),
                float(power.sum()),
                float(amplitude.sum()),
                float(cosine[active].mean()) if active.any() else 0.0,
                float(np.quantile(power, 0.01)),
                float(power.sum()),
                float((stress_norm * frobenius).sum()),
                float(np.median(cosine[active])) if active.any() else 0.0,
                float(np.quantile(cosine[active], 0.1)) if active.any() else 0.0,
            ))
            previous_stress = new_stress
            scores[state] = float(
                np.linalg.norm(measured[state] - simulated) / defect[state]
            )
        return {
            "per_state": {str(k): v for k, v in scores.items()},
            "fitted": float(np.mean([scores[s] for s in training])),
            "held_out": float(np.mean([scores[s] for s in sorted(holdout)])),
            "final_state": scores[states[-1]],
            "dissipation": {
                "negative_point_fraction": float(np.asarray(negative)[:, 0].mean()),
                "negative_power_share": float(np.asarray(negative)[:, 1].mean()),
                "net_dissipation": float(np.asarray(negative)[:, 2].sum()),
                "accumulated_equivalent_plastic_strain": float(
                    np.asarray(negative)[:, 3].sum() / points
                ),
                "mean_stress_alignment_where_active": float(
                    np.asarray(negative)[:, 4].mean()
                ),
                "first_percentile_of_power": float(np.asarray(negative)[:, 5].min()),
                "global_stress_alignment": float(
                    np.asarray(negative)[:, 6].sum()
                    / max(np.asarray(negative)[:, 7].sum(), 1e-300)
                ),
                "median_stress_alignment_where_active": float(
                    np.asarray(negative)[:, 8].mean()
                ),
                "tenth_percentile_stress_alignment": float(
                    np.asarray(negative)[:, 9].mean()
                ),
            },
        }

    channels = 2 * (3 + 3 + 3 + 1)
    results = {}
    for rank in arguments.ranks:
        for kind in ("krylov", "aligned"):
            entry = baseline(kind, rank)
            results[f"{kind}_r{rank}"] = entry
            print(f"{kind:8s} r={rank:2d}: fitted {entry['fitted']:.4f}  "
                  f"held out {entry['held_out']:.4f}  negative "
                  f"{100 * entry['dissipation']['negative_point_fraction']:.1f} % of points, "
                  f"{100 * entry['dissipation']['negative_power_share']:.1f} % of power  "
                  f"p_eq {entry['dissipation']['accumulated_equivalent_plastic_strain']:.3e}  "
                  f"chi {entry['dissipation']['mean_stress_alignment_where_active']:+.3f} "
                  f"(global {entry['dissipation']['global_stress_alignment']:+.3f})",
                  flush=True)
    for rank in arguments.ranks:
        network = Generator(channels=channels, rank=rank).double()
        if arguments.load_weights is not None:
            # Diagnostics added after a campaign started can be recovered from
            # the weights rather than by paying for the training again.
            network.load_state_dict(torch.load(
                arguments.load_weights.with_suffix(f".r{rank}.weights"), weights_only=True
            ))
            with torch.no_grad():
                _, scores, negative, conditioning = rollout(network, rank, grad=False)
            results[f"learned_r{rank}"] = {
                "per_state": {str(k): v for k, v in scores.items()},
                "fitted": float(np.mean([scores[s] for s in training])),
                "held_out": float(np.mean([scores[s] for s in sorted(holdout)])),
                "final_state": scores[states[-1]],
                "dissipation": negative,
                "smallest_over_largest_singular_value": conditioning,
            }
            print(f"scored r={rank}: fitted {results[f'learned_r{rank}']['fitted']:.4f}  "
                  f"held out {results[f'learned_r{rank}']['held_out']:.4f}", flush=True)
            continue
        optimiser = torch.optim.Adam(network.parameters(), lr=arguments.learning_rate)
        start = time.time()
        for step in range(1, arguments.steps + 1):
            optimiser.zero_grad()
            loss, _, _, _ = rollout(network, rank, grad=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimiser.step()
            if step % max(arguments.steps // 8, 1) == 0:
                with torch.no_grad():
                    _, scores, negative, conditioning = rollout(network, rank, grad=False)
                held = float(np.mean([scores[s] for s in sorted(holdout)]))
                fitted = float(np.mean([scores[s] for s in training]))
                print(
                    f"  r={rank:2d} step {step:4d}: loss {float(loss.detach()):.4e}  "
                    f"fitted {fitted:.4f}  held out {held:.4f}  "
                    f"negative {100 * negative['negative_point_fraction']:.1f} % of "
                    f"points, {100 * negative['negative_power_share']:.1f} % of power  "
                    f"p_eq {negative['accumulated_equivalent_plastic_strain']:.3e}  "
                    f"chi {negative['mean_stress_alignment_where_active']:+.3f} "
                    f"(global {negative['global_stress_alignment']:+.3f})  "
                    f"({time.time() - start:.0f} s)",
                    flush=True,
                )
        with torch.no_grad():
            _, scores, negative, conditioning = rollout(network, rank, grad=False)
        results[f"learned_r{rank}"] = {
            "per_state": {str(k): v for k, v in scores.items()},
            "fitted": float(np.mean([scores[s] for s in training])),
            "held_out": float(np.mean([scores[s] for s in sorted(holdout)])),
            "final_state": scores[states[-1]],
            "dissipation": negative,
            "smallest_over_largest_singular_value": conditioning,
        }
        # Without the weights nothing can be re-scored later, which is how the
        # magnitude of the negative dissipation went unmeasured on the first
        # pass of this very script.
        torch.save(
            network.state_dict(), arguments.output.with_suffix(f".r{rank}.weights")
        )
        print(f"learned r={rank}: fitted {results[f'learned_r{rank}']['fitted']:.4f}  "
              f"held out {results[f'learned_r{rank}']['held_out']:.4f}", flush=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "states": states,
                "holdout_states": sorted(holdout),
                "ranks": arguments.ranks,
                "steps": arguments.steps,
                "adjoint_discrepancy": discrepancy,
                "metric": (
                    "share of the elastic defect surviving, per increment: "
                    "|eps_measured - eps_sim| / |eps_measured - eps_elastic|"
                ),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
