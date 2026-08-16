# What the reduction attempts established, and why the question changed

Three reduced representations were tried on the measured equivalent-strain
morphology of P43. All three failed, and the way they failed is the result.

## The data they were tried on

Full-field DIC recomputed from the forty-two speckle images, converged DISFlow
settings (`finest_scale` 0, patch 4, stride 1, alpha 15, epsilon 0.01, a hundred
refinement iterations), 41 states of 3600x3100 referred to the undeformed
reference. Noise floor measured from the repeated final state: 0.148 pixel on
displacement, 0.100 on the equivalent strain, with signal above noise down to a
two-pixel wavelength. The morphology maps are `H_t = EVM_t / mean(EVM_t)`.

## The three failures

**POD.** Temporal holdout saturates at 0.115 from rank 2 to rank 32 while the
training error reaches exactly 0.000 at rank 32 -- thirty-one modes interpolate
thirty-two snapshots by construction.

**Convolutional autoencoder.** Flat from step 6000 to 20000 at a gradient error
of 0.97, with the four evaluations agreeing to within ten percent. Converged to
a poor solution rather than undertrained. At s32/d8 it carried 86016 numbers per
state and still destroyed everything below thirty-two pixels.

**CROM-style neural field.** Gradient error 1.0000 at step 5000, field error not
improving. One of my two attempts was invalidated by a bandwidth bug -- a
Fourier scale of 12 resolved nothing finer than forty-five pixels -- and the
corrected version at scale 129 did no better.

## The measurement that explains all three

POD fitted separately to each band of a Laplacian pyramid, same states, same
temporal holdout, error relative to the structure in that band:

| band | rank 1 | rank 4 | rank 16 | rank 31 | train, rank 31 |
|---|---|---|---|---|---|
| 2-8 px | 0.775 | 0.644 | 0.578 | 0.562 | 0.000 |
| 8-32 px | 0.689 | 0.591 | 0.517 | 0.507 | 0.000 |
| 32-128 px | 0.725 | 0.617 | 0.553 | 0.544 | 0.000 |
| above 128 px | 0.491 | 0.364 | 0.330 | 0.320 | 0.000 |

No band is compressible. At rank 31, essentially the full training basis, the
held-out error is 0.32 to 0.56 everywhere, including the coarse mechanical
scales, while the training error is exactly zero in all four.

**The 0.115 global figure was an artefact of the normalisation.** `H` is close
to one everywhere, so a relative error against its norm mostly measures having
recovered a constant level. Per band, with that level removed, POD generalises
poorly at every scale. Every comparison in this campaign that used 0.115 as the
bar was measured against a meaningless number.

## What this does and does not show

It does not show that the fields are unmodellable. Training error of zero with
holdout error of a half is the signature of a manifold whose points do not lie
in any fixed subspace -- which is what happens if a band appears in one place at
state 18 and elsewhere at state 19. A shared *local* rule has no such problem.

It does show that thirty-two states cannot support a temporal holdout at this
level: with a rank-31 basis for thirty-two snapshots there is no statistical
room, and the experiment tests interpolation of a manifold sampled by thirty-two
points rather than compactness.

The statistical power of this dataset is spatial, not temporal. Ten million
pixels, forty states.

## Two diagnostics worth keeping

**The fine band is not a fixed pattern.** Correlating the 2-8 px maps between
states gives 0.820 between neighbours decaying to 0.188 between extremes, with
the across-state mean holding 47.8 % of the band energy. Half is a common
pattern, half evolves. An earlier proposal of mine to low-pass it away is
withdrawn: constant band energy does not mean a constant map, and the evolving
half is where a late fine plastic structure would appear.

**The strain follows the Schmid factor, weakly but monotonically.**
`corr(EVM_t, S)` rises from +0.008 at state 1 to +0.116 at state 38, positive
throughout. A registration or imaging artefact would be constant; a correlation
that starts at zero under no load and grows with plastification is the expected
signature of strain concentrating where slip is easiest. The EBSD registration
is declared but not verified, which remains the dominant uncertainty.

## The question that replaces the old one

Not "does the field admit a low-dimensional global representation" but:

> is there a spatially transferable local rule which, coupled to equilibrium
> and thermodynamics, reproduces the measured heterogeneity?

The field itself may then have an enormous dimension; what has to be compact is
the rule that generates it, applied to local mechanical states that equilibrium
supplies.

Two guardrails, both of which this campaign would have crossed without noticing:

* **local inpainting is not a local law.** A network shown a large context and
  asked for a small core can interpolate. Any morphological transfer test needs
  a bicubic or kriging baseline, and beating it proves a transferable local
  regularity, nothing more.
* **P43 qualifies the software, not the hypothesis.** The small crop is where
  the primal, the adjoint, Kelvin and the dissipation get verified; spatial
  transferability can only be established on the full field.

And the identifiability of the plastic field is not restored by equilibrium: the
operator is surjective onto zero-boundary fields, so fitting the DIC is
guaranteed and proves nothing. What breaks the degeneracy is weight sharing --
one function used at ten million places -- together with incompressibility and
positive dissipation. That is a statistical guarantee, not a mechanical one, and
it should be stated as such.
