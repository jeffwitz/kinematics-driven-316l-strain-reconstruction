## Purpose

Describe the case-study requirement addressed by this change.

## Validation

- [ ] `ruff check .`
- [ ] `mypy src/fem_inhouse`
- [ ] `pytest --cov=fem_inhouse --cov-branch`
- [ ] wheel build
- [ ] scientific conventions and units unchanged or documented
- [ ] performance checked if array allocation or sparse assembly changed

## Numerical formula changes

- [ ] no numerical formula changed
- [ ] or: independent mechanics reviewer identified
- [ ] closed-form, finite-difference, or independent-reference test added
- [ ] previous/new result comparison attached

## Scientific limitations

List missing Abaqus/DIC references or other claims that this change does not
validate.

## Numerical-method changes

- [ ] Existing qualified reference path identified
- [ ] No production robustness logic duplicated in benchmark/prototype code
- [ ] Transaction and cutback semantics preserved or differences documented
- [ ] Candidate compared against the reference before performance claims
