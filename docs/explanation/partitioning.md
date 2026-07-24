# Why the ROI is partitioned

## Scale of the problem

The ROI contains

$$
3600\times3100=11{,}160{,}000
$$

pixel-sized CPS4 elements. A monolithic nonlinear solve would require a very
large sparse system plus four constitutive points per element. The article and
this repository therefore solve independent overlapping subdomains.

```{image} ../_static/partitioning.*
:alt: Ten-by-ten core layout and the retained core and padded solved region of corner partition zero.
:width: 96%
:align: center
```

## Core and solved region

Every partition has:

**Core**
: A unique, non-overlapping portion of the global ROI. This is the only part
  retained during stitching.

**Solved region**
: The core expanded by `padding` elements, clipped at global boundaries.
  DIC displacement is prescribed on the outer boundary of this larger region.

For corner partition 0 in the 100-partition layout:

```text
core:          360 × 310 elements
padding:       150 elements
solved region: 510 × 460 elements
```

An interior partition has padding on both sides and is larger than a corner
partition.

## Why overlap helps

If each core were solved directly with DIC displacement on its own boundary,
measurement noise and incompatible local conditions would act immediately next
to the retained result. Expanding the domain moves that artificial boundary
away from the core.

The effect is related to the decay of boundary perturbations in elliptic
problems. This is inspired by overlapping domain decomposition, but the
current workflow is not an iterative Schwarz coupling: subdomains do not
exchange updated interface values. Each one remains an independent
DIC-Dirichlet problem.

The article’s padding study selected approximately 150 elements. The exact
published BGE formula is not available, so this repository names its own
diagnostic `interface_gradient_ratio` rather than claiming formula identity.

## Unique ownership

Overlapping values are never averaged. The ownership rule is:

- each element belongs to exactly one core;
- a nodal interface is assigned deterministically to one neighbouring core;
- only the owner writes the global value.

This choice has three advantages:

1. every global location is written once;
2. stitching is independent of task order;
3. interface discontinuities remain visible instead of being hidden by an
   arbitrary blending rule.

Changing ownership or introducing weighted overlap would define a different
scientific post-processing method.

## Deterministic numbering

The domain is divided into balanced intervals on each axis. For indices
`index_x` and `index_y`:

```text
partition_id = index_x * parts_y + index_y
```

The 100-partition layout uses `parts_y=10`. Its manifest records core bounds,
solved bounds, shapes, and padding for all IDs before any solve starts.

## Resumption and memory

Partitions are independent calculation units:

- each writes six arrays atomically;
- `status.json` is written last;
- every array and manifest is fingerprinted;
- corrupted or missing output becomes pending;
- valid partitions survive interruption.

Stitching uses `numpy.memmap` for both local and global arrays. It therefore
does not require all 100 partitions or the complete global field to reside in
memory at once.

Partitioning controls per-job memory and enables job arrays. It does not reduce
the total scientific work, and smaller cores increase the relative influence
of boundaries. Partition count and padding must be treated as numerical
parameters, not merely infrastructure settings.
