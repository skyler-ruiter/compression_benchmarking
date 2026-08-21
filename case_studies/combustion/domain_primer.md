# Combustion domain primer for the S3D pilot

Checked 2026-08-21.  This document explains enough domain structure to design the
compression study; it is not a substitute for validation by a combustion expert.

## 1. What this simulation represents

The local dataset describes a statistically planar, lean premixed methane-air
flame computed with S3D.  “Premixed” means fuel and oxidizer are mixed before
reaching the flame.  “Statistically planar” does not mean the instantaneous flame
is flat: turbulence wrinkles a thin reacting layer separating colder reactants
from hotter products.  The present fields are one 500 x 500 x 500 snapshot:

- `CH4` and `O2`: reactants consumed across the flame;
- `CO2` and `H2O`: major products formed across the flame;
- `CO`: an intermediate/minor product with a narrower spatial profile;
- `N2`: mostly inert diluent, nearly constant here;
- `TEMP`: thermal state, approximately 300 K on the unburned side and 1840 K on
  the burned side in this snapshot;
- `PRES`: almost constant near one atmosphere;
- `U`, `V`, `W`: velocity components carrying the turbulent flow structure.

The values and metadata are consistent with a published S3D configuration of a
stationary planar lean methane-air flame at equivalence ratio 0.7, 300 K, and one
atmosphere.  More decisively, Rai et al. describe the same six-species, eleven-
variable, 500^3, 400-snapshot S3D dataset.  This makes the identification very
strong, although the SDRBench archive should still be traced to an explicit
provenance record before publication.

## 2. Reaction progress and the flame surface

A reaction progress variable converts thermochemical state into a scalar that is
approximately zero in fresh reactants and one in fully burned products.  A common
species-based definition is

```text
c = (Y - Y_reactants) / (Y_products - Y_reactants)
```

where `Y` is a selected species mass fraction.  With a consumed reactant such as
O2, the denominator is negative; the same normalization still maps reactants to
zero and products to one.  Product combinations such as CO2 + H2O + CO are also
used in combustion research.

A flame surface is then an isosurface `c = c*`.  The value `c* = 0.5` is a useful
exploratory convention, but not universal.  Published methane-air S3D work has,
for example, used an O2-based progress variable with `c* = 0.65` because it
coincided with maximum heat release for that configuration.  We lack heat-release
or reaction-rate data in this archive, so we cannot establish the physically
preferred isovalue from the snapshot alone.

For this dataset, an O2-based `c` is the best first proxy because O2 is monotonic
across the observed front and avoids CH4's near-zero burned-side numerical noise.
It should be cross-checked against normalized temperature, CO2, and H2O.  End
states should ultimately come from the simulation setup or a one-dimensional
laminar reference flame, not from per-snapshot extrema.

## 3. The flame brush and distance coordinates

The instantaneous flame surface is thin and wrinkled.  Over time or across a
turbulent ensemble, its range of positions forms the flame brush.  The Sarasota
report also uses “within the flame brush” operationally for the region where
accurate level sets and distance functions matter.

There are two useful but distinct ROI constructions:

1. **Progress interval:** mark cells with `c_low <= c <= c_high`.  This captures
   the thermochemical transition but does not give physical distance.
2. **Signed-distance band:** extract a validated `c = c*` surface, compute signed
   distance to it, and protect cells within a chosen distance.  This directly
   supports conditional statistics upstream and downstream of the flame.

The second construction matches the Sarasota analysis description more closely.
Its width must be defined in grid cells until physical coordinates/grid spacing
are recovered.

## 4. Quantities a combustion case study should preserve

### Minimal credible set

- progress-variable error and selected species/temperature error;
- displacement of the `c = c*` isosurface;
- flame surface area and enclosed reactant/product volume;
- gradient magnitude `|grad c|`, which is related to flame surface density;
- conditional means and PDFs of temperature, species, and velocity versus signed
  distance or progress variable;
- ROI-specific errors inside and outside the flame band.

### Stronger set

- surface normal `n = grad(c) / |grad(c)|` and normal-angle error;
- mean and Gaussian curvature distributions;
- connected components, reactant/product pockets, and tunnel/pinch-off events;
- velocity gradients, vorticity, strain rate, and their alignment with the flame
  normal;
- merge-tree or Morse--Smale preservation, if topology tooling and a defensible
  scalar field are available.

These derived quantities amplify spatially incoherent error.  A compressor can
satisfy a pointwise or global NRMSE target while moving a thin surface, changing
its area, or corrupting gradients.  This is why PSNR alone would make the case
study scientifically weak.

## 5. What composition could buy us

The spatial structures of O2, CH4, CO2, H2O, and temperature are highly redundant
in the first snapshot.  That suggests one detector or mask might be amortized
across multiple fields.  A candidate FZGM workflow is:

```text
O2 (or supplied progress variable)
    -> block-local flame detector / mask
    -> stringent quantization inside + relaxed quantization outside
    -> encode
```

The scientific comparison must include the mask/detector cost and compare at
matched compressed size or matched QoI.  The strongest result would show that
composition reaches a point unavailable to uniform GPU compressors—not merely
that automatic fusion removes FZGM's own intermediate-memory traffic.

## 6. Questions requiring domain confirmation

1. Is this archive the stationary planar methane-air case described in the
   corresponding S3D literature, and what are its grid spacing and timestep units?
2. Which progress-variable definition and end-state values were used by the
   original investigators?
3. Which isovalue corresponds to peak heat release, and what spatial width defines
   the flame brush?
4. Are the saved six species sufficient for the intended analysis, or are missing
   radicals/reaction rates essential?
5. Which surface, gradient, conditional-statistics, and topology tolerances would
   make reconstructed data scientifically usable?
6. Are all five archived snapshots physically distinct enough for temporal
   validation, and where do they sit in the statistically stationary period?

## Primary starting points

- J. H. Chen et al., S3D/direct numerical simulation work on turbulent combustion.
- R. A. C. Griffiths et al., [*Three-dimensional topology of turbulent premixed
  flame interaction*](https://doi.org/10.1016/j.proci.2014.08.003): progress
  variables, flame surfaces, and critical-point topology.
- S. Trivedi et al., [*Topology of pocket formation in turbulent premixed
  flames*](https://doi.org/10.1016/j.proci.2018.06.197): progress-variable
  gradients, surface density, normals, curvature, and pocket topology.
- H. Kolla and collaborators' statistically stationary planar lean methane-air
  S3D studies, available through the DOE OSTI record
  [SAND2017-7035J](https://www.osti.gov/biblio/1372305).
- P. Rai et al., [*Randomized Functional Sparse Tucker Tensor for Compression
  and Fast Visualization of Scientific Data*](https://arxiv.org/abs/1907.05884):
  prior compression work on this same 500^3, eleven-variable, 400-snapshot S3D
  dataset, including spatial-temporal tensor variants.
- Cappello et al., [Sarasota application requirements
  report](https://arxiv.org/html/2503.20031v1), Section II-B.
