# Polymerizing Model B

Model B with phases whose monomers crosslink into load-bearing gel networks. Each such phase splits into sol and gel, carries its degree-of-polymerization distribution as internal state, and reacts in the irreversible step. This is the last document: read it after [Model B](../model_b/README.md), whose `CHPhaseField` it reuses for the phases that do not react.

## `PolymerizingB`

[system.py](system.py)

- **`make_phase_fields()`** builds Model B's `CHPhaseField` for each `sol` entry, then a `PolymerizingPhaseField` for each `polymerizing` entry.
- **`make_couplings()`** builds `PolymerizingFloryHugginsCoupling`, `GelEntanglements` and `Drag`.
- **`irreversible_timestep(dt)`** sub-cycles the chemistry at the smallest of the gels' own bounds (`gelation_max_dt`). Within each sub-step every coupling steps before any gel, since locking reads both networks' moduli.

Its parameters ([parameters.py](parameters.py)) turn the roster into a block of two typed lists, `sol` and `polymerizing`: which list a phase is in is what says whether it crosslinks. They also add the three couplings' blocks.

## `PolymerizingPhaseField`

[phase_field/field.py](phase_field/field.py)

- **`declare_velocity_elements()`**: a sol velocity `v_s` and a gel velocity `v_g`, each with its own multipliers.
- **`declare_state_elements()`**, all DG0 and `positive`:
  - live, because the entropy reads them at $k+1$: the crosslinkable-site concentration `c_x`, and the moments `u_bar` and `u_trace` of the generating function;
  - lagging: the generating function `u` itself (one value per bin of the gelation coordinate $z$), `gel_shear_modulus`, and `gel_strain_moment` (symmetric, positive in the semidefinite sense).
- **`flux()`**: the sol $\phi_s = \nu m_1$ on `v_s` and the gel $\phi - \phi_s$ on `v_g`. The two partition `phi` exactly, and only the sol is injected.
- **`transported()`** adds, beside `phi`:
  - each site-counting state, split between the two velocities the same way, with the gel carrying the state's value at $z = 1$;
  - the elastic state, riding `v_g` alone: `gel_shear_modulus` as a `Transported` with a live selector, and `gel_strain_moment` as a `LieTransported` stretched by $\nabla v_g$.
- **`energy_density()`** declares two terms on top of the base terms:
  - the mixing entropy in the moments, convex-split;
  - the affinity, in the reacted-site fraction.
- **`energy_rate(terms)`** adds the elastic stress power directly, in place of declaring an elastic energy.
  - The energy cannot be declared. The strain energy in full, $\frac12\int(\operatorname{tr}{\sf m} - d\vartheta - 2\ell)$, needs the volumetric moment $\ell$, which the package does not evolve. Its power needs only the modulus and the strain moment: $\ell$'s rate law contributes exactly the $-\vartheta\,\nabla\cdot v_g$ in $({\sf m} - \vartheta I):\nabla v_g$.
  - Entering at `energy_rate` short-circuits the abstract pattern, in which energy is declared, passed up the tree and handed back down. That is admissible for a term that verifiably needs neither leg: this one generates forces only on the phase's own gel velocity, and is the potential of no transported variable. Such a term may be written directly as a power in the live velocities.
- **`dissipation()`** adds, for each velocity, what it pays:
  - the sol pays the Darcy drag; the gel, which has none, takes the floor deficit instead;
  - each pays surface drag, Brinkman screening and surface viscosity;
  - the gel also pays drag on $\Sigma$.
- **`irreversible_timestep(dt, ...)`** is the crosslinking chemistry, after which the moments are re-synced from `u`:
  - an exact Riccati step for `c_x`;
  - forward Euler for the elastic birth;
  - one SSP-RK3 MUSCL step of the Burgers equation for `u` in $z$.
- **`set_initial_conditions_on()`** starts all monomer and unreacted. **`declare_saveable()`** adds `phi_sol`, `phi_gel`, `reaction_extent` and the gel stresses.

## Couplings

[couplings/](couplings)

- **`PolymerizingFloryHugginsCoupling`** is Model B's mixing on a larger state space. `slots()` gives each gel two slots, `phi` and `c_x`, and `hessian()` builds the four coefficients `chi_00` to `chi_11` of each pair into a Hessian that is constant there. So the base's fixed convex split still applies.
- **`GelEntanglements`** owns fields. For each locked pair of networks it declares a locking modulus and two moments, as lagging `StateElement`s (`declare_elements`).
  - They are carried by the first network's velocity and fed by the pair's relative motion (`LockingTransported`, which overrides `transport_residual`).
  - It declares no energy, only the locking stress power in `energy_rate`: the same short-circuit, on the pair's two gel velocities, since its moments are lagged and differentiated by nothing.
  - The locking is born in `irreversible_timestep`.
- **`Drag`** is a dissipation only: each network's friction against every phase moving past it, weighted by both densities.
