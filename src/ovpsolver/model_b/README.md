# Model B

Multi-component Cahn–Hilliard: phases that demix by flowing, and nothing else. This is the third document: read it after the [abstract structure](../phase_field_system/README.md), whose methods it fills in. Everything not listed here is the abstract layer's: saturation, transport, the initial condition, the step and its guarantees.

## `ModelB`

[system.py](system.py)

- **`make_phase_fields()`** builds one `CHPhaseField` per roster entry.
- **`make_couplings()`** builds one `CHFloryHugginsCoupling`, from the spec's `couplings.flory_huggins` block.
- **`name()`** is `model_b`. Nothing else is overridden, and there is no irreversible step.

Its parameters ([parameters.py](parameters.py)) narrow the roster to `CHPhaseFieldParameters` and add the Flory–Huggins block.

## `CHPhaseField`

[phase_field/field.py](phase_field/field.py)

A phase that moves as a whole on one velocity.

- **`declare_velocity_elements()`**: one P2 vector velocity, `v`.
- **`flux()`**: the whole phase on `v`, with lagged density `prev.phi`. One monomer volume arrives with each injected monomer, in both injections:
  - at the inclusion surfaces, as `surface_flux`, per unit bulk volume;
  - through $\Sigma$, as `boundary_flux` from the parameter `boundary_flux_density`, per unit boundary area.
- **`energy_density()`**: two convex bulk terms on top of the base terms:
  - the ideal mixing entropy $k_BT\ (\phi/N)\log\phi$, with the logarithm regularized below `log_reg_delta`;
  - the bulk affinity $k_BT\ \omega\ \phi$.
- **`dissipation()`**: on top of the base terms (crossing penalty and inclusion drag), four terms, each against `prev.phi`:
  - the Darcy drag with diffusivity $D$;
  - the tangential surface drag;
  - Brinkman screening, with $\eta = \ell_\eta^2/D$;
  - the surface viscosity.

Its parameters ([phase_field/parameters.py](phase_field/parameters.py)) are `diffusivity`, `monomer_polymerization_degree`, `bulk_affinity`, `surface_drag`, the two screening lengths, and the two imposed flux densities.

## `CHFloryHugginsCoupling`

[couplings/flory_huggins.py](couplings/flory_huggins.py)

It supplies the two methods the abstract [`FloryHugginsCoupling`](../phase_field_system/couplings/flory_huggins.py) asks for:

- **`hessian()`**: the matrix of `chi`, one coefficient per unordered pair of phases.
- **`slots(time_level)`**: each phase's `phi` handle.

The base splits the Hessian convex–concave once, before the mesh exists. It declares $\frac{k_BT}{2}\ y^T H y$ as a single `EnergyDensity` written in the phases' handles, so its potentials reach the phases' rows through `energy_rate(terms)` without the coupling naming them. It owns no fields and declares no dissipation.
