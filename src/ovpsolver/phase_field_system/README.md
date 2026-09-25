# The abstract mixture

This package is the layer every mixture is built on. [`PhaseFieldSystem`](system.py) is the mixture and [`PhaseField`](phase_field/field.py) one of its phases. Between them they assemble the discrete Rayleighian $\mathcal R = \dot E + \Psi + \mathcal C$ out of what a concrete subclass declares, and derive every row of the step from it. A concrete mixture ([`ModelB`](../model_b), [`PolymerizingB`](../polymerizing_b)) declares physics and nothing else.

This is the second of four documents. The [package README](../../../README.md) runs the examples. This one states what the abstract layer does with a mixture's declarations, and what it guarantees in return. [Model B](../model_b/README.md) and then [Polymerizing Model B](../polymerizing_b/README.md) state what each concrete mixture declares, in the terms set out here. The docstrings of each file carry the reasoning.

## 1. Structure

Three kinds of object contribute to $\mathcal R$:

- **`PhaseFieldSystem`**, the mixture. It owns what no phase can: the diffuse geometry every phase lives on, and the pressure enforcing $\sum_i \phi_i = 1$.
- **`PhaseField`**, one conserved phase. It owns its volume fraction `phi`, the potential `aux_potential` its transport is paired against, the velocities that carry it with their multipliers, and any internal states.
- **`Coupling`** ([couplings/base.py](couplings/base.py)), a term belonging to no single phase, such as Flory–Huggins mixing or inter-gel locking. It may own fields and a transport law of its own.

Each is assembled from three bases, one concern apiece (§7): `Dissipative` for its terms of $\mathcal R$, `ElementOwner` for the finite elements it declares and the functions bound to them, and `ParametricSaveable` for the spec block it is read from and what it can write out (a `Coupling` is `Parametric` only).

**The energy goes up, then comes back down.** Each contributor declares its terms without reference to the others. `energy_density()` builds the mixture's free energy up the tree into one list, while `dissipation()` and `constraints()` are summed up and stop there. The assembled list then comes back down as the argument of every contributor's `energy_rate(terms)`, and through it into each transported variable's energy row. So each variable's potential is the derivative of the mixture's entire energy. That is how a Flory–Huggins term declared by a coupling reaches the phases' rows. The whole pattern is [`Dissipative.rayleighian()`](dissipative.py#L95-L106), `energy_rate(energy_density()) + dissipation() + constraints()`, called on the mixture.
- The one local read is the wetting term: a phase differentiates its own diffuse-surface terms for its potential row.
- A term that verifiably needs neither leg may short-circuit the pattern, entering at `energy_rate` directly as a power in the live velocities. That means a term generating forces only on its owner's own velocities, and the potential of no transported variable. The gel's elastic stress power is the case ([Polymerizing Model B](../polymerizing_b/README.md)), and it has to match the transport that releases it (§2).

**The Rayleighian is differentiated once, so no node writes its own residual in that sense.** The system differentiates the assembled $\mathcal R$ against every rate at once (`stationarity`). It has to, because the pressure and interphase drag couple every phase to every other, and a per-contributor derivative would drop exactly those cross terms. The only residuals a node states are:
- its transport laws, which `Transported` writes from the flux list, and which a subclass writes by hand only by overriding `transport_residual`;
- the definition of `aux_potential` (`PhaseField.aux_residual`), the one potential assembled by hand rather than computed as a derivative of the declared energy. The Korteweg stress, the biharmonic term of Cahn–Hilliard, needs this intermediate variable on a cell-constant phase. It enters the energy as the pairing `aux_potential * phi`, and from there reaches the rows like every other potential.

### The step

The driver ([solver/solver.py](../solver/solver.py)) takes each sub-step as a Lie split:

1. `irreversible_timestep(dt)` advances whatever is not the Onsager problem, such as reaction kinetics.
2. `snapshot()` sets `prev ← next`, and each lagged velocity to its last solved value.
3. **The rate solve** is Newton on `rate_residual()`: the stationarity of $\mathcal R$ in every `StaticElement`, plus the rows of the live states (`aux_residual`, and the transport of every `ENERGY_LIVE` variable).
4. **The transport step** advances every `ENERGY_LAGGING` variable by its own `Transported.solve()`, cell by cell. Keeping these variables out of the rate solve spares it their unknowns, and the step itself is exact with no solver at all: one diagonal update per variable.
5. **The verdict.** Every transported variable reports `cfl_limit()` from the velocities just solved. A step longer than that bound allows, or a failed solve, is rolled back (`checkpoint`/`restore`) and retried shorter. An accepted step then has every `positive` variable's `floor_value()` checked against `positivity_floor_throw`, and a violation stops the run.

## 2. Guarantees

What a concrete mixture inherits without writing anything, what it is refused for breaking, and what it has to get right itself.

### By construction

- **Saturation.** The system declares the pressure (DG0) and pairs it, in `compression_penalty`, with the volumetric rate of every phase's flux. So $\sum_i \phi_i = 1$ holds up to the compression rate $\Phi p / \eta_b$ whatever the phases are, and a subclass adding kinds of phase never touches it. The initial condition is saturated exactly: `wetting_equilibrium` solves with $\sum_i \phi_i = 1$ as a row of its own, and `add_phase_noise` renormalizes pointwise.
- **One transport rule.** Every term that reads a transported variable across a facet takes the triples `(selector, live velocity, lagged density)` from `Transported.upwind()`: its transport row, its energy rate, the pressure's volumetric rate and its step bound. None of the three can be chosen by a caller. So the flux leaving one cell enters its neighbour by construction, each phase is conserved exactly, and the phase sum in a cell changes by exactly the compression the pressure permits.
- **No motion, no change.** The velocity in every triple is the live one and the density the lagged one, so zero velocities change nothing.
- **A convex rate problem.** Upwind selectors read the lagged velocity, which keeps every row of the rate problem affine in the rates. The convex part of each energy is taken live and the concave part lagged.
- **Energy release matches transport.** A live variable's energy rate is $-\int \mu\ \nabla\cdot J$ on the same facet numbers as its transport row, with $\mu$ the convex-split derivative of the mixture's whole energy (`Transported.energy_rate`, `potential_of`).
- **Cross terms reach their rows.** $\mathcal R$ is differentiated once against every rate unknown (`stationarity`), and the energy handles from `ElementOwner.variable` are memoized. So a term one object writes in another's variables lands in that object's rows without naming them.
- **Flux conditions on every velocity.** Each `VelocityElement` brings its own multipliers, and `PhaseField` writes three terms for every velocity: the normal-flux pin on $\Sigma$, the membrane crossing dissipation at the inclusion surfaces, and the inclusion drag. A concrete phase cannot forget one.
- **Wetting as a natural condition.** `aux_residual` defines the potential as $-\kappa \Delta\phi + C/\chi_\varepsilon$, whose natural boundary condition at the diffuse surfaces is the contact angle. Nothing is penalized.
- **Positivity.** Every transported state is DG0 under an upwind flux. `cfl_limit` bounds the step from the flux the row actually applied, and the driver retries any step that exceeded it.
- **Weak constraints only.** No unknown carries a Dirichlet condition; every constraint is a declared multiplier or relaxation.
- **Rollback, resume, saving.** Checkpoints, time-level rotation and restart points cover every declared element. A mixture is resumable by having declared its state, and every bulk element is saveable as its own dofs.

### Checked, and refused

When the spec is read ([parameters.py](parameters.py)):
- an empty roster, or a phase missing `name`, `initial_volume_fraction`, `monomer_volume`, `surface_flux_density` or `save`;
- duplicate phase names, or initial volume fractions that do not sum to one;
- injections that are not volume-neutral: $\sum_i \nu_i q_i \neq 0$ at the inclusion surfaces, or $\sum_i \nu_i Q_i \neq 0$ on $\Sigma$.

When the mixture is built:
- phases that do not match the roster, and a `save` name nothing offers.

When the solver is built, before the first solve:
- a `Transported` whose element is `Solve.NONE`; a live one without flux pieces; one with neither flux pieces nor its own `transport_residual`; `upwind_live` on a live one;
- an `ENERGY_LAGGING` element carried by other than exactly one `Transported`, or a lagging row reading another's next value;
- a lagging row that is not affine in its own next value, or whose next value enters anywhere but a diagonal mass;
- a transported variable that is not cell-constant, or a potential that is not;
- a lagging variable whose next value the energy depends on;
- a phase declaring different numbers of velocities and fluxes;
- an `EnergyDensity` that already carries a measure;
- a surface injection read at a `flux_bc_quadrature` other than the indicator's degree;
- a surface affinity that is nonlinear in `phi`, or an initial wetting equilibrium that leaves the simplex.

During the run:
- a step longer than its bound, a failed Newton or linear solve, or a non-finite transport step is retried shorter;
- an accumulation that vanishes in some cell, or a positive variable below `positivity_floor_throw`, stops the run.

### User's responsibility

- **Convex-split.** Both parts of every `EnergyDensity` must be separately convex (jointly convex in the full state space). The unconditional energy descent of the step rests on it, and nothing can check it.
- **Lagged densities.** Every `Flux.lagged_density` must be lagged, as its name says. Only some violations are caught: a lagging row whose flux reads its own next value, or another lagging variable's.
- **Hand-written laws.** An overridden `transport_residual` owes the scheme's shape: explicit in the state, and affine in the rates if the variable is live. It also owes some route by which the work it does reaches the velocity rows. The transport step checks the first of these for lagging variables only.
- **Energy the rate solve does not carry.** Power stored in a lagged state, such as elastic stress power, is declared by overriding `energy_rate`, and has to match the transport that releases it.
- **The irreversible step** is not energy-accounted and has to stay inside the constraint set. Positivity is checked after it, at the end of the step; saturation is not re-imposed.
- **Save declarations.** `Saveable.static` is a claim the saver trusts.

## 3. `PhaseFieldSystem`

[system.py](system.py)

### What it does

- **Construction.** It sets every bulk measure to the quadrature rule the indicator $\chi_\varepsilon$ is tabulated on, and builds the `DiffuseDomain`. It then calls `make_phase_fields()` and `make_couplings()`, in that order because a coupling is about the phases, checks the phases against the roster, and checks every `save` name.
- **The pressure.** `declare_elements()` declares one element, `pressure`, a DG0 `StaticElement`. It is a multiplier conjugate to the mixture's volumetric rate and never an energy variable. DG0 against degree-2 velocities is an inf-sup stable pair.
- **Assembling $\mathcal R$.** `energy_density()` collects every contributor's terms into one list. `energy_rate(terms)`, `dissipation()` and `constraints()` sum the contributors' own, with `dissipation()` adding `compression_penalty()`, the pressure's share. `rayleighian()` adds the three.
- **The rate problem.** `stationarity(R)` differentiates $\mathcal R$ against every `StaticElement` of every owner in one pass. `rate_residual()` adds the rows stated outright: each phase's `aux_residual()`, and the transport row of every `ENERGY_LIVE` variable. `bregman_deficit()` is the dissipation the step itself adds by lagging the concave parts, reported alongside the solve.
- **Transport.** `transported()` lists every advected variable, the phases' before the couplings'. `total_phase()` is $\sum_i \phi_i$, the saturation residual, offered for saving as `system.total_phase`.
- **Initial conditions.** `set_initial_conditions()` solves `wetting_equilibrium()`, roughens it with `add_phase_noise()`, and gives each phase its `phi`; each coupling then sets its own fields.
  - The equilibrium is solved in the run's own discretization: the potential is defined term for term as `aux_residual` defines it, and the contact angle is in the run's operator.
  - So the first step is driven by the bulk energy alone, not by the difference between two discretizations of the same force.
- **Saving.** `savers()` are the mixture, the diffuse domain and the phases; a coupling publishes through a phase. The mixture files its own fields under `system` whatever its class is called.
- **The command line.** `run`, `plot` and `main` are classmethods that read a spec against the class, so a subclass inherits every entry point.

### What a subclass writes

- **`make_phase_fields()`** *(abstract)*. One `PhaseField` per roster entry, in roster order, each given `solver_parameters`, its parameters block, `diffuse_domain` and `k_B_T`. This is the only place kinds of phase are distinguished.
- **`make_couplings()`** *(abstract)*. One `Coupling` per block of the spec's `couplings`, each also given `phase_fields`. It returns an empty tuple if nothing couples the phases.
- **A parameters class**, named among the bases (`class MyMixture(PhaseFieldSystem, ParametricSaveable[MyMixtureParameters])`), and **`name()`**.

### What a subclass may override

- **`irreversible_timestep(dt)`**, the non-OVP half of the split. It is empty by default and dispatches to nothing: a mixture that has one calls whatever its phases and couplings need. It is handed the whole `dt` so it can sub-cycle at its own stability limit.
- **`entry_points()`**, to append an entry point only this mixture has, or to withhold one.
- **`declare_saveable()`**, extended as `{**super().declare_saveable(), ...}` with mixture-level diagnostics.
- **`plot_params`**, the figure styling, restating only the keys that change.

`stationarity`, `rate_residual`, `compression_penalty` and the summing methods are where §2 is made; overriding them trades those guarantees away.

## 4. `PhaseField`

[phase_field/field.py](phase_field/field.py)

### What it does

- **Elements.** `declare_elements()` declares `phi` and `aux_potential`, then each declared velocity followed by its multipliers, then the internal states.
  - `phi` is DG0, `positive` and `ENERGY_LIVE`, since the convex energy reads it at $k+1$.
  - `aux_potential` is DG0 and `ENERGY_LIVE`, and not positive.
- **Transport.** `transported()` returns `phi` as a `Transported` on `flux()`. A subclass appends its internal states.
- **Free energy.** `energy_density()` declares what every phase has: the wetting affinity on the diffuse surfaces, and the gradient penalty $\frac{\kappa}{2}|\nabla\phi|^2$ as the pairing `aux_potential * phi`.
  - This is the example of a free-energy term specified by a potential–field pairing rather than explicitly. On a cell-constant phase the Laplacian in the chemical potential needs an intermediate variable to be represented at all, so `aux_residual` defines it.
  - Differentiating the pairing against `phi` then hands that potential to the rows, exactly as every other term's derivative is handed over.
  - The pairing's value is not the penalty's, and nothing reads it.
- **Dissipation.** For every velocity and the flux piece that answers it, `dissipation()` adds two terms:
  - `surface_crossing_penalty`, the membrane resistance at the inclusion surfaces, written as a Legendre transform over its DG0 multiplier;
  - `inclusion_drag`, which holds the velocity to the inclusions' own inside them.
- **Constraints.** `constraints()` first refuses unequal numbers of velocities and fluxes. It then pins, for every velocity, the normal flux through $\Sigma$ to its flux's `boundary_flux` (`boundary_constraint`).
- **The potential row.** `aux_residual()` defines $\mu = -\kappa\Delta\phi + C/\chi_\varepsilon$ weakly, from a two-point Laplacian and the contact-angle defect $C$ of `contact_angle_defect()`. This imposes the wetting condition as the natural one. `surface_potential()` is the derivative of the declared wetting energy, read by this row and by the initial condition.
- **The energy rate.** `energy_rate(terms)` sums the release of every transported variable against `terms`, the mixture's whole energy.
- **Initial conditions and saving.** `set_initial_conditions(phi)` starts both time levels at `phi` with the lagged velocities at zero. `declare_saveable()` offers every bulk element, plus `drag_floor_indicator`.

### What a subclass writes

- **`declare_velocity_elements()`** *(abstract)*. One `VelocityElement` per independent velocity: one for a phase that moves as a whole, two for a sol/gel split. The multipliers follow automatically.
- **`flux()`** *(abstract)*. One `Flux` per velocity, in the same order, summing to the transport of the whole phase.
  - Each `lagged_density` is lagged, as its name says: `self.prev.phi` for the phase itself.
  - Each piece carries its own injection. `surface_flux()` is the smeared number flux of monomers injected at the surfaces, which each piece scales by what one monomer carries of its variable.
- **`energy_density()`**. The free-energy terms, convex-split and written in `self.variable(...)` handles, returned alongside `super().energy_density()` (the gradient-penalty pairing and the wetting term). A subclass may also change or drop those.
- **`dissipation()`**. `super().dissipation()` plus what each velocity pays, composed from the building blocks below.

### Building blocks for `dissipation()`

Each takes a velocity's lagged density and live velocity, as its flux piece holds them, so a phase with several velocities pays for each separately.

- **`darcy_dissipation(carried, v, D)`**, the Darcy drag. It is measured by the floored `drag_measure`, $\sqrt{(\chi_\varepsilon\phi)^2 + (\chi\phi)_\ast^2}$, which keeps a velocity row solvable where its phase vanishes.
- **`drag_floor_deficit(carried, v, D)`**, the same floor for a velocity with no Darcy term of its own.
- **`bulk_viscosity(carried, v, eta)`**, Brinkman screening of grid-scale modes, on the same floored measure.
- **`surface_viscosity(carried, v, length)`**, resistance to a normal velocity that varies along a surface. It costs nothing when `length` is zero.
- **`diffuse_domain.drag(density, v)`**, tangential drag against the inclusion surfaces.

### What a subclass may override

- **`declare_state_elements()` and `transported()`**, the latter through `super()`, for internal states carried by the phase with their fluxes or own laws.
- **`set_initial_conditions_on(fields, phi)`**, for those states' initial values at each time level.
- **`declare_saveable()`**, extended with diagnostics.
- **`energy_rate(terms)`**, only to add power stored in something the rate solve does not carry, such as elastic stress power, as `super().energy_rate(terms) + ...`.
- **`surface_affinity(phi)`**, which must stay linear in `phi`. The initial wetting solve is linear, and refuses anything else.

## 5. `Transported`

[transport/transported.py](transport/transported.py)

One advected variable, a phase's `phi` or any internal state, and every row that moves it. One type for all of them is what lets the solver treat transport as a loop.

### Declaration

`Transported(element, flux=(), upwind_live=False, chi_weighted=True)`:

- **`element`**, the variable's `StateElement`. Its `solve` decides where the row goes: `ENERGY_LIVE` into the rate problem, `ENERGY_LAGGING` into the transport step.
- **`flux`**, its `Flux` pieces. They are required of a live variable; a lagging one may instead override `transport_residual`.
- **`upwind_live`**, to select the upwind cell by the solved velocity rather than the lagged one.
  - It is for lagging variables only, and relaxes their step bound: a lagged selector that disagrees with the solved velocity reads the downstream cell, and the bound shortens the step to match.
  - A lagging variable that live ones are functions of has to share their selector, and so leaves it off.
- **`chi_weighted`**: whether the variable is a density of the free fluid, weighted by $\chi_\varepsilon$, or an intrinsic property of a phase, unweighted.

Its potential is always the derivative of the mixture's declared energy (`potential_of`); nothing is passed in beside it.

### Rows

- **`upwind(boundary=False)`**. The variable's flux as `(selector, live velocity, lagged density)` facet triples: the single source of every facet term (§2).
- **`transport_residual()`**. The conservation law $F(\chi\rho) + \nabla\cdot(\chi \sum_\alpha v_\alpha \rho_\alpha) = \sum_\alpha s_\alpha$, with an implicit accumulation and a flux live in the velocity and lagged in the density. The flux through $\Sigma$ is substituted by its imposed value rather than integrated. This is the one residual a subclass may write by hand.
- **`energy_rate(terms)`**. $-\int \mu\ \nabla\cdot J$ in the live velocities.
  - It integrates the live flux through $\Sigma$, so the velocity rows feel the energy an influx releases.
  - It is nothing for a lagging variable, which is refused if the energy depends on its next value.

### The transport step

- **`solve()`** advances a lagging variable exactly, with no matrix, in one update $x \leftarrow x - F(x)/\operatorname{diag}(M)$. This is exact because, once the rates are known, the row is affine in its own next value with a diagonal mass.
- **`transport_forms()`** compiles the row and establishes that exactness at build. It checks that the variable is lagging and the row affine in its own next value, and probes that the mass is diagonal to roundoff.
- **`transport_step(variables, lagging)`** collects the lagging variables at build and checks what no single row can. Each lagging element must be carried exactly once, and no row may read another's next value. That is what makes solving them one at a time exact.

### The step bound and verdict

- **`cfl_limit(cfl)`** is the largest step over which the row stays non-negative.
  - It is assembled after the solve, from the flux the row applied: content over outflow, per cell and per component, with draining sources counted as outflow.
  - It is `inf` for anything without entrywise positivity.
- **`floor_value()`** is the minimum as positivity means it: per entry for a scalar or vector, and per eigenvalue for a square tensor.

### `LieTransported`

A tensor stretched by a deforming network, $\partial_t M + \nabla\cdot(Mv) = \nabla v\ M + M \nabla v^{T}$.

- The conservation part stays a flux. `transport_residual` adds the stretching as the congruence $G M G^{T}$, $G = I + dt\ \nabla v$, which keeps a semidefinite moment semidefinite at any step.
- `cfl_limit` bounds growth rather than positivity.
- The phase that owns the moment declares the matching stress power.

## 6. `Flux`

[transport/flux.py](transport/flux.py)

One velocity's term of $J = \sum_\alpha v_\alpha \rho_\alpha$. A flux is a list of these pieces rather than fractions of one total, for two reasons. The upwind selector picks a density per velocity. And additive pieces partition a phase exactly: they sum to it identically, whatever its internal state.

- **`live_velocity`**, the rate unknown, which supplies every flux magnitude.
- **`lagged_velocity`**, the last solved value of the same rate (a `StaticElement` with `store_lagged`), which supplies only the upwind direction.
- **`lagged_density`**, the density $\rho_\alpha$ at the previous time level. The requirement is in the name because nothing can check it: a density explicit against an implicit accumulation is what makes transport positivity-preserving under a step bound.
- **`surface_flux`**, the injection at the inclusion surfaces, per unit *bulk volume*: the spec's per-area number, smeared per inclusion by the diffuse surface measure.
- **`boundary_flux`**, the injection through $\Sigma$, per unit *boundary area*: the spec's number as it is, since $\Sigma$ is a real surface of the mesh. It is also the target of the velocity's $\Sigma$ constraint.

The spec's own numbers carry a `_density` suffix, `surface_flux_density` and `boundary_flux_density`, marking a flux per unit area. Nothing converts a smeared flux back into one per unit area.

## 7. Inherited contracts

### `Dissipative`

[dissipative.py](dissipative.py)

The contributor interface. Every method defaults to an empty form, so a contributor writes only the terms it has:

- **`energy_density()`**, convex-split densities. They carry no measure; each `EnergyDensity` names its domain.
- **`energy_rate(terms)`**, $\dot E$ in the live rates, given the mixture's whole energy.
- **`dissipation()`**, $\Psi$, quadratic in the live rates. It is also the norm the excess Rayleighian is measured in.
- **`constraints()`**, $\mathcal C$, hard constraints only; a relaxed constraint is declared as a dissipation.
- **`aux_residual()`**, rows defining auxiliary unknowns. The system adds the phases' rows to the rate problem verbatim.
- **`nothing()`**, a zero form to sum onto.

One method is concrete: **`rayleighian()`** adds the energy rate of the contributor's own `energy_density()`, its `dissipation()` and its `constraints()`.

Nothing here is differentiated. Subclassing is the declaration, so an object cannot enter $\mathcal R$ merely by having a method of the right name.

An `EnergyDensity` ([energy.py](energy.py)) has three fields:
- **`convex`**, evaluated at $k+1$;
- **`concave`**, subtracted and evaluated at $k$;
- **`domain`**: `BULK` integrates against $\chi_\varepsilon\ dx$, and `DIFFUSE_SURFACE` against $d\Gamma_\varepsilon\ dx$.

Its `potential(next, prev)` is the convex-split derivative against one variable. A term may also be declared by its tangent, a potential paired with its field, where the potential is what the discrete space can hold; `PhaseField`'s gradient penalty is declared this way.

### `ElementOwner`

[fem/elements/owner.py](../fem/elements/owner.py)

Declares finite elements, and holds the functions the solver binds to them.

- **`name()`** and **`declare_elements()`** *(abstract)*.
- **`element_specs()`**, the declarations stamped with their owner and built once, so that comparing two by identity means something.
- **`element(name)`**, **`fields_at(level)`**, and the bundles `prev`, `next` and `rates`.
- **`variable(name, time_level)`**, the memoized `ufl.variable` handle that energy terms are written in.
- **`element_owners()`**, the owners the solver walks. For the system this is itself, its phases and its couplings.
- **Time levels**: `snapshot()`, `lag_rates()`, `zero_lagged_rates()` and `lag_to_live()`. **Rollback**: `checkpoint()` and `restore(keep_rates=...)`.
- **Resuming**: `resume_specs()` and `resume_from(read)`.

The element types are declared in [fem/elements/spec.py](../fem/elements/spec.py):

- **`StaticElement`**, a rate or multiplier. It is one function in `rates`; `store_lagged` also keeps its last solved value in `prev`.
- **`VelocityElement`**, a `StaticElement` that carries a phase. It is lagged by default, and names its two multipliers:
  - `Lambda`, P1 on $\Sigma$;
  - `lambda_`, DG0 in the bulk, and only where there are inclusions.
- **`StateElement`**, a time-evolved field, with `prev` and `next`. Its `solve` defaults to `ENERGY_LAGGING`; a state the energy reads at $k+1$ is declared `ENERGY_LIVE`.
- **`Solve`**: `ENERGY_LIVE` for an unknown of the rate problem, `ENERGY_LAGGING` for the transport step, and `NONE` for a coefficient or diagnostic.
- **`positive`**, which puts a variable under the step bound and the floor check.
- **`ElementDomain`**: `BULK` or `SURFACE` ($\Sigma$) for an element; `DIFFUSE_SURFACE` is for energy terms only.

The solver builds each space once, binds the functions onto the bundles, and binds each test function onto the attribute its element names. The rate problem's unknowns are every `StaticElement` plus the `ENERGY_LIVE` states, split for PETSc into its `static` and `state` halves by element type. The transport step's unknowns are the `ENERGY_LAGGING` states.

### `ParametricSaveable`

[fem/saveable.py](../fem/saveable.py)

- **From `Parametric[P]`** ([parametric/parameters.py](../parametric/parameters.py)): naming a parameters class among the bases binds the two, as `cls.Parameters` and `P.parametric`. So the class defines its spec block's schema, the block builds the object (`parameters.build()`), and `self.parameters` is the block.
- **`declare_saveable()`**, a mapping of name to `Saveable(element, value=None, static=False)`.
  - It is empty by default.
  - An element owner starts from `own_elements(self)`, which gives every bulk element as its own dofs, and adds derived entries, each in a declared space.
- **`requested()`** and **`check_saveable_names()`**: the block's `save` list, and the build-time check that every name in it is offered.
- **`saved(name)`**, the current value and its declaration, for the saver.
- **`save_qualifier()`** and **`qualified(owner, name)`**, which file each field as `<qualifier>.<name>`. One function is shared by writing, resuming and reading.

A `Coupling` is `Parametric` without `Saveable`: no spec block names its fields, and whatever it has worth saving it publishes through a phase.

## 8. A new mixture, in outline

```python
import numpy as np

from ovpsolver.fem.elements import StateElement, VelocityElement
from ovpsolver.fem.elements.dg0 import DISCONTINUOUS_LAGRANGE
from ovpsolver.fem.saveable import ParametricSaveable
from ovpsolver.parametric import Nonnegative, ParametersList
from ovpsolver.phase_field_system import PhaseFieldSystem
from ovpsolver.phase_field_system.energy import EnergyDensity
from ovpsolver.phase_field_system.parameters import PhaseFieldSystemParameters
from ovpsolver.phase_field_system.phase_field import PhaseField, PhaseFieldParameters
from ovpsolver.phase_field_system.transport import Flux, Transported


class MyPhaseFieldParameters(PhaseFieldParameters):
    # One phase's spec block, adding a kinetic constant. The irreversible step
    # reads it pointwise rather than in a form, so it is a plain float.
    decay_rate: float = Nonnegative(1.0, coefficient=False)


class MyMixtureParameters(PhaseFieldSystemParameters):
    phase_fields = ParametersList(MyPhaseFieldParameters())


class MyPhaseField(PhaseField, ParametricSaveable[MyPhaseFieldParameters]):
    _v_test: "Expr"
    _c_test: "Expr"

    def declare_velocity_elements(self):
        dim = self.solver_parameters.dolfinx_mesh.geometry.dim
        return [VelocityElement("v", "_v_test", degree=2, shape=(dim,))]

    def declare_state_elements(self):
        # A concentration carried by the phase: cell-constant, non-negative, and
        # lagging by default, since the energy does not read it at k+1.
        return [
            StateElement(
                "c", "_c_test", family=DISCONTINUOUS_LAGRANGE, degree=0, positive=True
            )
        ]

    def flux(self):
        return [
            Flux(
                live_velocity=self.rates.v,
                lagged_velocity=self.prev.v,
                lagged_density=self.prev.phi,
            )
        ]

    def transported(self):
        # c rides the phase's velocity. Being positive puts it under the step
        # bound and the floor check; being lagging, the transport step solves it.
        c = Transported(
            self.element("c"),
            flux=[
                Flux(
                    live_velocity=self.rates.v,
                    lagged_velocity=self.prev.v,
                    lagged_density=self.prev.c,
                )
            ],
        )
        return [*super().transported(), c]

    def set_initial_conditions_on(self, fields, phi):
        super().set_initial_conditions_on(fields, phi)
        # Proportional to the phase, vanishing where it does, as a
        # concentration the phase carries should.
        fields.c.x.array[:] = 0.1 * fields.phi.x.array

    def irreversible_timestep(self, dt):
        # The source dc/dt = -k c, cell by cell: a DG0 dof is its cell's value.
        # Integrated exactly, so c stays positive at any dt.
        self.next.c.x.array[:] *= np.exp(-self.parameters.decay_rate * dt)

    def energy_density(self):
        phi = self.variable("phi", time_level="next")
        return [EnergyDensity(convex=...), *super().energy_density()]

    def dissipation(self):
        return super().dissipation() + self.darcy_dissipation(
            self.prev.phi, self.rates.v, 1.0
        )


class MyMixture(PhaseFieldSystem, ParametricSaveable[MyMixtureParameters]):
    def name(self):
        return "my_mixture"

    def make_phase_fields(self):
        return tuple(
            MyPhaseField(self.solver_parameters, phase, self.diffuse_domain, self.k_B_T)
            for phase in self.parameters.phase_fields
        )

    def make_couplings(self):
        return ()

    def irreversible_timestep(self, dt):
        # The base dispatches to nothing: a mixture calls what its phases need.
        for field in self.phase_fields:
            field.irreversible_timestep(dt)


if __name__ == "__main__":
    MyMixture.main()
```

The spec is read against these classes, so the new constant appears under each phase:

```yaml
solver:
  ...
  irreversible_step:
    skip: false  # or the decay never runs; Model B's specs skip the step
system:
  ...
  phase_fields:
    - name: water
      ...
      decay_rate: 2.0
    - name: lipid
      ...
      decay_rate: 0.5
```

**Declared state.** Everything declared is reached through the owner's bundles:
- a state `c` as `self.prev.c` and `self.next.c`:
  - `self.next.c` is the value being advanced, first by the irreversible step and then by the solve;
  - `self.prev.c` is its value at the start of the OVP half-step, where `snapshot()` copies `next` into it, and is what lagged terms read;
- a rate `v` as `self.rates.v`, with its last solved value as `self.prev.v` if it keeps one (every `VelocityElement` does);
- an energy term instead uses the handles `self.variable("c", time_level=...)`, so that it can be differentiated.

For every declared element the driver guarantees:
- **rollback.** Each owner's `prev` and `next` are checkpointed before a sub-step, and restored when an attempt is rejected, including whatever the irreversible step did. So a retry starts from exactly the pre-step state; the rates are kept as the retry's starting guess until the selector freezes.
- **resume.** The state is written to restart points and read back by `--resume`.
- **saving.** A bulk element can be named in its block's `save` list.

State kept anywhere else, as a plain attribute or array, has none of this: a rejected step would not undo it, and a resumed run would not restore it.

[Model B](../model_b/README.md) is this outline written out in full, without the state; [Polymerizing Model B](../polymerizing_b/README.md) declares several.
