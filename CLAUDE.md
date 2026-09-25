# OVPSolver

This mixed Python-C++ package contains a mixed-finite element solver for multi-phase systems which can be described by an Onsager variational principle (see e.g. Doi, https://doi.org/10.1088/0953-8984/23/28/284118). The main primitive of the system is therefore the Rayleighian,

R = \dot E + \Psi + C

over rate-variables (velocities, pressure, and other Lagrange multipliers). The motion of each phase \phi_i, which should satisfy the individual conservation laws d/dt[\int_\Omega \phi_i] = 0 and overall saturation property \sum_i \phi_i = 1 pointwise in space, are given by the transport equations 

\partial_t \phi_i + \div J_i = 0

in the bulk, with various user-specified conditions on the flux J_i at the surface.
The fluxes J_i are determined by, among other user-specified factors, the stationarity condition for the rate variables 

\delta R / \delta (rates) = 0

which is physically a force-balance law. This is the primary variational form comprising the finite element method. Together, the above three equations constitute an Onsager variational principle. We call every timestep of the above an "OVP step".

The primary aim of this package is to provide a declare-and-forget means of describing Cahn-Hilliard type systems, where the pieces specified by the user are used to construct a scheme which silently and automatically satisfies:

- positivity of the phases phi_i. This is accomplished through the representation of the primary variables, the phases phi_i, as discontinuous Galerkin (DG) elements. We use DG0, which means that (viewed through just the phase-fields) this is a finite-volume scheme in disguise. We use upwinded transport along with a CFL condition to ensure that the scheme either always produces physically valid (nonnegative) data or breaks. The scheme has less work to do vis. CFL sub-stepping, and will generally be more consistent in its timesteps, when positivity is energetically preferred, for example through the use of log-barriers.

- saturation and conservation of the phases phi_i. This is accomplished through the use of a pressure variable, rather than imposing a particular mobility. This means, simultaneously: (i) the user can state an arbitrary mobility, (ii) fluxes satisfy the weaker constraint \sum_i div J_i = 0 rather than the classical \sum_i J_i = 0, and (iii) an arbitrary number of phases is possible without additional theoretical development. Therefore, the scheme realizes (for Cahn-Hilliard systems) the pressure-constrained model of E et al (https://link.aps.org/doi/10.1103/PhysRevE.55.R3844, https://doi.org/10.1063/1.474153). Pressure is also DG0; inf-sup (LBB) stability is attained by choosing the velocity elements (here P2) accordingly.

- every "OVP step" satisfies an unconditional energy descent property. That is, independent of the timestep or any other parameters, E_live - E_lagged \le 0. This is central to the physicality of the scheme's data. This is achieved by demanding that the user specify any energy density in a convex-split form. Not all energy densities admit this. However, the clever introduction of auxiliary state variables can enable convex-splitting, which is up to the user to divine. Then, the scheme implements Eyre's (http://link.springer.com/10.1557/PROC-529-39) method: the stationarity condition above is always a convex(-concave, with Lagrange multipliers) problem for unknown rates, with coefficients of the problem changing depending upon the concave part. In other words, the convex part of the energy density is live; the concave part is lagged. This is supplemented by the universal use of semi-implicit timestepping schemes of the form 

(phi_live - phi_lagged) / Delta t + div (density_lagged * velocity_live) = 0

which is affine in the unknown velocity_live. This corresponds to the well-known scheme of "lagged mobilities" for Cahn-Hilliard type equations.
An important structural property enforced throughout the scheme is therefore that phi_live is _always_ transported the same way, whether in evaluations of the convex part of the free energy or in transportation (which are distinct parts of the residual arising from distinct test functions; in the former, the velocity's test-fn and in the latter, phi's test-fn). A third example is in the saturation constraint. Any inconsistency in transport (e.g. in these three examples) breaks unconditional energy descent and the constraints. It is easy to accidentally break this, e.g. by upwinding the operator 

div (density * velocity)

in one place but not in another. To this end, the codebase centralizes the definitions of such operators to their element files, e.g. @src/ovpsolver/fem/elements/dg0.py. Lastly, importantly, the timestepping scheme as above satisfies the "no-motion" property, which is that zero-velocities always produce zero changes across the step. This is critical to the energy descent; an easy to miss breakage of this property is the use of fully-lagged (explicit) fluxes. A Flux/Transported API which decomposes the pieces of the flux is provided to ensure this is met; arbitrary user-defined residuals can of course break it.

These constitute the core guarantees of the scheme.

Beyond these basic guarantees, the package provides the following somewhat novel features. The primary aim of this feature set is to support an extension of Model B called "Polymerizing-Model B." They are:

- a Lie-split "irreversible" step. Termed so as it is simply the dynamics 

\partial_t (state) + (ovp_step) = ("irreversible" non-OVP step)

which is not governed by the OVP, i.e., energetically unaccounted for. (Therefore the above energy-descent guarantee is only for each OVP "substep".) The main use-case is in chemical reactions. Chemistry generally acts pointwise in space so this split is particularly convenient. It can include spatial updates; polyB does not require this and is the user's discretion. The actual implementation of the step is also dramatically simplified by the unified use of DG0 elements. Every DOF of the state is simply the cell-average value of the approximation solution, hence can be manipulated pointwise trivially without requiring any explicit back-projection. (PolyB uses precisely this property to implement via explicit timestepping a polymerization reaction, which itself requires a nontrivial finite-volume scheme in an auxiliary coordinate to resolve.) In general, this is where arbitrary "effectful" operations live, ranging from full variational solves to simple manipulation of the state as numpy arrays. It is implicitly assumed that the step respects the constraints imposed during the OVP step; leaving the constraint set during this step results in undefined behavior.

- first-class support for auxiliary state variables alongside the phase-fields, which are also typically DG0 elements. (These are used in PolyB to represent internal chemical and mechanical state variables, such as the extent of polymerization and stress of a gel phase.) A user declares the flux transporting these variables, and a "positivity" flag re-uses the same upwinding/CFL method to retain positivity of these fields. Positivity in the sense of LMI, i.e. PSD-preserving transport, of tensorial fields is also supported under motion by stretching terms, @src/ovpsolver/phase_field_system/transport/transported.py:730-799 (we call such fields, such as stresses and strain moments, "LieTransported" elements). Examples in PolyB include concentrations, shear moduli, and strain moments. The evolution of these state variables must also be Lie-split per the above. One correct pattern is, e.g. for a state variable of the form

\partial_t (state) + (transport) = (reaction/source)

is

(a) \partial_t (state) + (transport) = 0 during the OVP substep, in which (transport) is either implied by kinematic slaving to phase-fields or more generally given by velocities deduced from the OVP, as in the above semi-implicit transport law
(b) \partial_t (state) = (reaction/source) during the irreversible substep. 

This pattern is evident in PolyB. As a detail, the package is compiled with `nanobind` which in this case is used to defined accelerate C++ functions (which can leverage OpenMP on numpy data) which implement the reaction step (see e.g. @src/burgers.cpp).

- native support for various regularizations which improve the health of Newton during the OVP-step. These are to help ensure well-posedness of the rate solve. In concert with positivity preservation, these collectively promote solver behavior which merely gracefully degrades in accuracy with changing geometry and dynamics, as opposed to diverging or exhibiting poor Newton convergence.

- the representation of complex geometries on a relatively simple fixed mesh through the use of the diffuse-domain method (DDM) (https://pmc.ncbi.nlm.nih.gov/articles/PMC3097555/). Essentially the indicator of the domain is tanh(r/eps) where r is a signed distance function to the boundary. This indicator is respected first-class throughout all variational forms. It is kept in exact analytical form by demanding that the user construct signed distance functions (SDFs) in UFL (unified form language). This simultaneously facilitates autodiff and proper point evaluation and quadrature of the functions. These can in turn be generated from e.g. constructive solid geometry (CSG) e.g. from CAD.

- first-class support for Korteweg (interfacial) stresses. This is the (-kappa * Laplace) contribution to the chemical potential making CH fourth-order. This is accomplished as is common in CH schemes by defining an auxiliary chemical-potential like (though we emphasize that this is not the full chemical potential) variable, now in DG0 by using a finite-volume Laplacian. The interfacial penalty kappa |grad phi|^2 is supported with separate constants for each phase. When kappa is positive, this leads to the variational boundary condition as described next.

- wetting and flux conditions for phase-fields (these are the two BC's required for a fourth-order system, grad phi . n and J . n) against the complex geometry defined by the DDM. In phase separation, wetting is the additional symmetry-breaking mechanism beyond phase noise.

In PolyB, the above features are exercised together to resolve wetting against a relative complex configuration of disks, representing biological cells, with no changes to the mesh.

- native saving, loading, analysis, and plotting capabilities. These make efforts to save data within the solver's state as faithfully as possible to disk. These include e.g. cell average values of DG0 elements along with the mesh data. They facilitate: (i) plotting and rendering simulation data (ii) quantifications along the run, including diagnostics such as constraint satisfaction and "indicators of solve health", and (iii) archival for reproducibility purposes. Resuming a run is supported.

- finally, and perhaps most importantly for research purposes, a fully declarative format for dictating what we want out of a run. This includes all physical and solver parameters. The spec (with examples e.g. @examples/tips.yaml) is in YAML format. The tree structure of the spec itself is self-generating. Meaning, users implement new physics by subclassing key components -- at a minimum PhaseFieldSystem @src/ovpsolver/phase_field_system/system.py, see for example of a subclass ModelB @src/ovpsolver/model_b/system.py -- and declaring any new or changed parameters (see for example @src/ovpsolver/model_b/phase_field/parameters.py). These parameters determine the parser for the spec automatically.

The two primary examples using the solver are bundled with the package:
- "Model B", which implements the classical Cahn-Hilliard multi phase system with a Flory-Huggins free energy density of mixing
- "Polymerizing Model B," which is the extension of Model B to phases which polymerize, splitting into sol and gel, arresting patterns at finite lengths while thermodynamic forces continue to act upon the system away from equilibrium. This is used as a simple model of biological self-assembly of an extracellular matrix.

The structure of solver package makes the realization of these models essentially a declaration of the physics.

# Code structure.

From the frontend (user-side) to the backend, it goes:
- entrypoint in @src/ovpsolver/phase_field_system/cli.py (e.g. `run`, @src/ovpsolver/phase_field_system/run.py) calls the relevant `PhaseFieldSystem` @src/ovpsolver/phase_field_system/system.py on a user-specified spec,
- a YAML file e.g. @examples/tips.yaml, which specifies atomically (i) the domain geometry, (ii) solver parameters, and (iii) the physics of the run, with the latter specified by:
- each subclass of `PhaseFieldSystem`, e.g. `ModelB` @src/ovpsolver/model_b/system.py, which by declaring parameters actually sets the format of (the physics section of) the spec.
- these parameters in this sense _define_ the structure of the spec, so the spec schema is self-generating: e.g. @src/ovpsolver/phase_field_system/phase_field/parameters.py extended by @src/ovpsolver/model_b/parameters.py 
- then, the core solver logic, including construction of the problems and timestepping, is contained in @src/ovpsolver/solver/solver.py 
- this in turn possess and calls various methods of the "OVP state tree." Each node is generally a product-type of `Parametric`, `Dissipative`, and `ElementOwner`.
- this goes from the abstract (`PhaseFieldSystem`, @src/ovpsolver/phase_field_system/system.py, handling generic things e.g. pressure-enforced mixture saturation) to specific (e.g. `ModelB`, @src/ovpsolver/model_b/system.py) which build own contributions to the Rayleighian.
- the next level of the tree generally consists of individual `PhaseFields` (abstract, @src/ovpsolver/phase_field_system/phase_field/field.py, to specific e.g. `CHPhaseField`, @src/ovpsolver/model_b/phase_field/field.py), but the exact structure depends upon the declared parametric structure (e.g. `PolymerizingB`, which splits phase-fields into sol and polymerizing-type, @src/ovpsolver/polymerizing_b/parameters.py).

# Coding conventions and standards 

- prefer literally named, public functions over poetically named, private helpers. Along these lines, heavily concentrate and deduplicate utilities and helpers from local specialized forms to global well-abstracted forms located in shallow-filetree-depth utils-type files.
- collapse the call tree when possible. If one function is only ever called by one other without iteration or recursion, inline the former unless the result is too indigestible.
- imports occurring within functions are generally only to be used if an import cycle would necessitate too large or undesirable of a restructuring. Along these lines, the pattern: "import XXX.dg0 as dgo; dg0.foo()" is preferred over "from XXX.dg0 import dg0_foo()".
- in function docstrings, give proper justification for the implementation of the function which cannot be gleaned from its name alone. However, do not "react to the past": document common misconceptions, counterfactuals, or avoided pitfalls, but do not reference or state a difference with respect to a prior implementation unless it falls in one of the former categories.
- I personally prefer to split a class (especially a subclass)'s methods by comment-line into "overrides," "public methods," and "private helpers." The last conventionally has leading underscores. Along these lines, prefer public over private methods and functions in general. Every function / method should satisfy a reasonable standard of "justifying and documenting its existence"; opaque, numerous private helpers create spaghetti code. This also leads to an overall preference to minimality, subject to the task.
- dependencies should go in environment.yml, and in pyproject.toml only if really necessary.
- be careful with the usage of abstraction patterns when constructing UFL forms. Mathematically and programmatically they are evocative; yet they can easily result in poorer performing numerical expressions (e.g. those implying a cancellation).
- the library's structure prefers to consist of a tight core supplemented by a flat, collection-of-applications hierarchy that use it. For example, the CLI entry point of PhaseFieldSystem, @src/ovpsolver/phase_field_system/system.py.

# Outstanding issues and TODOs