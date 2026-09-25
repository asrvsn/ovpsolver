#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

// Periodic finite-difference viscous Burgers. The solver does not use these;
// tests/test_burgers.py calls them to check that the extension builds and loads.

double burgers_flux(double u) {
    return 0.5 * u * u;
}

std::vector<double> burgers_rhs(const std::vector<double> &u, double dx,
                                double viscosity) {
    if (u.size() < 3) {
        throw std::invalid_argument("burgers_rhs requires at least 3 values");
    }
    if (dx <= 0.0) {
        throw std::invalid_argument("dx must be positive");
    }
    if (viscosity < 0.0) {
        throw std::invalid_argument("viscosity must be non-negative");
    }

    const std::size_t n = u.size();
    std::vector<double> rhs(n);
    const double inv_2dx = 0.5 / dx;
    const double inv_dx2 = 1.0 / (dx * dx);

    for (std::size_t i = 0; i < n; ++i) {
        const double left = u[(i + n - 1) % n];
        const double center = u[i];
        const double right = u[(i + 1) % n];

        const double advection = -center * (right - left) * inv_2dx;
        const double diffusion = viscosity * (right - 2.0 * center + left) * inv_dx2;
        rhs[i] = advection + diffusion;
    }

    return rhs;
}

std::vector<double> burgers_step(const std::vector<double> &u, double dt,
                                 double dx, double viscosity) {
    if (dt < 0.0) {
        throw std::invalid_argument("dt must be non-negative");
    }

    std::vector<double> rhs = burgers_rhs(u, dx, viscosity);
    std::vector<double> next(u.size());

    for (std::size_t i = 0; i < u.size(); ++i) {
        next[i] = u[i] + dt * rhs[i];
    }

    return next;
}

// The gelation step's finite-volume Burgers transport in z. A state is (N, B):
// one row of B cell averages u_b per mesh cell, on the edges z_{b-1/2}
// partitioning [0, 1]. The wave speed gamma u is non-negative, so every
// interface takes its left (upwind) value.

using ConstVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using ConstMatrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig>;
using Vector = nb::ndarray<double, nb::ndim<1>, nb::c_contig>;
using Matrix = nb::ndarray<double, nb::ndim<2>, nb::c_contig>;

struct StateGrid {
    std::size_t num_systems;
    std::size_t num_cells;
    const double *edges;
    const double *u;
};

StateGrid validate_state_grid(ConstMatrix state, ConstVector edge_positions) {
    const std::size_t num_systems = state.shape(0);
    const std::size_t num_cells = state.shape(1);

    if (edge_positions.shape(0) != num_cells + 1) {
        throw std::invalid_argument(
            "edge_positions must contain the B + 1 finite-volume edges");
    }
    if (num_cells == 0) {
        throw std::invalid_argument("state must contain at least one finite-volume cell");
    }

    const double *edges = edge_positions.data();
    if (edges[0] != 0.0 || edges[num_cells] != 1.0) {
        throw std::invalid_argument("edge_positions must start at 0 and end at 1");
    }
    for (std::size_t b = 0; b < num_cells; ++b) {
        if (edges[b + 1] <= edges[b]) {
            throw std::invalid_argument("edge_positions must be strictly increasing");
        }
    }

    return {num_systems, num_cells, edges, state.data()};
}

void validate_matrix_shape(Matrix matrix, std::size_t rows, std::size_t cols,
                           const char *name) {
    if (matrix.shape(0) != rows || matrix.shape(1) != cols) {
        throw std::invalid_argument(std::string(name) + " must have shape (N, B)");
    }
}

void validate_vector_shape(Vector vector, std::size_t size, const char *name) {
    if (vector.shape(0) != size) {
        throw std::invalid_argument(std::string(name) + " must have shape (B,)");
    }
}

// Van Leer's harmonic-mean slope of cell b, zero at an extremum. The left
// one-sided slope of the first cell reads u(0) = 0; the right one of the last
// cell is zero, which makes the left trace at z = 1 the last cell average.
inline double vanleer_limited_slope(const double *u_row, const double *edges,
                                    std::size_t b, std::size_t num_cells) {
    const double z_left_edge = edges[b];
    const double z_right_edge = edges[b + 1];
    const double z_center = 0.5 * (z_left_edge + z_right_edge);

    const double m_left =
        b == 0
            ? u_row[0] / z_center
            : (u_row[b] - u_row[b - 1]) /
                  (z_center - 0.5 * (edges[b - 1] + z_left_edge));

    const double m_right =
        b + 1 == num_cells
            ? 0.0
            : (u_row[b + 1] - u_row[b]) /
                  (0.5 * (z_right_edge + edges[b + 2]) - z_center);

    return m_left * m_right > 0.0 ? 2.0 * m_left * m_right / (m_left + m_right)
                                  : 0.0;
}

// The reconstruction's value at the right edge of cell b, u^-_{b+1/2}.
inline double upwind_value(const double *u_row, const double *edges, std::size_t b,
                           std::size_t num_cells) {
    const double dz = edges[b + 1] - edges[b];
    return u_row[b] + 0.5 * vanleer_limited_slope(u_row, edges, b, num_cells) * dz;
}

inline void fill_upwind_values_row(const double *u_row, const double *edges,
                                   std::size_t num_cells, double *upwind_row) {
    for (std::size_t b = 0; b < num_cells; ++b) {
        upwind_row[b] = upwind_value(u_row, edges, b, num_cells);
    }
}

double min_cell_width(const double *edges, std::size_t num_cells) {
    double dz_min = edges[1] - edges[0];
    for (std::size_t b = 1; b < num_cells; ++b) {
        dz_min = std::min(dz_min, edges[b + 1] - edges[b]);
    }
    return dz_min;
}

// du_b/dt of the finite-volume scheme, with the upwind flux h = gamma (u^-)^2 / 2
// and no inflow at z = 0.
void burgers_rhs_muscl_vanleer(ConstMatrix state, Matrix rhs,
                               ConstVector edge_positions, double gamma) {
    const StateGrid grid = validate_state_grid(state, edge_positions);

    validate_matrix_shape(rhs, grid.num_systems, grid.num_cells, "rhs");
    if (gamma < 0.0) {
        throw std::invalid_argument(
            "gamma must be non-negative for the right-going upwind flux");
    }

    double *du_dt = rhs.data();

#pragma omp parallel for if(grid.num_systems > 1)
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(grid.num_systems); ++n) {
        const double *u_row =
            grid.u + static_cast<std::size_t>(n) * grid.num_cells;
        double *rhs_row = du_dt + static_cast<std::size_t>(n) * grid.num_cells;
        // rhs_row holds the upwind values until each is consumed: entry b is
        // read, then overwritten, before entry b + 1 is reached.
        fill_upwind_values_row(u_row, grid.edges, grid.num_cells, rhs_row);

        double flux_left = 0.0;
        for (std::size_t b = 0; b < grid.num_cells; ++b) {
            const double dz = grid.edges[b + 1] - grid.edges[b];
            const double flux_right = 0.5 * gamma * rhs_row[b] * rhs_row[b];

            rhs_row[b] = -(flux_right - flux_left) / dz;
            flux_left = flux_right;
        }
    }
}

// The explicit scheme's step bound before its CFL factor, taken conservatively as
// the narrowest cell over the largest upwind value; infinite when nothing moves.
double max_cfl_timestep(ConstMatrix state, ConstVector edge_positions, double gamma) {
    const StateGrid grid = validate_state_grid(state, edge_positions);
    if (gamma < 0.0) {
        throw std::invalid_argument(
            "gamma must be non-negative for the right-going upwind flux");
    }

    double max_upwind_value = 0.0;

#pragma omp parallel for reduction(max : max_upwind_value) if(grid.num_systems > 1)
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(grid.num_systems); ++n) {
        const double *u_row =
            grid.u + static_cast<std::size_t>(n) * grid.num_cells;
        for (std::size_t b = 0; b < grid.num_cells; ++b) {
            max_upwind_value =
                std::max(max_upwind_value, upwind_value(u_row, grid.edges, b,
                                                        grid.num_cells));
        }
    }

    if (gamma == 0.0 || max_upwind_value <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    return min_cell_width(grid.edges, grid.num_cells) / (gamma * max_upwind_value);
}

// The upwind values and the right edges they sit at, for plotting the
// reconstruction against the characteristics.
void burgers_upwind_values_muscl_vanleer(ConstMatrix state, Vector upwind_z,
                                         Matrix upwind_u_vals,
                                         ConstVector edge_positions) {
    const StateGrid grid = validate_state_grid(state, edge_positions);
    validate_vector_shape(upwind_z, grid.num_cells, "upwind_z");
    validate_matrix_shape(upwind_u_vals, grid.num_systems, grid.num_cells,
                          "upwind_u_vals");

    double *point_data = upwind_z.data();
    for (std::size_t b = 0; b < grid.num_cells; ++b) {
        point_data[b] = grid.edges[b + 1];
    }

    double *upwind_data = upwind_u_vals.data();

#pragma omp parallel for if(grid.num_systems > 1)
    for (std::int64_t n = 0; n < static_cast<std::int64_t>(grid.num_systems); ++n) {
        const double *u_row =
            grid.u + static_cast<std::size_t>(n) * grid.num_cells;
        double *upwind_row = upwind_data + static_cast<std::size_t>(n) * grid.num_cells;
        fill_upwind_values_row(u_row, grid.edges, grid.num_cells, upwind_row);
    }
}

}  // namespace

void register_burgers(nb::module_ &m) {
    using namespace nb::literals;

    m.def("burgers_flux", &burgers_flux, "u"_a,
          "Return the inviscid Burgers flux f(u) = u^2 / 2.");
    m.def("burgers_rhs", &burgers_rhs, "u"_a, "dx"_a, "viscosity"_a = 0.0,
          "Return a periodic finite-difference RHS for 1D viscous Burgers.");
    m.def("burgers_step", &burgers_step, "u"_a, "dt"_a, "dx"_a,
          "viscosity"_a = 0.0,
          "Advance 1D viscous Burgers by one explicit Euler step.");
    m.def("burgers_rhs_muscl_vanleer", &burgers_rhs_muscl_vanleer, "state"_a,
          "rhs"_a, "edge_positions"_a, "gamma"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Fill rhs in-place using the MUSCL/van Leer finite-volume scheme.");
    m.def("max_cfl_timestep", &max_cfl_timestep, "state"_a,
          "edge_positions"_a, "gamma"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Return the maximum Burgers timestep before applying a CFL factor.");
    m.def("burgers_upwind_values_muscl_vanleer", &burgers_upwind_values_muscl_vanleer,
          "state"_a, "upwind_z"_a, "upwind_u_vals"_a, "edge_positions"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Fill upwind interface positions and MUSCL upwind values in-place.");
}
