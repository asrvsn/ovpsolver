#include <nanobind/nanobind.h>

namespace nb = nanobind;

void register_burgers(nb::module_ &m);

NB_MODULE(ovpsolver_ext, m) {
    m.doc() = "C++ kernels for the ovpsolver package";
    register_burgers(m);
}
