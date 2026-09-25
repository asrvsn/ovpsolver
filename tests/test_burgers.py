import ovpsolver as ps


def test_burgers_flux():
    assert ps.burgers_flux(2.0) == 2.0


def test_burgers_rhs_periodic_shape():
    rhs = ps.burgers_rhs([1.0, 0.5, 0.0, -0.5], dx=1.0, viscosity=0.1)
    assert len(rhs) == 4


def test_burgers_step_periodic_shape():
    next_values = ps.burgers_step([1.0, 0.5, 0.0, -0.5], dt=0.01, dx=1.0)
    assert len(next_values) == 4
