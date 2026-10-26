from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import yaml

from ovpsolver.phase_field_system.cli import build
from ovpsolver.phase_field_system.sweep import PRODUCT, ZIP, Sweep, SweepError
from ovpsolver.phase_field_system.visualize import figure as figures
from ovpsolver.phase_field_system.visualize.layers import Layer
from ovpsolver.phase_field_system.visualize.style import resolved
from ovpsolver.phase_field_system.visualize.sweep import (
    VisualizeSweep,
    grid_axes,
    open_run,
)
from ovpsolver.polymerizing_b import PolymerizingB

PIPS = Path(__file__).resolve().parents[1] / "examples" / "pips.yaml"
KAPPAS = (
    ZIP,
    ["phase_fields.sol[0].kappa=[1e-3, 2e-3]", "phase_fields.polymerizing[0].kappa=[1e-3, 2e-3]"],
)
RATES = (PRODUCT, ["phase_fields.polymerizing[0].crosslinking_rate=[1.5, 3, 6]"])


def sweep_of(*groups):
    return Sweep.from_groups([(kind, list(words)) for kind, words in groups])


def test_with_no_axes_named_x_is_the_last_axis_and_y_the_first():
    (across, columns), (up, rows) = grid_axes(sweep_of(KAPPAS, RATES), None, None)
    assert (across, columns.values) == (1, (1.5, 3, 6))
    assert (up, rows.address) == (0, ("phase_fields", "sol", 0, "kappa"))


def test_an_address_on_a_zipped_axis_labels_it_and_the_other_axis_is_y():
    (across, columns), (up, rows) = grid_axes(
        sweep_of(KAPPAS, RATES), "phase_fields.polymerizing[0].kappa", None
    )
    assert (across, columns.address) == (0, ("phase_fields", "polymerizing", 0, "kappa"))
    assert (up, rows.values) == (1, (1.5, 3, 6))


def test_x_and_y_on_one_axis_are_refused():
    with pytest.raises(SweepError, match="step together"):
        grid_axes(
            sweep_of(KAPPAS, RATES),
            "phase_fields.sol[0].kappa",
            "phase_fields.polymerizing[0].kappa",
        )


def test_an_address_the_sweep_does_not_sweep_is_refused_with_those_it_does():
    with pytest.raises(SweepError, match=r"not swept; the swept addresses are phase_fields\.sol"):
        grid_axes(sweep_of(KAPPAS, RATES), "phase_fields.sol[0].surface_affinity", None)


@pytest.mark.parametrize("groups", [(KAPPAS,), (KAPPAS, RATES, (PRODUCT, ["rng_seed=[0, 1]"]))])
def test_a_sweep_of_other_than_two_axes_is_refused(groups):
    with pytest.raises(SweepError, match="grid of two axes"):
        grid_axes(sweep_of(*groups), None, None)


def test_the_directories_are_where_the_variants_save(tmp_path):
    sweep = sweep_of(KAPPAS, RATES)
    saved = [
        yaml.safe_load(variant.path.read_text())["solver"]["save"]["path"]
        for variant in sweep.variants(PIPS, tmp_path)
    ]
    assert [str(directory) for directory in sweep.directories(PIPS)] == saved


def test_a_run_with_nothing_to_draw_says_why(tmp_path):
    assert open_run(PolymerizingB, tmp_path / "absent", -1) == "not run"
    # What a run that failed while building leaves: its log, and no frames.
    (tmp_path / "failed").mkdir()
    (tmp_path / "failed" / "log.txt").write_text("")
    assert open_run(PolymerizingB, tmp_path / "failed", -1) == "nothing saved"


def test_the_command_line_reads_the_sweep_and_the_grid():
    entries = {"visualize.sweep": VisualizeSweep()}
    parser = build(PolymerizingB, entries)
    arguments = vars(
        parser.parse_args(
            [
                "visualize.sweep", str(PIPS), "--sweep-zip", *KAPPAS[1], "--sweep", *RATES[1],
                "--x", "phase_fields.sol[0].kappa", "--x-label", r"$\kappa$",
                "--field", r"protein.phi_gel=$\phi_{\rm gel}$", "--save",
            ]
        )
    )
    arguments.pop("entry_point")
    prepared = VisualizeSweep().prepare(arguments)
    assert len(prepared["sweep"]) == 6
    assert (prepared["x_label"], prepared["frame"], prepared["show"]) == (r"$\kappa$", -1, False)
    assert (prepared["field"], prepared["field_label"]) == ("protein.phi_gel", r"$\phi_{\rm gel}$")


@pytest.mark.parametrize(
    ("given", "field", "label"),
    [("protein.phi_gel", "protein.phi_gel", None), ("water.phi=$t=0$", "water.phi", "$t=0$")],
)
def test_a_field_is_named_by_its_saved_name_or_by_what_follows_the_first_equals(given, field, label):
    prepared = VisualizeSweep().prepare(
        {"sweep": [(PRODUCT, ["rng_seed=[0, 1]"])], "save": None, "field": given}
    )
    assert (prepared["field"], prepared["field_label"]) == (field, label)


def test_the_command_line_needs_the_sweep_that_made_the_runs():
    with pytest.raises(SweepError, match="--sweep and --sweep-zip"):
        VisualizeSweep().prepare({"sweep": [], "save": None})


def test_a_grid_leaves_an_empty_cell_empty_and_shapes_every_cell_as_its_panel():
    style = resolved(PolymerizingB, {"panel_pixels": 40, "dpi": 40.0})
    panel = np.zeros((40, 60, 4), dtype=np.uint8)
    figure = figures.grid(
        [[panel, None, panel], [panel, panel, panel]],
        titles=[[None, "nothing saved", None], [None, None, None]],
        column_labels=("1.5", "3", "6"),
        row_labels=("0.002", "0.001"),
        x_label=r"$\gamma$",
        y_label=r"$\kappa$",
        title="protein.phi_gel",
        layer=Layer(name="protein.phi_gel", field="protein.phi_gel"),
        value_range=(0.0, 1.0),
        colorbar_label=r"$\phi_{\rm gel}$",
        aspect=1.5,
        style=style,
    )
    images = [axis for axis in figure.axes if axis.images]
    # Five panels and the colourbar's ramp; the empty cell draws nothing.
    assert len(images) == 6
    # Each panel's cell is its image's shape, 60 by 40 pixels at this dpi.
    width, height = images[0].get_position().size * figure.get_size_inches() * figure.dpi
    assert (round(width), round(height)) == (60, 40)
