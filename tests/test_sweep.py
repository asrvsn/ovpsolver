from pathlib import Path

import pytest
import yaml

from ovpsolver.phase_field_system.cli import build
from ovpsolver.phase_field_system.run import RunSpec, claim, read
from ovpsolver.phase_field_system.sweep import (
    PRODUCT,
    ZIP,
    Sweep,
    SweepError,
    SweepFailed,
    assignments,
    parse_address,
)
from ovpsolver.polymerizing_b import PolymerizingB

PIPS = Path(__file__).resolve().parents[1] / "examples" / "pips.yaml"
CHI_01 = 'couplings.flory_huggins.(water, protein).chi_01'


def sweep_of(*groups):
    return Sweep.from_groups([(kind, list(words)) for kind, words in groups])


def test_addresses_read_keys_indices_and_quoted_keys():
    assert parse_address("phase_fields.sol[0].kappa") == ("phase_fields", "sol", 0, "kappa")
    # Spaces and parentheses are part of a bare key; quotes are needed only for
    # the characters the grammar uses.
    assert parse_address(CHI_01) == parse_address(
        'couplings.flory_huggins."(water, protein)".chi_01'
    )
    assert parse_address("a.'b.c'[2]") == ("a", "b.c", 2)


@pytest.mark.parametrize("text", ["", "a.", ".a", "a..b", "a[x]", "a[0", "[0]", "a.[0]", 'a."b'])
def test_malformed_addresses_are_refused(text):
    with pytest.raises(SweepError):
        parse_address(text)


def test_values_read_as_the_spec_reads_them_but_1e_3_is_a_number():
    (one,) = assignments(["x=[1e-3, 1.0e-3, 2, true, null, phi]"])
    assert one.values == (0.001, 0.001, 2, True, None, "phi")
    assert assignments(["x=1,2,3"])[0].values == (1, 2, 3)
    assert assignments(["x=5"])[0].values == (5,)
    assert assignments(['x=[["phi"], ["phi", "v"]]'])[0].values == (["phi"], ["phi", "v"])


def test_an_assignment_the_shell_split_is_rejoined():
    words = ["a=[1,", "2,", "3]", "b=[4,", "5]"]
    first, second = assignments(words)
    assert (first.address, first.values) == (("a",), (1, 2, 3))
    assert (second.address, second.values) == (("b",), (4, 5))


def test_runs_are_every_combination_of_the_axes_in_the_order_given():
    sweep = sweep_of((ZIP, ["a=[1, 2]", "b=[10, 20]"]), (PRODUCT, ["c=[7, 8, 9]"]))
    assert len(sweep) == 6
    values = [tuple(value for _, value in point) for point in sweep.points()]
    assert values == [
        (1, 10, 7), (1, 10, 8), (1, 10, 9),
        (2, 20, 7), (2, 20, 8), (2, 20, 9),
    ]


def test_zipped_assignments_need_as_many_values_each():
    with pytest.raises(SweepError, match="as many values"):
        sweep_of((ZIP, ["a=[1, 2]", "b=[1, 2, 3]"]))


@pytest.mark.parametrize("second", ["a=[3]", "a.b=[3]"])
def test_an_address_is_swept_once_and_not_inside_another(second):
    with pytest.raises(SweepError):
        sweep_of((PRODUCT, ["a=[1, 2]", second]))


def test_labels_are_the_shortest_trailing_part_that_tells_them_apart():
    sweep = sweep_of(
        (ZIP, ["phase_fields.sol[0].kappa=[1]", "phase_fields.polymerizing[0].kappa=[1]"]),
        (PRODUCT, [f"{CHI_01}=[1]"]),
    )
    assert list(sweep.labels().values()) == ["sol.0.kappa", "polymerizing.0.kappa", "chi_01"]


def test_a_variant_is_the_spec_with_its_values_in_place(tmp_path):
    sweep = sweep_of(
        (ZIP, ["phase_fields.sol[0].kappa=[0.5e-3, 1.5e-3]", "phase_fields.polymerizing[0].kappa=[0.5e-3, 1.5e-3]"]),
        (PRODUCT, [f"{CHI_01}=[1, 2, 3]"]),
    )
    variants = sweep.variants(PIPS, tmp_path)
    assert len(variants) == 6
    last = variants[-1]
    assert last.path.name == "pips-sol.0.kappa=0.0015-polymerizing.0.kappa=0.0015-chi_01=3.yaml"

    text = last.path.read_text()
    document = yaml.safe_load(text)
    expected = yaml.safe_load(PIPS.read_text())
    expected["system"]["phase_fields"]["sol"][0]["kappa"] = 1.5e-3
    expected["system"]["phase_fields"]["polymerizing"][0]["kappa"] = 1.5e-3
    expected["system"]["couplings"]["flory_huggins"]["(water, protein)"]["chi_01"] = 3
    expected["solver"]["save"]["path"] = str(
        (PIPS.parent / "scratch" / last.path.stem).resolve()
    )
    assert document == expected
    # Every comment of the original survives, under a header saying what was swept.
    comments = [line for line in PIPS.read_text().splitlines() if line.lstrip().startswith("#")]
    assert all(line in text for line in comments)
    assert text.startswith("# Run 6 of 6 of a sweep over the spec")


def test_a_variant_reads_against_the_spec_it_varies(tmp_path):
    (variant,) = sweep_of((PRODUCT, [f"{CHI_01}=[2.5]"])).variants(PIPS, tmp_path)
    parameters = read(PolymerizingB.Parameters, variant.path, relative_to=variant.origin)
    # The geometry is the one beside the original, not one beside the variant.
    assert parameters.solver.diffuse_domain.geometry_path == PIPS.parent / "geometry_disks_triplet.py"
    assert parameters.solver.save.directory.name == "pips-chi_01=2.5"
    # The variant is what is archived, so the run's own spec carries its values.
    assert parameters.source == variant.path.resolve()


def test_a_missing_key_is_reported_with_the_keys_that_are_there(tmp_path):
    with pytest.raises(SweepError, match="has no 'kapa'; it has name, color"):
        sweep_of((PRODUCT, ["phase_fields.sol[0].kapa=[1]"])).variants(PIPS, tmp_path)
    with pytest.raises(SweepError, match=r"has 1 entries, so \[3\] is past its end"):
        sweep_of((PRODUCT, ["phase_fields.sol[3].kappa=[1]"])).variants(PIPS, tmp_path)


def test_the_command_line_keeps_the_axes_in_the_order_given():
    entries = {"run": RunSpec()}
    parser = build(PolymerizingB, entries)
    arguments = vars(
        parser.parse_args(
            ["run", str(PIPS), "--sweep", "a=[1, 2]", "--sweep-zip", "b=[1]", "c=[2]", "--sweep", "d=[3]"]
        )
    )
    arguments.pop("entry_point")
    sweep = RunSpec().prepare(arguments)["sweep"]
    assert [[a.address for a in axis] for axis in sweep.axes] == [
        [("a",)], [("b",), ("c",)], [("d",)]
    ]


def test_checking_a_directory_clears_nothing(tmp_path):
    (variant,) = sweep_of((PRODUCT, [f"{CHI_01}=[2.5]"])).variants(PIPS, tmp_path)
    save = read(PolymerizingB.Parameters, variant.path, relative_to=variant.origin).solver.save
    save.directory = tmp_path / "run"
    save.directory.mkdir()
    (save.directory / "spec.yaml").write_text("")
    with pytest.raises(FileExistsError):
        claim(save, check_only=True)
    claim(save, overwrite=True, check_only=True)
    assert (save.directory / "spec.yaml").exists()


@pytest.fixture
def spec_copy(tmp_path):
    """``pips.yaml`` and its geometry, copied where its runs would save under ``tmp_path``.

    So a test that checks output directories reads none a real run left.
    """

    for name in (PIPS.name, "geometry_disks_triplet.py"):
        (tmp_path / name).write_text((PIPS.parent / name).read_text())
    return tmp_path / PIPS.name


class HalfFailing(PolymerizingB):
    """Runs nothing; fails the variant that sets chi_01 to 2."""

    calls: list[str] = []

    @classmethod
    def run(cls, spec, **options):
        cls.calls.append(Path(spec).stem)
        if "chi_01=2" in Path(spec).stem:
            raise RuntimeError("no admissible sub-step\nsecond line")
        return 0.5


def test_a_sweep_goes_on_past_a_failed_run_and_reports_it_at_the_end(spec_copy, capsys):
    HalfFailing.calls.clear()
    sweep = sweep_of((PRODUCT, [f"{CHI_01}=[1, 2, 3]"]))
    with pytest.raises(SweepFailed) as failed:
        RunSpec()(HalfFailing, specs=[spec_copy], sweep=sweep)
    assert HalfFailing.calls == ["pips-chi_01=1", "pips-chi_01=2", "pips-chi_01=3"]
    assert failed.value.reached == [0.5, None, 0.5]
    ((directory, why),) = failed.value.failures
    assert directory.name == "pips-chi_01=2"
    assert why == "RuntimeError: no admissible sub-step"
    assert "run 2 of 3 failed, going on to the next" in capsys.readouterr().err


def test_a_queue_of_specs_still_stops_at_the_first_failure(spec_copy):
    HalfFailing.calls.clear()
    second = spec_copy.with_name("second.yaml")
    second.write_text(spec_copy.read_text().replace("path: ./scratch/pips", "path: ./scratch/second"))

    class AlwaysFailing(HalfFailing):
        @classmethod
        def run(cls, spec, **options):
            cls.calls.append(Path(spec).stem)
            raise RuntimeError("diverged")

    with pytest.raises(RuntimeError, match="diverged"):
        RunSpec()(AlwaysFailing, specs=[spec_copy, second])
    assert HalfFailing.calls == ["pips"]
