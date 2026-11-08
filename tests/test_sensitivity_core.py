from pathlib import Path
import tempfile
import unittest

from tsgen.sensitivity.ffield import FFieldDocument
from tsgen.sensitivity.fort_parser import build_evaluation_result
from tsgen.sensitivity.models import ParameterBound, ParameterCoordinate
from tsgen.sensitivity.params_ml import parse_params_ml, validate_parameter_bounds
from tsgen.sensitivity.sampling import generate_samples, write_sample_ffield
from tsgen.sensitivity.trainset_map import parse_geo_inventory, parse_trainset, validate_training_inputs


def minimal_ffield_text() -> str:
    return (
        "Reactive MD-force field\n"
        "  2 ! Number of general parameters\n"
        "    1.0000 ! general one\n"
        "    2.0000 ! general two\n"
        "  1 ! Nr of atoms\n"
        "    ! atom header 2\n"
        "    ! atom header 3\n"
        "    ! atom header 4\n"
        " C    1.0000   2.0000   3.0000   4.0000   5.0000   6.0000   7.0000   8.0000\n"
        "      9.0000  10.0000  11.0000  12.0000  13.0000  14.0000  15.0000  16.0000\n"
        "     17.0000  18.0000  19.0000  20.0000  21.0000  22.0000  23.0000  24.0000\n"
        "     25.0000  26.0000  27.0000  28.0000  29.0000  30.0000  31.0000  32.0000\n"
        "  1 ! Nr of bonds\n"
        "    ! bond header 2\n"
        "  1  1   1.0000   2.0000   3.0000   4.0000   5.0000   6.0000   7.0000   8.0000 ! bond values\n"
        "          9.0000  10.0000  11.0000  12.0000  13.0000  14.0000  15.0000  16.0000\n"
        "  1 ! off diagonal\n"
        "  1  1   1.0000   2.0000   3.0000   4.0000   5.0000   6.0000\n"
        "  1 ! angles\n"
        "  1  1  1   1.0000   2.0000   3.0000   4.0000   5.0000   6.0000   7.0000\n"
        "  1 ! torsions\n"
        "  1  1  1  1   1.0000   2.0000   3.0000   4.0000   5.0000   6.0000   7.0000\n"
        "  1 ! hydrogen bonds\n"
        "  1  1  1   1.0000   2.0000   3.0000   4.0000\n"
    )


class CoreSensitivityTests(unittest.TestCase):
    def test_ffield_round_trip_and_single_update(self):
        text = minimal_ffield_text()
        document = FFieldDocument.from_text(text)
        self.assertEqual(document.render({}), text)
        counts = {}
        for parameter in document.parameters:
            section = parameter.coordinate.section
            counts[section] = counts.get(section, 0) + 1
        self.assertEqual(counts, {1: 2, 2: 32, 3: 16, 4: 6, 5: 7, 6: 7, 7: 4})
        updated = FFieldDocument.from_text(
            document.render({ParameterCoordinate(3, 1, 4): 4.25})
        )
        self.assertAlmostEqual(updated.get(ParameterCoordinate(3, 1, 4)), 4.25)

    def test_params_ml_normalizes_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params-ml"
            path.write_text("3 1 4 0.1 9.0 1.0\n", encoding="utf-8")
            bounds = parse_params_ml(path)
            validate_parameter_bounds(bounds, FFieldDocument.from_text(minimal_ffield_text()))
            self.assertEqual((bounds[0].lower_bound, bounds[0].upper_bound), (1.0, 9.0))

    def test_trainset_mapping_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geo = root / "geo"
            geo.write_text(
                "BIOGRF 200\nDESCRP mol_a.1\nHETATM 1 C 0.0 0.0 0.0\nEND\n",
                encoding="utf-8",
            )
            trainset = root / "trainset.in"
            trainset.write_text(
                "CHARGE\nmol_a.1 0.1 1 -0.2\nENDCHARGE\n"
                "GEOMETRY\nmol_a.1 1.0 1 2 1.5 !bond\nENDGEOMETRY\n"
                "ENERGY\n1.0 + mol_a.1/1 2.5\nENDENERGY\n",
                encoding="utf-8",
            )
            inventory = parse_geo_inventory(geo)
            items = parse_trainset(trainset)
            validate_training_inputs(inventory, items)
            self.assertEqual([item.section for item in items], ["CHARGE", "ENERGY", "GEOMETRY"])
            bound = ParameterBound(ParameterCoordinate(3, 1, 4), 0.1, 1.0, 9.0)
            samples = generate_samples([bound], [4.0], 8, 42, True)
            self.assertEqual(len(samples), 8)
            self.assertEqual(samples[0].physical_values, (4.0,))
            output = root / "sample" / "ffield"
            write_sample_ffield(
                FFieldDocument.from_text(minimal_ffield_text()),
                [bound],
                samples[1],
                output,
            )
            self.assertTrue(output.exists())

    def test_fort_outputs_map_to_section_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainset = root / "trainset.in"
            trainset.write_text(
                "ENERGY\n1.0 + mol_a/1 2.0\nENDENERGY\n"
                "GEOMETRY\nmol_a 0.5 1 2 1.5\nENDGEOMETRY\n",
                encoding="utf-8",
            )
            items = parse_trainset(trainset)
            (root / "fort.13").write_text("2.0000\n", encoding="utf-8")
            rows = []
            running = 0.0
            for item in items:
                running += 1.0
                label = item.item_label if item.section == "ENERGY" else "mol_a Bond distance: 1 2"
                rows.append(
                    f"{label:60s}{0.0:12.4f}{item.reference_value:12.4f}"
                    f"{item.weight:12.4f}{1.0:12.4f}{running:12.4f}\n"
                )
            (root / "fort.99").write_text("".join(rows), encoding="utf-8")
            result = build_evaluation_result(
                "sample_000001",
                root / "fort.13",
                root / "fort.99",
                items,
                0.2,
                0,
                "abc",
            )
            self.assertEqual(result.status, "success")
            self.assertEqual(result.section_losses, {"ENERGY": 1.0, "GEOMETRY": 1.0})


if __name__ == "__main__":
    unittest.main()
