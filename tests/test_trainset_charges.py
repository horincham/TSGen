from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from tsgen.data.xyz import Frame
from tsgen.trainset.charges import (
    load_reference_charges,
    parse_adch_charges,
    parse_hirshfeld_charges,
)
from tsgen.trainset.config import GaussianCfg, TrainsetBuildCfg
from tsgen.trainset.gaussian import GaussianScanRunner
from tsgen.trainset.models import ScanPoint
from tsgen.trainset.reaxff_io import write_trainset_in


class ChargeTargetTests(unittest.TestCase):
    def test_adch_parser_reads_multiwfn_style_atom_charge_table(self):
        text = """
        Atomic charges:
          1 C   -0.214321
          2 H    0.071440
          3 H    0.071441
          4 H    0.071440
        Sum of atomic charges: 0.000000
        """

        charges = parse_adch_charges(text, natoms=4)

        self.assertEqual(len(charges), 4)
        self.assertAlmostEqual(charges[0], -0.214321)
        self.assertAlmostEqual(sum(charges), 0.000000, places=5)

    def test_hirshfeld_parser_reads_gaussian_population_block(self):
        text = """
        Hirshfeld charges, spin densities, dipoles, and CM5 charges
             Atom     Q-H        S-H        Dx
              1  C   -0.130000   0.000000   0.0
              2  H    0.043333   0.000000   0.0
              3  H    0.043333   0.000000   0.0
              4  H    0.043334   0.000000   0.0
        Tot           0.000000
        """

        charges = parse_hirshfeld_charges(text, natoms=4)

        self.assertEqual(len(charges), 4)
        self.assertAlmostEqual(charges[0], -0.13)
        self.assertAlmostEqual(sum(charges), 0.0, places=5)

    def test_adch_loader_falls_back_to_hirshfeld_when_adch_missing(self):
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            (job_dir / "input.log").write_text(
                """
                Hirshfeld charges, spin densities, dipoles, and CM5 charges
                     Atom     Q-H        S-H        Dx
                      1  O   -0.420000   0.000000   0.0
                      2  H    0.210000   0.000000   0.0
                      3  H    0.210000   0.000000   0.0
                Tot           0.000000
                """,
                encoding="utf-8",
            )

            charges, source = load_reference_charges(
                job_dir,
                natoms=3,
                method="adch",
                fallback_method="hirshfeld",
            )

        self.assertEqual(source, "hirshfeld")
        self.assertEqual(len(charges), 3)
        self.assertAlmostEqual(charges[0], -0.42)

    def test_adch_loader_runs_configured_postprocess_command_when_wavefunction_present(self):
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            (job_dir / "input.fchk").write_text("wavefunction fixture", encoding="utf-8")
            command = (
                f'"{sys.executable}" -c '
                '"from pathlib import Path; '
                "Path('adch_charges.txt').write_text(chr(10).join(['1 O -0.420000','2 H 0.210000','3 H 0.210000']) + chr(10), encoding='utf-8')\""
            )

            charges, source = load_reference_charges(
                job_dir,
                natoms=3,
                method="adch",
                fallback_method="hirshfeld",
                adch_command=command,
            )

        self.assertEqual(source, "adch")
        self.assertEqual(len(charges), 3)
        self.assertAlmostEqual(charges[0], -0.42)

    def test_gaussian_input_writes_checkpoint_for_charge_postprocessing(self):
        frame = Frame(
            symbols=["O", "H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
            comment="water",
        )
        point = ScanPoint(
            id="water_eq",
            parent_id="water_opt",
            coord_id="eq",
            scan_type="equilibrium",
            family="H2O",
            point_index=0,
            target_value=0.0,
            eq_value=0.0,
            atom_indices=(),
            frame=frame,
        )
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            GaussianScanRunner(GaussianCfg()).write_job(job_dir, point)
            text = (job_dir / "input.gjf").read_text(encoding="utf-8")

        self.assertIn("%chk=input.chk", text)

    def test_trainset_writer_emits_charge_section_before_geometry_and_energy(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "trainset.in"
            write_trainset_in(
                out,
                geometry_rows=[],
                energy_rows=[],
                charge_rows=[
                    {"id": "water", "weight": 0.1, "atom_index_1based": 1, "charge": -0.42},
                    {"id": "water", "weight": 0.1, "atom_index_1based": 2, "charge": 0.21},
                ],
            )

            text = out.read_text(encoding="utf-8")

        self.assertIn("CHARGE", text)
        self.assertIn("ENDCHARGE", text)
        self.assertIn("water", text)
        self.assertIn("-0.420000", text)

    def test_trainset_config_enables_hirshfeld_by_default(self):
        cfg = TrainsetBuildCfg()

        self.assertTrue(cfg.trainset.charge.enabled)
        self.assertEqual(cfg.trainset.charge.method, "hirshfeld")
        self.assertEqual(cfg.trainset.charge.fallback_method, "hirshfeld")


if __name__ == "__main__":
    unittest.main()
