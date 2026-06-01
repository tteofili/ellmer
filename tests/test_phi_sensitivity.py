import unittest

from ellmer.phi_sensitivity import parse_phi_from_model_key, phi_tag
from ellmer.post_hoc.lemon_masked import lime_sample_budget_for_n


class TestPhiSensitivityHelpers(unittest.TestCase):
    def test_phi_tag(self):
        self.assertEqual(phi_tag(0.25), "0p25")
        self.assertEqual(phi_tag(0.5), "0p5")
        self.assertEqual(phi_tag(0.75), "0p75")

    def test_lime_sample_budget_for_n(self):
        self.assertEqual(lime_sample_budget_for_n(1), 500)
        self.assertEqual(lime_sample_budget_for_n(17), max(min(30 * 17, 3000), 500))
        self.assertEqual(lime_sample_budget_for_n(200), 3000)

    def test_parse_phi_from_model_key(self):
        self.assertAlmostEqual(parse_phi_from_model_key("hybrid_certa_phi0p25_sample"), 0.25)
        self.assertAlmostEqual(parse_phi_from_model_key("hybrid_lemon_minun_phi0p5_x"), 0.5)
        self.assertAlmostEqual(parse_phi_from_model_key("hybrid_lemon_minun_phi0p75_run"), 0.75)
        self.assertAlmostEqual(parse_phi_from_model_key("hybrid_certa_phi1_sample"), 1.0)
        self.assertIsNone(parse_phi_from_model_key("hybrid_sample"))


if __name__ == "__main__":
    unittest.main()
