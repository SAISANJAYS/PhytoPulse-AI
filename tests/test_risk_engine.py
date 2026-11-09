import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.risk_engine import (
    normalize_sensor_readings,
    calculate_environmental_risk_score,
    calculate_disease_risk,
)


class TestRiskEngine(unittest.TestCase):
    def test_formula_weights_sum_to_one(self):
        from src.config import (
            WEIGHT_LWD,
            WEIGHT_HUMIDITY,
            WEIGHT_TEMPERATURE,
            WEIGHT_MOISTURE,
        )
        total_weight = WEIGHT_LWD + WEIGHT_HUMIDITY + WEIGHT_TEMPERATURE + WEIGHT_MOISTURE
        self.assertAlmostEqual(total_weight, 1.00, places=4)

    def test_low_risk_calculation(self):
        # Dry conditions: rain=1 (no rain -> LWD=0.0), hum=50% (0.5), temp=25°C (0.5), dry soil 900
        packet = {
            "temperature_c": 25.0,
            "humidity_pct": 50.0,
            "rain_raw": 1,
            "soil_raw": 900,
        }
        res = calculate_environmental_risk_score(packet)
        # R = 100 * (0.53*0 + 0.25*0.5 + 0.15*0.5 + 0.07*(100/700)) = 100 * (0.125 + 0.075 + 0.01) = 21.0
        self.assertLess(res["score"], 30.0)
        self.assertEqual(res["category"], "Low")
        self.assertEqual(res["level"], "LOW")

    def test_moderate_risk_calculation(self):
        # Higher humidity & moist soil without rain: hum=85% (0.85), temp=28°C (0.60), wet soil 400
        packet = {
            "temperature_c": 28.0,
            "humidity_pct": 85.0,
            "rain_raw": 1,
            "soil_raw": 400,
        }
        res = calculate_environmental_risk_score(packet)
        self.assertGreaterEqual(res["score"], 30.0)
        self.assertLess(res["score"], 60.0)
        self.assertEqual(res["category"], "Moderate")
        self.assertEqual(res["level"], "MODERATE")

    def test_high_risk_calculation(self):
        # Rain detected: rain=0 (LWD=1.00), hum=80% (0.80), temp=26°C (0.53)
        packet = {
            "temperature_c": 26.0,
            "humidity_pct": 80.0,
            "rain_raw": 0,
            "soil_raw": 350,
        }
        res = calculate_environmental_risk_score(packet)
        self.assertGreaterEqual(res["score"], 60.0)
        self.assertEqual(res["category"], "High")
        self.assertEqual(res["level"], "HIGH")

    def test_healthy_prediction_does_not_claim_active_spread(self):
        packet = {
            "temperature_c": 26.0,
            "humidity_pct": 80.0,
            "rain_raw": 0,
            "soil_raw": 350,
        }
        prediction = {
            "class_name": "Corn_(maize)___healthy",
            "confidence": 0.98,
        }
        res = calculate_disease_risk(prediction, packet)
        self.assertTrue(res["is_healthy"])
        self.assertTrue(any("healthy" in r.lower() for r in res["reasons"]))
        self.assertTrue(any("no active disease" in r.lower() for r in res["reasons"]))

    def test_no_prediction_message(self):
        packet = {
            "temperature_c": 25.0,
            "humidity_pct": 60.0,
            "rain_raw": 1,
            "soil_raw": 700,
        }
        res = calculate_disease_risk(None, packet)
        self.assertEqual(res["level"], "UNKNOWN")
        self.assertIn("Upload a leaf image and run disease prediction to assess disease spread risk.", res["reasons"])


if __name__ == "__main__":
    unittest.main()
