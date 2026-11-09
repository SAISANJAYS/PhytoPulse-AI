import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.sensor_reader import ArduinoSensorReader


class TestSerialProtocol(unittest.TestCase):
    def setUp(self):
        self.reader = ArduinoSensorReader("COM99", 9600)

    def test_parse_valid_sensor_packet(self):
        packet = self.reader._process_incoming_line("24.7,78.1,1,1022")
        self.assertIsNotNone(packet)
        self.assertEqual(packet["temperature_c"], 24.7)
        self.assertEqual(packet["humidity_pct"], 78.1)
        self.assertEqual(packet["rain_raw"], 1)
        self.assertEqual(packet["soil_raw"], 1022)
        self.assertEqual(self.reader.latest_sensor_packet, packet)

    def test_parse_risk_score_line(self):
        packet = self.reader._process_incoming_line("RISK_SCORE,28.8")
        self.assertIsNone(packet)  # Not a 4-value sensor packet
        self.assertEqual(self.reader.latest_risk_score, 28.8)

    def test_parse_calculated_risk_line(self):
        packet = self.reader._process_incoming_line("CALCULATED_RISK,LOW")
        self.assertIsNone(packet)
        self.assertEqual(self.reader.latest_calculated_risk, "LOW")

    def test_ignore_status_and_error(self):
        self.assertIsNone(self.reader._process_incoming_line("STATUS,STARTED"))
        self.assertIsNone(self.reader._process_incoming_line("ERROR,DHT"))


if __name__ == "__main__":
    unittest.main()
