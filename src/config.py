
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Model settings
MODEL_PATH = ROOT_DIR / "models" / "plant_disease_model_candidate.h5"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)

# Serial / Arduino settings
DEFAULT_SERIAL_PORT = "COM5"
SERIAL_BAUD_RATE = 9600
SENSOR_INTERVAL = 5
SERIAL_TIMEOUT = 10
SENSOR_POLL_TIMEOUT = 0.5
SENSOR_READ_RETRIES = 10

# Prediction settings
CONFIDENCE_THRESHOLD = 0.60

# Disease risk thresholds
HIGH_HUMIDITY_THRESHOLD = 80
MODERATE_HUMIDITY_THRESHOLD = 60
DRY_SOIL_THRESHOLD = 700
VERY_DRY_SOIL_THRESHOLD = 850
RAIN_DETECTED_VALUE = 0
NO_RAIN_VALUE = 1

# Disease Spread Risk formula weights
# R = 100 * (0.53 * LWD_norm + 0.25 * Humidity_norm
#            + 0.15 * Temperature_norm + 0.07 * Moisture_norm)
WEIGHT_LWD = 0.53
WEIGHT_HUMIDITY = 0.25
WEIGHT_TEMPERATURE = 0.15
WEIGHT_MOISTURE = 0.07

# Risk category thresholds
# Low: 0 to < 30
# Moderate: 30 to < 60
# High: 60 to 100
RISK_THRESHOLD_LOW_MAX = 30.0
RISK_THRESHOLD_MODERATE_MAX = 60.0

# Sensor normalization parameters
TEMP_NORM_MIN = 10.0
TEMP_NORM_MAX = 40.0
HUMIDITY_NORM_MIN = 0.0
HUMIDITY_NORM_MAX = 100.0
SOIL_RAW_WET = 300
SOIL_RAW_DRY = 1000
