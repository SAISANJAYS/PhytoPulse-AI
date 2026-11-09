
import time
from pathlib import Path

from .config import SENSOR_READ_INTERVAL, TEST_IMAGES_DIR
from .sensor_reader import SensorReader
from .image_predictor import ImagePredictor
from .arduino_controller import ArduinoController
from .risk_engine import calculate_risk
from .utils import display_result, log_result


def main():
    print("Starting MPMC Leaf Disease Prediction System...")

    predictor = ImagePredictor()

    print(f"\nTest images folder: {TEST_IMAGES_DIR}")
    image_input = input(
        "Enter the leaf image path (or press Enter to type a filename): "
    ).strip()

    if not image_input:
        filename = input(
            "Enter filename inside data/test_images/: "
        ).strip()
        image_path = TEST_IMAGES_DIR / filename
    else:
        image_path = Path(image_input).expanduser()

    # Run the CNN once for the selected image.
    prediction = predictor.predict(image_path)

    print("\nImage prediction complete.")
    print(f"Class      : {prediction['disease']}")
    print(f"Confidence : {prediction['confidence']:.2%}")

    sensor_reader = None
    controller = None

    try:
        sensor_reader = SensorReader()
        controller = ArduinoController(
            connection=sensor_reader.connection
        )

        print("\nArduino connected.")
        print("Reading sensors. Press Ctrl+C to stop.")

        last_logged_signature = None

        while True:
            sensors = sensor_reader.read_sensors()

            if sensors is None:
                print("Waiting for a valid Arduino sensor line...")
                time.sleep(0.5)
                continue

            risk_result = calculate_risk(prediction, sensors)

            controller.send_risk(risk_result["risk"])
            display_result(prediction, sensors, risk_result)

            # Log when readings/risk change, to avoid excessive duplicates.
            signature = (
                prediction["disease"],
                round(sensors["temperature"], 1),
                round(sensors["humidity"], 1),
                sensors["rain"],
                sensors["soil"],
                risk_result["risk"],
            )

            if signature != last_logged_signature:
                log_result(prediction, sensors, risk_result)
                last_logged_signature = signature

            time.sleep(SENSOR_READ_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping the system...")

    finally:
        if controller is not None:
            controller.close()

        if sensor_reader is not None:
            sensor_reader.close()

        print("System stopped.")


if __name__ == "__main__":
    main()
