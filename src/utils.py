
import csv
from datetime import datetime

from .config import LOGS_DIR


LOG_FILE = LOGS_DIR / "prediction_log.csv"


def display_result(prediction, sensors, risk_result):
    print("\n" + "=" * 48)
    print("       MPMC LEAF DISEASE PREDICTION")
    print("=" * 48)

    print(
        f"Predicted class : {prediction['disease']}"
    )
    print(
        f"Confidence      : {prediction['confidence']:.2%}"
    )
    print(f"Temperature     : {sensors['temperature']:.1f} °C")
    print(f"Humidity        : {sensors['humidity']:.1f} %")
    print(f"Rain sensor     : {sensors['rain']}")
    print(f"Soil reading    : {sensors['soil']}")
    print(f"Risk level      : {risk_result['risk']}")
    print(f"Risk score      : {risk_result['score']}")

    print("Reasons:")
    for reason in risk_result["reasons"]:
        print(f"  - {reason}")

    print("=" * 48)


def log_result(prediction, sensors, risk_result):
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "image_class",
                "confidence",
                "temperature",
                "humidity",
                "rain",
                "soil",
                "risk",
                "risk_score",
            ],
        )

        if not file_exists or LOG_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_class": prediction["disease"],
            "confidence": round(prediction["confidence"], 4),
            "temperature": sensors["temperature"],
            "humidity": sensors["humidity"],
            "rain": sensors["rain"],
            "soil": sensors["soil"],
            "risk": risk_result["risk"],
            "risk_score": risk_result["score"],
        })
