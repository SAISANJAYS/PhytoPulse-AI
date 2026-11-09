"""Disease Spread Risk Engine for MPMC Leaf Disease Prediction.

Calculates the environmental Disease Spread Risk using the calibrated formula:
R = 100 × (0.53 × LWD_norm + 0.25 × Humidity_norm + 0.15 × Temperature_norm + 0.07 × Moisture_norm)

Risk categories (out of 100):
- Low:      0 to < 30
- Moderate: 30 to < 60
- High:     60 to 100
"""

from .config import (
    WEIGHT_LWD,
    WEIGHT_HUMIDITY,
    WEIGHT_TEMPERATURE,
    WEIGHT_MOISTURE,
    RISK_THRESHOLD_LOW_MAX,
    RISK_THRESHOLD_MODERATE_MAX,
    TEMP_NORM_MIN,
    TEMP_NORM_MAX,
    HUMIDITY_NORM_MIN,
    HUMIDITY_NORM_MAX,
    SOIL_RAW_WET,
    SOIL_RAW_DRY,
    RAIN_DETECTED_VALUE,
    CONFIDENCE_THRESHOLD,
)


def normalize_sensor_readings(packet: dict) -> dict:
    """Normalize raw sensor packet readings into bounded [0.0, 1.0] components.

    Parameters
    ----------
    packet : dict
        Sensor dictionary containing keys:
        - ``temperature_c`` (or ``temperature``)
        - ``humidity_pct`` (or ``humidity``)
        - ``rain_raw`` (or ``rain``, where 0 = rain detected, 1 = no rain)
        - ``soil_raw`` (or ``soil``, raw analog value ~300 wet to ~1000 dry)

    Returns
    -------
    dict
        Normalized values and raw readings:
        - ``lwd_norm``: 1.0 if rain/wetness detected, else 0.0 (rain sensor proxy)
        - ``humidity_norm``: normalized relative humidity [0.0, 1.0]
        - ``temperature_norm``: normalized temperature [0.0, 1.0]
        - ``moisture_norm``: normalized soil moisture based on calibrated wet/dry bounds [0.0, 1.0]
        - ``moisture_calibrated_pct``: soil moisture as calibrated percentage (0-100%)
        - ``temperature_c``, ``humidity_pct``, ``rain_raw``, ``soil_raw``
    """
    temp_c = float(packet.get("temperature_c") if "temperature_c" in packet else packet.get("temperature", 25.0))
    hum_pct = float(packet.get("humidity_pct") if "humidity_pct" in packet else packet.get("humidity", 50.0))
    rain_val = int(packet.get("rain_raw") if "rain_raw" in packet else packet.get("rain", 1))
    soil_val = int(packet.get("soil_raw") if "soil_raw" in packet else packet.get("soil", 800))

    # 1. LWD_norm: Leaf Wetness Duration estimated via rain sensor proxy
    # In hardware: 0 = rain/moisture detected, 1 = dry / no rain
    is_rain = (rain_val == RAIN_DETECTED_VALUE)
    lwd_norm = 1.0 if is_rain else 0.0

    # 2. Humidity_norm: Relative humidity bounded [0.0, 1.0]
    humidity_norm = max(
        0.0,
        min(1.0, (hum_pct - HUMIDITY_NORM_MIN) / (HUMIDITY_NORM_MAX - HUMIDITY_NORM_MIN))
    )

    # 3. Temperature_norm: Temperature bounded [0.0, 1.0]
    temp_norm = max(
        0.0,
        min(1.0, (temp_c - TEMP_NORM_MIN) / (TEMP_NORM_MAX - TEMP_NORM_MIN))
    )

    # 4. Moisture_norm: Calibrated soil moisture
    # Lower raw analog reading = wetter soil; higher raw reading = drier soil
    moisture_range = max(1.0, float(SOIL_RAW_DRY - SOIL_RAW_WET))
    moisture_norm = max(0.0, min(1.0, (SOIL_RAW_DRY - soil_val) / moisture_range))
    moisture_pct = round(moisture_norm * 100.0, 1)

    return {
        "lwd_norm": lwd_norm,
        "humidity_norm": humidity_norm,
        "temperature_norm": temp_norm,
        "moisture_norm": moisture_norm,
        "moisture_calibrated_pct": moisture_pct,
        "temperature_c": temp_c,
        "humidity_pct": hum_pct,
        "rain_raw": rain_val,
        "soil_raw": soil_val,
        "is_rain": is_rain,
    }


def calculate_environmental_risk_score(packet: dict) -> dict:
    """Calculate environmental disease spread risk score using the formula:

    R = 100 × (0.53 × LWD_norm + 0.25 × Humidity_norm + 0.15 × Temperature_norm + 0.07 × Moisture_norm)

    Returns
    -------
    dict
        Keys:
        - ``score``: float risk score in [0.0, 100.0] rounded to 1 decimal place
        - ``score_str``: formatted string e.g. "36.3 / 100"
        - ``category``: "Low", "Moderate", or "High"
        - ``level``: "LOW", "MODERATE", or "HIGH" (Arduino-compatible uppercase)
        - ``breakdown``: dictionary of normalized components
    """
    norms = normalize_sensor_readings(packet)

    r_score = 100.0 * (
        WEIGHT_LWD * norms["lwd_norm"]
        + WEIGHT_HUMIDITY * norms["humidity_norm"]
        + WEIGHT_TEMPERATURE * norms["temperature_norm"]
        + WEIGHT_MOISTURE * norms["moisture_norm"]
    )
    r_score = max(0.0, min(100.0, round(r_score, 1)))

    if r_score < RISK_THRESHOLD_LOW_MAX:
        category = "Low"
        level = "LOW"
    elif r_score < RISK_THRESHOLD_MODERATE_MAX:
        category = "Moderate"
        level = "MODERATE"
    else:
        category = "High"
        level = "HIGH"

    return {
        "score": r_score,
        "score_str": f"{r_score:.1f} / 100",
        "category": category,
        "level": level,
        "breakdown": norms,
    }


def calculate_disease_risk(prediction: dict | None, packet: dict) -> dict:
    """Evaluate Disease Spread Risk combining environmental score with leaf prediction.

    Parameters
    ----------
    prediction : dict or None
        Image prediction result dictionary containing:
        - ``class_name``: str
        - ``confidence``: float in [0.0, 1.0]
    packet : dict
        Sensor reading packet.

    Returns
    -------
    dict
        ``level``       – "LOW", "MODERATE", or "HIGH"
        ``risk``        – alias of level
        ``category``    – "Low", "Moderate", or "High"
        ``score``       – float risk score (0.0 to 100.0)
        ``score_str``   – string e.g. "36.3 / 100"
        ``reasons``     – list of assessment explanation strings
        ``disease``     – predicted disease class or "None"
        ``confidence``  – model confidence float
        ``is_healthy``  – bool or None
        ``breakdown``   – normalized sensor parameters
    """
    env_risk = calculate_environmental_risk_score(packet)
    score = env_risk["score"]
    level = env_risk["level"]
    category = env_risk["category"]
    norms = env_risk["breakdown"]

    # If no leaf prediction has been completed yet
    if not prediction:
        return {
            "level": "UNKNOWN",
            "risk": "UNKNOWN",
            "category": "Unknown",
            "score": score,
            "score_str": env_risk["score_str"],
            "reasons": [
                "Upload a leaf image and run disease prediction to assess disease spread risk."
            ],
            "disease": "None",
            "confidence": 0.0,
            "is_healthy": None,
            "breakdown": norms,
        }

    disease_name = str(
        prediction.get("class_name")
        or prediction.get("disease")
        or "Unknown"
    )
    confidence = float(prediction.get("confidence", 0.0))
    is_healthy = "healthy" in disease_name.lower()

    reasons: list[str] = []

    # 1. Image classification assessment
    if is_healthy:
        reasons.append(
            f"Leaf image diagnosed as Healthy with {confidence * 100:.1f}% confidence. "
            "No active disease is currently spreading from this plant."
        )
    else:
        reasons.append(
            f"Active disease '{disease_name}' predicted with {confidence * 100:.1f}% confidence."
        )

    # 2. Environmental Disease Spread Risk assessment
    reasons.append(
        f"Environmental Disease Spread Risk calculated as {category} ({score:.1f} / 100) "
        "using current microclimate conditions."
    )

    # 3. Factor explanations
    # Leaf Wetness Duration / Rain proxy
    if norms["is_rain"]:
        reasons.append(
            "Rain detected via sensor (LWD_norm = 1.00); standing water creates ideal conditions for "
            "fungal spore germination and rain-splash dispersal to neighboring foliage."
        )
    else:
        reasons.append(
            "No rain detected (LWD_norm = 0.00); absence of free water limits splash spore dissemination."
        )

    # Humidity
    hum = norms["humidity_pct"]
    hum_norm = norms["humidity_norm"]
    if hum >= 80.0:
        reasons.append(
            f"High relative humidity ({hum:.1f}%, Humidity_norm = {hum_norm:.2f}) sustains spore viability and lesion expansion."
        )
    elif hum >= 60.0:
        reasons.append(
            f"Moderate relative humidity ({hum:.1f}%, Humidity_norm = {hum_norm:.2f}) supports pathogen persistence."
        )
    else:
        reasons.append(
            f"Low relative humidity ({hum:.1f}%, Humidity_norm = {hum_norm:.2f}) suppresses rapid spore sporulation."
        )

    # Temperature
    temp = norms["temperature_c"]
    temp_norm = norms["temperature_norm"]
    if 18.0 <= temp <= 32.0:
        reasons.append(
            f"Temperature ({temp:.1f}°C, Temperature_norm = {temp_norm:.2f}) falls within optimal fungal metabolic range."
        )
    else:
        reasons.append(
            f"Temperature ({temp:.1f}°C, Temperature_norm = {temp_norm:.2f}) is outside peak pathogen transmission kinetics."
        )

    # Soil moisture
    soil_raw = norms["soil_raw"]
    moisture_pct = norms["moisture_calibrated_pct"]
    reasons.append(
        f"Soil moisture reading: {soil_raw} raw ({moisture_pct:.1f}% calibrated moisture, Moisture_norm = {norms['moisture_norm']:.2f})."
    )

    return {
        "level": level,
        "risk": level,
        "category": category,
        "score": score,
        "score_str": env_risk["score_str"],
        "reasons": reasons,
        "disease": disease_name,
        "confidence": confidence,
        "is_healthy": is_healthy,
        "breakdown": norms,
    }


# Backwards compatibility aliases
calculate_risk = calculate_disease_risk


def assess_environment(packet: dict, prediction: dict | None = None) -> dict:
    return calculate_disease_risk(prediction, packet)
