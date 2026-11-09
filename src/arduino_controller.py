

def send_risk_to_arduino(reader, risk_level: str):
    """Send a risk-level command string to the Arduino via *reader*.

    Parameters
    ----------
    reader : ArduinoSensorReader
        An already-connected ``ArduinoSensorReader`` instance.
    risk_level : str
        One of ``"LOW"``, ``"MODERATE"``, ``"HIGH"``, or ``"UNKNOWN"``.
    """
    allowed_levels = {"LOW", "MODERATE", "HIGH", "UNKNOWN"}

    if risk_level not in allowed_levels:
        raise ValueError(
            f"Unsupported risk level '{risk_level}'. "
            f"Must be one of {sorted(allowed_levels)}."
        )

    reader.send_command(risk_level)


def send_manual_alert(reader):
    """Send MANUAL_ALERT command string to the Arduino via *reader*."""
    reader.send_command("MANUAL_ALERT")


def send_reset_alert(reader):
    """Send RESET_ALERT command string to the Arduino via *reader*."""
    reader.send_command("RESET_ALERT")

