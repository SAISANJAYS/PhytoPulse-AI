
import time

import serial
import serial.tools.list_ports

from .config import SERIAL_TIMEOUT, SENSOR_POLL_TIMEOUT, SENSOR_READ_RETRIES


def available_ports():
    """Return a list of currently available serial port names."""
    return [port.device for port in serial.tools.list_ports.comports()]


class ArduinoSensorReader:
    """Manages a serial connection to the Arduino and reads sensor packets.

    Typical usage
    -------------
    ::

        reader = ArduinoSensorReader("COM5", 9600)
        reader.connect()          # opens port, waits for reset
        reader.start_prediction() # sends START\\n, waits for STATUS,STARTED
        packet = reader.read_sensor_packet()
        reader.stop_prediction()  # sends STOP\\n
        reader.close()

    Parameters
    ----------
    port : str
        Serial port name, e.g. ``"COM5"`` or ``"/dev/ttyUSB0"``.
    baud_rate : int
        Baud rate matching the Arduino sketch (9600).
    """

    def __init__(self, port: str, baud_rate: int = 9600):
        self.port = port
        self.baud_rate = baud_rate
        self._connection: serial.Serial | None = None
        self.latest_sensor_packet: dict | None = None
        self.latest_risk_score: float | None = None
        self.latest_calculated_risk: str | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        """Open the serial port and wait for the Arduino to finish booting.

        The Arduino Uno resets when a new serial connection is opened; a
        2-second pause ensures it is ready before any commands are sent.
        Calling ``connect()`` on an already-open port is a no-op.
        """
        if self._connection and self._connection.is_open:
            return

        self._connection = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=SERIAL_TIMEOUT,
        )
        # Arduino Uno resets on DTR assertion – wait for it to boot.
        time.sleep(2)
        self._connection.reset_input_buffer()

    def close(self):
        """Close the serial port if it is open."""
        if self._connection and self._connection.is_open:
            self._connection.close()

    def _require_open(self):
        if self._connection is None or not self._connection.is_open:
            raise RuntimeError("Not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Prediction lifecycle commands
    # ------------------------------------------------------------------

    def start_prediction(self):
        """Send ``START`` to the Arduino and confirm it responded.

        The Arduino replies with ``STATUS,STARTED`` before it begins
        sending sensor data every ``SENSOR_INTERVAL`` milliseconds.
        This method waits up to ``SENSOR_READ_RETRIES`` lines for that
        acknowledgement; it does **not** block for the first data packet.

        Raises
        ------
        RuntimeError
            If the acknowledgement is not received within the retry limit.
        """
        self._require_open()
        self._connection.reset_input_buffer()
        self.send_command("START")

        for _ in range(SENSOR_READ_RETRIES):
            line = self._readline_stripped()
            if line == "STATUS,STARTED":
                return
            # Process any other lines that might arrive
            self._process_incoming_line(line)

        raise RuntimeError(
            "Arduino did not confirm START within the expected number of "
            f"lines ({SENSOR_READ_RETRIES}). "
            "Check wiring, baud rate, and that the correct sketch is uploaded."
        )

    def stop_prediction(self):
        """Send ``STOP`` to the Arduino.

        The Arduino will stop sending sensor data after receiving this.
        """
        self._require_open()
        self.send_command("STOP")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _process_incoming_line(self, line: str) -> dict | None:
        """Process a stripped line from the Arduino serial stream.

        Recognises:
        - ``STATUS,...`` messages (e.g. ``STATUS,STARTED``, ``STATUS,STOPPED``)
        - ``ERROR,...`` messages (e.g. ``ERROR,DHT``)
        - ``RISK_SCORE,<float>`` (e.g. ``RISK_SCORE,28.8``) -> updates ``latest_risk_score``
        - ``CALCULATED_RISK,<level>`` (e.g. ``CALCULATED_RISK,LOW``) -> updates ``latest_calculated_risk``
        - ``temperature,humidity,rain,soil`` (e.g. ``24.7,78.1,1,1022``) -> updates
          ``latest_sensor_packet`` and returns the dictionary.

        Returns
        -------
        dict | None
            A parsed 4-value sensor packet, or ``None`` for non-packet lines.
        """
        if not line:
            return None

        # Ignore STATUS lines
        if line.startswith("STATUS,"):
            return None

        # Ignore sensor read errors
        if line.startswith("ERROR,"):
            return None

        # Parse RISK_SCORE lines if emitted by Arduino
        if line.startswith("RISK_SCORE,"):
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    self.latest_risk_score = float(parts[1].strip())
                except ValueError:
                    pass
            return None

        # Parse CALCULATED_RISK lines if emitted by Arduino
        if line.startswith("CALCULATED_RISK,"):
            parts = line.split(",")
            if len(parts) >= 2:
                risk_level = parts[1].strip().upper()
                if risk_level in ("LOW", "MODERATE", "HIGH", "UNKNOWN"):
                    self.latest_calculated_risk = risk_level
            return None

        # Attempt to parse as 4-value sensor packet
        packet = self._parse_data_line(line)
        if packet is not None:
            self.latest_sensor_packet = packet
            return packet

        return None

    def read_sensor_packet(self) -> dict | None:
        """Read the next valid sensor packet from the Arduino.

        Skips status, error, and risk lines and retries up to ``SENSOR_READ_RETRIES`` times.
        Updates ``latest_risk_score`` and ``latest_calculated_risk`` if encountered.

        Returns
        -------
        dict | None
            Parsed sensor packet dict, or ``None`` if no valid packet was received.
        """
        self._require_open()

        for attempt in range(SENSOR_READ_RETRIES):
            try:
                line = self._readline_stripped()
            except serial.SerialException as exc:
                raise RuntimeError(f"Serial read error: {exc}") from exc

            packet = self._process_incoming_line(line)
            if packet is not None:
                return packet

        return None

    def try_read_packet(self) -> dict | None:
        """Non-blocking read attempt for use in polling."""
        self._require_open()

        original_timeout = self._connection.timeout
        self._connection.timeout = SENSOR_POLL_TIMEOUT
        try:
            try:
                line = self._readline_stripped()
            except serial.SerialException:
                return None
        finally:
            self._connection.timeout = original_timeout

        return self._process_incoming_line(line)

    def read_latest_packet(self, timeout: float = SENSOR_POLL_TIMEOUT) -> dict | None:
        """Read and return the newest valid sensor packet from the serial connection.

        Drains all available lines from the input buffer, correctly processes
        ``STATUS,*``, ``ERROR,*``, ``RISK_SCORE,*``, and ``CALCULATED_RISK,*``
        lines without confusing them for sensor packets, and returns the newest
        valid 4-value sensor packet.
        """
        self._require_open()

        original_timeout = self._connection.timeout
        self._connection.timeout = timeout
        latest_packet = None

        try:
            while True:
                try:
                    line = self._readline_stripped()
                except (serial.SerialException, OSError):
                    break

                if not line:
                    break

                packet = self._process_incoming_line(line)
                if packet is not None:
                    latest_packet = packet

                if self._connection.in_waiting == 0:
                    break
        finally:
            self._connection.timeout = original_timeout

        return latest_packet

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def send_command(self, command: str):
        """Send a newline-terminated command string to the Arduino.

        Parameters
        ----------
        command : str
            Command string without trailing newline, e.g. ``"START"``,
            ``"STOP"``, ``"HIGH"``.
        """
        self._require_open()
        self._connection.write(f"{command}\n".encode("utf-8"))
        self._connection.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _readline_stripped(self) -> str:
        """Read one line and return it stripped of whitespace."""
        raw = self._connection.readline()
        return raw.decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _parse_data_line(line: str) -> dict | None:
        """Try to parse *line* as a ``temperature,humidity,rain,soil`` packet.

        Parameters
        ----------
        line : str
            A single stripped line from the Arduino.

        Returns
        -------
        dict | None
            Parsed packet dict or ``None`` if the line is not valid data.

        Examples
        --------
        >>> ArduinoSensorReader._parse_data_line("23.9,78.7,1,1023")
        {'temperature_c': 23.9, 'humidity_pct': 78.7, 'rain_raw': 1, 'soil_raw': 1023, ...}
        >>> ArduinoSensorReader._parse_data_line("STATUS,STARTED") is None
        True
        """
        parts = line.split(",")
        if len(parts) != 4:
            return None

        try:
            temperature_c = float(parts[0])
            humidity_pct = float(parts[1])
            rain_raw = int(parts[2])
            soil_raw = int(parts[3])
        except ValueError:
            return None

        # Sanity-check ranges
        if not (-10 <= temperature_c <= 80):
            return None
        if not (0 <= humidity_pct <= 100):
            return None
        if rain_raw not in (0, 1):
            return None
        if not (0 <= soil_raw <= 1023):
            return None

        return {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rain_raw": rain_raw,
            "soil_raw": soil_raw,
            "timestamp": time.time(),
        }
