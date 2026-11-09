"""MPMC Leaf Disease Prediction – Streamlit dashboard.

Auto-refresh design
-------------------
When prediction is active, the sensor readings and Disease Spreading Risk Level
section automatically refresh every 5 seconds using Streamlit's native
``@st.fragment(run_every=5)``.

On each refresh:
1. Python reads directly from the existing Arduino serial connection via
   ``reader.read_latest_packet()`` using a short non-blocking timeout.
2. Any waiting buffer lines (including STATUS, ERROR, RISK_SCORE, and
   CALCULATED_RISK lines) are drained without lag.
3. If a valid new packet arrives, the metrics, Risk Score, Disease Spreading Risk Level,
   and "Last update" timestamp are automatically updated.
4. When Stop Prediction is clicked, STOP is sent to the Arduino and the
   automatic refresh stops immediately (``run_every=None``).
"""

import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    DEFAULT_SERIAL_PORT,
    SERIAL_BAUD_RATE,
    SENSOR_INTERVAL,
)
from src.image_predictor import LeafDiseasePredictor
from src.sensor_reader import ArduinoSensorReader, available_ports
from src.risk_engine import (
    calculate_disease_risk,
    calculate_environmental_risk_score,
)
from src.arduino_controller import (
    send_risk_to_arduino,
    send_manual_alert,
    send_reset_alert,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MPMC Leaf Disease Prediction",
    page_icon="🌿",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "sensor_packet": None,
    "last_update_ts": None,
    "arduino_reader": None,
    "last_prediction": None,
    "uploaded_leaf_image": None,
    "_last_uploaded_file_id": None,
    "disease_risk": None,
    "environmental_risk": None,
    "prediction_active": False,
    "manual_alert_active": False,
    "_reader_error": None,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ---------------------------------------------------------------------------
# Sidebar – Arduino connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Arduino connection")

    try:
        ports = available_ports()
    except Exception as e:
        ports = []
        st.warning(f"Could not list serial ports: {e}")

    port_options = list(dict.fromkeys(ports + [DEFAULT_SERIAL_PORT]))

    chosen_port = st.selectbox(
        "Serial port",
        port_options,
        index=(
            port_options.index(DEFAULT_SERIAL_PORT)
            if DEFAULT_SERIAL_PORT in port_options
            else 0
        ),
    )

    if st.button("Connect to Arduino", use_container_width=True):
        try:
            old_reader = st.session_state.arduino_reader
            if old_reader:
                try:
                    if st.session_state.prediction_active:
                        old_reader.stop_prediction()
                except Exception:
                    pass
                old_reader.close()

            reader = ArduinoSensorReader(chosen_port, SERIAL_BAUD_RATE)
            reader.connect()

            st.session_state.arduino_reader = reader
            st.session_state.sensor_packet = None
            st.session_state.last_update_ts = None
            st.session_state.environmental_risk = None
            st.session_state.disease_risk = None
            st.session_state.prediction_active = False
            st.session_state._reader_error = None

            st.success(f"Connected to {chosen_port}")
            st.rerun()

        except Exception as e:
            st.session_state.arduino_reader = None
            st.error(f"Could not connect: {e}")

    if st.button("Disconnect", use_container_width=True):
        reader = st.session_state.arduino_reader

        if reader:
            try:
                if st.session_state.prediction_active:
                    reader.stop_prediction()
            except Exception:
                pass
            try:
                reader.close()
            except Exception as e:
                st.warning(f"Error while closing connection: {e}")

        st.session_state.arduino_reader = None
        st.session_state.sensor_packet = None
        st.session_state.last_update_ts = None
        st.session_state.prediction_active = False
        st.session_state.manual_alert_active = False
        st.session_state.disease_risk = None
        st.session_state.environmental_risk = None
        st.session_state._reader_error = None

        st.info("Arduino disconnected.")
        st.rerun()

    st.caption("Close Arduino IDE Serial Monitor before connecting here.")

    # Live status badge
    if st.session_state.prediction_active:
        st.success(f"🟢 Live – auto-refreshing every {SENSOR_INTERVAL} s")
    elif st.session_state.arduino_reader:
        st.info("⚪ Connected – press ▶ Start Prediction")
    else:
        st.warning("🔴 Not connected")

# ---------------------------------------------------------------------------
# Main content – two-column layout
# ---------------------------------------------------------------------------
left, right = st.columns(2)

# ── Left column: leaf image prediction ──────────────────────────────────────
with left:
    st.subheader("1. Leaf image")

    upload = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_image_uploader",
    )

    if upload is not None:
        current_file_id = f"{upload.name}_{upload.size}"
        if st.session_state.get("_last_uploaded_file_id") != current_file_id:
            st.session_state["_last_uploaded_file_id"] = current_file_id
            st.session_state["last_prediction"] = None
            st.session_state["disease_risk"] = None

        try:
            upload.seek(0)
            uploaded_image = Image.open(upload).convert("RGB")
            st.session_state["uploaded_leaf_image"] = uploaded_image
        except Exception as e:
            st.session_state["uploaded_leaf_image"] = None
            st.session_state["last_prediction"] = None
            st.error(f"Could not open uploaded image: {e}")
    else:
        st.session_state["uploaded_leaf_image"] = None
        st.session_state["_last_uploaded_file_id"] = None

    current_image = st.session_state.get("uploaded_leaf_image")

    if current_image is not None:
        st.image(
            current_image,
            caption="Uploaded leaf",
            use_container_width=True,
        )

    predict_clicked = st.button(
        "Predict Disease",
        type="primary",
        disabled=(current_image is None),
        use_container_width=True,
    )

    if predict_clicked:
        if current_image is None:
            st.warning("Please upload a valid leaf image first.")
        else:
            try:
                with st.spinner("Running CNN model inference…"):
                    predictor = LeafDiseasePredictor(
                        MODEL_PATH,
                        CLASS_NAMES_PATH,
                    )
                    result = predictor.predict(current_image)
                st.session_state.last_prediction = result

                # When a new image is predicted, recalculate disease risk using latest sensor readings
                if st.session_state.sensor_packet is not None:
                    risk = calculate_disease_risk(
                        result, st.session_state.sensor_packet
                    )
                    st.session_state.disease_risk = risk

                    # Send calculated risk level to Arduino
                    reader = st.session_state.arduino_reader
                    if reader is not None and st.session_state.prediction_active:
                        try:
                            send_risk_to_arduino(reader, risk["level"])
                        except Exception:
                            pass
                else:
                    st.session_state.disease_risk = None

            except Exception as e:
                st.session_state.last_prediction = None
                st.session_state.disease_risk = None
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Check the model files, TensorFlow installation, "
                    "class_names.json, and image preprocessing."
                )

    result = st.session_state.last_prediction

    if result:
        st.divider()
        st.markdown("### Leaf Disease Detection")

        is_healthy = "healthy" in result["class_name"].lower()
        status_label = "Healthy" if is_healthy else "Diseased"
        status_color = "🟢" if is_healthy else "🔴"

        col_stat, col_conf = st.columns(2)
        col_stat.metric("Predicted Status", f"{status_color} {status_label}")
        col_conf.metric("CNN Confidence", f"{result['confidence'] * 100:.2f}%")

        if not is_healthy:
            clean_name = result["class_name"].replace("___", " – ").replace("_", " ")
            st.error(f"**Specific Disease:** {clean_name}")
        else:
            st.success("**Predicted Condition:** Healthy leaf (no disease detected)")

        st.write("**Top predictions:**")
        st.dataframe(
            [
                {
                    "Class": item["class_name"].replace("___", " – ").replace("_", " "),
                    "Score (%)": round(item["score"] * 100, 2),
                }
                for item in result["all_scores"][:5]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Upload a leaf image and click 'Predict Disease' to run CNN classification.")

# ── Right column: live sensor readings & disease spreading risk level ───────
with right:
    st.subheader("2. Sensor readings")

    # When prediction is active, fragment runs every SENSOR_INTERVAL seconds (5 s).
    # When stopped, run_interval is None so the automatic refresh stops.
    run_interval = SENSOR_INTERVAL if st.session_state.prediction_active else None

    @st.fragment(run_every=run_interval)
    def render_sensor_readings():
        reader = st.session_state.arduino_reader
        packet = st.session_state.sensor_packet
        last_ts = st.session_state.last_update_ts

        if reader is not None:

            # ── Start / Stop controls ─────────────────────────────────────
            col_start, col_stop = st.columns(2)

            if col_start.button(
                "▶ Start Prediction",
                use_container_width=True,
                disabled=st.session_state.prediction_active,
            ):
                try:
                    with st.spinner("Sending START to Arduino…"):
                        reader.start_prediction()

                    st.session_state.prediction_active = True
                    st.session_state.sensor_packet = None
                    st.session_state.last_update_ts = None
                    st.session_state.environmental_risk = None
                    st.session_state._reader_error = None

                    # Perform an initial read attempt to catch immediate data
                    try:
                        new_packet = reader.read_latest_packet(timeout=1.0)
                        if new_packet is not None:
                            st.session_state.sensor_packet = new_packet
                            st.session_state.last_update_ts = new_packet["timestamp"]
                            env_risk = calculate_environmental_risk_score(new_packet)
                            st.session_state.environmental_risk = env_risk

                            if st.session_state.last_prediction is not None:
                                risk = calculate_disease_risk(
                                    st.session_state.last_prediction, new_packet
                                )
                                st.session_state.disease_risk = risk
                                send_risk_to_arduino(reader, risk["level"])
                            else:
                                send_risk_to_arduino(reader, env_risk["level"])
                    except Exception:
                        pass

                    st.rerun()

                except Exception as e:
                    st.error(f"Could not start prediction: {e}")

            if col_stop.button(
                "⏹ Stop Prediction",
                use_container_width=True,
                disabled=not st.session_state.prediction_active,
            ):
                try:
                    reader.stop_prediction()
                except Exception as e:
                    st.warning(f"STOP command failed: {e}")

                st.session_state.prediction_active = False
                st.session_state.manual_alert_active = False
                st.info("Prediction stopped. Sensor values frozen.")
                st.rerun()

            # ── Background error banner ───────────────────────────────────
            reader_err = st.session_state._reader_error
            if reader_err:
                st.error(f"Serial read error: {reader_err}")

            # ── Active serial read on each refresh ─────────────────────────
            # Actually read from the existing serial connection before display
            if st.session_state.prediction_active:
                try:
                    new_packet = reader.read_latest_packet()
                    if new_packet is not None:
                        st.session_state.sensor_packet = new_packet
                        st.session_state.last_update_ts = new_packet["timestamp"]
                        st.session_state._reader_error = None

                        # Calculate environmental risk score using the formula
                        env_risk = calculate_environmental_risk_score(new_packet)
                        st.session_state.environmental_risk = env_risk

                        if st.session_state.last_prediction is not None:
                            risk = calculate_disease_risk(
                                st.session_state.last_prediction, new_packet
                            )
                            st.session_state.disease_risk = risk
                            try:
                                send_risk_to_arduino(reader, risk["level"])
                            except Exception:
                                pass
                        else:
                            st.session_state.disease_risk = None
                            try:
                                send_risk_to_arduino(reader, env_risk["level"])
                            except Exception:
                                pass
                except Exception as exc:
                    st.session_state._reader_error = str(exc)

            # ── Live 5-card sensor metrics ─────────────────────────────────
            packet = st.session_state.sensor_packet
            last_ts = st.session_state.last_update_ts

            if packet:
                # Ensure environmental risk is computed
                if st.session_state.environmental_risk is None:
                    st.session_state.environmental_risk = calculate_environmental_risk_score(packet)

                env_risk = st.session_state.environmental_risk
                risk_score_display = env_risk["score_str"]

                col1, col2 = st.columns(2)

                temperature = packet.get("temperature_c")
                humidity    = packet.get("humidity_pct")
                rain        = packet.get("rain_raw")
                soil        = packet.get("soil_raw")

                # 1. Temperature
                col1.metric(
                    "🌡 Temperature",
                    f"{temperature:.1f} °C" if temperature is not None else "N/A",
                )
                # 2. Humidity
                col2.metric(
                    "💧 Humidity",
                    f"{humidity:.1f}%" if humidity is not None else "N/A",
                )
                # 3. Rain sensor
                col1.metric(
                    "🌧 Rain sensor",
                    ("Rain detected" if rain == 0 else "No rain detected")
                    if rain is not None else "N/A",
                )
                # 4. Soil moisture (raw analog value with clear label)
                col2.metric(
                    "🌱 Soil moisture",
                    f"{soil} (raw)" if soil is not None else "N/A",
                    help="Raw analog sensor reading (0 = saturated wet, 1023 = completely dry air). Calibrated percentage is used in the risk formula.",
                )
                # 5. Risk Score out of 100
                col1.metric(
                    "📊 Risk Score",
                    risk_score_display,
                    help="Environmental Disease Spread Risk score out of 100 based on LWD proxy, humidity, temperature, and calibrated moisture.",
                )

                if last_ts:
                    st.caption(
                        f"Last sensor update: {time.strftime('%H:%M:%S', time.localtime(last_ts))}"
                    )

            else:
                if st.session_state.prediction_active:
                    st.info(
                        "⏳ Waiting for first sensor packet from Arduino "
                        f"(sent every {SENSOR_INTERVAL} s after START)…"
                    )
                else:
                    st.info("Press **▶ Start Prediction** to begin reading sensors.")

        else:
            st.info("Connect the Arduino from the sidebar to read live sensors.")

        # ── Color-coded Disease Spreading Risk Level section ─────────────────
        st.divider()
        st.subheader("Disease Spreading Risk Level")

        if packet is None:
            st.info("⏳ Waiting for valid sensor readings")
        else:
            env_risk = st.session_state.environmental_risk
            if env_risk is None:
                env_risk = calculate_environmental_risk_score(packet)
                st.session_state.environmental_risk = env_risk

            score = env_risk["score"]
            category = env_risk["category"]

            # Threshold styling:
            # Low: 0 to < 30 (Green)
            # Moderate: 30 to < 60 (Yellow/Amber)
            # High: 60 to 100 (Red)
            if category == "Low":
                accent_color = "#10b981"
                bg_color = "rgba(16, 185, 129, 0.14)"
                icon = "🟢"
            elif category == "Moderate":
                accent_color = "#f59e0b"
                bg_color = "rgba(245, 158, 11, 0.14)"
                icon = "🟡"
            else:
                accent_color = "#ef4444"
                bg_color = "rgba(239, 68, 68, 0.18)"
                icon = "🔴"

            risk_card_html = f"""
            <div style="
                border: 1.5px solid {accent_color};
                background-color: {bg_color};
                border-radius: 10px;
                padding: 16px 20px;
                margin: 8px 0 16px 0;
            ">
                <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {accent_color};">
                    Disease Spreading Risk Level
                </div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #ffffff; margin: 4px 0 8px 0;">
                    {icon} {category} ({score:.1f} / 100)
                </div>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45;">
                    This score estimates environmental conditions favorable to disease spread and does not guarantee that disease will spread.
                </div>
            </div>
            """
            st.markdown(risk_card_html, unsafe_allow_html=True)

            # Disease-specific assessment logic
            if st.session_state.last_prediction is None:
                st.info("Upload a leaf image and run disease prediction to assess disease spread risk.")
            else:
                risk = st.session_state.disease_risk
                if risk is None:
                    risk = calculate_disease_risk(st.session_state.last_prediction, packet)
                    st.session_state.disease_risk = risk

                if risk["is_healthy"]:
                    st.success(
                        "🌿 **Leaf Status: Healthy.** The CNN classifies this leaf as healthy. "
                        "No specific disease is spreading from this plant. "
                        f"The environmental risk above ({category}) represents general microclimate favorability "
                        "for foliar pathogens if disease spores were introduced."
                    )
                else:
                    st.warning(
                        f"⚠️ **Assessment for {risk['disease']}:** Active disease confirmed with "
                        f"{risk['confidence'] * 100:.1f}% confidence. Environmental conditions favor spread at a **{category}** level."
                    )

                with st.expander("Detailed Environmental Assessment Factors", expanded=False):
                    for reason in risk.get("reasons", []):
                        st.write(f"• {reason}")

                if reader is not None and st.button("📡 Send risk alert to Arduino"):
                    send_risk_to_arduino(reader, risk["level"])
                    st.success(f"Sent {risk['level']} alert to Arduino.")

        # ── Manual Emergency Alert Controls ──────────────────────────────────
        st.divider()
        st.subheader("Manual Emergency Alert")

        col_alert, col_reset = st.columns(2)

        if col_alert.button(
            "🚨 Send Manual Alert",
            use_container_width=True,
            type="primary",
        ):
            if reader is None:
                st.warning("Arduino is not connected. Connect the Arduino from the sidebar before sending an alert.")
            elif not st.session_state.prediction_active:
                st.warning("Prediction is not active. Connect Arduino and click '▶ Start Prediction' before sending a manual emergency alert.")
            else:
                try:
                    send_manual_alert(reader)
                    st.session_state.manual_alert_active = True
                except Exception as e:
                    st.error(f"Failed to send MANUAL_ALERT command to Arduino: {e}")

        if col_reset.button(
            "🔕 Reset Manual Alert",
            use_container_width=True,
        ):
            if reader is None:
                st.warning("Arduino is not connected. Connect the Arduino from the sidebar before resetting alerts.")
            elif not st.session_state.prediction_active:
                st.warning("Prediction is not active. Connect Arduino and start prediction to manage alerts.")
            else:
                try:
                    send_reset_alert(reader)
                    st.session_state.manual_alert_active = False
                    st.success("🔕 Alert reset sent. Indicators restored to current automatic risk level.")
                except Exception as e:
                    st.error(f"Failed to send RESET_ALERT command to Arduino: {e}")

        if st.session_state.manual_alert_active:
            st.error("**⚠️ EMERGENCY ALERT ON**")

    render_sensor_readings()

# ---------------------------------------------------------------------------
# Section: How the result is used
# ---------------------------------------------------------------------------
st.divider()
st.subheader("How the result is used")

st.write(
    "The CNN model analyzes the uploaded leaf image to identify whether the leaf is healthy "
    "or diseased and, when diseased, predicts the specific disease with a confidence score. "
    "Separately, the Disease Spread Risk engine uses environmental sensor readings to estimate "
    "conditions that may favor the spread or progression of disease to other plants. The image "
    "classification and environmental risk assessment are separate results. Environmental "
    "conditions do not change the CNN's predicted disease."
)

st.markdown("#### Disease Spread Risk Formula")

# Styled formula container matching the dark dashboard aesthetic
formula_html = """
<div style="
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0 16px 0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #38bdf8;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
">
    R = 100 × (0.53 × LWD_norm + 0.25 × Humidity_norm + 0.15 × Temperature_norm + 0.07 × Moisture_norm)
</div>
"""
st.markdown(formula_html, unsafe_allow_html=True)

st.markdown(
    """
**Variable explanations:**
- **`LWD_norm`**: Normalized leaf-wetness duration. Since the hardware uses a rain sensor rather than a dedicated leaf-wetness sensor, this is calculated as an estimate or proxy (where detected rain = 1.00 and no rain = 0.00).
- **`Humidity_norm`**: Normalized relative humidity bounded between 0.0 and 1.0 based on current ambient humidity readings.
- **`Temperature_norm`**: Normalized temperature bounded between 0.0 and 1.0 across standard agricultural ranges (10°C to 40°C).
- **`Moisture_norm`**: Normalized soil moisture based on calibrated sensor readings (accounting for analog dry/wet calibration points), not the raw analog value treated as a percentage.

> **Note:** This is a demonstration risk estimate and is not a validated epidemiological prediction.
"""
)