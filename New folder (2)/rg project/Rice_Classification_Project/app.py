import streamlit as st
import numpy as np
import cv2
from prediction import RicePredictor

# Streamlit Page Config
st.set_page_config(
    page_title='Professional Rice Classification',
    page_icon='🌾',
    layout='centered'
)

# Initialize predictor
if 'predictor' not in st.session_state:
    st.session_state.predictor = RicePredictor()

st.title("🌾 Rice Image Classification")
st.write("Upload a clear image of rice grains for AI-based analysis.")

uploaded_file = st.file_uploader(
    "Choose a high-quality image (JPG, PNG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # 🔥 ALWAYS allow analysis (NO blocking)
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing rice grains..."):
            try:
                result = st.session_state.predictor.process_and_predict(image_rgb)

                st.subheader("📊 Analysis Results")

                # ❗ Case 1: No grains detected
                if result["status"] == "NO_GRAINS":
                    st.error("❌ No rice grains detected in the image.")
                    st.info("Try a clearer image with visible rice grains.")
                
                else:
                    # ✅ Show prediction
                    st.success(f"✅ Prediction: **{result['predicted_class']}**")

                    # ✅ Confidence
                    if "confidence" in result:
                        st.write(f"**Confidence:** {result['confidence'] * 100:.2f}%")

                    # ✅ Grain count
                    if "grain_count" in result:
                        st.write(f"**Grains detected:** {result['grain_count']}")

                    # 🔥 Show individual grain predictions (advanced)
                    if "all_predictions" in result:
                        st.markdown("---")
                        st.markdown("### 🔍 Individual Grain Predictions")

                        for i, item in enumerate(result["all_predictions"][:10]):  # limit to 10
                            st.write(
                                f"Grain {i+1}: {item['class']} ({item['confidence']*100:.1f}%)"
                            )

            except Exception as e:
                st.error(f"⚠️ Error during prediction: {str(e)}")

# Footer
st.markdown("---")
st.caption("YOLO + CNN Based Rice Classification System 🚀")