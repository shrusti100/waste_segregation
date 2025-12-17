import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64

# Load trained model
model = tf.keras.models.load_model("waste_classifier_mobilenetv2.h5")

# Define class labels
class_labels = ['battery', 'biological', 'clothes', 'metal', 'paper', 'plastic']

# PAGE CONFIG
st.set_page_config(
    page_title="♻️ Waste Segregation Classifier",
    page_icon="♻️",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 2.3em;
            font-weight: bold;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .predCard {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
            padding: 15px;
            text-align: center;
            width: 100%;
            height: 370px;
        }
        .predCard img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
        }
        .predClass {
            font-weight: 600;
            font-size: 1.1em;
            color: #0072ff;
        }
    </style>
""", unsafe_allow_html=True)

#HEADER
st.markdown("<h1 class='title'>♻️ Waste Segregation Classifier</h1>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("ℹ️ About the App")
st.sidebar.write("""
This app uses **Deep Learning CNN Model** to classify waste into:
- Battery  
- Biological  
- Clothes  
- Metal  
- Paper  
- Plastic  
""")

# MULTI FILE UPLOAD
uploaded_files = st.file_uploader(
    " Upload one or more images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# CLASSIFICATION
if uploaded_files:
    cols = st.columns(3)  # ---------> 3 COLUMNS FIXED

    index = 0
    for uploaded_file in uploaded_files:
        col = cols[index % 3]

        with col:
            img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            # Card HTML
            st.markdown(f"""
                <div class='predCard'>
                    <img src='data:image/png;base64,{img_b64}' />
                    <div class='predClass'>{predicted_class.upper()}</div>
                    <div style='color:gray;'>Confidence: {confidence:.2f}%</div>
                    <div style='width:100%; background:#e9ecef; border-radius:10px; height:10px;'>
                        <div style='width:{confidence:.2f}%; background:#0072ff; height:10px; border-radius:10px;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        index += 1

else:
    st.info(" Upload one or more images to start classification.")
