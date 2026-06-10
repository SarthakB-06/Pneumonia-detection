import streamlit as st
import requests
from PIL import Image

# Configure the page appearance
st.set_page_config(
    page_title="Pneumonia Detection AI", 
    page_icon="🫁",
    layout="centered"
)

# Set the API endpoint (This will change to a Docker service name later)
import os

API_HOST = os.getenv("BACKEND_API_HOST", "http://127.0.0.1:8000")
API_URL = f"{API_HOST}/predict"
# API_URL = "http://127.0.0.1:8000/predict"

st.title("🫁 Clinical Pneumonia Detection AI")
st.markdown("""
This tool uses a fine-tuned ResNet50V2 Deep Learning model to analyze chest X-rays.
Upload a standard DICOM-converted JPEG/PNG to receive an instant probabilistic assessment.
""")

st.divider()

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)
    
    # Analyze button
    if st.button("Run AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing scan parameters..."):
            try:
                # Prepare the file to be sent via HTTP POST
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send the request to FastAPI
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence_score"] * 100
                    
                    st.divider()
                    st.subheader("Diagnostic Results")
                    
                    # Display results with dynamic colors
                    if prediction == "Pneumonia":
                        st.error(f"**Detected:** {prediction}")
                        st.warning(f"**Confidence Score:** {confidence:.2f}%")
                        st.info("Recommendation: Review with attending pulmonologist.")
                    else:
                        st.success(f"**Detected:** {prediction}")
                        st.success(f"**Confidence Score:** {confidence:.2f}%")
                        st.info("Recommendation: Standard protocol.")
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Is the FastAPI server running?")