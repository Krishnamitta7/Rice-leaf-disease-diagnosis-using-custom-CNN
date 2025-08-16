import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import base64

# Load your trained model
model = load_model('C:/B.Tech/B.Tech Projects/CSP/best csp model/final_model.h5')  # Update the path

# Define the labels and remedies
all_labels = ['Bacterial leaf blight', 'Blast', 'Brown spot']
remedies = {
    'Blast': """
    **Remedies for Blast:**
    - Use resistant rice varieties.
    - Apply fungicides early during detection.
    - Optimize irrigation and avoid water logging.
    - Practice crop rotation and residue management.
    """,
    'Bacterial leaf blight': """
    **Remedies for Bacterial Leaf Blight:**
    - Use disease-free seeds.
    - Avoid high nitrogen fertilizers.
    - Proper water management is crucial.
    - Remove infected plants promptly.
    """,
    'Brown spot': """
    **Remedies for Brown Spot:**
    - Apply appropriate nitrogen fertilizers.
    - Treat seeds with fungicides.
    - Rotate crops and maintain field hygiene.
    - Ensure proper water drainage to avoid stress.
    """
}

# Function to process and predict the uploaded image
def convert_image_to_array(image):
    # Convert image to RGB if not already in RGB format
    image = image.convert('RGB')
    # Resize to match model input size
    image = image.resize((128, 128))
    image = np.array(image)
    # Normalize image (match training preprocessing)
    image = image.astype('float32') / 255.0
    return image

def predict_disease(image):
    image = convert_image_to_array(image)
    # Ensure the image has the shape (1, 128, 128, 3)
    image = image.reshape(1, 128, 128, 3)
    predictions = model.predict(image)
    # Return the predicted label
    return all_labels[np.argmax(predictions)]

# Load and encode local background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Background image setup with linear gradient
local_image_path = "C:/B.Tech/B.Tech Projects/CSP/best csp model/background.jpg"
base64_image = get_base64_image(local_image_path)
page_bg_img = f'''
<style>
    .appview-container, .css-1outpf7 {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("data:image/jpg;base64,{base64_image}");
    }}

    /* Black transparent overlay for text readability */
    .black-overlay {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
    }}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar with team details
st.sidebar.title("Team Members")
st.sidebar.markdown("""
- **M.V.N Amruth Sai** - 99220040641
- **K.Gopi Krishna** - 99220040583
- **M.Krishna Kumar Reddy** - 99220040635
- **M.Rama Yogi Reddy** - 99220040627
""")

# Main app layout
st.markdown("""
<div class="black-overlay" style="text-align: center; font-size: 36px; color: #4CAF50; font-weight: bold;">
    🌾 Rice Leaf Disease Prediction 🌾
</div>
""", unsafe_allow_html=True)

st.markdown("""
<hr style="border: 1px solid #4CAF50;">
""", unsafe_allow_html=True)

st.markdown("""
<div class="black-overlay" style="text-align: center; font-size: 18px; color: #FFFFFF;">
 Our aim is to develop a user-friendly web application that utilizes deep learning techniques, particularly Convolutional Neural Networks (CNNs), to accurately identify and classify rice leaf diseases from images. The platform will allow users to upload images of rice leaves and receive real-time predictions of potential diseases, along with recommendations for treatment and prevention.
</div>
""", unsafe_allow_html=True)

# File uploader to get image input
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict and display results
    predicted_disease = predict_disease(image)
    st.markdown(f"""
    <div class="black-overlay" style="text-align: center; font-size: 28px; color: #4CAF50; font-weight: bold;">
        Predicted Disease: {predicted_disease}
    </div>
    """, unsafe_allow_html=True)

    # Show remedies with black transparent background
    with st.expander("View Remedies"):
        st.markdown(f"""
        <div class="black-overlay">
            {remedies[predicted_disease]}
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Please upload an image to start the prediction.")

# Feedback Section
st.markdown("""
<hr style="border: 1px solid #4CAF50;">
""", unsafe_allow_html=True)

st.markdown("""
<div class="black-overlay" style="text-align: center; font-size: 20px; color: #FFFFFF;">
    We value your feedback! Please share your thoughts and suggestions below.
</div>
""", unsafe_allow_html=True)


feedback = st.radio(
    "How was your experience with the prediction?",
    options=["😊 Good", "😞 Bad"],
    help="Select an option to rate your experience."
)

if feedback == "😊 Good":
    st.success("Thank you for your positive feedback! 😊")
elif feedback == "😞 Bad":
    st.warning("We're sorry to hear that! 😞 We appreciate your feedback to improve our service.")

feedback = st.text_area("Your Feedback", "", height=150)

if feedback:
    st.success("Thank you for your feedback!")
