import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# **Step 1: Load the trained model**
model = tf.keras.models.load_model('chicken_fecal_model.h5')

# **Step 2: Define the class names**
class_names = ['Healthy', 'coccidiosis']

# **Step 3: Streamlit UI for uploading an image**
st.title("🐔 Chicken Fecal Disease Prediction")
st.write("Upload an image to predict if the chicken is healthy or has coccidiosis.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # **Step 4: Preprocess the image**
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # **Step 5: Display the uploaded image**
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # **Step 6: Predict the class**
    preds = model.predict(img_array)
    predicted_label = np.argmax(preds[0])
    confidence = preds[0][predicted_label] * 100

    # **Step 7: Show the prediction result with confidence**
    st.success(f"🧠 Prediction: **{class_names[predicted_label]}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")

    # **Optional: Show prediction scores for all classes**
    st.subheader("Prediction Scores:")
    for i, score in enumerate(preds[0]):
        st.write(f"{class_names[i]}: {score * 100:.2f}%")

    # **Step 8: Plot the image with prediction**
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_label]} ({confidence:.2f}%)")
    plt.axis('off')
    st.pyplot(plt)
