import streamlit as st
from model_helper import predict

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image_path = 'temp_file.jpg'

    # Save the file locally so the predict function can read it
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the image to the user
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    # Run the prediction
    with st.spinner('Analyzing damage...'):
        prediction = predict(image_path)

    # Display the result!
    st.success(f"**Prediction:** {prediction.replace('_', ' ')}")

