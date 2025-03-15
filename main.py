import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Converting single image into a batch, the  in (1, 128, 128, 3) means this is the first batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction) # this will retrieve the maximum index of the matrix of this prediction
    return result_index

# Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About', 'Disease Recognition'])

# Home Page
if(app_mode=='Home'):
    st.header('PLANT DISEASE RECOGNITION SYSTEM')
    image_path = 'home_page.jpeg'
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif(app_mode=='About'):
    st.header('About')
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 images)
    3. Test (33 images)
    """)

# Prediction Page
elif(app_mode=='Disease Recognition'):
    st.header('Disease Recognition')
    test_image = st.file_uploader("Choose an image:")
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    # Predict button
    if st.button('Predict'):
        # st.balloons()
        st.write('Our Prediction')
        result_index = model_prediction(test_image)
        # Define Class
        class_name = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
    
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Confidence threshold (adjust based on testing)
# CONFIDENCE_THRESHOLD = 0.7  # 70% confidence

# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('trained_model.keras')

# def is_leaf_image(prediction):
#     """Check if the prediction confidence meets threshold requirements"""
#     max_confidence = np.max(prediction)
#     return max_confidence >= CONFIDENCE_THRESHOLD

# def model_prediction(test_image):
#     try:
#         model = load_model()
#         img = Image.open(test_image)
        
#         # Preprocess image
#         img = img.resize((128, 128))
#         input_arr = tf.keras.preprocessing.image.img_to_array(img)
#         input_arr = np.array([input_arr]) / 255.0
        
#         # Make prediction
#         prediction = model.predict(input_arr)
        
#         if is_leaf_image(prediction):
#             return np.argmax(prediction), np.max(prediction)
#         return None, None
        
#     except Exception as e:
#         st.error(f"Error processing image: {str(e)}")
#         return None, None

# # ... (keep the rest of your existing code the same until prediction page)

# elif app_mode == 'Disease Recognition':
#     st.header('Plant Disease Diagnosis 🩺')
#     test_image = st.file_uploader("Upload leaf image:", type=['jpg', 'jpeg', 'png'])
    
#     if test_image:
#         st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
#         if st.button('Diagnose'):
#             with st.spinner('Analyzing...'):
#                 result_index, confidence = model_prediction(test_image)
                
#             if result_index is None or confidence is None:
#                 st.error("⚠️ Unable to identify plant leaf. Please upload a clear image of a plant leaf.")
#                 st.image('examples/leaf_example.jpg', caption="Example of a proper leaf image", width=300)
#             else:
#                 st.success(f"Diagnosis: **{CLASS_NAMES[result_index]}**")
#                 st.metric("Confidence", f"{confidence*100:.2f}%")
                
#                 if "healthy" in CLASS_NAMES[result_index]:
#                     st.balloons()
#                     st.markdown("### 🎉 Healthy Plant Detected!")
#                 else:
#                     st.markdown("### 🚨 Treatment Recommendations")
#                     st.write("1. Isolate affected plants\n2. Apply copper-based fungicide\n3. Remove infected leaves")