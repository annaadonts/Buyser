import streamlit as st
import tensorflow as tf
import numpy as np

# Add custom CSS
st.markdown("""
<style>
    /* Remove cursor from selectbox */
    div[data-baseweb="select"] > div:first-child {
        caret-color: transparent !important;
    }
    
    /* Armenian font settings */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Armenian:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans Armenian', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def validate_image(uploaded_file):
    """Validate uploaded file is a valid image"""
    if uploaded_file is None:
        return None
    if uploaded_file.type not in ["image/png", "image/jpeg", "image/jpg"]:
        return None
    return uploaded_file

def model_prediction(test_image):
    """Return predicted class index and confidence score"""
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction), np.max(prediction)

# ========================
# LANGUAGE CONFIGURATION
# ========================
LANGUAGES = {
    'en': {
        'dashboard_title': 'Dashboard',
        'page_select': 'Select Page',
        'pages': ['Home', 'About', 'Disease Recognition'],
        'home_title': 'PLANT DISEASE RECOGNITION SYSTEM',
        'about_title': 'About',
        'prediction_title': 'Disease Recognition',
        'file_uploader': 'Choose an image:',
        'show_image_btn': 'Show Image',
        'predict_btn': 'Predict',
        'prediction_result': "Model prediction: {}",
        'upload_warning': "Please upload an image first",
        'home_content': """Welcome to the Plant Disease Recognition System! 🌿🔍

Our mission is to help identify plant diseases efficiently. Upload a plant image, and our system will analyze it for disease signs. Let's protect our crops together!

### How It Works
1. **Upload Image:** Go to **Disease Recognition** and upload an image
2. **Analysis:** Our AI will process the image
3. **Results:** Get diagnosis and recommendations""",
        'about_content': """#### About Dataset
This dataset contains plant disease images with three main folders:

#### Content
- Train (70,295 images)
- Validation (17,572 images)
- Test (33 images)""",
        'class_names': [
            'Apple - Apple scab', 'Apple - Black rot', 'Apple - Cedar apple rust', 'Apple - Healthy',
            'Blueberry - Healthy', 'Cherry - Powdery mildew', 'Cherry - Healthy',
            'Corn - Cercospora leaf spot', 'Corn - Common rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
            'Grape - Black rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf blight', 'Grape - Healthy',
            'Orange - Huanglongbing (Citrus greening)', 'Peach - Bacterial spot', 'Peach - Healthy',
            'Bell Pepper - Bacterial spot', 'Bell Pepper - Healthy', 'Potato - Early blight',
            'Potato - Late blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy',
            'Squash - Powdery mildew', 'Strawberry - Leaf scorch', 'Strawberry - Healthy',
            'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - Late blight',
            'Tomato - Leaf Mold', 'Tomato - Septoria leaf spot', 'Tomato - Spider mites',
            'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic virus', 'Tomato - Healthy'
        ]
    },
    'am': {
        'dashboard_title': 'Վահանակ',
        'page_select': 'Ընտրել Էջը',
        'pages': ['Գլխավոր', 'Մեր մասին', 'Հիվանդության որոշում'],
        'home_title': 'ԲՈՒՍԱԿԱՆ ՀԻՎԱՆԴՈՒԹՅՈՒՆՆԵՐԻ ԶՆՆՄԱՆ ՀԱՄԱԿԱՐԳ',
        'about_title': 'Մեր մասին',
        'prediction_title': 'Հիվանդության ախտորոշում',
        'file_uploader': 'Ընտրել նկարը',
        'show_image_btn': 'Ցույց տալ նկարը',
        'predict_btn': 'Ախտորոշել',
        'prediction_result': "Արդյունք՝ {}",
        'upload_warning': "Խնդրում ենք նկարը վերբեռնել",
        'home_content': """Բարի գալուստ Բուսական հիվանդությունների զննման համակարգ! 🌿🔍

Մեր նպատակն է օգնել նույնականացնել բույսերի հիվանդությունները: Վերբեռնեք բույսի նկարը, և մեր համակարգը կվերլուծի այն: Եկեք միասին պաշտպանենք մեր բերքը:

### Ինչպես Աշխատել
1. **Նկարի վերբեռնում:** Անցեք «Հիվանդության Որոշում» բաժինը
2. **Վերլուծություն:** Համակարգը կվերլուծի նկարը
3. **Արդյունքներ:** Ստացեք ախտորոշում և առաջարկություններ""",
        'about_content': """#### Տվյալների Հավաքածու
Այս հավաքածուն պարունակում է բույսերի հիվանդությունների նկարներ:

#### Պարունակություն
- Դասավորված (70,295 նկար)
- Վավերացում (17,572 նկար)
- Փորձարկում (33 նկար)""",
        'class_names': [
            'Խնձոր - Խնձորի խառնարան', 'Խնձոր - Սև փտում', 'Խնձոր - Կեդրոնի ժանգ', 'Խնձոր - Առողջ',
            'Հապալաս - Առողջ', 'Բալ - Ճերմակ փոշի', 'Բալ - Առողջ',
            'Եգիպտացորեն - Տերևի բծեր', 'Եգիպտացորեն - Սովորական ժանգ', 'Եգիպտացորեն - Հյուսիսային տերևի այրում', 'Եգիպտացորեն - Առողջ',
            'Խաղող - Սև փտում', 'Խաղող - Սև կարմրախտ', 'Խաղող - Տերևի այրում', 'Խաղող - Առողջ',
            'Նարինջ - Խիտրոսային կանաչություն', 'Դեղձ - Բակտերիալ բիծ', 'Դեղձ - Առողջ',
            'Բուլղարական պղպեղ - Բակտերիալ բիծ', 'Բուլղարական պղպեղ - Առողջ',
            'Կարտոֆիլ - Վաղ այրում', 'Կարտոֆիլ - Ուշ այրում', 'Կարտոֆիլ - Առողջ',
            'Ազնվամորի - Առողջ', 'Սոյա - Առողջ', 'Դդում - Ճերմակ փոշի',
            'Ելակ - Տերևի այրում', 'Ելակ - Առողջ', 'Լոլիկ - Բակտերիալ բիծ',
            'Լոլիկ - Վաղ այրում', 'Լոլիկ - Ուշ այրում', 'Լոլիկ - Տերևի բորբոս',
            'Լոլիկ - Septoria բիծ', 'Լոլիկ - Սարդոստայն տիզ', 'Լոլիկ - Թիրախային բիծ',
            'Լոլիկ - Դեղին տերևի գանգրություն', 'Լոլիկ - Մոզաիկ վիրուս', 'Լոլիկ - Առողջ'
        ]
    }
}

# ========================
# APP INITIALIZATION
# ========================
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_confidence' not in st.session_state:
    st.session_state.prediction_confidence = None

# Load current language
lang = LANGUAGES[st.session_state.lang]

# ========================
# SIDEBAR
# ========================
st.sidebar.title(lang['dashboard_title'])
st.session_state.lang = st.sidebar.radio(
    "Լեզու / Language",
    ['en', 'am'],
    index=0 if st.session_state.lang == 'en' else 1
)
app_mode = st.sidebar.selectbox(lang['page_select'], lang['pages'])

# ========================
# PAGE CONTENT
# ========================
if app_mode == lang['pages'][0]:  # Home
    st.header(lang['home_title'])
    st.image('home_page.jpeg', use_container_width=True)
    st.markdown(lang['home_content'])

elif app_mode == lang['pages'][1]:  # About
    st.header(lang['about_title'])
    st.markdown(lang['about_content'])

elif app_mode == lang['pages'][2]:  # Prediction
    st.header(lang['prediction_title'])
    
    # Initialize session state
    session_defaults = {
        'show_image': False,
        'prediction_result': None,
        'uploaded_image': None,
        'prediction_confidence': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # File uploader
    test_image = st.file_uploader(lang['file_uploader'], type=["png", "jpg", "jpeg"])
    
    # Handle image upload
    if test_image is not None:
        validated_image = validate_image(test_image)
        if validated_image is None:
            st.error("Invalid file type! Please upload a valid image file (PNG, JPG, JPEG).")
            st.session_state.uploaded_image = None
            st.session_state.show_image = False
            st.session_state.prediction_result = None
            st.session_state.prediction_confidence = None
        else:
            if st.session_state.uploaded_image != validated_image:
                st.session_state.uploaded_image = validated_image
                st.session_state.show_image = False
                st.session_state.prediction_result = None
                st.session_state.prediction_confidence = None

    # Two-column layout
    col1, col2 = st.columns(2)
    
    # Image column
    with col1:
        if st.button(lang['show_image_btn'], 
                    disabled=st.session_state.uploaded_image is None,
                    help="Display the uploaded image"):
            st.session_state.show_image = True
        
        if st.session_state.show_image and st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, use_container_width=True)
        elif st.session_state.uploaded_image and st.session_state.show_image:
            st.warning("Image failed to load")

    # Prediction column
    with col2:
        if st.button(lang['predict_btn'], 
                    disabled=st.session_state.uploaded_image is None,
                    help="Analyze the uploaded image"):
            result_index, confidence = model_prediction(st.session_state.uploaded_image)
            st.session_state.prediction_result = lang['class_names'][result_index]
            st.session_state.prediction_confidence = confidence
        
        if st.session_state.prediction_result and st.session_state.prediction_confidence:
            confidence_percent = st.session_state.prediction_confidence * 100
            st.success(
                f"{lang['prediction_result'].format(st.session_state.prediction_result)} "
                f"({confidence_percent:.2f}% confidence)"
            )
            st.session_state.show_image = True
        elif st.session_state.uploaded_image and not st.session_state.prediction_result:
            st.info("Click 'Predict' to analyze the image")

    # Force image display if prediction exists
    if st.session_state.prediction_result and not st.session_state.show_image:
        st.session_state.show_image = True
