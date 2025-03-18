import streamlit as st
import tensorflow as tf
import numpy as np


# Add at the top after imports
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Armenian:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans Armenian', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LANGUAGE CONFIGURATION
# ==============================
LANGUAGES = {
    'en': {
        # App Interface
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
        
        # Content
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

        # Complete English class names
        'class_names': [
            'Apple - Apple scab',
            'Apple - Black rot',
            'Apple - Cedar apple rust',
            'Apple - Healthy',
            'Blueberry - Healthy',
            'Cherry - Powdery mildew',
            'Cherry - Healthy',
            'Corn - Cercospora leaf spot',
            'Corn - Common rust',
            'Corn - Northern Leaf Blight',
            'Corn - Healthy',
            'Grape - Black rot',
            'Grape - Esca (Black Measles)',
            'Grape - Leaf blight',
            'Grape - Healthy',
            'Orange - Huanglongbing (Citrus greening)',
            'Peach - Bacterial spot',
            'Peach - Healthy',
            'Bell Pepper - Bacterial spot',
            'Bell Pepper - Healthy',
            'Potato - Early blight',
            'Potato - Late blight',
            'Potato - Healthy',
            'Raspberry - Healthy',
            'Soybean - Healthy',
            'Squash - Powdery mildew',
            'Strawberry - Leaf scorch',
            'Strawberry - Healthy',
            'Tomato - Bacterial spot',
            'Tomato - Early blight',
            'Tomato - Late blight',
            'Tomato - Leaf Mold',
            'Tomato - Septoria leaf spot',
            'Tomato - Spider mites',
            'Tomato - Target Spot',
            'Tomato - Yellow Leaf Curl Virus',
            'Tomato - Mosaic virus',
            'Tomato - Healthy'
        ]
    },
    'am': {
        # App Interface
        'dashboard_title': 'Վահանակ',
        'page_select': 'Ընտրել Էջը',
        'pages': ['Գլխավոր', 'Մեր Մասին', 'Հիվանդության Որոշում'],
        'home_title': 'ԲՈՒՍԱԿԱՆ ՀԻՎԱՆԴՈՒԹՅՈՒՆՆԵՐԻ ԶՆՆՄԱՆ ՀԱՄԱԿԱՐԳ',
        'about_title': 'Մեր Մասին',
        'prediction_title': 'Հիվանդության Ախտորոշում',
        'file_uploader': 'Ընտրել Նկարը',
        'show_image_btn': 'Ցույց Տալ Նկարը',
        'predict_btn': 'Ախտորոշել',
        'prediction_result': "Արդյունք՝ {}",
        'upload_warning': "Խնդրում ենք նկարը վերբեռնել",
        
        # Content
        'home_content': """Բարի գալուստ Բուսական հիվանդությունների զննման համակարգ! 🌿🔍

Մեր նպատակն է օգնել նույնականացնել բույսերի հիվանդությունները: Վերբեռնեք բույսի նկարը, և մեր համակարգը կվերլուծի այն: Եկեք միասին պաշտպանենք մեր բերքը:

### Ինչպես Աշխատել
1. **Նկարի Վերբեռնում:** Անցեք «Հիվանդության Որոշում» բաժինը
2. **Վերլուծություն:** Համակարգը կվերլուծի նկարը
3. **Արդյունքներ:** Ստացեք ախտորոշում և առաջարկություններ""",

        'about_content': """#### Տվյալների Հավաքածու
Այս հավաքածուն պարունակում է բույսերի հիվանդությունների նկարներ:

#### Պարունակություն
- Դասավորված (70,295 նկար)
- Վավերացում (17,572 նկար)
- Փորձարկում (33 նկար)""",

        # Armenian class names (COMPLETE TRANSLATIONS)
        'class_names': [
            'Խնձոր - Խնձորի խառնարան',
            'Խնձոր - Սև փտում',
            'Խնձոր - Կեդրոնի ժանգ',
            'Խնձոր - Առողջ',
            'Թուշ - Առողջ',
            'Բալ - Ճերմակ փոշի',
            'Բալ - Առողջ',
            'Եգիպտացորեն - Տերևի բծեր',
            'Եգիպտացորեն - Սովորական ժանգ',
            'Եգիպտացորեն - Հյուսիսային տերևի այրում',
            'Եգիպտացորեն - Առողջ',
            'Ուրց - Սև փտում',
            'Ուրց - Սև կարմրախտ',
            'Ուրց - Տերևի այրում',
            'Ուրց - Առողջ',
            'Նարինջ - Խիտրոսային կանաչություն',
            'Բալ - Բակտերիալ բիծ',
            'Բալ - Առողջ',
            'Վարունգ - Բակտերիալ բիծ',
            'Վարունգ - Առողջ',
            'Կարտոֆիլ - Վաղ այրում',
            'Կարտոֆիլ - Ուշ այրում',
            'Կարտոֆիլ - Առողջ',
            'Ազնվամորի - Առողջ',
            'Սոյա - Առողջ',
            'Դդում - Ճերմակ փոշի',
            'Մորի - Տերևի այրում',
            'Մորի - Առողջ',
            'Պոմիդոր - Բակտերիալ բիծ',
            'Պոմիդոր - Վաղ այրում',
            'Պոմիդոր - Ուշ այրում',
            'Պոմիդոր - Տերևի բորբոս',
            'Պոմիդոր - Septoria բիծ',
            'Պոմիդոր - Սարդոստայն տիզ',
            'Պոմիդոր - Թիրախային բիծ',
            'Պոմիդոր - Դեղին տերևի գանգրություն',
            'Պոմիդոր - Մոզաիկ վիրուս',
            'Պոմիդոր - Առողջ'
        ]
    }
}

# ==============================
# APP INITIALIZATION
# ==============================
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

# Load current language
lang = LANGUAGES[st.session_state.lang]

# ==============================
# MODEL FUNCTION
# ==============================
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title(lang['dashboard_title'])
st.session_state.lang = st.sidebar.radio(
    "Լեզու / Language",
    ['en', 'am'],
    index=0 if st.session_state.lang == 'en' else 1
)
app_mode = st.sidebar.selectbox(lang['page_select'], lang['pages'])

# ==============================
# PAGE CONTENT
# ==============================
if app_mode == lang['pages'][0]:  # Home
    st.header(lang['home_title'])
    st.image('home_page.jpeg', use_column_width=True)
    st.markdown(lang['home_content'])

elif app_mode == lang['pages'][1]:  # About
    st.header(lang['about_title'])
    st.markdown(lang['about_content'])

elif app_mode == lang['pages'][2]:  # Prediction
    st.header(lang['prediction_title'])
    test_image = st.file_uploader(lang['file_uploader'])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(lang['show_image_btn']):
            if test_image:
                st.image(test_image, use_column_width=True)
            else:
                st.warning(lang['upload_warning'])
    
    with col2:
        if st.button(lang['predict_btn']):
            if test_image:
                result_index = model_prediction(test_image)
                st.success(f"{lang['prediction_result'].format(lang['class_names'][result_index])}")
            else:
                st.warning(lang['upload_warning'])
