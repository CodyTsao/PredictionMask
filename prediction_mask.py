import streamlit as st 
from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from IPython.display import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

model = tf.keras.models.load_model('model_maskdetect.h5')

st.title("上傳圖片口罩辨識")

uploaded_file = st.file_uploader("上傳圖片")
if uploaded_file is not None: 
    
    # 任意一張圖片測試
    #img_path = './test_image/127-with-mask.jpg'
    #img_path = './test_image/11.jpg'
    # 載入圖檔，並縮放寬高為 (224, 224)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    # 加一維，變成 (1, 224, 224, 3)，最後一維是色彩
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
     
    st.image(uploaded_file)
    st.write("predict...")
    predictions = f'戴口罩機率：{round(model.predict(x)[0][1] * 100, 2):.2f}'+'%'
    st.write(predictions)
    


