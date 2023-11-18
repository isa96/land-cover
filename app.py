"""
reference:
    - https://docs.streamlit.io/library/api-reference/widgets
"""

# load libraries 
import os
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from numpy import asarray
from model import ShuffleNetV2
import matplotlib.pyplot as plt
from warnings import filterwarnings

# ignore warnings
filterwarnings("ignore")

# set style matplotlib
plt.style.use("ggplot")
pd.set_option("display.max_columns", None)


# global variable
model = ShuffleNetV2(include_top=True, input_shape=(32, 32, 3),load_model=True, classes=3)
alter_df = pd.DataFrame({'Model':['Bare', 'Sedang', 'Tinggi']})


_class = {0 : "bare", 1 : "sedang", 2 : "tinggi"}

# helper function
def preprocess_image(image_path):
    
    # resize image into 32 x 32 x 3
    # image = cv.resize(image_array, (32, 32))
    image = Image.open(image_path).resize((32, 32))
    image = asarray(image)
    print(image.shape)

    # normalize image
    image = image / 255.0

    # return image with shape 1 x 32 x 32 x 3
    image = image.reshape(1, 32, 32, 3)

    # predict
    result = model.predict(image) 
    # print(result)
    # update acc
    alter_df['Probs'] = result.tolist()[0]
    # print(result.tolist()[0])
    # show plot of probs
    fig, ax = plt.subplots()
    ax.bar('Model', 'Probs', data = alter_df)
    ax.grid(True)

    # add label
    for i in range(3):
        y = alter_df["Probs"][i]
        ax.text(alter_df['Model'][i], y, np.round(y*100, 2), ha = "center")

    # st.pyplot(fig)

    idx = int(result.argmax(axis=1))
    
    return _class[idx], result[0][idx]

# # helper function for video
# def preprocess_video(tempfile_io):

#     # read video file
#     cam = cv.VideoCapture(tempfile_io)

#     # define blank frame as output in streamlit
#     result_frame = st.empty()

#     while cam.isOpened():

#         ret, frame = cam.read()

#         if not ret:
#             break 

#         # preprocess frame
#         frame_preprocess = cv.resize(frame, (32, 32))
#         frame_preprocess = frame_preprocess / 255
#         frame_preprocess = frame_preprocess.reshape(1, 32, 32, 3)

#         # predict frame
#         prediction_conf  = model.predict(frame_preprocess)
#         prediction_class = _class[int(prediction_conf.argmax(axis = 1))]

#         # create frame
#         frame = cv.rectangle(frame, (5, 5), (700, 250), (255, 255, 255), -1)
#         frame = cv.putText(frame, 
#                            f"{prediction_class} - {np.round(prediction_conf[0][int(prediction_conf.argmax(axis = 1))] * 100, 2)}", 
#                            (10, 150), cv.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 2)

#         # print(f"{prediction_class} - {np.round(prediction_conf[0][int(prediction_conf.argmax(axis = 1))] * 100, 2)}")

#         # update frame into text
#         result_frame.image(frame) 
        

# helper function for folder
def preprocess_image_folder(folder_path):

    _class = {0 : "bare", 1 : "sedang", 2 : "tinggi"}
    preprocess = lambda x: Image.open(x).resize((32, 32))
    
    # read image
    # list_image_array = [cv.resize(cv.imread(folder_path + file)[:, :, ::-1], (32, 32)) for file in os.listdir(folder_path)]
    list_image_array = [preprocess(file) for file in os.listdir(folder_path)]

    # convert list as array
    list_image_array = np.array(list_image_array)

    # normalize image
    list_image_array = list_image_array / 255.0

    # predict
    result = model.predict_on_batch(list_image_array)

    # get result
    label = result.argmax(axis = 1)
    convert_label = [_class[i].capitalize() for i in label]

    return convert_label
    

# interface
st.header("KLASIFIKASI TUTUPAN LAHAN")
select_mode = st.sidebar.selectbox("Pilih Jenis Input",
                                   ["", "Gambar", "Folder"])

if select_mode == "Gambar":
    file_image = st.file_uploader("Pilih Gambar")
    if file_image is not None:
        col1, col2, col3 = st.columns([2, 5, 2])
        image = Image.open(file_image)
        col2.image(image, width = 400)
        col_1, col_2, col_3 = st.columns([5, 3, 3])
        isPredict = col_2.button("Prediksi")
        if isPredict:
            # preprocess image
            _class, score = preprocess_image(file_image)
            st.success(f"Prediksi: {_class} ") 
            # (f"{_class} : {np.round(score, 2)}")

# if select_mode == "Video":
#     file_video = st.file_uploader("Input File")
    
#     if file_video is not None:
#         tempfile_video = tempfile.NamedTemporaryFile(delete = False)
#         tempfile_video.write(file_video.read())

#         preprocess_video(tempfile_video.name)

#     # st.video(file_video)

if select_mode == "Folder":
    folder = st.text_input("Masukkan alamat lokasi folder :", "Lokal Disk/nama folder/")

    # replace \ with /
    folder = folder.replace("\\", "/")

    # check if not endswith /
    folder = folder if folder.endswith("/") else folder + "/"

    col_1, col_2, col_3 = st.columns([5, 3, 3])
    isPredict = col_2.button("Prediksi")
    if os.path.isdir(folder) and isPredict:
        
        # preprocess image in folder
        get_label = preprocess_image_folder(folder)

        # define dataframe
        final_df = pd.DataFrame({
            'File' : [folder + i for i in os.listdir(folder)],
            'Label': get_label
        })

        # download csv file
        save_csv = st.download_button("Simpan File CSV", final_df.to_csv(), 
                                      file_name = 'Hasil_Prediksi.csv', 
                                      mime = 'text/csv')
                    
        # show dataframe
        st.dataframe(final_df, width=400)