import streamlit as st
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def main():
    st.title('Traffic Image Analyser')

    upload_image = st.file_uploader("Upload an image to analyse" , type=["jpg" , "png"])

    if upload_image:
        st.image(upload_image , caption="Uploaded Image" , use_column_width=True)

    if upload_image is not None:
        with open(upload_image.name, 'wb') as f:
            f.write(upload_image.read())

    if st.button('Analysis'):
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'
        model = cv2.dnn_DetectionModel(config_file , frozen_model)
        classLabel = []
        file_name = 'labels.txt'
        with open(file_name, 'rt') as fpt:
            classLabel = fpt.read().rstrip('\n').split('\n')
        
        model.setInputSize(320,320)
        model.setInputScale(1.0/127.5)
        model.setInputMean((127.5,127.5,127.5))
        model.setInputSwapRB(True)

        img = cv2.imread(upload_image.name)
        classIndex , confidence , bbox = model.detect(img , confThreshold=0.5)
        font_scale = 3 
        font = cv2.FONT_HERSHEY_PLAIN
        for classInd , conf , boxes in zip(classIndex.flatten() , confidence.flatten() , bbox):
            cv2.rectangle(img , boxes,(255,0,0),2)
            cv2.putText(img , classLabel[classInd-1] , (boxes[0]+10 , boxes[1]+40) , font , fontScale=font_scale , color=(0,255,0) , thickness=3)
        
        st.image(cv2.cvtColor(img , cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    main()