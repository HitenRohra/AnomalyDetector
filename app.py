import streamlit as st
import detect
import cv2
import shutil
import os
import helper

frame_th = 50
helper.del_dir('temp')

def save_video(video_file):
    with open("./sample/uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())
    return "uploaded_video.mp4"

def read_video(path):
    video_file = open(path, 'rb')
    video_bytes = video_file.read()
    return video_bytes


st.title('Anaomaly detection')


video_file = st.file_uploader("Upload video", type=["mp4"])
if st.button('Predict'):
    if video_file is not None:
        vid_path = save_video(video_file)
        status = detect.main()
        os.system("ffmpeg -i ./sample/output_video.mp4 -vcodec libx264 output1_video.mp4 -y")
        
        if status:
            st.warning('Anomalous activity detected')
    
        
        vid = read_video('output1_video.mp4')
        st.video(vid)
        
        
    else:
        st.error('Video file not uploaded')
