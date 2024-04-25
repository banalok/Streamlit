  
import streamlit as st
from utils import * 
from streamlit_drawable_canvas import st_canvas##############################new########################
import plotly.io
import math
import tifffile
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns
import statistics as stat
import os
import numpy as np
import cv2
from io import BytesIO
import imutils
from matplotlib import pyplot as plt
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import measure, color, io
import plotly.express as px
from skimage import (
    filters,  morphology, img_as_float, img_as_ubyte, img_as_uint, exposure, restoration
)
from skimage.draw import polygon
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import time
import subprocess
import shutil
import re   
#from st_pages import show_pages_from_config

#show_pages_from_config()

cwd=os.getcwd()+'/'
os.makedirs('temp dir', exist_ok = True)

if 'first_raw_image' not in st.session_state:
    st.session_state.first_raw_image = None
    
if 'overlayed_image' not in st.session_state:
    st.session_state['overlayed_image'] = None
                             
if 'raw_file_overlay' not in st.session_state:
    st.session_state['raw_file_overlay'] = None
    
if 'first_raw_overlay' not in st.session_state:
    st.session_state['first_raw_overlay'] = None

if "raw_file" not in st.session_state:
    st.session_state.raw_file = None

if "multi" not in st.session_state:
    st.session_state.multi = False

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
    
# if "show_frames" not in st.session_state:
#     st.session_state.show_frames = False

if "button_clicked_allframes" not in st.session_state:
    st.session_state.button_clicked_allframes = False

if "button_clicked_roi" not in st.session_state:
    st.session_state.button_clicked_roi = False
     
if 'display_table' not in st.session_state:
    st.session_state.display_table = False
    
if 'all_param_table' not in st.session_state:
    st.session_state.all_param_table = False
        
def callback_off():
    st.session_state.button_clicked = False
    st.session_state.button_clicked_allframes = False
    st.session_state.display_table = False    

def callback_allframes():
     st.session_state.button_clicked_allframes = True
     st.session_state.button_clicked_roi = False
     
def callback_roi():
     st.session_state.button_clicked_roi = True    
  
def callback():
    #Button was clicked
    st.session_state.button_clicked = True

def callback_multi():
    st.session_state.multi = True

def callback_table():
   st.session_state.display_table = True
   
def callback_all_param_table():
   st.session_state.all_param_table = True   

def main():
    Segment()

def Segment():
    st.title('**_DL-SCAN (Deep Learning- based Segmentation of Cells and Analysis)_**')
    
    if st.session_state.raw_file is not None:
        st.warning('Please reload the page to upload a new file')        
    else:
        st.session_state.raw_file = st.file_uploader("*_Choose a TIFF-stack file_*")        
    
    if st.session_state.raw_file is not None:
        if st.session_state['first_raw_image'] is None:
            file_path = os.path.join('temp dir', st.session_state.raw_file.name)
    
            with open(file_path, "wb") as f:
                f.write(st.session_state.raw_file.read())

            raw_name=cwd+'temp dir/'+st.session_state['raw_file'].name
            st.session_state['first_raw_image'] = load_image(raw_name)
            f.close()
            shutil.rmtree("temp dir")
            raw_image = st.session_state['first_raw_image']
        else:
            raw_image = st.session_state['first_raw_image']
            shutil.rmtree("temp dir")
         
        if raw_image.dtype != 'uint8':
            img_list = []
            for frames_64 in range(0, raw_image.shape[0]):
                frame_rescale = exposure.rescale_intensity(raw_image[frames_64], out_range=(0, 255)).astype('uint8')
                img_list.append(frame_rescale)
            raw_image = np.array(img_list)            
            
        if (len(raw_image.shape) == 2):
            st.warning("Please upload a tiff stack with multiple frames")
            
        if (len(raw_image.shape) ==3 and raw_image.shape[-1]!=3) or (len(raw_image.shape) ==4 and raw_image.shape[-1]!=3):
            raw_image_1D = raw_image
            raw_image = np.zeros((raw_image_1D.shape[0], raw_image_1D.shape[1], raw_image_1D.shape[2], 3), dtype=np.uint8)
            for i in range(raw_image.shape[0]):
                raw_image[i] = cv2.cvtColor(img_as_ubyte(raw_image_1D[i]), cv2.COLOR_GRAY2RGB)

        model = load_model()        
        extension = st.session_state['raw_file'].name.split(".")[-1]        
        raw_image_ani = raw_image       
        st.session_state['raw_img_ani_pg_2'] = raw_image_ani        
            
        if extension == 'tif' and len(raw_image_ani.shape)==4:            
            st.write("*_Original Frames_*")
            image_frame_num = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0,max_value = raw_image_ani.shape[0]-1, value = 0, step = 1, key='num_1')
            if image_frame_num==0:
                st.image(raw_image_ani[0],use_column_width=True,clamp = True)
            elif image_frame_num >= 1:
                st.image(raw_image_ani[image_frame_num],use_column_width=True,clamp = True)
            
            if 'bc_corr_check' not in st.session_state:
                st.session_state.bc_corr_check = st.radio("Select", ['No background correction', 'Background correction'])  
                
            else:
                st.session_state.bc_corr_check = st.radio("Select an option", ['No background correction', 'Background correction'], index=0 if st.session_state.bc_corr_check == 'No background correction' else 1)
            
            
            if st.session_state.bc_corr_check == 'No background correction':
                background_corr_img = raw_image_ani
                                                   
            # Load the image
            elif st.session_state.bc_corr_check == 'Background correction':
                background_corr_img = np.zeros(raw_image_ani.shape,dtype=np.int32)  
                                
                if 'canvas_data' not in st.session_state:
                    st.session_state.canvas_data = None
                                                 
                st.markdown("Draw one region on the image below", help= "The original stack is background corrected based on the drawn region. This corrected stack is not used for segmentation")
                image_draw = Image.fromarray(raw_image_ani[0])  
                canvas_height = raw_image_ani.shape[1]
                canvas_width = raw_image_ani.shape[2]
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
                    stroke_width=2,
                    stroke_color="rgba(255, 0, 0, 0.7)",
                    background_image=image_draw,
                    update_streamlit=True,
                    display_toolbar=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="canvas",                        
                    )
                if canvas_result.json_data != st.session_state.canvas_data:
                    st.session_state.canvas_data = canvas_result.json_data    
                               
                    
                for bc in range(0,raw_image_ani.shape[0]):
                    #rerun_flag = False
                    #if canvas_result.json_data is not None:
                    if st.session_state.canvas_data is not None:
                        objects = pd.json_normalize(st.session_state.canvas_data["objects"])
                        
                        if len(objects) == 0:
                            background_corr_img = raw_image_ani                            
                            break
                                            
                        elif len(objects) == 1:
                            bc_selected_region = raw_image_ani[bc][int(objects['top']):int(objects['top'])+int(objects['height']), int(objects['left']):int(objects['left'])+ int(objects['width'])]
                            background_corr_img[bc] = np.subtract((raw_image_ani[bc]).astype(np.int32), (np.mean(bc_selected_region)).astype(np.int32))
                            background_corr_img[bc] = np.clip(background_corr_img[bc], 0, 255)
                            
                        elif len(objects) > 1: 
                            background_corr_img = raw_image_ani                            
                            break
                background_corr_img = background_corr_img.astype(np.uint8)
                
                if len(objects) > 1:                    
                    st.error('Please select only one region. Use the toolbar to either undo or delete the selected region(s)')                
                if len(objects) == 1:
                    st.write("*_Background Corrected Image_*")
                    bc_image_frame_num = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0,max_value = raw_image_ani.shape[0]-1, value = 0, step = 1, key='bc_1')
                    if bc_image_frame_num==0:
                        st.image(background_corr_img[0],use_column_width=True,clamp = True)
                    elif bc_image_frame_num >= 1:
                        st.image(background_corr_img[bc_image_frame_num],use_column_width=True,clamp = True)
                   
            #raw_image_ani = background_corr_img
            st.session_state['background_corr_pg_2'] = background_corr_img

            if 'gauss_x' not in st.session_state:
                st.session_state.gauss_x = None
            
            if 'med_x' not in st.session_state:
                st.session_state.med_x = None 
                
            if 'bri_x' not in st.session_state:
                st.session_state.bri_x = None
    
            if 'con_x' not in st.session_state:
                st.session_state.con_x = None
    
            if 'hist_x' not in st.session_state:
                st.session_state.hist_x = None

            if st.session_state.gauss_x is None:
                st.session_state.gauss_x = st.slider("*_Gaussian Blur Kernel size_*", min_value = -1,max_value = 100, step = 2,help = "Filters are applied to all frames. Check 'Processed Frames' below. -1 means no blur.", on_change=callback_off)
            else:
                st.session_state.gauss_x = st.slider('*_Gaussian Blur Kernel size_*', min_value = -1,max_value = 100, value = st.session_state.gauss_x, step = 2,help = "Filters are applied to all frames. Check 'Processed Frames' below. -1 means no blur.", on_change=callback_off)            
    
            if st.session_state.med_x is None:
                st.session_state.med_x = st.slider("*_Median Blur Kernel size_*", min_value = -1,max_value = 100, step = 2,help = "Filters are applied to all frames. Check 'Processed Frames' below. -1 means no blur.", on_change=callback_off)
            else:
                st.session_state.med_x = st.slider('*_Median Blur Kernel size_*', min_value = -1,max_value = 100, value = st.session_state.med_x, step = 2,help = "Filters are applied to all frames. Check 'Processed Frames' below. -1 means no blur.", on_change=callback_off)
                
            if st.session_state.bri_x is None:            
                st.session_state.bri_x = st.slider('*_Change Brightness_*',min_value = -255,max_value = 255,value = 0, on_change=callback_off)
            else:
                st.session_state.bri_x = st.slider('*_Change Brightness_*',min_value = -255,max_value = 255, value = st.session_state.bri_x, on_change=callback_off)                
            
            if st.session_state.con_x is None:
                st.session_state.con_x = st.slider('*_Change Contrast_*',min_value = -255,max_value = 255,value = 0, on_change=callback_off)
            else:
                st.session_state.con_x = st.slider('*_Change Contrast_*',min_value = -255,max_value = 255, value = st.session_state.con_x, on_change=callback_off)
                
            if st.session_state.hist_x is None:   
                st.session_state.hist_x = st.slider("*_Histogram Equalization cliplimit factor (CLAHE)_*", min_value = 1,max_value = 20, step = 1, on_change=callback_off)
            else:
                st.session_state.hist_x = st.slider("*_Histogram Equalization cliplimit factor (CLAHE)_*", min_value = 1,max_value = 20, step = 1, value = st.session_state.hist_x, on_change=callback_off)
            
            if f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}" not in st.session_state and f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}" not in st.session_state:
                st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = None
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = None
                            
            if (st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] is None) and (st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] is None):
                
                st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = []
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = np.zeros_like(raw_image_ani[0][:,:,0])   #size of one of the images
                
                weight = 1/raw_image_ani.shape[0]                      
                for frame_num in range(0,raw_image_ani.shape[0]):                       
                    raw_image = raw_image_ani[frame_num]
                    if st.session_state.gauss_x == -1:
                       blur_gauss = raw_image 
                    else:
                       blur_gauss = cv2.GaussianBlur(raw_image, (st.session_state.gauss_x, st.session_state.gauss_x), sigmaX=0)
                                        
                    if st.session_state.med_x == -1:
                        blur_median_proc2 = blur_gauss
                    else:
                        blur_median_proc2 = cv2.medianBlur(blur_gauss, st.session_state.med_x)               
                                
                    bri_con2 = apply_brightness_contrast(blur_median_proc2, st.session_state.bri_x, st.session_state.con_x)
                                        
                    lab_img2= cv2.cvtColor(bri_con2, cv2.COLOR_RGB2LAB)
                    l2, a2, b2 = cv2.split(lab_img2)
                    equ2 = cv2.equalizeHist(l2)
                    updated_lab_img2 = cv2.merge((equ2,a2,b2))
                    hist_eq_img2 = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2RGB)
                    
                    clahe2 = cv2.createCLAHE(clipLimit=st.session_state.hist_x, tileGridSize=(8,8))
                    clahe_img2 = clahe2.apply(l2)
                    updated_lab_img22 = cv2.merge((clahe_img2,a2,b2))
                    CLAHE_img = cv2.cvtColor(updated_lab_img22, cv2.COLOR_LAB2RGB)
                    CLAHE_img = CLAHE_img[:,:,0] 
                    st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"].append(CLAHE_img) 
                    st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = np.maximum(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"], CLAHE_img)    
                
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"].astype(np.int32)
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]>255]=255
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"].astype(np.uint8)
                
                st.write('*_Processed Frames_*')
                image_frame_num_pro = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0, max_value = raw_image_ani.shape[0]-1, step = 1,key='num_2')
                if image_frame_num==0:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] [0],use_column_width=True,clamp = True)
                elif image_frame_num >= 1:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] [image_frame_num_pro],use_column_width=True,clamp = True)
            
                st.markdown("**_The Collapsed Image_**")
                
                st.image(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"], use_column_width=True,clamp = True) 
                st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
 
            else:
                st.write('*_Processed Frames_*')
                image_frame_num_pro = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0, max_value = raw_image_ani.shape[0]-1, step = 1,key='num_2')
                if image_frame_num==0:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0],use_column_width=True,clamp = True)
                elif image_frame_num >= 1:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][image_frame_num_pro],use_column_width=True,clamp = True)
            
                st.markdown("**_The Collapsed Image_**")
                
                st.image(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"], use_column_width=True,clamp = True) 
                st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
     
            if st.button("*_Segment and generate labels_*", key='frame_btn',on_click = callback_allframes) or st.session_state.button_clicked_allframes:
                with st.expander("Inhomogeneous pixel distribution?"):
                    segment_check = st.radio("Select one:", ["Segment on the collapsed image", "Segment on the first image"])
                    rb_check = st.radio("Rolling ball background correction (RBBC)", ['No RBBC', 'RBBC'], help='Select "RBBC" for rolling ball background correction on the selected image, otherwise, select No "RBBC"')
                    #if f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg" not in st.session_state:
                    if segment_check == "Segment on the collapsed image":
                        if rb_check == 'No RBBC':
                            st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                            seg_im = stardist_seg(st.session_state['Collapsed_Image'],model)
                            #st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] = stardist_seg(st.session_state['Collapsed_Image'],model)
                        elif rb_check == 'RBBC':
                            st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                            radius_x = st.slider("Ball Radius", min_value = 1, max_value = 50, step=1, value = 25)                        
                            background_to_remove = restoration.rolling_ball(st.session_state['Collapsed_Image'], radius=radius_x)
                            st.session_state['Collapsed_Image'] = st.session_state['Collapsed_Image'] - background_to_remove
                            st.image(st.session_state['Collapsed_Image'], use_column_width=True,clamp = True)  
                            seg_im = stardist_seg(st.session_state['Collapsed_Image'],model)
                            #st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] = stardist_seg(st.session_state['Collapsed_Image'],model)
                        #st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] = stardist_seg(st.session_state['Collapsed_Image'],model)
                        st.image(render_label(seg_im, img=st.session_state['Collapsed_Image']), use_column_width=True,clamp = True)
                        #st.image(render_label(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"], img=st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0]), use_column_width=True,clamp = True)
                        label = seg_im
                        #label = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] 
                        st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                    elif segment_check == "Segment on the first image":
                        if rb_check == 'No RBBC':                             
                            #st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                            seg_first = st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0]
                            #st.session_state['Collapsed_Image'] = seg_first
                            seg_first_im = stardist_seg(seg_first, model)
                        elif rb_check == 'RBBC':                             
                            radius_x = st.slider("Ball Radius", min_value = 1, max_value = 50, step=1, value = 25)                        
                            background_to_remove = restoration.rolling_ball(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0], radius=radius_x)
                            seg_first = st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0] - background_to_remove
                            #st.session_state['Collapsed_Image'] = seg_first
                            seg_first_im = stardist_seg(seg_first, model)
                            #st.image(seg_first, use_column_width=True,clamp = True) 
                        #st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] = stardist_seg(seg_first,model)
                        
                        st.image(render_label(seg_first_im, img=st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0]), use_column_width=True,clamp = True)
                        label = seg_first_im
                        #label = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] 
                        st.session_state['Collapsed_Image'] = st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0]          
                    # else:
                    #     st.write("ff")
                    #     label = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"]
                
                props = measure.regionprops(label) 
               
                if len(props) == 0:
                    st.warning("No region of interests found. Please use another file or try pre-processing.")
                else:
                    diam = [props[obj_s]['equivalent_diameter_area'] for obj_s in range(0,len(props))]
                    labels_to_keep_len = len([prop['label'] for prop in props if prop['equivalent_diameter_area'] > 0.5*stat.mean(diam)])
                    labels_to_keep = list(range(1, labels_to_keep_len+1))
                    label_f = np.zeros_like(label, dtype= 'uint8')
                    for labels in labels_to_keep:
                        label_f[label==labels] = labels    
                    
                    props_to_sort = measure.regionprops(label_f)
                    centroid_positions = [prop_sort.centroid for prop_sort in props_to_sort]
                    sorted_indices = np.lexsort((np.array(centroid_positions)[:, 1], np.array(centroid_positions)[:, 0]))
                    
                    final_label = np.zeros_like(label_f, dtype= 'uint8')
                    for i, idx in enumerate(sorted_indices, start=1):
                        final_label[label_f == (idx + 1)] = i                                                      
                    super_im_rgb = cv2.cvtColor(st.session_state['Collapsed_Image'] , cv2.COLOR_GRAY2RGB)
                    label_list_len = len([prop['label'] for prop in props_to_sort if prop['label']])
                    label_list = list(range(1,label_list_len+1))
                    st.session_state['label_list_pg_2'] = label_list
                    
                    st.session_state['final_label_pg_2'] = final_label
                    labels_rgb = np.expand_dims(final_label, axis=2)
                    final_label_rgb = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)
                    for label in np.unique(label_list):                                                           
                            if label == 0:
                                continue                
                            mask = np.zeros(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0].shape, dtype="uint8")
                            mask[final_label == label] = 255
                            #detect contours in the mask and grab the largest one
                            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
                            cnts = imutils.grab_contours(cnts)
                            c = max(cnts, key=cv2.contourArea)
                            # draw a circle enclosing the object
                            ((x, y), r) = cv2.minEnclosingCircle(c)
                            #cv2.circle(final_label_rgb, (int(x), int(y)), int(r), (255, 0, 0), 1)
                            coords = np.argwhere(final_label==label)                            
                            # Create a polygon from the coordinates
                            poly = polygon(coords[:, 0], coords[:, 1])
                            # Set the color of the polygon to red
                            color_poly = (255, 0, 0)
        
                            # Color the polygon in the color image
                            final_label_rgb[poly] = color_poly
                            cv2.putText(final_label_rgb, "{}".format(label), (int(x) - 10, int(y)),
                             	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            cv2.circle(super_im_rgb, (int(x), int(y)), int(r), (255, 0, 0), 1)
                            cv2.putText(super_im_rgb, "{}".format(label), (int(x) - 10, int(y)),
                             	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)                        
                    
                    st.write("*_Automatically labeled objects on the selected image_*")                    
                    st.image(super_im_rgb,use_column_width=True,clamp = True)
                    st.write("*_Automatically segmented and labeled objects on a black background_*")
                    st.image(final_label_rgb,use_column_width=True,clamp = True)
                    st.session_state['final_label_rgb_pg_2'] = final_label_rgb
                    st.session_state['super_im_rgb_pg_2'] = super_im_rgb
                   
                    with st.expander("*_Object(s) not detected?_*"):                        
                        if st.button('Draw ROI', key='roi_btn',help = "Use this option (only when it's absolutely necessary) to draw ROI on the labeled collapsed image", on_click = callback_roi) or st.session_state.button_clicked_roi:
                            
                            image_draw_2 = Image.fromarray(super_im_rgb) 
                            canvas_height_s = super_im_rgb.shape[1]
                            canvas_width_s = super_im_rgb.shape[2]
                            if 'canvas_data_2' not in st.session_state:
                                st.session_state.canvas_data_2 = None
                            st.warning('Use left-click to assign points for the shape boundaries, and right-click to finalize the ROI. Undo, redo or delete the drawn regions as needed.')
                            canvas_result_2 = st_canvas(
                                fill_color="rgba(255, 0, 0, 0.3)",  
                                stroke_width=1,
                                stroke_color="rgb(255, 0, 0)",
                                background_image=image_draw_2,
                                display_toolbar=True,
                                drawing_mode="polygon",
                                key="canvas_2",
                                width = image_draw_2.size[0],
                                height = image_draw_2.size[1],
                            )
                            
                            if canvas_result_2 is None:
                                pass
                            
                            # Once the user has drawn the ROI, process and update the image
                            if canvas_result_2.json_data != st.session_state.canvas_data_2:
                                st.session_state.canvas_data_2 = canvas_result_2.json_data  
                                
                                                   
                            if st.session_state.canvas_data_2 is not None:
                                # Get the JSON data from canvas_result 
                                objects = pd.json_normalize(st.session_state.canvas_data_2["objects"])
                                for col in objects.select_dtypes(include=['object']).columns:
                                    objects[col] = objects[col].astype("str")
                                
                                if len(objects) == 0:
                                    pass
                                                                                                                 
                                elif len(objects) >= 1:
                                    final_label_rgb_with_outline = final_label_rgb.copy()
                                    added_roi_list = []
                                    # Iterate through the data to extract consecutive pairs of numbers
                                    for roi_num in range(0, len(objects)):                                        
                                        roi_list = eval(objects['path'][roi_num])
                                        coordinates_roi = extract_coordinates(roi_list)
                                        array_0 = np.array([coordinates_roi[i][0] for i in range(len(coordinates_roi))], dtype=np.int64)
                                        array_1 = np.array([coordinates_roi[i][1] for i in range(len(coordinates_roi))], dtype=np.int64)
                                        poly = list(zip(array_0, array_1))
                                        # Create a mask for the polygon outline
                                        polygon_mask = Image.new('L', (final_label_rgb_with_outline.shape[1], final_label_rgb_with_outline.shape[0]) , 0)
                                        draw = ImageDraw.Draw(polygon_mask)
                                        polygon_points = poly
                                        draw.polygon(polygon_points, outline=1, fill=1)        
                                        # Convert the mask to a NumPy array
                                        polygon_mask_np = np.array(polygon_mask)                                        
                                        # Create a boolean mask for points inside the polygon
                                        points_inside_polygon = polygon_mask_np != 0        
                                        # Color the outline in the color image (final_label_rgb)
                                        color_outline = (255, 0, 0)
                                        final_label_rgb_with_outline = final_label_rgb_with_outline
                                        final_label_rgb_with_outline[polygon_mask_np > 0] = color_outline        
                                        # Color the points inside the polygon
                                        color_points_inside = (255, 0, 0)  # Green color for points inside
                                        final_label_rgb_with_outline[points_inside_polygon] = color_points_inside
                                        cv2.putText(final_label_rgb_with_outline, "{}".format(len(np.unique(label_list))+roi_num+1), (int(max(array_0)) - 15, int(max(array_1))-10),
                           	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                                        final_label[points_inside_polygon] = len(np.unique(label_list))+roi_num + 1
                                        added_roi_list.append(len(np.unique(label_list))+roi_num + 1)
                                    st.write("Automatically and manually segmented labels")
                                    st.image(final_label_rgb_with_outline,use_column_width=True,clamp = True)           
                                    st.session_state['final_label_pg_2'] = final_label
                                    st.session_state['final_label_rgb_pg_2'] = final_label_rgb_with_outline
                                    st.session_state['label_list_pg_2'] = label_list + added_roi_list
                                    st.warning('This combination of automatic and manual object identification will be used to extract image properties. To continue only with the automatically generated labels, click "Segment and generate labels" above one more time.')   
                                               
                    
                    with st.expander("*_Cell-specific Analysis? (Expand if needed to overlay a dye positive image (uint8) on top of the labeled image for efficient cell selection_*"):
                         
                        st.session_state['overlayed_image'] = None    # for now it is None all the time, but the code is prepared in case the session state needs to get applied to these parameters
                        st.session_state['raw_file_overlay'] = None
                        st.session_state['first_raw_overlay']= None
                        if st.session_state['overlayed_image'] is None:
                            os.makedirs('temp dir_2', exist_ok = True)
                            if st.session_state['raw_file_overlay'] is None:
                                st.session_state['raw_file_overlay'] = st.file_uploader('*_Upload a dye-positive image_*', help='Use this option to overlay a dye positive image on top of the labeled image for efficient cell selection', type=['jpeg', 'jpg', 'png', 'tif', 'tiff'])
                                if st.session_state['raw_file_overlay'] is not None:                                    
                                    overlay_file_path = os.path.join('temp dir_2', st.session_state.raw_file_overlay.name)
                                    
                                    with open(overlay_file_path, "wb") as fl:
                                       fl.write(st.session_state['raw_file_overlay'].read())
                                       raw_name_overlay=cwd+'temp dir_2/'+st.session_state['raw_file_overlay'].name
                                       st.session_state['first_raw_overlay'] = load_single_image(raw_name_overlay)                                  
                                       st.session_state['first_raw_overlay'] = cv2.resize(st.session_state['first_raw_overlay'], (st.session_state['final_label_rgb_pg_2'].shape[1], st.session_state['final_label_rgb_pg_2'].shape[0]))
                                       if len(st.session_state['first_raw_overlay'].shape)==3 and st.session_state['first_raw_overlay'].shape[2] == 3:
                                           st.session_state['overlayed_image'] = cv2.addWeighted(st.session_state['final_label_rgb_pg_2'], 0.2 , st.session_state['first_raw_overlay'], 1, 0)
                                       elif len(st.session_state['first_raw_overlay'].shape)==2 or (len(st.session_state['first_raw_overlay'].shape)==3 and st.session_state['first_raw_overlay'].shape[2] == 1):
                                           st.session_state['overlayed_image'] = cv2.addWeighted(st.session_state['final_label_rgb_pg_2'], 0.2 , cv2.cvtColor(st.session_state['first_raw_overlay'], cv2.COLOR_GRAY2RGB), 1, 0)
                                       st.image(st.session_state['overlayed_image'],use_column_width=True,clamp = True) 
                                       fl.close()
                                       shutil.rmtree("temp dir_2")
                            else:                                   
                                st.image(st.session_state['overlayed_image'],use_column_width=True,clamp = True)                          
                                                        
                    if 'df_pro' in st.session_state:   
                        st.session_state.pop('df_pro')
                        
                    if 'area_thres_x' in st.session_state:
                        st.session_state.pop('area_thres_x')
                        



if __name__ == "__main__":
    main()      