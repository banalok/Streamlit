  
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_drawable_canvas import st_canvas##############################new########################
#from st_pages import Page, show_pages, hide_pages
import plotly.io
#import PIL 
import math
import tifffile
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import seaborn as sns
import statistics as stat
import os
import numpy as np
#import segmentation_models as sm
import cv2
from io import BytesIO
#import base64 
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
import imutils
# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
#from keras.preprocessing import image
from matplotlib import pyplot as plt
import pandas as pd
#from smooth_blending_functions import predict_img_with_smooth_windowing
#import scipy as sp
#from scipy.signal import medfilt
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage import measure, color, io
#from skimage.segmentation import clear_border
#from skimage import segmentation
import plotly.express as px
#from scipy import ndimage
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
#st.set_page_config(initial_sidebar_state="collapsed")

#hide_pages(['Multiple_Intensity_Traces'])
#from streamlit_extras.stateful_button import button
# for k, v in st.session_state.items():
#     st.session_state[k] = v

# if 'selected_row' not in st.session_state:
#     st.session_state['selected_row'] = 0
    
#get current directory
cwd=os.getcwd()+'/'
os.makedirs('temp dir', exist_ok = True)

if 'first_raw_image' not in st.session_state:
    st.session_state.first_raw_image = None

if "raw_file" not in st.session_state:
    st.session_state.raw_file = None

# for keys, v in st.session_state.items():
#     st.session_state[keys] = v
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
    
# def callback_show():
#      st.session_state.show_frames = True  
 
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
    
@st.cache_resource(max_entries=1, show_spinner=False, ttl = 2*60)
def load_model():
    model = StarDist2D.from_pretrained('2D_versatile_fluo') 
    #model = StarDist2D.from_pretrained('2D_paper_dsb2018')
    return model

@st.cache_data(max_entries=1, show_spinner=False, ttl = 2*60)
def load_image(images):
     img = io.imread(images, plugin='tifffile')
     # re, img = cv2.imreadmulti(images, flags=cv2.IMREAD_UNCHANGED)
     # img = np.array(img)
     return img

@st.cache_data(max_entries=1, show_spinner=False, ttl = 2*60)
def stardist_seg(im,_model):
    img_labels, img_det = _model.predict_instances(normalize(im), prob_thresh=0.6)
    return img_labels

def main():
    # selected_box = st.sidebar.selectbox(
    #     'Segment the images',
    #     ('Process a single file', 'Process multiple frames' )
    #     )
    # if selected_box == 'Process multiple frames':
    Segment()

def Segment():
    st.title('**_Segmentation of a tiff stack_**')
    
    if st.session_state.raw_file is not None:
        st.warning('Please reload the page to upload a new file')        
    else:
        st.session_state.raw_file = st.file_uploader("*_Choose an image file_*")
        
    #st.write(st.session_state.raw_file)
    if st.session_state.raw_file is not None:
        if st.session_state['first_raw_image'] is None:
            file_path = os.path.join('temp dir', st.session_state.raw_file.name)
    
            with open(file_path, "wb") as f:
                f.write(st.session_state.raw_file.read())
            #plt.save(raw_file, cwd)
            ######use this script to load the image on the deployed app############
            #file_bytes = BytesIO(raw_file.read())
            #st.image(file_bytes,use_column_width=True,clamp = True) 
            ############use this script to load the image on the deployed app############################
            #st.image(raw_file,use_column_width=True,clamp = True) 
            raw_name=cwd+'temp dir/'+st.session_state['raw_file'].name
            #st.write(raw_name)      #needs to be (none, none, 3)
            #raw_image = load_image(file_bytes) #use this script to load the image on the deployed app
            st.session_state['first_raw_image'] = load_image(raw_name)
            shutil.rmtree("temp dir")
            raw_image = st.session_state['first_raw_image']
        else:
            raw_image = st.session_state['first_raw_image']
            shutil.rmtree("temp dir")
        #raw_image = io.imread(raw_name) 
        if raw_image.dtype != 'uint8':
            img_list = []
            for frames_64 in range(0, raw_image.shape[0]):
                frame_rescale = exposure.rescale_intensity(raw_image[frames_64], out_range=(0, 255)).astype('uint8')
                img_list.append(frame_rescale)
            raw_image = np.array(img_list)
            #raw_image = exposure.rescale_intensity(raw_image, out_range=(0, 255)).astype('uint8')
            
        if (len(raw_image.shape) == 2):
            st.warning("Please upload a tiff stack with multiple frames")
            
        if (len(raw_image.shape) ==3 and raw_image.shape[-1]!=3) or (len(raw_image.shape) ==4 and raw_image.shape[-1]!=3):
            raw_image_1D = raw_image
            raw_image = np.zeros((raw_image_1D.shape[0], raw_image_1D.shape[1], raw_image_1D.shape[2], 3), dtype=np.uint8)
            # convert each 2D image from grayscale to RGB and stack them into the 4D array
            for i in range(raw_image.shape[0]):
                raw_image[i] = cv2.cvtColor(img_as_ubyte(raw_image_1D[i]), cv2.COLOR_GRAY2RGB)

        model = load_model()        
        extension = st.session_state['raw_file'].name.split(".")[-1]
        # if raw_file.name.split(".")[1] == 'tif':
        raw_image_ani = raw_image
        #st.write(raw_image_ani.shape)
        st.session_state['raw_img_ani_pg_2'] = raw_image_ani
        if extension == 'tif' and len(raw_image_ani.shape)==4:
            #with st.expander("**_Show Frames_**"):
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
                #st.write(background_corr_img.shape)                    
            # Load the image
            elif st.session_state.bc_corr_check == 'Background correction':
                background_corr_img = np.zeros(raw_image_ani.shape,dtype=np.int32)  
                                
                if 'canvas_data' not in st.session_state:
                    st.session_state.canvas_data = None
                                                 
                st.markdown("Draw one region on the image below", help= "The original stack is background corrected based on the drawn region. This corrected stack is not used for segmentation")
                image_draw = Image.fromarray(raw_image_ani[0])  # Replace 'image.jpg' with the path to your image
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
                        objects = pd.json_normalize(st.session_state.canvas_data["objects"]) # need to convert obj to str because PyArrow
                        
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
                #st.write(np.clip(np.subtract(raw_image_ani[0,2:10,2:10], (np.mean(bc_selected_region))), 0, 255).astype(np.uint8))
                #st.write(background_corr_img[0,2:10,2:10])
                if len(objects) > 1:
                    # Delete the first drawn rectangle by removing it from the JSON data
                    st.error('Please select only one region. Use the toolbar to either undo or delete the selected region(s)')                
                if len(objects) == 1:
                    st.write("*_Background Corrected Image_*")
                    bc_image_frame_num = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0,max_value = raw_image_ani.shape[0]-1, value = 0, step = 1, key='bc_1')
                    if bc_image_frame_num==0:
                        st.image(background_corr_img[0],use_column_width=True,clamp = True)
                    elif bc_image_frame_num >= 1:
                        st.image(background_corr_img[bc_image_frame_num],use_column_width=True,clamp = True)
                
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
                        
                    #diam_s = [props_final[obj_s]['equivalent_diameter_area'] for obj_s in range(0,len(props_final))]
                    #raw_image_f = raw_image_ani[frame_num]  
                    raw_image = raw_image_ani[frame_num]
                    # raw_second_pixels = min(raw_image_f[:,:,0][raw_image_f[:,:,0][:,:]!=0])            
                    # raw_image_f[raw_image_f > (255 - raw_second_pixels)] = 255 - raw_second_pixels
                    # raw_image = raw_image_f + raw_second_pixels
                    if st.session_state.gauss_x == -1:
                       blur_gauss = raw_image 
                    else:
                       blur_gauss = cv2.GaussianBlur(raw_image, (st.session_state.gauss_x, st.session_state.gauss_x), sigmaX=0)
                    #mean_of_mean.append(cv2.mean(raw_image[:,:,0])[0])  #find mean pixel value of each frame of the whole gray image
                    
                    if st.session_state.med_x == -1:
                        blur_median_proc2 = blur_gauss
                    else:
                        blur_median_proc2 = cv2.medianBlur(blur_gauss, st.session_state.med_x)               
                                
                    bri_con2 = apply_brightness_contrast(blur_median_proc2, st.session_state.bri_x, st.session_state.con_x)
                    #st.image(bri_con2,use_column_width=True,clamp = True)
                    
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
                    
                    #super_im = super_im + weight*CLAHE_img 
                    st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = np.maximum(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"], CLAHE_img)    
                
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"].astype(np.int32)
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]>255]=255
                st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"].astype(np.uint8)
                #with st.expander("**_Show Processed Frames_**"):
                st.write('*_Processed Frames_*')
                image_frame_num_pro = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0, max_value = raw_image_ani.shape[0]-1, step = 1,key='num_2')
                if image_frame_num==0:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] [0],use_column_width=True,clamp = True)
                elif image_frame_num >= 1:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] [image_frame_num_pro],use_column_width=True,clamp = True)
            
                st.markdown("**_The Collapsed Image_**")
                #with st.expander("*_Show_*"):
                st.image(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"], use_column_width=True,clamp = True) 
                st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                # with st.expander("Inhomogeneous pixel distribution?"):
                #     rb_check = st.radio("Rolling ball background correction (RBBC)", ['No RBBC', 'RBBC'], help='Select "RBBC" for rolling ball background correction on the collapsed image, otherwise, select No "RBBC"', on_change=callback_off)
                #     if rb_check == 'No RBBC':
                #         st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                #     elif rb_check == 'RBBC':
                #         radius_x = st.slider("Ball Radius", min_value = 1, max_value = 50, step=1, value = 25, on_change=callback_off)                        
                #         background_to_remove = restoration.rolling_ball(st.session_state['Collapsed_Image'], radius=radius_x)
                #         st.session_state['Collapsed_Image'] = st.session_state['Collapsed_Image'] - background_to_remove
                #         st.image(st.session_state['Collapsed_Image'], use_column_width=True,clamp = True )   
            else:
                st.write('*_Processed Frames_*')
                image_frame_num_pro = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0, max_value = raw_image_ani.shape[0]-1, step = 1,key='num_2')
                if image_frame_num==0:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0],use_column_width=True,clamp = True)
                elif image_frame_num >= 1:
                    st.image(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][image_frame_num_pro],use_column_width=True,clamp = True)
            
                st.markdown("**_The Collapsed Image_**")
                #with st.expander("*_Show_*"):
                st.image(st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"], use_column_width=True,clamp = True) 
                st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                # _, binary_image = cv2.threshold(st.session_state['Collapsed_Image'], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # st.image(binary_image,use_column_width=True,clamp = True )
     
            if st.button("*_Segment and generate labels_*", key='frame_btn',on_click = callback_allframes) or st.session_state.button_clicked_allframes:
                with st.expander("Inhomogeneous pixel distribution?"):
                    segment_check = st.radio("Select one:", ["Segment on the collapsed image", "Segment on the first image"])
                    rb_check = st.radio("Rolling ball background correction (RBBC)", ['No RBBC', 'RBBC'], help='Select "RBBC" for rolling ball background correction on the selected image, otherwise, select No "RBBC"')
                    if f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_seg" not in st.session_state:
                        if segment_check == "Segment on the collapsed image":
                            if rb_check == 'No RBBC':
                                st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                            elif rb_check == 'RBBC':
                                radius_x = st.slider("Ball Radius", min_value = 1, max_value = 50, step=1, value = 25)                        
                                background_to_remove = restoration.rolling_ball(st.session_state['Collapsed_Image'], radius=radius_x)
                                st.session_state['Collapsed_Image'] = st.session_state['Collapsed_Image'] - background_to_remove
                                st.image(st.session_state['Collapsed_Image'], use_column_width=True,clamp = True)  
                            st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] = stardist_seg(st.session_state['Collapsed_Image'],model)
                            label = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] 
                        elif segment_check == "Segment on the first image":
                            if rb_check == 'No RBBC':
                                st.session_state['Collapsed_Image'] = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]
                            elif rb_check == 'RBBC':
                                radius_x = st.slider("Ball Radius", min_value = 1, max_value = 50, step=1, value = 25)                        
                                background_to_remove = restoration.rolling_ball(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0], radius=radius_x)
                                st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0] = st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0] - background_to_remove
                        
                            st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] = stardist_seg(st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0],model)
                            label = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"] 
                            st.session_state['Collapsed_Image'] = st.session_state[f"CLAHE_img_array_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"][0]          
                    else:
                        label = st.session_state[f"super_im_{st.session_state.gauss_x}_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}_seg"]
                
                #p.write("Done!")  
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
                    #st.image(label_f,use_column_width=True,clamp = True)
                    props_to_sort = measure.regionprops(label_f)
                    centroid_positions = [prop_sort.centroid for prop_sort in props_to_sort]
                    sorted_indices = np.lexsort((np.array(centroid_positions)[:, 1], np.array(centroid_positions)[:, 0]))
                    # label the regions based on the sorted indices
                    final_label = np.zeros_like(label_f, dtype= 'uint8')
                    for i, idx in enumerate(sorted_indices, start=1):
                        final_label[label_f == (idx + 1)] = i
    
                        # st.write(seg_im.shape)
                        # st.write("Segmented image")
                        # st.image(seg_im,use_column_width=True,clamp = True) 
                        # rgba_image = Image.fromarray(seg_im, "RGB")
                        # #rgb_image = seg_im.convert("RGB")
                        # rgb_image = np.array(rgba_image)
                        # get_image_download_link(rgb_image,"segmented.png")
                    #st.write(final_label.shape)
                    #st.image(final_label,use_column_width=True,clamp = True)
                    #st.session_state['final_label_pg_2'] = final_label
                    #final_label_rgb = cv2.cvtColor(final_label, cv2.COLOR_GRAY2RGB)                                       
                    super_im_rgb = cv2.cvtColor(st.session_state['Collapsed_Image'] , cv2.COLOR_GRAY2RGB)
                    label_list_len = len([prop['label'] for prop in props_to_sort if prop['label']])
                    label_list = list(range(1,label_list_len+1))
                    st.session_state['label_list_pg_2'] = label_list
                    # ####new to display red labeled image#################
                    st.session_state['final_label_pg_2'] = final_label
                    labels_rgb = np.expand_dims(final_label, axis=2)
                    final_label_rgb = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)
                    for label in np.unique(label_list):                                 ####chek label list####                            
                      	 #if the label is zero, we are examining the 'background'
                          #so simply ignore it
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
                            #st.write(coords)
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
                    #with st.expander("*_Show the segmented and labeled image_*"):
                    st.write("*_Automatically labeled objects on the selected image_*")
                    #overlay_image = super_im + final_label_rgb[:,:,0]
                    st.image(super_im_rgb,use_column_width=True,clamp = True)
                    st.write("*_Automatically segmented and labeled objects on a black background_*")
                    st.image(final_label_rgb,use_column_width=True,clamp = True)
                    st.session_state['final_label_rgb_pg_2'] = final_label_rgb
                    st.session_state['super_im_rgb_pg_2'] = super_im_rgb
                   
                    with st.expander("*_Object(s) not detected?_*"):
                        #st.write(raw_image.shape)
                        # Add a button to allow the user to start drawing the ROI
                        if st.button('Draw ROI', key='roi_btn',help = "Use this option (only when it's absolutely necessary) to draw ROI on the labeled collapsed image", on_click = callback_roi) or st.session_state.button_clicked_roi:
                            
                            image_draw_2 = Image.fromarray(super_im_rgb)  # Replace 'image.jpg' with the path to your image
                            #st.write(image_draw_2.size)
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
                                # Get the JSON data from canvas_result (replace with your variable name)
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
                                        #st.write(f"roi:{roi_num}")
                                        roi_list = eval(objects['path'][roi_num])
                                        coordinates_roi = extract_coordinates(roi_list)
                                        #roi_label = len(np.unique(label_list))+len(objects)
                                        #label_list.append(roi_label)
                                        # #st.write(label_list)
                                        array_0 = np.array([coordinates_roi[i][0] for i in range(len(coordinates_roi))], dtype=np.int64)
                                        array_1 = np.array([coordinates_roi[i][1] for i in range(len(coordinates_roi))], dtype=np.int64)
                                        #st.write(array_0)
                                        #st.write(array_1)
                                        poly = list(zip(array_0, array_1))
                                        #st.write(array_0)
                                        # Create a mask for the polygon outline
                                        polygon_mask = Image.new('L', (final_label_rgb_with_outline.shape[1], final_label_rgb_with_outline.shape[0]) , 0)
                                        draw = ImageDraw.Draw(polygon_mask)
                                        polygon_points = poly
                                        draw.polygon(polygon_points, outline=1, fill=1)
        
                                        # Convert the mask to a NumPy array
                                        polygon_mask_np = np.array(polygon_mask)
                                        #st.write(polygon_mask_np.shape)
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
                                               
                    if 'df_pro' in st.session_state:   
                        st.session_state.pop('df_pro')
                        
                    if 'area_thres_x' in st.session_state:
                        st.session_state.pop('area_thres_x')
                        
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def image_stdev(region, intensities):
    return np.std(intensities[region])

def image_mode(region, intensities):
    return stat.mode(intensities[region])

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
   
   if brightness != 0:
       if brightness > 0:
           shadow = brightness
           highlight = 255
       else:
           shadow = 0
           highlight = 255 + brightness
       alpha_b = (highlight - shadow)/255
       gamma_b = shadow
       
       buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
       buf =  np.clip(buf, 0, 255).astype(np.uint8) 
   else:
       buf = input_img.copy()
   
   if contrast != 0:
       f = 131*(contrast + 127)/(127*(131-contrast))
       alpha_c = f
       gamma_c = 127*(1-f)
       
       buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
       buf =  np.clip(buf, 0, 255).astype(np.uint8)
   return buf 


def get_image_download_link(img,filename_with_extension):
    result = Image.fromarray(img.astype(np.uint8))
    buffered = BytesIO()
    result.save(buffered, format="PNG")
    byte_im_2 = buffered.getvalue()
    #img_str = base64.b64encode(buffered.getvalue()).decode()
    #href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    st.download_button("Press to Download", byte_im_2, filename_with_extension, "image/JPEG")
    #return href   


def intensity(df_1, multi_tif_img, window):
    img_frames_list = list(range(0,multi_tif_img.shape[0]))
    img_frames = pd.DataFrame(img_frames_list, columns = ['Frame'])
    mean_intensity = []
    p_count = []
    #change_in_F = []
    for frames_pro in range(0,multi_tif_img.shape[0]):
            #new_df = pd.DataFrame(frames_pro, df_pro[f'intensity_mean_{frames_pro}'].mean(),  columns = ['Frames', 'Mean Intensity'])
        mean_intensity.append(df_1[f'intensity_mean_{frames_pro}'].mean()) #[0]
        p_count.append(df_1[f'Bright_pixel_area_{frames_pro}'].mean()) #[0]
            # new_df = pd.DataFrame.from_dict({'Frame': frames_pro, 'Mean Intensity': df_pro[f'intensity_mean_{frames_pro}'].mean()})
            # img_frames = pd.merge(img_frames, new_df, on = "Frame")
        #st.write(df_1[f'pixel_count_{frames_pro}'])
        #change_f = fluo_change(mean_intensity[frames_pro], baseline)
        #change_in_F.append(change_f)
        
    #change_F_df = pd.DataFrame(change_in_F, columns = ['delta_F/F'])
    smooth_F_df = pd.DataFrame(smooth_plot(mean_intensity, window), columns = ['Smoothed Mean Intensity'] ) #pd.DataFrame(smooth_df, columns = ['smoothed mean intensity'])
    mean_inten_df = pd.DataFrame(mean_intensity)
    pixel_count_df = pd.DataFrame(p_count, columns = ['Bright Pixel Area'])
    new_d = pd.concat([img_frames, mean_inten_df, smooth_F_df, pixel_count_df],axis=1)
    new_d.rename(columns = {0 : 'Mean Intensity'}, inplace=True)
    #new_d.rename(columns = {1 : 'Bright Pixel Number'}, inplace=True)
    return new_d 

def fluo_change(intensity_mean, baseline):
    delta_F = intensity_mean - baseline
    change_f = delta_F/baseline
    return change_f

def smooth_plot(unsmoothed_intensity, window):
    smooth_df = (np.convolve(unsmoothed_intensity, np.ones((window)), mode = 'valid'))/window #ndimage.median_filter(unsmoothed_intensity,7)
    return smooth_df

def mono_exp_decay(t, a, b):
    return a * np.exp(-b * t)

def extract_coordinates(data):
    coordinates = []
    current_x, current_y = None, None

    for item in data:
        if item[0] == 'M' or item[0] == 'L' :
            # Move To command
            current_x, current_y = item[1], item[2]
            coordinates.append((current_x, current_y))
        elif item[0] == 'z':
            current_x, current_y = data[0][1], data[0][2]
        elif item[0] == 'Q':
            # Quadratic Bezier Curve To command
            control_x, control_y, end_x, end_y = item[1], item[2], item[3], item[4]
            if current_x is not None and current_y is not None:
                # Interpolate points along the curve
                num_points = 100  # Adjust as needed for precision
                for t in range(1, num_points + 1):
                    t_normalized = t / num_points
                    x = (1 - t_normalized) ** 2 * current_x + 2 * (1 - t_normalized) * t_normalized * control_x + t_normalized ** 2 * end_x
                    y = (1 - t_normalized) ** 2 * current_y + 2 * (1 - t_normalized) * t_normalized * control_y + t_normalized ** 2 * end_y
                    coordinates.append((x, y))
                current_x, current_y = end_x, end_y

    return coordinates

# Function to fill the polygon and return the filled area as a mask
def fill_polygon(coordinates, canvas_size):
    image = Image.new("L", canvas_size, 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(coordinates, outline=1, fill=1)
    return np.array(image)


if __name__ == "__main__":
    main()      