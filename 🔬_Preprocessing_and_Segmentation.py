  
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_drawable_canvas import st_canvas
#from st_pages import Page, show_pages, hide_pages
import plotly.io
#import PIL 
import math
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
from skimage import measure, color, io
#from skimage.segmentation import clear_border
#from skimage import segmentation
import plotly.express as px
#from scipy import ndimage
from skimage import (
    filters,  morphology, img_as_float, img_as_ubyte, exposure
)
from skimage.draw import polygon
from stardist.models import StarDist2D
from stardist.plot import render_label
from csbdeep.utils import normalize
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import time
import subprocess

#st.set_page_config(initial_sidebar_state="collapsed")

#hide_pages(['Multiple_Intensity_Traces'])
#from streamlit_extras.stateful_button import button
# for k, v in st.session_state.items():
#     st.session_state[k] = v

# if 'selected_row' not in st.session_state:
#     st.session_state['selected_row'] = 0
    
#get current directory
cwd=os.getcwd()+'/'

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
    return model

#@st.cache_data(max_entries=1, show_spinner=False, ttl = 2*60)
def load_image(images):
     img = io.imread(images, plugin='tifffile')
     # re, img = cv2.imreadmulti(images, flags=cv2.IMREAD_UNCHANGED)
     # img = np.array(img)
     return img

@st.cache_data(max_entries=1, show_spinner=False, ttl = 2*60)
def stardist_seg(im,_model):
    img_labels, img_det = _model.predict_instances(normalize(im))
    return img_labels

# @st.cache(allow_output_mutation=True)
# def show_ani(images):
#     fig = px.imshow(
#         images,
#         animation_frame=0,
#         binary_string=True,
#         labels={'animation_frame': 'time point'}
#         )
#     #chart = st. plotly_chart(fig, use_container_width=False, sharing="streamlit", theme="streamlit")
#     return st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

# @st.cache(allow_output_mutation=True)
# def show_video(img):
#     #video_frames = []
#     output_file = "output.mp4"
#     # Add each video frame to the list as a PIL Image object
#     for i, name in enumerate(range(img.shape[0])):
#         #img_arr = img[i]
#         img_arr_name = f"frame_{i}.tif"
#         image = cv2.cvtColor(img[name],cv2.COLOR_RGB2GRAY)
#         Image.fromarray(image).save(img_arr_name)
#         #video_frames.append(img_v)
#     command = f"ffmpeg -r 30 -i frame_%d.tif -c:v libx264 -preset slow -crf 22 {output_file}"
#     subprocess.call(command, shell=True)
#     # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
#     # video_writer = cv2.VideoWriter(output_file_path, fourcc, 30.0, (video_frames[0].width, video_frames[0].height))
    
#     # for frame in video_frames:
#     #     frame = np.array(frame)
#     #     video_writer.write(frame) 
#     # video_writer.release()
#   # Read the output video file
#     with open(output_file, "rb") as f:
#         video_bytes = f.read()
#     return video_bytes


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
        st.image(st.session_state.raw_file,use_column_width=True,clamp = True)
    else:
        st.session_state.raw_file = st.file_uploader("*_Choose an image file_*")
        #st.image(st.session_state.raw_file,use_column_width=True,clamp = True)
    
    if st.session_state.raw_file is not None:
        #plt.save(raw_file, cwd)
        ######use this script to load the image on the deployed app############
        file_bytes = BytesIO(st.session_state.raw_file.read())
        file_bytes = ((np.frombuffer(file_bytes, dtype=np.uint8) / 255.0) * 255).astype(np.uint8)
        #st.write(type(file_bytes))
        #st.image(file_bytes,use_column_width=True,clamp = True) 
        ############use this script to load the image on the deployed app############################
        
        #st.image(raw_file,use_column_width=True,clamp = True) 
        #raw_name=cwd+st.session_state['raw_file'].name
        #st.write(raw_name)      #needs to be (none, none, 3)
        raw_image = load_image(file_bytes) #use this script to load the image on the deployed app
        #raw_image = load_image(raw_name)
        #raw_image = io.imread(raw_name) 
        if raw_image.dtype != 'uint8':
            raw_image = exposure.rescale_intensity(raw_image, out_range=(0, 255)).astype('uint8')      
            
        if (len(raw_image.shape) == 2) or (len(raw_image.shape) ==3 and raw_image.shape[-1]!=3) or (len(raw_image.shape) ==4 and raw_image.shape[-1]!=3):
            raw_image_1D = raw_image
            raw_image = np.zeros((raw_image_1D.shape[0], raw_image_1D.shape[1], raw_image_1D.shape[2], 3), dtype=np.uint8)
            # convert each 2D image from grayscale to RGB and stack them into the 4D array
            for i in range(raw_image.shape[0]):
                raw_image[i] = cv2.cvtColor(img_as_ubyte(raw_image_1D[i]), cv2.COLOR_GRAY2RGB)
           
        #st.write(raw_image.dtype)
        model = load_model()        
        extension = st.session_state['raw_file'].name.split(".")[1]
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
            
            #btn_clk = st.form('Show Frmaes', clear_on_submit=False)
            #if (st.button("Show frames", on_click=callback_frame) or st.session_state.button_clicked):
            # with st.form("my_form"):  
            #     submitted = st.form_submit_button("Show video")
            #     if submitted:  
            #        video_bytes = show_video(raw_image_ani) 
            #        st.video(video_bytes)
                # if btn_clk:# or st.session_state.show_frames:                
                #     st.subheader("Image frames")             
                #     show_ani(raw_image_ani) 
                
                #st.session_state['frames_displayed'] = True              

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
                               
                    
                for bc in range(0, raw_image_ani.shape[0]):
                    #rerun_flag = False
                    #if canvas_result.json_data is not None:
                    if st.session_state.canvas_data is not None:
                        objects = pd.json_normalize(st.session_state.canvas_data["objects"]) # need to convert obj to str because PyArrow
                        
                        if len(objects) == 0:
                            background_corr_img = raw_image_ani
                            break
                                            
                        elif len(objects) == 1:
                            #st.write(raw_image_ani[bc][int(objects['top']):int(objects['top'])+int(objects['height']), int(objects['left']):int(objects['left'])+ int(objects['width'])])
                            bc_selected_region = raw_image_ani[bc][int(objects['top']):int(objects['top'])+int(objects['height']), int(objects['left']):int(objects['left'])+ int(objects['width'])]
                            background_corr_img[bc] = np.subtract((raw_image_ani[bc]).astype(np.int32), (np.mean(bc_selected_region)).astype(np.int32))
                            background_corr_img[bc] = np.clip(background_corr_img[bc], 0, 255)
                            
                        elif len(objects) > 1: 
                            background_corr_img = raw_image_ani
                            break
                background_corr_img = background_corr_img.astype(np.uint8)
                       
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
    
            #mean_of_mean = []   
            if 'med_x' not in st.session_state:
                st.session_state.med_x = st.slider("*_Median Blur Kernel size_*", min_value = 1,max_value = 100, step = 2,help = "Filters are applied to all frames. Check 'Processed Frames' below", on_change=callback_off)
            else:
                st.session_state.med_x = st.slider('*_Median Blur Kernel size_*', min_value = 1,max_value = 100, value = st.session_state.med_x, step = 2,help = "Filters are applied to all frames. Check 'Processed Frames' below", on_change=callback_off)
                
            if 'bri_x' not in st.session_state:            
                st.session_state.bri_x = st.slider('*_Change Brightness_*',min_value = 0,max_value = 255, on_change=callback_off)
            else:
                st.session_state.bri_x = st.slider('*_Change Brightness_*',min_value = 0,max_value = 255, value = st.session_state.bri_x, on_change=callback_off)                
            
            if 'con_x' not in st.session_state:
                st.session_state.con_x = st.slider('*_Change Contrast_*',min_value = 0,max_value = 255, on_change=callback_off)
            else:
                st.session_state.con_x = st.slider('*_Change Contrast_*',min_value = 0,max_value = 255, value = st.session_state.con_x, on_change=callback_off)
                
            if 'hist_x' not in st.session_state:   
                st.session_state.hist_x = st.slider("*_Histogram Equalization cliplimit factor (CLAHE)_*", min_value = 1,max_value = 10, step = 1, on_change=callback_off)
            else:
                st.session_state.hist_x = st.slider("*_Histogram Equalization cliplimit factor (CLAHE)_*", min_value = 1,max_value = 10, step = 1, value = st.session_state.hist_x, on_change=callback_off)
            
            CLAHE_img_array = []
            super_im = np.zeros_like(raw_image_ani[0][:,:,0])   #size of one of the image
            weight = 1/raw_image_ani.shape[0]                      
            for frame_num in range(0,raw_image_ani.shape[0]):
                    
                #diam_s = [props_final[obj_s]['equivalent_diameter_area'] for obj_s in range(0,len(props_final))]
                #raw_image_f = raw_image_ani[frame_num]  
                raw_image = raw_image_ani[frame_num]
                # raw_second_pixels = min(raw_image_f[:,:,0][raw_image_f[:,:,0][:,:]!=0])            
                # raw_image_f[raw_image_f > (255 - raw_second_pixels)] = 255 - raw_second_pixels
                # raw_image = raw_image_f + raw_second_pixels
                                     
                #mean_of_mean.append(cv2.mean(raw_image[:,:,0])[0])  #find mean pixel value of each frame of the whole gray image
                
                blur_median_proc2 = cv2.medianBlur(raw_image, st.session_state.med_x)               
                            
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
                CLAHE_img_array.append(CLAHE_img) 
                
                super_im = super_im + weight*CLAHE_img 
            super_im = (np.round(super_im)).astype('uint8')
            super_im[super_im>255]=255
            #with st.expander("**_Show Processed Frames_**"):
            st.write('*_Processed Frames_*')
            image_frame_num_pro = st.number_input(f"(0 - {raw_image_ani.shape[0]-1})", min_value = 0, max_value = raw_image_ani.shape[0]-1, step = 1,key='num_2')
            if image_frame_num==0:
                st.image(CLAHE_img_array[0],use_column_width=True,clamp = True)
            elif image_frame_num >= 1:
                st.image(CLAHE_img_array[image_frame_num_pro],use_column_width=True,clamp = True)
            
            st.markdown("**_The Collapsed Image_**")
            #with st.expander("*_Show_*"):
            st.image(super_im, use_column_width=True,clamp = True) 
            st.session_state['Collapsed_Image'] = super_im
            if st.button("*_Segment and generate labels_*", key='frame_btn',on_click = callback_allframes) or st.session_state.button_clicked_allframes:
                #st.image(super_im,use_column_width=True,clamp = True)   
                #st.write(st.session_state.button_clicked_allframes)
                if f"super_im_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}" not in st.session_state:
                #st.write("It is not in session state")
                    st.session_state[f"super_im_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"] = stardist_seg(super_im,model)
                label = st.session_state[f"super_im_{st.session_state.med_x}_{st.session_state.bri_x}_{st.session_state.con_x}_{st.session_state.hist_x}"]      

                #p.write("Done!")  
                props = measure.regionprops(label)
                if len(props) == 0:
                    st.warning("No region of interests found. Please use another file or try pre-processing.")
                else:
                    diam = [props[obj_s]['equivalent_diameter_area'] for obj_s in range(0,len(props))]
                    labels_to_keep_len = len([prop['label'] for prop in props if prop['equivalent_diameter_area'] > 0.5*stat.mean(diam)])
                    labels_to_keep = list(range(1, labels_to_keep_len))
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
                    st.session_state['final_label_pg_2'] = final_label
                    final_label_rgb = cv2.cvtColor(final_label, cv2.COLOR_GRAY2RGB)
                    super_im_rgb = cv2.cvtColor(super_im, cv2.COLOR_GRAY2RGB)
                    label_list_len = len([prop['label'] for prop in props_to_sort if prop['label']])
                    label_list = list(range(1,label_list_len+1))
                    st.session_state['label_list_pg_2'] = label_list
                    for label in np.unique(label_list):                                 ####chek label list####                            
                      	 #if the label is zero, we are examining the 'background'
                          #so simply ignore it
                            if label == 0:
                                continue                
                            mask = np.zeros(CLAHE_img.shape, dtype="uint8")
                            mask[final_label == label] = 255
                            #detect contours in the mask and grab the largest one
                            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
                            cnts = imutils.grab_contours(cnts)
                            c = max(cnts, key=cv2.contourArea)
                            # draw a circle enclosing the object
                            ((x, y), r) = cv2.minEnclosingCircle(c)
                            cv2.circle(final_label_rgb, (int(x), int(y)), int(r), (255, 0, 0), 1)
                            cv2.putText(final_label_rgb, "{}".format(label), (int(x) - 10, int(y)),
                             	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            cv2.circle(super_im_rgb, (int(x), int(y)), int(r), (255, 0, 0), 1)
                            cv2.putText(super_im_rgb, "{}".format(label), (int(x) - 10, int(y)),
                             	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)                          
                    #with st.expander("*_Show the segmented and labeled image_*"):
                    st.markdown("*_Segmented and labeled image overlayed on the collapsed image_*", help = 'The red circles just show the location of the segmented labels and are not the actual segmented labels themselves')
                    st.image(super_im_rgb,use_column_width=True,clamp = True)
                    st.markdown("*_Segmented and labeled image overlayed on a black background_*", help = 'The red circles just show the location of the segmented labels and are not the actual segmented labels themselves')
                    st.image(final_label_rgb,use_column_width=True,clamp = True)
                    st.session_state['final_label_rgb_pg_2'] = final_label_rgb
                    st.session_state['super_im_rgb_pg_2'] = super_im_rgb        
                    
                    # col_pg_1, col_pg_2, col_pg_3 = st.columns(3)
                    # with col_pg_1:
                    #     if st.button("**_Single Intensity Traces_**"):
                    #         switch_page('Single intensity traces')
                    # with col_pg_3:
                    #     if st.button("**_Multiple Intensity Traces_**"):
                    #         switch_page('📉 Multiple intensity traces')                                
                
         
                                                                         
                           
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
   else:
       buf = input_img.copy()
   
   if contrast != 0:
       f = 131*(contrast + 127)/(127*(131-contrast))
       alpha_c = f
       gamma_c = 127*(1-f)
       
       buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

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


if __name__ == "__main__":
    st.set_page_config(page_title="Segmentation", page_icon=None, layout="centered", initial_sidebar_state="expanded", menu_items=None)
    main()      