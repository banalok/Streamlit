  
import streamlit as st
import plotly.io
#import PIL 
import math
from PIL import Image
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
#from streamlit_extras.stateful_button import button
# for k, v in st.session_state.items():
#     st.session_state[k] = v

# if 'selected_row' not in st.session_state:
#     st.session_state['selected_row'] = 0
    
#get current directory
cwd=os.getcwd()+'/'

# for keys in st.session_state.keys():
#     st.write(keys) 


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
    
# if "show_frames" not in st.session_state:
#     st.session_state.show_frames = False

if "button_clicked_allframes" not in st.session_state:
    st.session_state.button_clicked_allframes = False
    
if 'display_table' not in st.session_state:
    st.session_state.display_table = False
    
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

def callback_table():
   st.session_state.display_table = True
    
@st.cache(allow_output_mutation=True, max_entries=1, show_spinner=False, ttl = 2*60)
def load_model():
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    return model

@st.cache(allow_output_mutation=True, max_entries=1, show_spinner=False, ttl = 2*60)
def load_image(images):
     img = io.imread(images)
     # re, img = cv2.imreadmulti(images, flags=cv2.IMREAD_UNCHANGED)
     # img = np.array(img)
     return img

@st.cache(allow_output_mutation=True, max_entries=1, show_spinner=False, ttl = 2*60)
def stardist_seg(im,model):
    img_labels, img_det = model.predict_instances(normalize(im))
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
    st.title('Segmentation of a tiff stack')
    raw_file = st.file_uploader("Choose an image file")
    #st.write(raw_file)
    if raw_file is not None:
        
        #plt.save(raw_file, cwd)
        ######use this script to load the image on the deployed app############
        file_bytes = BytesIO(raw_file.read())
        #st.image(file_bytes,use_column_width=True,clamp = True) 
        ############use this script to load the image on the deployed app############################
        
        #st.image(raw_file,use_column_width=True,clamp = True) 
        #raw_name=cwd+raw_file.name
        #st.write(raw_name)      #needs to be (none, none, 3)
        raw_image = load_image(file_bytes) #use this script to load the image on the deployed app
        #raw_image = load_image(raw_name)
        #raw_image = io.imread(raw_name) 
        #st.write(raw_image.shape)
        if (len(raw_image.shape) == 2) or (len(raw_image.shape) ==3 and raw_image.shape[-1]!=3) or (len(raw_image.shape) ==4 and raw_image.shape[-1]!=3):
            raw_image = cv2.cvtColor(img_as_ubyte(raw_image), cv2.COLOR_GRAY2RGB)
           
        #st.write(raw_image.dtype)
        model = load_model()        
        extension = raw_file.name.split(".")[1]
        # if raw_file.name.split(".")[1] == 'tif':
        raw_image_ani = raw_image
        #st.write(raw_image_ani.shape)
        
        if extension == 'tif' and len(raw_image_ani.shape)==4:
            image_frame_num = st.number_input(f"Show frames (0 - {raw_image_ani.shape[0]-1})", min_value = 0,max_value = raw_image_ani.shape[0], value = 0, step = 1)
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
            #mean_of_mean = []    
            med_x = st.slider("Median Blur Kernel size", min_value = 1,max_value = 100, step = 2, on_change=callback_off)
            bri_x = st.slider('Change Brightness',min_value = 0,max_value = 255, on_change=callback_off)
            con_x = st.slider('Change Contrast',min_value = 0,max_value = 255, on_change=callback_off)
            
            #diam_f = []
            label_list = []
            centroid_set_first = []
            final_label_1 = np.zeros((raw_image_ani.shape[1], raw_image_ani.shape[2]),dtype=np.uint8)
            #raw_image_1 = raw_image_ani[0]           
            raw_image_first = raw_image_ani[0] 
            # raw_first_pixels = min(raw_image_1[:,:,0][raw_image_1[:,:,0][:,:]!=0])            
            # raw_image_1[raw_image_1 > (255 - raw_first_pixels)] = 255 - raw_first_pixels
            # raw_image_first = raw_image_1 + raw_first_pixels
            
            #mean_of_mean.append(cv2.mean(raw_image_first[:,:,0])[0])
            #raw_image_first = raw_image_first[:,:,0]
            
            # st.write(raw_image.shape)
            # st.image(raw_image,use_column_width=True,clamp = True)
            #processed_tiff = []        
                  
            blur_median_proc1 = cv2.medianBlur(raw_image_first,med_x)               
                            
            bri_con1 = apply_brightness_contrast(blur_median_proc1, bri_x, con_x)
            st.image(bri_con1,use_column_width=True,clamp = True)
            
            lab_img1= cv2.cvtColor(bri_con1, cv2.COLOR_RGB2LAB)
            l1, a1, b1 = cv2.split(lab_img1)
            equ1 = cv2.equalizeHist(l1)
            updated_lab_img1 = cv2.merge((equ1,a1,b1))
            hist_eq_img1 = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2RGB)
            
            clahe1 = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
            clahe_img1 = clahe1.apply(l1)
            updated_lab_img11 = cv2.merge((clahe_img1,a1,b1))
            CLAHE_img_first = cv2.cvtColor(updated_lab_img11, cv2.COLOR_LAB2RGB)
            CLAHE_img_first = CLAHE_img_first[:,:,0]
                
            st.write("The processed frame (after histogram equalization)")                   
            st.image(CLAHE_img_first,use_column_width=True,clamp = True)   
            st.subheader("Segmentation")              
            
            #stardist_labels = f"CLAHE_img_first_{med_x}_{bri_x}_{con_x}"
            if f"CLAHE_img_first_{med_x}_{bri_x}_{con_x}" not in st.session_state:
                #st.write("It is not in session state")
                st.session_state[f"CLAHE_img_first_{med_x}_{bri_x}_{con_x}"] = stardist_seg(CLAHE_img_first,model)
                
            # else:    
            #     st.write("It is in session state")    
            label_first = st.session_state[f"CLAHE_img_first_{med_x}_{bri_x}_{con_x}"]                      
            props_first = measure.regionprops(label_first)
            label_val = 1
   
            for obj_first in range(0,len(props_first)): 
                centroid_coord_first = props_first[obj_first]['centroid']
                #diameter_first = props_first[obj_first]['equivalent_diameter_area']
                #diam_f.append(diameter_first)
                centroid_set_first.append(centroid_coord_first)
                diameter_first = props_first[obj_first]['equivalent_diameter_area']
                if diameter_first > 10:
                    #if (centroid_coord_first[0]<200 and centroid_coord_first[0]>0)  and (centroid_coord_first[1]>0 and centroid_coord_first[1]<500):
                    object_coords = np.argwhere(label_first==props_first[obj_first]['label'])
                    label_value_first = polygon(object_coords[:, 0], object_coords[:, 1])
                    final_label_1[label_value_first] = label_val        #object_label = props[objects]['label'] 
                    label_list.append(label_val)
                    label_val += 1
            
            #if button("Show segmented frame 1", key='seg_btn'): 
            #st.write(st.session_state.button_clicked)
            if st.button("Show segmented frame 1 (Overlayed on the original image)", key='seg_btn', on_click = callback) or st.session_state.button_clicked:
                #st.write(st.session_state.button_clicked)
                seg_im = render_label(final_label_1, img=raw_image_first)
                st.image(seg_im,use_column_width=True,clamp = True)  
                #st.write(label_list)
                cutoff = 10
                p = st.empty()
                #st.write(st.session_state.button_clicked_allframes)
                #if button("Apply to all frames", key='frame_btn'):
                if st.button("Apply to all frames", key='frame_btn',on_click = callback_allframes) or st.session_state.button_clicked_allframes:
                    #st.write(st.session_state.button_clicked_allframes)
                    
                    for frame_num in range(1,raw_image_ani.shape[0]):
                        #raw_image_f = raw_image_ani[frame_num]  
                        raw_image = raw_image_ani[frame_num]
                        # raw_second_pixels = min(raw_image_f[:,:,0][raw_image_f[:,:,0][:,:]!=0])            
                        # raw_image_f[raw_image_f > (255 - raw_second_pixels)] = 255 - raw_second_pixels
                        # raw_image = raw_image_f + raw_second_pixels
                                             
                        #mean_of_mean.append(cv2.mean(raw_image[:,:,0])[0])  #find mean pixel value of each frame of the whole gray image
                        
                        blur_median_proc2 = cv2.medianBlur(raw_image,med_x)               
                                    
                        bri_con2 = apply_brightness_contrast(blur_median_proc2, bri_x, con_x)
                        #st.image(bri_con2,use_column_width=True,clamp = True)
                        
                        lab_img2= cv2.cvtColor(bri_con2, cv2.COLOR_RGB2LAB)
                        l2, a2, b2 = cv2.split(lab_img2)
                        equ2 = cv2.equalizeHist(l2)
                        updated_lab_img2 = cv2.merge((equ2,a2,b2))
                        hist_eq_img2 = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2RGB)
                        
                        clahe2 = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
                        clahe_img2 = clahe2.apply(l2)
                        updated_lab_img22 = cv2.merge((clahe_img2,a2,b2))
                        CLAHE_img = cv2.cvtColor(updated_lab_img22, cv2.COLOR_LAB2RGB)
                        CLAHE_img = CLAHE_img[:,:,0]
        
                        
                        
                        p.write(f"Working on frame number {frame_num+1}") 
                        #time.sleep(0.05) 
                        #st.write(f"Working on frame number {frame_num+1}")
                        #stardist_labels_rest = f"CLAHE_img_{med_x}_{bri_x}_{con_x}"
                        if f"CLAHE_img_{med_x}_{bri_x}_{con_x}" not in st.session_state:
                            st.session_state[f"CLAHE_img_{med_x}_{bri_x}_{con_x}"] = stardist_seg(CLAHE_img,model)
                        labels = st.session_state[f"CLAHE_img_{med_x}_{bri_x}_{con_x}"]
                        #labels = stardist_seg(raw_image,model)
                        #plt.imsave(f"F:/Intensity measurement/check labels/label_{frame_num}.jpg", labels)
                        props = measure.regionprops(labels) 
                                                
                        for objects in range(0,len(props)): 
                            centroid_coord = props[objects]['centroid']
                            diameter = props[objects]['equivalent_diameter_area']
                            
                            flag = False
                            for item in range(0, len(centroid_set_first)):
                                if (centroid_coord in centroid_set_first) or ((abs(centroid_coord[0]-centroid_set_first[item][0])<=cutoff) and (abs(centroid_coord[1]-centroid_set_first[item][1])<=cutoff)) or diameter < 10: # or (centroid_coord[0]>=200 and centroid_coord[1]>=0) :
                                    #st.write(f"I'm here {objects}")
                                    flag = True
                                    break
                                else:                                    
                                    continue
                                
                            if not flag:                               
                                object_coords = np.argwhere(labels==props[objects]['label'])
                                label_value = polygon(object_coords[:, 0], object_coords[:, 1])
                                final_label_1[label_value] = label_val
                                label_list.append(label_val)
                                #plt.imsave(f"F:/Intensity measurement/check labels/label_{frame_num}_{label_val}.jpg", final_label==label_val)
                                label_val += 1 
                                centroid_set_first.append(centroid_coord)
                    p.write("Done!")  
                    #st.image(final_label_1,use_column_width=True,clamp = True)
                    props_to_sort = measure.regionprops(final_label_1)
                    centroid_positions = [prop_sort.centroid for prop_sort in props_to_sort]
                    sorted_indices = np.lexsort((np.array(centroid_positions)[:, 1], np.array(centroid_positions)[:, 0]))
                    # label the regions based on the sorted indices
                    final_label = np.zeros_like(final_label_1)
                    for i, idx in enumerate(sorted_indices, start=1):
                        final_label[final_label_1 == (idx + 1)] = i

                    # st.write(seg_im.shape)
                    # st.write("Segmented image")
                    # st.image(seg_im,use_column_width=True,clamp = True) 
                    # rgba_image = Image.fromarray(seg_im, "RGB")
                    # #rgb_image = seg_im.convert("RGB")
                    # rgb_image = np.array(rgba_image)
                    # get_image_download_link(rgb_image,"segmented.png")
                    final_label_rgb = cv2.cvtColor(final_label, cv2.COLOR_GRAY2RGB)
                    for label in np.unique(label_list):
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
                    
                    st.image(final_label_rgb,use_column_width=True,clamp = True)
                        
                    if st.button("Show intensity table", on_click = callback_table) or st.session_state.display_table:
                        
                        #total_mean = np.sum(mean_of_mean)/raw_image_ani.shape[0]
                        #st.write(total_mean)
                        #st.write(mean_of_mean)
                        st.write("Intensity table for segmented Image") 
                        #st.image(labels,use_column_width=True,clamp = True)          
                                   
                        data = list(np.unique(label_list)) 
                        
                        df_pro = pd.DataFrame(data, columns=['label'])
                        
                        for frames_pro in range(0,raw_image_ani.shape[0]):
                            props_pro = measure.regionprops_table(final_label, intensity_image=raw_image_ani[frames_pro][:,:,0],   #markers
                                                                  properties=['label','intensity_mean'])
                            
                            df_single = pd.DataFrame(props_pro)
                            df_single['intensity_mean'] = np.round(df_single['intensity_mean'],3)
                            df_single.rename(columns = {'intensity_mean' : f'intensity_mean_{frames_pro}'}, inplace=True)
                            #df_single.rename(columns = {'solidity' : f'solidity_{frames_pro}'}, inplace=True)
                            df_pro = pd.merge(df_pro, df_single, on = 'label', how = 'outer')                                                
                            
                        #df_pro.drop([0], inplace=True)
                        
                        #df_pro = df_pro.drop(df_pro[df_pro['label'] == 255].index)
                            
                            ###############Interactive table################################################################
                        st.write(df_pro)
                        get_data_indi = convert_df(df_pro)
                        st.download_button("Press to Download", get_data_indi, 'label_intensity_data.csv', "text/csv", key='label_download-get_data')                      
                        st.write('Select a label to explore')
                        
                        gb = GridOptionsBuilder.from_dataframe(df_pro)                       
                        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                        gb.configure_side_bar() #Add a sidebar
                        #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                        gb.configure_selection(selection_mode="single", use_checkbox=True, groupSelectsChildren="Group checkbox select children", pre_selected_rows=[0]) #list(range(0, len(df_pro))))  #[str(st.session_state.selected_row)]
                                   
                        gridOptions = gb.build()
                        #gridOptions["columnDefs"][0]["checkboxSelection"]=True
                        #gridOptions["columnDefs"][0]["headerCheckboxSelection"]=True
                        
                        grid_response = AgGrid(
                            df_pro,
                            gridOptions=gridOptions,
                            data_return_mode='AS_INPUT', 
                            update_mode=GridUpdateMode.SELECTION_CHANGED,    #'MODEL_CHANGED',
                            update_on='MANUAL',
                            #data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            fit_columns_on_grid_load=False,
                            theme='alpine', #Add theme color to the table
                            enable_enterprise_modules=True,
                            height=350, 
                            width='100%',
                            key='table_key'
                        )
                        
                        data = grid_response['data']
                        selected = grid_response['selected_rows'] 
                        # if len(selected) != 0:
                        #     st.write(selected[0])
                            # st.session_state.selected_row = selected_rows[0]['_selectedRowNodeInfo']['nodeRowIndex']
                        df_selected = pd.DataFrame(selected)
                        labels_rgb = np.expand_dims(final_label, axis=2)
                        labels_rgbb = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)
                        
                        if selected: 
                            #st.write(df_selected)# Loop over the selected indices and draw polygons on the color image
                            for i in df_selected['label']:
                                # Extract the coordinates of the region boundary
                                coords = np.argwhere(final_label==i)
                            
                                # Create a polygon from the coordinates
                                poly = polygon(coords[:, 0], coords[:, 1])
                            
                                # Set the color of the polygon to red
                                color_poly = (255, 0, 0)
                            
                                # Color the polygon in the color image
                                labels_rgbb[poly] = color_poly
                            
                            # Display the color image with the selected regions highlighted
                            st.image(labels_rgbb,use_column_width=True,clamp = True)
                            df_selected = df_selected.drop(columns = ['_selectedRowNodeInfo'])

                            nested_dict = {'Label':[], "Number of Events":[], "Rise time":[], "Decay time":[], "Duration":[], "Amplitude":[]}
                            plot_df = intensity(df_selected, raw_image_ani)                                                      
                            #smoothed_plot_df = plot_df['Smoothed Mean Intensity']                           
                            missing_values = pd.isna(plot_df["Smoothed Mean Intensity"])
                            plot_df.loc[missing_values, "Smoothed Mean Intensity"] = plot_df.loc[missing_values, "Mean Intensity"]  
                            #smooth_mode = stat.mode(smoothed_plot_df)
                            #sd_smooth = plot_df['Smoothed Mean Intensity'].std()
                            #smooth_baseline_mean_sd = smooth_mode + sd_smooth
                            #st.write(smooth_mode)
                            #st.write(smooth_baseline_mean_sd)
                            plot_df["Smoothed Mean Intensity"] = np.round(plot_df["Smoothed Mean Intensity"],3)                            
                            plot_df_smooth_mode =plot_df.mode()['Smoothed Mean Intensity'].min() #stat.mode(new_df_selected_transposed_smooth[f"smooth cell {i}"])
                            plot_df_smooth_sd = plot_df['Smoothed Mean Intensity'].std()
                            baseline_each = plot_df_smooth_mode + plot_df_smooth_sd   
                            plot_df['delta_f/f_0'] = (plot_df['Smoothed Mean Intensity'] - baseline_each)/baseline_each  
                            unsmooth_mode = stat.mode(plot_df['Mean Intensity'])
                            sd_unsmooth = plot_df['Mean Intensity'].std()
                            unsmooth_baseline_mean_sd = unsmooth_mode + sd_unsmooth
                            # st.write(baseline_each)
                            # st.write(plot_df_smooth_mode)
                            # st.write(plot_df_smooth_sd)
                            keyval = {}
                            amp_keyval = {}
                            prev_intensity = 0
                            flag = 1
                            
                            for frame_key, intensity_val in enumerate(plot_df['Smoothed Mean Intensity']):
                                if prev_intensity == 0 and intensity_val > baseline_each:
                                    continue
                                elif intensity_val >= baseline_each:
                                    keyval[frame_key] = intensity_val
                                    break
                                else:
                                    if frame_key==len(plot_df.index)-1:
                                        flag = 0
                                    else:
                                        prev_intensity = intensity_val
                                        continue
                            if flag==0:
                                prev_intensity = 0
                                st.write("Error! The trace doesn't cross the baseline. No parameters can be found")
                                st.subheader("Data for intensity of selected label")
                                st.write(plot_df)
    
                                unsmoothed_figure =  px.line(
                                                    plot_df,
                                                    x="Frame",
                                                    y="Mean Intensity"
                                                    #color="sepal_length",
                                                    #color=plot_df['Mean Intensity'],
                                                )
                                unsmoothed_figure.add_shape(type='line',
                                                    x0=0,
                                                    y0=unsmooth_baseline_mean_sd,
                                                    x1=raw_image_ani.shape[0],
                                                    y1=unsmooth_baseline_mean_sd,
                                                    line=dict(color='Green',),
                                                    xref='x',
                                                    yref='y')    
                                                            
                                smoothed_figure =  px.line(
                                                    plot_df,
                                                    x="Frame",
                                                    y='Smoothed Mean Intensity'
                                                    #color="sepal_length",
                                                    #color=plot_df['Mean Intensity'],
                                                )
                                smoothed_figure.add_shape(type='line',
                                                    x0=0,
                                                    y0=baseline_each,
                                                    x1=raw_image_ani.shape[0],
                                                    y1=baseline_each,
                                                    line=dict(color='Red',),
                                                    xref='x',
                                                    yref='y')                            
                            

                        
                                csv = convert_df(plot_df)           
                                st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv')
                                #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
                                #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
                                st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)     
                                st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)      
                            else:
                                first_key = frame_key
                                first_intensity = keyval[frame_key]
                                
                                prev_intensity = keyval[frame_key]
                                for frame_key_2, intensity_val_2 in enumerate(plot_df['Smoothed Mean Intensity']):                               
                                    if frame_key_2 <= frame_key:
                                        continue
                                    elif frame_key_2 > frame_key:
                                        if intensity_val_2 >= prev_intensity:
                                            if intensity_val_2 < baseline_each:
                                                frame_key = frame_key_2
                                                continue
                                            elif intensity_val_2 >= baseline_each:
                                                if prev_intensity < baseline_each:
                                                    first_key = frame_key_2
                                                    first_intensity = intensity_val_2
                                                    keyval[first_key] = first_intensity
                                                    frame_key = frame_key_2
                                                    prev_intensity = intensity_val_2
                                                else:
                                                    frame_key = frame_key_2
                                                    prev_intensity = intensity_val_2
                                                
    
                                        elif intensity_val_2 < prev_intensity:
                                            if intensity_val_2 > baseline_each:
                                                frame_key = frame_key_2                                         
                                            elif intensity_val_2 <= baseline_each:
                                                if prev_intensity <= baseline_each:
                                                    frame_key = frame_key_2
                                                    continue
                                                else:
                                                    keyval[frame_key_2] = intensity_val_2
                                                    frame_key = frame_key_2
                                                    #start_key = plot_df.query(f'"Smoothed Mean Intensity" == {prev_intensity}')['Frame']
                                                    amp_key_vals = plot_df[plot_df['Smoothed Mean Intensity']==prev_intensity]['Frame']
                                                    amp_key_vals = amp_key_vals[amp_key_vals>=first_key].iloc[0]
                                                    amp_key = str(amp_key_vals)
                                                    amplitude = prev_intensity - baseline_each
                                                    keyval[amp_key] = prev_intensity
                                                    prev_intensity = intensity_val_2
                                                    if (first_key == int(amp_key)): #or (int(amp_key) == frame_key): 
                                                        first_key = int(amp_key)-1
                                                        amp_keyval[f"{first_key}-{amp_key}-{frame_key}"] = amplitude
                                                    else:
                                                        amp_keyval[f"{first_key}-{amp_key}-{frame_key}"] = amplitude                
                                
                                #st.write(keyval)    
                                #st.write(amp_keyval)
                                count_items = 0
                                for item in amp_keyval.items():
                                    #st.write(item[0].split('-')) 
                                    if len(item[0].split('-'))==3:
                                        count_items += 1
                                        signal_start_frame = item[0].split('-')[0]
                                        #st.write(f"The signal start frame is {int(signal_start_frame)}")
                                        peak_frame = item[0].split('-')[1]
                                        #st.write(f"The peak frame is {int(peak_frame)}")
                                        signal_decay_frame = item[0].split('-')[2]
                                        #st.write(f"The signal decay frame is {int(signal_decay_frame)}")
                                        event_num = count_items
                                        amplitude_each = item[1]
                                        signal_rise = int(peak_frame)-int(signal_start_frame)
                                        signal_decay = int(signal_decay_frame)-int(peak_frame)
                                        signal_duration = int(signal_decay_frame)-int(signal_start_frame)
                                        nested_dict["Label"].append(i)
                                        nested_dict["Number of Events"].append(event_num)
                                        nested_dict["Rise time"].append(signal_rise)
                                        nested_dict["Decay time"].append(signal_decay)
                                        nested_dict["Duration"].append(signal_duration)
                                        nested_dict["Amplitude"].append(amplitude_each)
                                
                                
                                st.subheader("Data for intensity of selected label")
                                st.write(plot_df)
    
                                unsmoothed_figure =  px.line(
                                                    plot_df,
                                                    x="Frame",
                                                    y="Mean Intensity"
                                                    #color="sepal_length",
                                                    #color=plot_df['Mean Intensity'],
                                                )
                                unsmoothed_figure.add_shape(type='line',
                                                    x0=0,
                                                    y0=unsmooth_baseline_mean_sd,
                                                    x1=raw_image_ani.shape[0],
                                                    y1=unsmooth_baseline_mean_sd,
                                                    line=dict(color='Green',),
                                                    xref='x',
                                                    yref='y')    
                                                            
                                smoothed_figure =  px.line(
                                                    plot_df,
                                                    x="Frame",
                                                    y='Smoothed Mean Intensity'
                                                    #color="sepal_length",
                                                    #color=plot_df['Mean Intensity'],
                                                )
                                smoothed_figure.add_shape(type='line',
                                                    x0=0,
                                                    y0=baseline_each,
                                                    x1=raw_image_ani.shape[0],
                                                    y1=baseline_each,
                                                    line=dict(color='Red',),
                                                    xref='x',
                                                    yref='y')                            
                            

                        
                                csv = convert_df(plot_df)           
                                st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv')
                                #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
                                #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
                                st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)     
                                st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)                           
     
                                nested_dict = (pd.DataFrame.from_dict(nested_dict)) 
                                if nested_dict.empty:
                                    st.write("No parameter information for the selected label can be found based on the trace")
                                else:                                
                                    st.subheader("Parameters for selected label across all frames")
                                    st.write(nested_dict)
                                    individual_csv = convert_df(nested_dict)           
                                    st.download_button("Press to Download", individual_csv, 'individual_para_data.csv', "text/csv", key='individual_download-csv')
                                    average_rise_time = np.round(nested_dict['Rise time'].mean(),4)
                                    st.write(f"The average rise time based on the selected labels across all frames is {average_rise_time}")
                                    average_decay_time = np.round(nested_dict['Decay time'].mean(),4)
                                    st.write(f"The average decay time based on the selected labels across all frames is {average_decay_time}")
                                    average_duration = np.round(nested_dict['Duration'].mean(),4)
                                    st.write(f"The average duration based on the selected labels across all frames is {average_duration}")
                                    average_amplitude = np.round(nested_dict['Amplitude'].mean(),4)
                                    st.write(f"The average amplitude based on the selected labels across all frames is {average_amplitude}")
                                    st.subheader("Distribution plots based on selected label")
                                    sns.displot(data = nested_dict, x="Rise time",kind='hist')
                                    st.pyplot(plt.gcf())
                                    sns.displot(data = nested_dict, x="Decay time",kind='hist')
                                    st.pyplot(plt.gcf())
                                    sns.displot(data = nested_dict, x="Duration",kind='hist')
                                    st.pyplot(plt.gcf())
                                    sns.displot(data = nested_dict, x="Amplitude",kind='hist')
                                    st.pyplot(plt.gcf())
                       
####################################  Parameter calcualtion for all the detected cells  ###############################################################################
                       
                        new_df_pro_transposed_smooth = df_pro.transpose()
                        new_df_pro_transposed_smooth.columns = new_df_pro_transposed_smooth.iloc[0]
                        new_df_pro_transposed_smooth.drop(new_df_pro_transposed_smooth.index[0], inplace=True)  
                        #st.write(new_df_pro_transposed_smooth)      
                        for i in df_pro['label']: 
                            df_pro_transposed_smooth = pd.DataFrame(smooth_plot(new_df_pro_transposed_smooth[i]),columns = [f'smooth cell {i}'])
                            new_df_pro_transposed_smooth = pd.concat([new_df_pro_transposed_smooth.reset_index(drop=True), (np.round(df_pro_transposed_smooth[f'smooth cell {i}'],3)).reset_index(drop=True)],axis=1)
                            new_df_missing_values = pd.isna(new_df_pro_transposed_smooth[f"smooth cell {i}"])
                            new_df_pro_transposed_smooth.loc[new_df_missing_values, f'smooth cell {i}'] = new_df_pro_transposed_smooth.loc[new_df_missing_values, i]                               
                            #st.write(new_df_pro_transposed)
                        new_df_pro_transposed_smooth['Frame'] = pd.DataFrame(list(range(0, df_pro.shape[1])))
                        
                        #get_data_indi = convert_df(new_df_pro_transposed_smooth)
                        #st.download_button("Press to Download", get_data_indi, 'indi_intensity_data.csv', "text/csv", key='indi_download-get_data')
                        nested_dict_pro = {'Label':[], "Number of Events":[], "Rise time":[], "Decay time":[], "Duration":[], "Amplitude":[]}
                        for i in df_pro['label']:
                            new_df_pro_transposed_smooth_mode = new_df_pro_transposed_smooth.mode()[f"smooth cell {i}"].min() #stat.mode(new_df_pro_transposed_smooth[f"smooth cell {i}"])
                            new_df_pro_transposed_smooth_sd = new_df_pro_transposed_smooth[f"smooth cell {i}"].std()
                            baseline_each = new_df_pro_transposed_smooth_mode + new_df_pro_transposed_smooth_sd
                            keyval = {}
                            amp_keyval = {}
                            prev_intensity = 0
                            flag = 1
                            for frame_key, intensity_val in enumerate(new_df_pro_transposed_smooth[f'smooth cell {i}']):
                                if prev_intensity == 0 and intensity_val > baseline_each:
                                    continue
                                elif intensity_val >= baseline_each:
                                    keyval[frame_key] = intensity_val
                                    break
                                else:
                                    if frame_key==len(new_df_pro_transposed_smooth.index)-1:
                                        flag = 0
                                    else:
                                        prev_intensity = intensity_val
                                        continue
                            if flag==0:
                                prev_intensity = 0
                                continue
                            
                            first_key = frame_key
                            first_intensity = keyval[frame_key]
                            
                            prev_intensity = keyval[frame_key]
                            for frame_key_2, intensity_val_2 in enumerate(new_df_pro_transposed_smooth[f'smooth cell {i}']):                               
                                if frame_key_2 <= frame_key:
                                    continue
                                elif frame_key_2 > frame_key:
                                    if intensity_val_2 >= prev_intensity:
                                        if intensity_val_2 < baseline_each:
                                            frame_key = frame_key_2
                                            continue
                                        elif intensity_val_2 >= baseline_each:
                                            if prev_intensity < baseline_each:
                                                first_key = frame_key_2
                                                first_intensity = intensity_val_2
                                                keyval[first_key] = first_intensity
                                                frame_key = frame_key_2
                                                prev_intensity = intensity_val_2
                                            else:
                                                frame_key = frame_key_2
                                                prev_intensity = intensity_val_2
                                            
    
                                    elif intensity_val_2 < prev_intensity:
                                        if intensity_val_2 > baseline_each:
                                            frame_key = frame_key_2                                         
                                        elif intensity_val_2 <= baseline_each:
                                            if prev_intensity <= baseline_each:
                                                frame_key = frame_key_2
                                                continue
                                            else:
                                                keyval[frame_key_2] = intensity_val_2
                                                frame_key = frame_key_2
                                                #start_key = plot_df.query(f'"Smoothed Mean Intensity" == {prev_intensity}')['Frame']
                                                amp_key_vals = new_df_pro_transposed_smooth[new_df_pro_transposed_smooth[f'smooth cell {i}']==prev_intensity]['Frame']
                                                amp_key_vals = amp_key_vals[amp_key_vals>=first_key].iloc[0]
                                                amp_key = str(amp_key_vals)
                                                #amp_key = str(new_df_pro_transposed_smooth[new_df_pro_transposed_smooth[f'smooth cell {i}']==prev_intensity]['Frame'].iloc[0])
                                                amplitude = prev_intensity - baseline_each
                                                keyval[amp_key] = prev_intensity
                                                prev_intensity = intensity_val_2
                                                if (first_key == int(amp_key)): #or (int(amp_key) == frame_key): 
                                                    first_key = int(amp_key)-1
                                                    amp_keyval[f"{first_key}-{amp_key}-{frame_key}"] = amplitude
                                                else:
                                                    amp_keyval[f"{first_key}-{amp_key}-{frame_key}"] = amplitude                
 
                            count_items = 0
                            for item in amp_keyval.items():
                                #st.write(item[0].split('-')) 
                                if len(item[0].split('-'))==3:
                                    count_items += 1
                                    signal_start_frame = item[0].split('-')[0]
                                    #st.write(f"The signal start frame is {int(signal_start_frame)}")
                                    peak_frame = item[0].split('-')[1]
                                    #st.write(f"The peak frame is {int(peak_frame)}")
                                    signal_decay_frame = item[0].split('-')[2]
                                    #st.write(f"The signal decay frame is {int(signal_decay_frame)}")
                                    event_num = count_items
                                    amplitude_each = item[1]
                                    signal_rise = int(peak_frame)-int(signal_start_frame)
                                    signal_decay = int(signal_decay_frame)-int(peak_frame)
                                    signal_duration = int(signal_decay_frame)-int(signal_start_frame)
                                    nested_dict_pro["Label"].append(i)
                                    nested_dict_pro["Number of Events"].append(event_num)
                                    nested_dict_pro["Rise time"].append(signal_rise)
                                    nested_dict_pro["Decay time"].append(signal_decay)
                                    nested_dict_pro["Duration"].append(signal_duration)
                                    nested_dict_pro["Amplitude"].append(amplitude_each)
                                    

                        nested_dict_pro = (pd.DataFrame.from_dict(nested_dict_pro)) 
                        st.subheader("Parameters for all detected labels")
                        st.write(nested_dict_pro)  
                        all_csv = convert_df(nested_dict_pro)           
                        st.download_button("Press to Download", all_csv, 'all_data.csv', "text/csv", key='all_download-csv')
                        average_rise_time = np.round(nested_dict_pro['Rise time'].mean(),4)
                        st.write(f"The average rise time based on all detected labels across all frames is {average_rise_time}")
                        average_decay_time = np.round(nested_dict_pro['Decay time'].mean(),4)
                        st.write(f"The average decay time based on all detected labels across all frames is {average_decay_time}")
                        average_duration = np.round(nested_dict_pro['Duration'].mean(),4)
                        st.write(f"The average duration based on all detected labels across all frames is {average_duration}")
                        average_amplitude = np.round(nested_dict_pro['Amplitude'].mean(),4)
                        st.write(f"The average amplitude based on all detected labels across all frames is {average_amplitude}")
                        st.subheader("Distribution plots based on all detected labels")
                        sns.displot(data = nested_dict_pro, x="Rise time")
                        st.pyplot(plt.gcf())
                        sns.displot(data = nested_dict_pro, x="Decay time")
                        st.pyplot(plt.gcf())
                        sns.displot(data = nested_dict_pro, x="Duration")
                        st.pyplot(plt.gcf())
                        sns.displot(data = nested_dict_pro, x="Amplitude")
                        st.pyplot(plt.gcf())
           
                                                                         
                           
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


def intensity(df_1, multi_tif_img):
    img_frames_list = list(range(0,multi_tif_img.shape[0]))
    img_frames = pd.DataFrame(img_frames_list, columns = ['Frame'])
    mean_intensity = []
    #change_in_F = []
    for frames_pro in range(0,multi_tif_img.shape[0]):
            #new_df = pd.DataFrame(frames_pro, df_pro[f'intensity_mean_{frames_pro}'].mean(),  columns = ['Frames', 'Mean Intensity'])
        mean_intensity.append(df_1[f'intensity_mean_{frames_pro}'].mean())
            # new_df = pd.DataFrame.from_dict({'Frame': frames_pro, 'Mean Intensity': df_pro[f'intensity_mean_{frames_pro}'].mean()})
            # img_frames = pd.merge(img_frames, new_df, on = "Frame")
        #st.write(np.array(mean_intensity).max())
        #change_f = fluo_change(mean_intensity[frames_pro], baseline)
        #change_in_F.append(change_f)
        
    #change_F_df = pd.DataFrame(change_in_F, columns = ['delta_F/F'])
    smooth_F_df = pd.DataFrame(smooth_plot(mean_intensity), columns = ['Smoothed Mean Intensity'] ) #pd.DataFrame(smooth_df, columns = ['smoothed mean intensity'])
    mean_inten_df = pd.DataFrame(mean_intensity)
    new_d = pd.concat([img_frames, mean_inten_df, smooth_F_df],axis=1)
    new_d.rename(columns = {0 : 'Mean Intensity'}, inplace=True)
    #new_d.rename(columns = {1 : 'delta_F/F'}, inplace=True)
    return new_d 

def fluo_change(intensity_mean, baseline):
    delta_F = intensity_mean - baseline
    change_f = delta_F/baseline
    return change_f

def smooth_plot(unsmoothed_intensity):
    smooth_df = (np.convolve(unsmoothed_intensity, np.ones((3)), mode = 'valid'))/3 #ndimage.median_filter(unsmoothed_intensity,7)
    return smooth_df
    

if __name__ == "__main__":
    main()      