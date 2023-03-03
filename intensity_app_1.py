  
import streamlit as st
import plotly.io
#import PIL 
import math
from PIL import Image
from matplotlib import pyplot as plt
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
#from scipy import ndimage
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
    
@st.cache(allow_output_mutation=True)
def load_model():
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    return model

@st.cache(allow_output_mutation=True)
def load_image(images):
     img = io.imread(images)
     # re, img = cv2.imreadmulti(images, flags=cv2.IMREAD_UNCHANGED)
     # img = np.array(img)
     return img

@st.cache(allow_output_mutation=True)
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

@st.cache(allow_output_mutation=True)
def show_video(img):
    video_frames = []
    output_file_path = "/app/output.mp4"
    # Add each video frame to the list as a PIL Image object
    for i in range(img.shape[0]):
        img_arr = img[i]
        img_v = Image.fromarray(img_arr)
        video_frames.append(img_v)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video_writer = cv2.VideoWriter(output_file_path, fourcc, 30.0, (video_frames[0].width, video_frames[0].height))
    
    for frame in video_frames:
        frame = np.array(frame)
        video_writer.write(frame) 
    video_writer.release()
  # Read the output video file
    with open(output_file_path, "rb") as f:
        video_bytes = f.read()
    return video_bytes
    
def main():
    # selected_box = st.sidebar.selectbox(
    #     'Segment the images',
    #     ('Process a single file', 'Process multiple frames' )
    #     )
    # if selected_box == 'Process multiple frames':
    Segment()
 

def Segment():
    st.title('Segmentation')
    raw_file = st.file_uploader("Choose an image file")
    st.write(raw_file)
    if raw_file is not None:
        
        #plt.save(raw_file, cwd)
        ######use this script to load the image on the deployed app############
        file_bytes = BytesIO(raw_file.read())
        st.image(file_bytes,use_column_width=True,clamp = True) 
        ############use this script to load the image on the deployed app############################
        
        #st.image(raw_file,use_column_width=True,clamp = True) 
        #raw_name=cwd+raw_file.name
        #st.write(raw_name)      #needs to be (none, none, 3)
        raw_image = load_image(file_bytes) #use this script to load the image on the deployed app
        #raw_image = load_image(raw_name)
        #raw_image = io.imread(raw_name) 
        st.write(raw_image.shape)
        if (len(raw_image.shape) == 2) or (len(raw_image.shape) ==3 and raw_image.shape[-1]!=3) or (len(raw_image.shape) ==4 and raw_image.shape[-1]!=3):
            raw_image = cv2.cvtColor(img_as_ubyte(raw_image), cv2.COLOR_GRAY2RGB)
           
        st.write(raw_image.dtype)
        model = load_model()
        raw_image_name = raw_file.name.split(".")[0]
        extension = raw_file.name.split(".")[1]
        # if raw_file.name.split(".")[1] == 'tif':
        raw_image_ani = raw_image
        st.write(raw_image_ani.shape)
        if extension == 'tif' and len(raw_image_ani.shape)==4:
            
            #btn_clk = st.form('Show Frmaes', clear_on_submit=False)
            #if (st.button("Show frames", on_click=callback_frame) or st.session_state.button_clicked):
            with st.form("my_form"):  
                submitted = st.form_submit_button("Show video")
                if submitted:  
                   video_bytes = show_video(raw_image_ani) 
                   st.video(video_bytes)
                # if btn_clk:# or st.session_state.show_frames:                
                #     st.subheader("Image frames")             
                #     show_ani(raw_image_ani) 
                
                #st.session_state['frames_displayed'] = True
                
            med_x = st.slider("Median Blur Kernel size", min_value = 1,max_value = 100, step = 2, on_change=callback_off)
            bri_x = st.slider('Change Brightness',min_value = 0,max_value = 255, on_change=callback_off)
            con_x = st.slider('Change Contrast',min_value = 0,max_value = 255, on_change=callback_off)
            
            #diam_f = []
            label_list = []
            centroid_set_first = []
            final_label = np.zeros((raw_image_ani.shape[1], raw_image_ani.shape[2]),dtype=np.uint8)
            raw_image_first = raw_image_ani[0]
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
                    final_label[label_value_first] = label_val        #object_label = props[objects]['label'] 
                    label_list.append(label_val)
                    label_val += 1
            
            #if button("Show segmented frame 1", key='seg_btn'): 
            #st.write(st.session_state.button_clicked)
            if st.button("Show segmented frame 1 (Overlayed on the original image)", key='seg_btn', on_click = callback) or st.session_state.button_clicked:
                #st.write(st.session_state.button_clicked)
                seg_im = render_label(final_label, img=raw_image_first)
                st.image(seg_im,use_column_width=True,clamp = True)  
                #st.write(label_list)
                cutoff = 10
                p = st.empty()
                #st.write(st.session_state.button_clicked_allframes)
                #if button("Apply to all frames", key='frame_btn'):
                if st.button("Apply to all frames", key='frame_btn',on_click = callback_allframes) or st.session_state.button_clicked_allframes:
                    #st.write(st.session_state.button_clicked_allframes)
                    for frame_num in range(1,raw_image_ani.shape[0]):
                        raw_image = raw_image_ani[frame_num] 
                        
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
                                final_label[label_value] = label_val
                                label_list.append(label_val)
                                #plt.imsave(f"F:/Intensity measurement/check labels/label_{frame_num}_{label_val}.jpg", final_label==label_val)
                                label_val += 1 
                                centroid_set_first.append(centroid_coord)
                    p.write("Done!")  
                    st.image(final_label,use_column_width=True,clamp = True)
        # st.write(seg_im.shape)
        # st.write("Segmented image")
        # st.image(seg_im,use_column_width=True,clamp = True) 
        # rgba_image = Image.fromarray(seg_im, "RGB")
        # #rgb_image = seg_im.convert("RGB")
        # rgb_image = np.array(rgba_image)
        # get_image_download_link(rgb_image,"segmented.png")
                    # final_label_rgb = cv2.cvtColor(final_label, cv2.COLOR_GRAY2RGB)
                    # for label in np.unique(label_list):
                  	 # #if the label is zero, we are examining the 'background'
                    #   #so simply ignore it
                    #     if label == 0:
                    #         continue                
                    #     mask = np.zeros(CLAHE_img.shape, dtype="uint8")
                    #     mask[final_label == label] = 255
                    #     #detect contours in the mask and grab the largest one
                    #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    #         cv2.CHAIN_APPROX_SIMPLE)
                    #     cnts = imutils.grab_contours(cnts)
                    #     c = max(cnts, key=cv2.contourArea)
                    #     # draw a circle enclosing the object
                    #     ((x, y), r) = cv2.minEnclosingCircle(c)
                    #     cv2.circle(final_label_rgb, (int(x), int(y)), int(r), (255, 0, 0), 1)
                    #     cv2.putText(final_label_rgb, "{}".format(label), (int(x) - 10, int(y)),
                    #      	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # st.image(final_label_rgb,use_column_width=True,clamp = True)
                        
                    if st.button("Show intensity table", on_click = callback_table) or st.session_state.display_table:
                        
                        
                        st.write("Intensity table for segmented Image (Click to see each object on the label and its mean frame intensity)") 
                        #st.image(labels,use_column_width=True,clamp = True)          
                                   
                        data = list(np.unique(label_list)) 
                        
                        df_pro = pd.DataFrame(data, columns=['label'])
                        
                        for frames_pro in range(0,raw_image_ani.shape[0]):
                            props_pro = measure.regionprops_table(final_label, intensity_image=raw_image_ani[frames_pro][:,:,0],   #markers
                                                                  properties=['label','intensity_mean'])
                            
                            df_single = pd.DataFrame(props_pro)
                            df_single.rename(columns = {'intensity_mean' : f'intensity_mean_{frames_pro}'}, inplace=True)
                            #df_single.rename(columns = {'solidity' : f'solidity_{frames_pro}'}, inplace=True)
                            df_pro = pd.merge(df_pro, df_single, on = 'label', how = 'outer')                                                
                            
                        #df_pro.drop([0], inplace=True)
                        
                        #df_pro = df_pro.drop(df_pro[df_pro['label'] == 255].index)
                            
                            ###############Interactive table################################################################
                            
                        gb = GridOptionsBuilder.from_dataframe(df_pro)
                        
                        
                        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                        gb.configure_side_bar() #Add a sidebar
                        #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                        gb.configure_selection(selection_mode="multiple", use_checkbox=True, groupSelectsChildren="Group checkbox select children", pre_selected_rows=[]) #list(range(0, len(df_pro))))  #[str(st.session_state.selected_row)]
                                   
                        gridOptions = gb.build()
                        gridOptions["columnDefs"][0]["checkboxSelection"]=True
                        gridOptions["columnDefs"][0]["headerCheckboxSelection"]=True
                        
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
                            st.write(df_selected) 
                
                            plot_df, _ = intensity(df_selected, raw_image_ani)
                            
                            figure = px.scatter(
                                                plot_df,
                                                x="Frame",
                                                y="Mean Intensity",
                                                #color="sepal_length",
                                                color_continuous_scale="reds",
                                            )
                            
                            csv = convert_df(plot_df)           
                            st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv')
                            st.plotly_chart(figure, theme="streamlit", use_container_width=True)
                                 
              
        
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
    for frames_pro in range(0,multi_tif_img.shape[0]):
            #new_df = pd.DataFrame(frames_pro, df_pro[f'intensity_mean_{frames_pro}'].mean(),  columns = ['Frames', 'Mean Intensity'])
        mean_intensity.append(df_1[f'intensity_mean_{frames_pro}'].mean())
            # new_df = pd.DataFrame.from_dict({'Frame': frames_pro, 'Mean Intensity': df_pro[f'intensity_mean_{frames_pro}'].mean()})
            # img_frames = pd.merge(img_frames, new_df, on = "Frame")
        #st.write(np.array(mean_intensity).max())
    mean_inten_df = pd.DataFrame(mean_intensity)
    new_d = pd.concat([img_frames, mean_inten_df],axis=1)
    new_d.rename(columns = {0 : 'Mean Intensity'}, inplace=True)
    return new_d, st.write(new_d)    

if __name__ == "__main__":
    main()      