import streamlit as st
from streamlit_extras.switch_page_button import switch_page   ########################newwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#from st_pages import Page, show_pages, hide_pages
import plotly.io
import plotly.graph_objs as go                  ########################newwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
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

# for keys, v in st.session_state.items():
#     st.session_state[keys] = v
if 'all_param_table' not in st.session_state:
    st.session_state.all_param_table = False
    
def callback_all_param_table():
   st.session_state.all_param_table = True
    
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
    new_d['Mean Intensity'] = np.round(new_d["Mean Intensity"],3) 
    #new_d.rename(columns = {1 : 'Bright Pixel Number'}, inplace=True)
    return new_d 

def fluo_change(intensity_mean, baseline):
    delta_F = intensity_mean - baseline
    change_f = delta_F/baseline
    return change_f

def smooth_plot(unsmoothed_intensity, window):
    smooth_df = (np.convolve(unsmoothed_intensity, np.ones((window)), mode = 'valid'))/window #ndimage.median_filter(unsmoothed_intensity,7)
    return smooth_df

if "button_clicked_movav" not in st.session_state:
    st.session_state.button_clicked_movav = False
    
def callback_movav():
    #Button was clicked
    st.session_state.button_clicked_movav = True
    
st.header('**_Intensity trace for single label_**')
#st.image(labels,use_column_width=True,clamp = True)      
if 'Collapsed_Image' not in st.session_state:
    pass
else:
    collapsed = st.session_state['Collapsed_Image']
    st.write('*_The Collapsed Image_*')
    st.image(collapsed,use_column_width=True,clamp = True)
    
if 'label_list_pg_2' not in st.session_state:
    pass
else:
    label_list_pg_2 = st.session_state['label_list_pg_2'] 
    
if 'raw_img_ani_pg_2' not in st.session_state:
    pass
else:
    raw_img_ani_pg_2 = st.session_state['raw_img_ani_pg_2']
    
if 'final_label_rgb_pg_2' not in st.session_state:
    st.warning("Please generate the labeled image and the intensity table from the 'Preprocessing and Segmentation' page, and click on 'Single Intensity Traces' before proceeding")
else:
    label = st.session_state['final_label_rgb_pg_2']
    st.write('*_Segmented and labeled image_*')
    st.image(label,use_column_width=True,clamp = True)
    
if 'final_label_pg_2' not in st.session_state:
    pass
else:
    label_fin = st.session_state['final_label_pg_2']
           
    data = list(np.unique(label_list_pg_2)) 
    
    df_pro = pd.DataFrame(data, columns=['label'])
    col_arr = []
    
    for frames_pro in range(0,raw_img_ani_pg_2.shape[0]):
        props_pro = measure.regionprops_table(label_fin, intensity_image=raw_img_ani_pg_2[frames_pro][:,:,0],   #markers
                                              properties=['label','intensity_mean','image_intensity'])
        col = []
        label_array = props_pro['label']
        intensity_im = props_pro['image_intensity']
        #col_arr.append(intensity_im)                        
        for lab in label_array:                          
            mask_label = label_array == lab
            intensity_values = intensity_im[mask_label]
            col.append(intensity_values)
        col_arr.append((np.array(col)).ravel())
        
        df_single = pd.DataFrame(props_pro)
        #df_single['area'] = df_single[df_single['area']>df_single['intensity_mean'].mean()]['area']
        df_single['intensity_mean'] = np.round(df_single['intensity_mean'],3)
        #df_single.rename(columns = {'area' : f'area_{frames_pro}'}, inplace=True)
        df_single.rename(columns = {'intensity_mean' : f'intensity_mean_{frames_pro}'}, inplace=True)
        df_single.rename(columns = {'image_intensity' : f'image_intensity_{frames_pro}'}, inplace=True)
        #df_single.rename(columns = {'solidity' : f'solidity_{frames_pro}'}, inplace=True)
        df_pro = pd.merge(df_pro, df_single, on = 'label', how = 'outer')                                                 
    #st.write(col_arr[0].shape)
    #df_pro.drop([0], inplace=True)
    
    ######## #################  ################# ###############Interactive table################################################################
    area_thres_x = st.slider("*_Choose the area threshold percentage_*", min_value=0.0, max_value=1.0, value=0.3, format="%0.1f", help = f"Default is 0.3. Pixels below 30% of the maximum ({np.amax(raw_img_ani_pg_2)}) are not counted to get the bright area of labels", key='area_thres')
    #df_pro = df_pro.drop(df_pro[df_pro['label'] == 255].index)
    for frame_col in range(0, raw_img_ani_pg_2.shape[0]):
        pixel_counts = []
        for label_val in df_pro['label']:
            intensity_image = col_arr[frame_col][label_val-1]
            count = np.sum(np.greater(intensity_image, area_thres_x*np.amax(raw_img_ani_pg_2[frame_col]))) #df_pro[f'intensity_mean_{frames_pro}'].mean()))
            pixel_counts.append(count)
        #st.write(type(np.amax(raw_image_ani[frame_col])))
        pixel_var = f'Bright_pixel_area_{frame_col}'
        #df_pro[pixel_var] = pixel_counts
        pixel_counts_df = pd.DataFrame(pixel_counts,columns = [pixel_var])
        df_pro = pd.concat((df_pro, pixel_counts_df),axis=1)   
                         
    
    # pixels_to_add = []   
    # for frame_col in range(0, raw_image_ani.shape[0]):                                    
    #     pixel_var = f'pixel_count_{frame_col}'
    #     pixel_counts = []
    #     for label_val in df_pro['label']:
    #         #st.write("HERE")
    #         intensity_image = col_arr[frame_col][label_val-1]
    #         count = np.sum(np.greater(intensity_image, 0.5*np.amax(raw_image_ani))) #df_pro[f'intensity_mean_{frames_pro}'].mean()))
    #         pixel_counts.append(count)  
    #     pixels_to_add.append({pixel_var: pixel_counts})
        
    # df_pro = pd.concat([df_pro, pd.DataFrame(pixels_to_add)], axis=1)  
    #st.write(df_pro["pixel_count_40"].dtype)
    
    for drop_frame in range(0, raw_img_ani_pg_2.shape[0]):  
       df_pro.drop([f'image_intensity_{drop_frame}'], axis=1, inplace=True) 
    #st.session_state['df_pro_pg_2'] = df_pro
    st.dataframe(df_pro, 1000, 200)
    get_data_indi = convert_df(df_pro)
    st.download_button("Press to Download", get_data_indi, 'label_intensity_data.csv', "text/csv", key='label_download-get_data')                      
    st.write('*_Select a label to explore_*')
    
    gb = GridOptionsBuilder.from_dataframe(df_pro)                       
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_selection(selection_mode="single", use_checkbox=True, groupSelectsChildren="Group checkbox select children", pre_selected_rows=[]) #list(range(0, len(df_pro))))  #[str(st.session_state.selected_row)]
               
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
    labels_rgb = np.expand_dims(label_fin, axis=2)
    labels_rgbb = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)
    
    if selected: 
        #st.write(df_selected)# Loop over the selected indices and draw polygons on the color image
        for i in df_selected['label']:
            # Extract the coordinates of the region boundary
            coords = np.argwhere(label_fin==i)
        
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
        
        st.subheader("**_Data for intensity of selected label_**")  
        smooth_plot_x = st.slider("*_Moving Average Window_*", min_value=1, max_value=5, help = "Select to smooth the intensity trace. Moving average of 1 would mean the original 'Mean Intensity' trace below")
        plot_df = intensity(df_selected, raw_img_ani_pg_2, smooth_plot_x)
        #st.write(plot_df)                                                      
        #smoothed_plot_df = plot_df['Smoothed Mean Intensity']                           
        missing_values = pd.isna(plot_df["Smoothed Mean Intensity"])
        plot_df.loc[missing_values, "Smoothed Mean Intensity"] = plot_df.loc[missing_values, "Mean Intensity"]  
        #smooth_mode = stat.mode(smoothed_plot_df)
        #sd_smooth = plot_df['Smoothed Mean Intensity'].std()
        #smooth_baseline_mean_sd = smooth_mode + sd_smooth
        #st.write(smooth_mode)
        #st.write(smooth_baseline_mean_sd)
        plot_df["Smoothed Mean Intensity"] = np.round(plot_df["Smoothed Mean Intensity"],3)                            
        plot_df_smooth_mode =stat.mode(plot_df['Smoothed Mean Intensity']) #stat.mode(new_df_selected_transposed_smooth[f"smooth cell {i}"])
        plot_df_smooth_sd = plot_df['Smoothed Mean Intensity'].std()
        
        baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the traces below. Baselines are calculated as: **_mode + n(S.D.)._** Parameters are calculated on the basis of 'Smoothed Mean Intensity'",  key='smooth')
        baseline_each = plot_df_smooth_mode + baseline_smooth_x*plot_df_smooth_sd   
        plot_df['delta_f/f_0'] = (plot_df['Smoothed Mean Intensity'] - baseline_each)/baseline_each 
        baseline_unsmooth_x = st.slider("*_Choose 'n' in n(S.D.) for Mean Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0, key='unsmooth')
        unsmooth_mode = stat.mode(plot_df['Mean Intensity'])
        sd_unsmooth = plot_df['Mean Intensity'].std()
        unsmooth_baseline_mean_sd = unsmooth_mode + baseline_unsmooth_x*sd_unsmooth
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
            
            st.dataframe(plot_df, 1000,200)
    
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
                                x1=raw_img_ani_pg_2.shape[0],
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
                                x1=raw_img_ani_pg_2.shape[0],
                                y1=baseline_each,
                                line=dict(color='Red',),
                                xref='x',
                                yref='y') 
            unsmoothed_area_figure =  px.line(
                                plot_df,
                                x="Frame",
                                y="Bright Pixel Area"
                                #color="sepal_length",
                                #color=plot_df['Mean Intensity'],
                            )
        
    
    
            csv = convert_df(plot_df)           
            st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv')
            #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
            #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
            col_13, col_14 = st.columns([3,1])
            col_17, col_18 = st.columns([3,1])
            col_19, col_20 = st.columns([3,1])
            with col_13:
                st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)
            with col_14:
                st.write(plot_df[['Frame', 'Smoothed Mean Intensity']])
            with col_17:
                st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
            with col_18:
                st.write(plot_df[['Frame', 'Mean Intensity']],use_container_width=True)
            with col_19:
                st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)  
            with col_20:
                st.write(plot_df[['Frame', 'Bright Pixel Area']])
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
            
            
            #st.subheader("**_Data for intensity of selected label_**")
            st.dataframe(plot_df, 1000,200)
    
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
                                x1=raw_img_ani_pg_2.shape[0],
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
                                x1=raw_img_ani_pg_2.shape[0],
                                y1=baseline_each,
                                line=dict(color='Red',),
                                xref='x',
                                yref='y')                            
            unsmoothed_area_figure =  px.line(
                                plot_df,
                                x="Frame",
                                y="Bright Pixel Area"
                                #color="sepal_length",
                                #color=plot_df['Mean Intensity'],
                            )
    
            
            csv = convert_df(plot_df)           
            st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv')
            #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
            #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
            col_15, col_16 = st.columns([3,1])
            col_21, col_22 = st.columns([3,1])
            col_23, col_24 = st.columns([3,1])                            
            with col_15:
                st.plotly_chart(smoothed_figure, theme="streamlit",use_container_width=True)
            with col_16:
                st.dataframe(plot_df[['Frame','Smoothed Mean Intensity']])                            
            with col_21:
                st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
            with col_22:
                st.write(plot_df[['Frame','Mean Intensity']])
            with col_23:
                st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)  
            with col_24:
                st.write(plot_df[['Frame','Bright Pixel Area']])                               
    
            nested_dict = (pd.DataFrame.from_dict(nested_dict)) 
            if nested_dict.empty:
                st.write("No parameter information for the selected label can be found based on the trace")
            else:
                st.subheader("**_Parameters for selected label across all frames_**")
                col_1, col_2 = st.columns(2)
                with col_1:
                    st.write(nested_dict)
                    individual_csv = convert_df(nested_dict)           
                    st.download_button("Press to Download", individual_csv, 'individual_para_data.csv', "text/csv", key='individual_download-csv')
                    
                with col_2:
                    average_rise_time = np.round(nested_dict['Rise time'].mean(),4)
                    st.write(f"The average rise time based on the selected labels across all frames is {average_rise_time}")
                    average_decay_time = np.round(nested_dict['Decay time'].mean(),4)
                    st.write(f"The average decay time based on the selected labels across all frames is {average_decay_time}")
                    average_duration = np.round(nested_dict['Duration'].mean(),4)
                    st.write(f"The average duration based on the selected labels across all frames is {average_duration}")
                    average_amplitude = np.round(nested_dict['Amplitude'].mean(),4)
                    st.write(f"The average amplitude based on the selected labels across all frames is {average_amplitude}")
                st.subheader("Distribution plots based on selected label")    
                col_3, col_4 = st.columns(2)
                with col_3:                                
                    sns.displot(data = nested_dict, x="Rise time",kind='hist')
                    st.pyplot(plt.gcf())
                with col_4: 
                    sns.displot(data = nested_dict, x="Decay time",kind='hist')
                    st.pyplot(plt.gcf())
                col_5, col_6 = st.columns(2)    
                with col_5: 
                    sns.displot(data = nested_dict, x="Duration",kind='hist')
                    st.pyplot(plt.gcf())
                with col_6:     
                    sns.displot(data = nested_dict, x="Amplitude",kind='hist')
                    st.pyplot(plt.gcf())
           
               
       
    ####################################  Parameter calcualtion for all the detected cells  ###############################################################################
        if st.button("**_Go to Multiple Intensity Traces_**", help = 'Clicking on this switches to a new page and all selection in the current page will be lost'):
            switch_page('📉 Multiple intensity traces')
            
