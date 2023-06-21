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
from scipy.optimize import curve_fit
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
import warnings

st.warning('Navigating to another page from the sidebar will remove all selections from the current page')
# for keys, v in st.session_state.items():
#     st.session_state[keys] = v

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

def mono_exp_decay(t, a, b):
    return a * np.exp(-b * t)

def mono_exp_rise(t, a, b):
    return a * np.exp(b * t)

def find_b_est_decay(x_data, y_data):
    decay_constants = []
    for i in range(len(x_data)-1):
        x1, y1 = x_data[i], y_data[i]
        x2, y2 = x_data[i+1], y_data[i+1]
        k = -math.log(y2/y1) / (x2-x1)
        decay_constants.append(k)
    
    # Calculate average decay rate
    avg_decay_rate = np.mean(decay_constants)
    
    return avg_decay_rate

def find_b_est_rise(x_data, y_data):
    
    decay_constants = []
    for i in range(len(x_data)-1):
        x1, y1 = x_data[i], y_data[i]
        x2, y2 = x_data[i+1], y_data[i+1]
        k = math.log(y2/y1) / (x2-x1)
        decay_constants.append(k)
    
    # Calculate average decay rate
    avg_rise_rate = np.mean(decay_constants)
    
    return avg_rise_rate

if "button_clicked_movav" not in st.session_state:
    st.session_state.button_clicked_movav = False
    
def callback_movav():
    #Button was clicked
    st.session_state.button_clicked_movav = True
    
if 'all_param_table' not in st.session_state:
    st.session_state.all_param_table = False
    
def callback_all_param_table():
   st.session_state.all_param_table = True
    
st.header('**_Intensity trace for multiple labels_**')
if 'raw_img_ani_pg_2' not in st.session_state:
    pass
else:
    raw_img_ani_pg_2 = st.session_state['raw_img_ani_pg_2']

if 'background_corr_pg_2' not in st.session_state:
    pass
else:
    background_corr_pg_2 = st.session_state['background_corr_pg_2']

if 'Collapsed_Image' not in st.session_state:
    pass
else:
    collapsed = st.session_state['Collapsed_Image']
    st.write('*_The Collapsed Image_*')
    st.image(collapsed,use_column_width=True,clamp = True)
    
if 'super_im_rgb_pg_2' not in st.session_state:
    pass
else:
    super_im_pg_2 = st.session_state['super_im_rgb_pg_2']
    st.write('*_Segmented and labeled image overlayed on the collapsed image_*')
    st.image(super_im_pg_2,use_column_width=True,clamp = True)

if 'final_label_rgb_pg_2' not in st.session_state:
    st.warning("Please generate the segmented and labeled image from the 'Preprocessing and Segmentation' page, and click on 'Single-cell Analysis' before proceeding")
else:
    label = st.session_state['final_label_rgb_pg_2']
    st.write('*_Segmented and labeled image on a black background_*')
    st.image(label,use_column_width=True,clamp = True)
    
if 'final_label_pg_2' not in st.session_state:
    pass
else:
    label_fin = st.session_state['final_label_pg_2']
    #st.image(label_fin,use_column_width=True,clamp = True)    

if 'label_list_pg_2' not in st.session_state:
    pass  
else:
    label_list_pg_2 = st.session_state['label_list_pg_2'] 
    
    data = list(np.unique(label_list_pg_2)) 
    
    df_pro = pd.DataFrame(data, columns=['label'])
    col_arr = []
    
    for frames_pro in range(0,raw_img_ani_pg_2.shape[0]):
        props_pro = measure.regionprops_table(label_fin, intensity_image=background_corr_pg_2[frames_pro][:,:,0],   #markers
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
    st.session_state['df_pro_pg_2'] = df_pro
    st.dataframe(df_pro, 1000, 200)
    get_data_indi = convert_df(df_pro)
    st.download_button("Press to Download", get_data_indi, 'label_intensity_data.csv', "text/csv", key='label_download-get_data')
    #st.dataframe(df_pro, 1000, 200)
    st.write('*_Select label(s) to explore_*')    
    gb = GridOptionsBuilder.from_dataframe(df_pro)                       
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_selection(selection_mode="multiple", use_checkbox=True, groupSelectsChildren="Group checkbox select children", pre_selected_rows=[]) #list(range(0, len(df_pro))))  #[str(st.session_state.selected_row)]
             
    gridOptions = gb.build()
    gridOptions["columnDefs"][0]["checkboxSelection"]=True
    gridOptions["columnDefs"][0]["headerCheckboxSelection"]=True
    
    grid_response_m = AgGrid(
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
        key='table_m_key'
    )
    
    data = grid_response_m['data']
    selected_m = grid_response_m['selected_rows'] 
    
    # if len(selected) != 0:
    #     st.write(selected[0])
        # st.session_state.selected_row = selected_rows[0]['_selectedRowNodeInfo']['nodeRowIndex']
    df_selected = pd.DataFrame(selected_m)
    labels_rgb = np.expand_dims(label_fin, axis=2)
    labels_rgbb_m = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)
    
    if selected_m:
        #st.write(selected_m)
        csv_selected = convert_df(df_selected)           
        st.download_button("Press to Download the Selected Data", csv_selected, 'selected_intensity_data.csv', "text/csv", key='download-selected-csv')
        #st.write(df_selected)# Loop over the selected indices and draw polygons on the color image
        for i in df_selected['label']:
            # Extract the coordinates of the region boundary
            coords = np.argwhere(label_fin==i)
        
            # Create a polygon from the coordinates
            poly = polygon(coords[:, 0], coords[:, 1])
        
            # Set the color of the polygon to red
            color_poly = (255, 0, 0)
        
            # Color the polygon in the color image
            labels_rgbb_m[poly] = color_poly
        st.image(labels_rgbb_m,use_column_width=True,clamp = True)
        df_selected_1 = df_selected.drop(columns = ['_selectedRowNodeInfo'])
        df_selected = df_selected.drop(columns = ['_selectedRowNodeInfo'])
        df_selected_remove = df_selected_1.drop(columns=df_selected_1.filter(regex='^Bright_pixel_area').columns)
        df_selected_transpose = df_selected_remove.transpose()
        df_selected_transpose.columns = (df_selected_transpose.iloc[0])
        df_selected_transpose = df_selected_transpose.drop(index = ['label'])
        df_selected_transpose['Frames'] = list(range(0,df_selected_transpose.shape[0]))
        #st.dataframe(df_selected_transpose)
        
        # create an empty list to store the traces
        traces = []
        # loop through each column (excluding the x column)
        for column in df_selected_transpose.columns[:-1]:    
            # create a trace for the current column
            trace = go.Scatter(x=df_selected_transpose['Frames'], y=df_selected_transpose[column], name=column)
            # add the trace to the list
            traces.append(trace)
        # create the plot
        fig = go.Figure(data=traces)
        # update the layout
        fig.update_layout(title='Intensity Traces', xaxis_title='Frame', yaxis_title='Mean Intensity',height=900)
        # display the plot
        st.plotly_chart(fig, use_container_width=True) 
        
        st.write("*_Select the parameters to be applied on all labels_*")
        
        bleach_corr_check = st.radio("Select one", ('No bleaching correction', 'Bleaching correction'), help='Analyze the trace as is (No bleaching correction) or fit mono-exponential curves and interpolate to correct for bleaching (Bleaching correction)')
        
        if bleach_corr_check == 'No bleaching correction':
            smooth_plot_x = st.slider("*_Moving Average Window_*", min_value=1, max_value=5, help = "Adjust to smooth the mean intensity trace below. Moving average of 1 would mean the original 'Mean Intensity' trace")
            baseline_smooth_x = st.slider("*_Choose frame number(s) to average their corresponding intensity values for baseline calculation_*", min_value = 0, max_value = raw_img_ani_pg_2.shape[0]-1, value = 10,  key='smooth_multi_0')
            
            if st.button("Obtain the parameters for selected labels",on_click=callback_movav) or st.session_state.button_clicked_movav:
                
                st.warning("The parameters for all labels are obtained using the same set of selections.")
                df_pro_pixel_remove = df_selected_1.drop(columns=df_selected.filter(regex='^Bright_pixel_area_').columns)
                #df_pro_pixel_remove = df_pro_pixel_remove.drop(columns=df_pro.filter(regex='^area').columns)
                new_df_pro_transposed_smooth = df_pro_pixel_remove.transpose()
                new_df_pro_transposed_smooth.columns = new_df_pro_transposed_smooth.iloc[0]
                new_df_pro_transposed_smooth.drop(new_df_pro_transposed_smooth.index[0], inplace=True)  
                
                
                #smooth_plot_x = st.slider("*_Moving Average Window_*", min_value=1, max_value=5, help = "Select to smooth the intensity trace. Moving average of 1 would mean the original 'Mean Intensity' trace below", key = 'mov_av')
                for i in df_selected['label']: 
                    
                    df_pro_transposed_smooth = pd.DataFrame(smooth_plot(new_df_pro_transposed_smooth[i],smooth_plot_x),columns = [f'smooth cell {i}'])
                    new_df_pro_transposed_smooth = pd.concat([new_df_pro_transposed_smooth.reset_index(drop=True), (np.round(df_pro_transposed_smooth[f'smooth cell {i}'],3)).reset_index(drop=True)],axis=1)
                    new_df_missing_values = pd.isna(new_df_pro_transposed_smooth[f"smooth cell {i}"])
                    new_df_pro_transposed_smooth.loc[new_df_missing_values, f'smooth cell {i}'] = new_df_pro_transposed_smooth.loc[new_df_missing_values, i]                               
                    
                    #st.write(new_df_pro_transposed)
                new_df_pro_transposed_smooth['Frame'] = pd.DataFrame(list(range(0, df_selected.shape[1])))
                new_df_pro_transposed_smooth = new_df_pro_transposed_smooth.iloc[:, [new_df_pro_transposed_smooth.shape[1] - 1] + list(range(new_df_pro_transposed_smooth.shape[1] - 1))]
                
                #get_data_indi = convert_df(new_df_pro_transposed_smooth)
                #st.download_button("Press to Download", get_data_indi, 'indi_intensity_data.csv', "text/csv", key='indi_download-get_data')
                
                #baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the 'Smoothed Mean Intensity' trace below. Baseline is calculated as: **_mode + n(S.D.)._**",  key='smooth')
                
                nested_dict_final = {}           
                nested_dict_pro = {'Label':[], "Number of Events":[], "Rise time":[], "Rise Rate":[], "Decay time":[], "Decay Rate":[], "Duration":[], "Amplitude":[]}

                for i in df_selected['label']:
                    baseline_each = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth['Frame'] >= 0) & (new_df_pro_transposed_smooth['Frame'] <= baseline_smooth_x), f'smooth cell {i}'].mean()
                    new_df_pro_transposed_smooth[f'delta_f/f_0_{i}'] = (new_df_pro_transposed_smooth[f'smooth cell {i}'] - baseline_each)/baseline_each 
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
                    
                    
                    max_df_value = new_df_pro_transposed_smooth[f'smooth cell {i}'].max()
                    #st.write(plot_df.dtypes)
                    #####test by setting a some equal high values#########plot_df.loc[plot_df['Frame'] == 39, 'Smoothed Mean Intensity'] = max_df_value ##plot_df.loc[plot_df['Frame'] == 69, 'Smoothed Mean Intensity'] = baseline_each
                    count_max = new_df_pro_transposed_smooth[f'smooth cell {i}'].eq(max_df_value).sum()
                    max_frame = new_df_pro_transposed_smooth.loc[new_df_pro_transposed_smooth[f'smooth cell {i}'] == max_df_value, 'Frame']
                    decay_df = pd.DataFrame()
                    rise_df = pd.DataFrame()
                    if ~((new_df_pro_transposed_smooth[f'smooth cell {i}'] <= baseline_each) & (new_df_pro_transposed_smooth['Frame'] > max(max_frame))).any(): ##trace crosses baseline but never comes back
                        #nested_dict_pro = {'Label':[], "Number of Events":[], "Rise time":[], "Rise Rate":[], "Decay time":[], "Decay Rate":[], "Duration":[], "Amplitude":[]}
                        continue
                        
                    else:
                        
                        if count_max == 1:
                            rise_df['Rise intensity'] = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth[f'smooth cell {i}'] <= max_df_value) & (new_df_pro_transposed_smooth[f'smooth cell {i}'] >= baseline_each) & (new_df_pro_transposed_smooth['Frame'] <= max(max_frame)) , f'smooth cell {i}']
                            decay_df['Decay intensity'] = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth[f'smooth cell {i}'] <= max_df_value) & (new_df_pro_transposed_smooth[f'smooth cell {i}'] >= baseline_each) & (new_df_pro_transposed_smooth['Frame'] >= max(max_frame)) , f'smooth cell {i}']
                            decay_df['Frame'] = decay_df.index
                            rise_df['Frame'] = rise_df.index
                            decay_df = decay_df[decay_df.columns[::-1]]
                            rise_df = rise_df[rise_df.columns[::-1]]
                            test_missing_value_df = next((decay_df['Frame'].iloc[i] + 1 for i in range(len(decay_df['Frame'])-1) if decay_df['Frame'].iloc[i+1] - decay_df['Frame'].iloc[i] > 1), None)
                            if test_missing_value_df is None:
                                missing_value_df = None
                            else:
                                missing_value_df = test_missing_value_df - 1 
                            missing_value_rise_df = (rise_df.loc[rise_df['Frame'].diff() > 1, 'Frame'].max())-1
                            #st.write(~rise_df['Rise intensity'].isin([baseline_each]).any())
                            # if ~decay_df['Decay intensity'].isin([baseline_each]).any():
                            #     new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_each}
                            #     decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_each]).any():
                            #     new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_each}
                            #     rise_df.loc[missing_value_rise_df] = new_row_rise
                            
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_df)
                            if not pd.isna(missing_value_df): #there is a missing value
                                #st.write('here')
                                decay_df = decay_df.loc[decay_df['Frame'] <= missing_value_df]
                             
                            else:
                                if (decay_df['Decay intensity'] == baseline_each).any():
                                    baseline_frame = max(decay_df.loc[decay_df['Decay intensity'] == baseline_each, 'Frame'])
                                    decay_df = decay_df.loc[(decay_df['Decay intensity'] >= baseline_each) & (decay_df['Frame'] <= baseline_frame)]  
                                else:
                                    decay_df = decay_df
                                
                            if not pd.isna(missing_value_rise_df):
                                #st.write('here')
                                rise_df = rise_df.loc[rise_df['Frame'] >= missing_value_rise_df]
                            else:
                                if (rise_df['Rise intensity'] == baseline_each).any():
                                    #st.write(rise_df)
                                    baseline_frame = max(rise_df.loc[rise_df['Rise intensity'] == baseline_each, 'Frame'])
                                    rise_df = rise_df.loc[(rise_df['Rise intensity'] >= baseline_each) & (rise_df['Frame'] >= baseline_frame)]  
                                else:
                                    rise_df = rise_df
                            #st.write(rise_df)
                            #st.write(missing_value_df)
                            
                        if count_max > 1: 
                            decay_df['Decay intensity'] = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth[f'smooth cell {i}'] <= max_df_value) & (new_df_pro_transposed_smooth[f'smooth cell {i}'] >= baseline_each) & (new_df_pro_transposed_smooth['Frame'] >= max(max_frame)) , f'smooth cell {i}']
                            last_index = decay_df.loc[decay_df['Decay intensity'] == max_df_value].index[-1]
                            rise_df['Rise intensity'] = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth[f'smooth cell {i}'] <= max_df_value) & (new_df_pro_transposed_smooth[f'smooth cell {i}'] >= baseline_each) & (new_df_pro_transposed_smooth['Frame'] <= max(max_frame)) , f'smooth cell {i}']
                            first_index = decay_df.loc[decay_df['Decay intensity'] == max_df_value].index[0]                    
                            decay_df.loc[last_index, 'Decay intensity'] *= 1.01
                            rise_df.loc[first_index, 'Rise intensity'] *= 1.01
                            decay_df['Frame'] = decay_df.index
                            rise_df['Frame'] = rise_df.index
                            decay_df = decay_df[decay_df.columns[::-1]]
                            rise_df = rise_df[rise_df.columns[::-1]]
                            #st.write(decay_df)
                            test_missing_value_df = next((decay_df['Frame'].iloc[i] + 1 for i in range(len(decay_df['Frame'])-1) if decay_df['Frame'].iloc[i+1] - decay_df['Frame'].iloc[i] > 1), None)
                            if test_missing_value_df is None:
                                missing_value_df = None
                            else:
                                missing_value_df = test_missing_value_df - 1                         
                            missing_value_rise_df = (rise_df.loc[rise_df['Frame'].diff() > 1, 'Frame'].max())-1
                            #st.write(np.any(rise_df['Rise intensity']) != baseline_each)
                            # if ~decay_df['Decay intensity'].isin([baseline_each]).any():
                            #     new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_each}
                            #     decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_each]).any():
                            #     new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_each}
                            #     rise_df.loc[missing_value_rise_df] = new_row_rise
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_rise_df)
                            if not pd.isna(missing_value_df):
                                #st.write('here')
                                decay_df = decay_df.loc[decay_df['Frame'] <= missing_value_df]
                            else:
                                if (decay_df['Decay intensity'] == baseline_each).any():
                                    baseline_frame = max(decay_df.loc[decay_df['Decay intensity'] == baseline_each, 'Frame'])
                                    decay_df = decay_df.loc[(decay_df['Decay intensity'] >= baseline_each) & (decay_df['Frame'] <= baseline_frame)]  
                                else:
                                    decay_df = decay_df
                                
                            if not pd.isna(missing_value_rise_df):
                                #st.write('here')
                                rise_df = rise_df.loc[rise_df['Frame'] >= missing_value_rise_df]
                            else:
                                if (rise_df['Rise intensity'] == baseline_each).any():
                                    #st.write(rise_df)
                                    baseline_frame = max(rise_df.loc[rise_df['Rise intensity'] == baseline_each, 'Frame'])
                                    rise_df = rise_df.loc[(rise_df['Rise intensity'] >= baseline_each) & (rise_df['Frame'] >= baseline_frame)]  
                                else:
                                    rise_df = rise_df             
                            #st.write(rise_df)
                        
                        a_est = decay_df['Decay intensity'].iloc[0]
                        b_est = find_b_est_decay(np.array(decay_df['Frame']), np.array(decay_df['Decay intensity']))
                        a_est_rise = rise_df['Rise intensity'].iloc[-1]
                        b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                        #bounds = ([0, 0], [100, 100])
                        #st.write(a_est)
                        try:
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                # Handle the error and continue to the next iteration
                                continue
                            #st.write("here")
                            # Replace the error with a warning message
                            else:
                                warning_message = "Fitting cannot be performed"
                                warnings.warn(warning_message, category=UserWarning)
                                popt_decay, pcov_decay = None, None
                        else: 
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            decay_curve_exp = np.round((mono_exp_decay(decay_df['Frame'], *popt_decay)),3)

                            #st.write(popt_decay)
                        try:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                continue
                            # Replace the error with a warning message
                            else:                           
                                warning_message = "Fitting cannot be performed"
                                warnings.warn(warning_message, category=UserWarning)
                                popt_rise, pcov_rise = None, None
                                #bounds = ([0, 0], [100, 100])
                                #st.write(a_est)
                        else:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            rise_curve_exp = np.round((mono_exp_rise(rise_df['Frame'], *popt_rise)),3)                
                            #st.write(popt_decay)
                            #st.write(popt_rise)
                            
                            
                            # st.write(i)
                            # st.write(nested_dict_final)
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
    
                                if (amplitude_each > 0.1*baseline_each) and (popt_rise is not None) and (popt_decay is not None):
                                    nested_dict_pro["Label"].append(i)
                                    nested_dict_pro["Number of Events"].append(event_num)
                                    nested_dict_pro["Rise time"].append(signal_rise)
                                    nested_dict_pro["Decay time"].append(signal_decay)
                                    nested_dict_pro["Duration"].append(signal_duration)
                                    nested_dict_pro["Amplitude"].append(amplitude_each) 
                                    rise_rate = np.round(popt_rise[1],4)
                                    nested_dict_pro["Rise Rate"].append(rise_rate)
                                    decay_rate = np.round(popt_decay[1],4)
                                    nested_dict_pro["Decay Rate"].append(decay_rate)
                                
                                nested_dict_final = nested_dict_pro.copy()
                #st.dataframe(new_df_pro_transposed_smooth, 1000,200)
                    
                st.write(new_df_pro_transposed_smooth)
                multi_csv = convert_df(new_df_pro_transposed_smooth)           
                st.download_button("Press to Download",  multi_csv, 'multi_cell_data.csv', "text/csv", key='download_multi_-csv')                
                #st.write(nested_dict_final)
                nested_dict_final = (pd.DataFrame.from_dict(nested_dict_final))
                
                if nested_dict_final.empty:
                    pass
                else:                              
                    st.subheader("**_Parameters for selected labels_**")
                    col_7, col_8 = st.columns(2)
                    
                    with col_7: 
                        nested_dict_final['Number of Events'] = nested_dict_final.groupby('Label')['Number of Events'].transform('count')

                        st.write(nested_dict_final)  
                        all_csv = convert_df(nested_dict_final)           
                        st.download_button("Press to Download", all_csv, 'all_data.csv', "text/csv", key='all_download-csv')
                    with col_8:
                        average_rise_time = np.round(nested_dict_final['Rise time'].mean(),4)
                        st.write(f"The average rise time based on selected labels across all frames is {average_rise_time}")
                        average_rise_rate = np.round(nested_dict_final['Rise Rate'].mean(),4)
                        st.write(f"The average rise rate based on selected labels across all frames is {average_rise_rate}")
                        average_decay_time = np.round(nested_dict_final['Decay time'].mean(),4)
                        st.write(f"The average decay time based on selected labels across all frames is {average_decay_time}")
                        average_decay_rate = np.round(nested_dict_final['Decay Rate'].mean(),4)
                        st.write(f"The average decay rate based on selected labels across all frames is {average_decay_rate}")
                        average_duration = np.round(nested_dict_final['Duration'].mean(),4)
                        st.write(f"The average duration based on selected labels across all frames is {average_duration}")
                        average_amplitude = np.round(nested_dict_final['Amplitude'].mean(),4)
                        st.write(f"The average amplitude based on selected labels across all frames is {average_amplitude}")
                        
                    st.subheader("Distribution plots based on selected labels")    
                    col_9, col_10 = st.columns(2)  
                    col_11, col_12 = st.columns(2) 
                    with col_9:
                        sns.displot(data = nested_dict_final, x="Rise time")
                        st.pyplot(plt.gcf())
                    with col_10:
                        sns.displot(data = nested_dict_final, x="Decay time")
                        st.pyplot(plt.gcf())
                    with col_11:
                        sns.displot(data = nested_dict_final, x="Duration")
                        st.pyplot(plt.gcf())
                    with col_12:    
                        sns.displot(data = nested_dict_final, x="Amplitude")
                        st.pyplot(plt.gcf())
                    st.warning('Navigating to another page from the sidebar will remove all selections from the current page')
                    # if st.button("**_Go to Single Intensity Traces_**", help = 'Clicking on this switches to a new page and all selection in the current page will be lost'):
                    #     switch_page('Single_Intensity_Trace')  
        
        if bleach_corr_check == 'Bleaching correction':
            smooth_plot_x = st.slider("*_Moving Average Window_*", min_value=1, max_value=5, help = "Adjust to smooth the mean intensity trace below. Moving average of 1 would mean the original 'Mean Intensity' trace")
            baseline_smooth_x = st.slider("*_Choose frame number(s) to average their corresponding intensity values for baseline calculation_*", min_value = 0, max_value = raw_img_ani_pg_2.shape[0]-1, value = 10,  key='smooth_multi')
            fit_first_x = st.slider("*_Choose the number of first few frame number(s) to fit a mono-exponential decay_*", min_value = 1, max_value = 30, value = 30,  key='smooth_fit_first_multi')
            fit_last_x = st.slider("*_Choose the number of last few frame number(s) to fit a mono-exponential decay_*", 1, 30, value = 30, key='smooth_fit_last_multi')
            fit_last_x = raw_img_ani_pg_2.shape[0] - 1 - fit_last_x
            
            if st.button("Obtain the parameters for selected labels",on_click=callback_movav) or st.session_state.button_clicked_movav:
                st.warning("The parameters for all labels are obtained using the same set of selections.")
                df_pro_pixel_remove = df_selected_1.drop(columns=df_selected.filter(regex='^Bright_pixel_area_').columns)
                #df_pro_pixel_remove = df_pro_pixel_remove.drop(columns=df_pro.filter(regex='^area').columns)
                new_df_pro_transposed_smooth = df_pro_pixel_remove.transpose()
                new_df_pro_transposed_smooth.columns = new_df_pro_transposed_smooth.iloc[0]
                new_df_pro_transposed_smooth.drop(new_df_pro_transposed_smooth.index[0], inplace=True)  
                
                
                #smooth_plot_x = st.slider("*_Moving Average Window_*", min_value=1, max_value=5, help = "Select to smooth the intensity trace. Moving average of 1 would mean the original 'Mean Intensity' trace below", key = 'mov_av')
                for i in df_selected['label']: 
                    
                    df_pro_transposed_smooth = pd.DataFrame(smooth_plot(new_df_pro_transposed_smooth[i],smooth_plot_x),columns = [f'smooth cell {i}'])
                    new_df_pro_transposed_smooth = pd.concat([new_df_pro_transposed_smooth.reset_index(drop=True), (np.round(df_pro_transposed_smooth[f'smooth cell {i}'],3)).reset_index(drop=True)],axis=1)
                    new_df_missing_values = pd.isna(new_df_pro_transposed_smooth[f"smooth cell {i}"])
                    new_df_pro_transposed_smooth.loc[new_df_missing_values, f'smooth cell {i}'] = new_df_pro_transposed_smooth.loc[new_df_missing_values, i]                               
                    
                    #st.write(new_df_pro_transposed)
                new_df_pro_transposed_smooth['Frame'] = pd.DataFrame(list(range(0, df_selected.shape[1])))
                new_df_pro_transposed_smooth = new_df_pro_transposed_smooth.iloc[:, [new_df_pro_transposed_smooth.shape[1] - 1] + list(range(new_df_pro_transposed_smooth.shape[1] - 1))]
                
                #get_data_indi = convert_df(new_df_pro_transposed_smooth)
                #st.download_button("Press to Download", get_data_indi, 'indi_intensity_data.csv', "text/csv", key='indi_download-get_data')
                
                #baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the 'Smoothed Mean Intensity' trace below. Baseline is calculated as: **_mode + n(S.D.)._**",  key='smooth')
                
                nested_dict_final = {}           
                nested_dict_pro = {'Label':[], "Number of Events":[], "Rise time":[], "Rise Rate":[], "Decay time":[], "Decay Rate":[], "Duration":[], "Amplitude":[]}  
                #st.write(new_df_pro_transposed_smooth)
                
                plot_df_corr = pd.DataFrame()
                plot_df_corr['Frame'] = new_df_pro_transposed_smooth['Frame']
                for i in df_selected['label']: 
                    column_corr_first = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth['Frame'] >= 0) & (new_df_pro_transposed_smooth['Frame'] <= fit_first_x), f'smooth cell {i}']
                    exp_df_1 = pd.DataFrame({f'Bleach intensity {i}': column_corr_first})
                    exp_df_1['Frames'] = new_df_pro_transposed_smooth[0:fit_first_x+1]['Frame']
                    column_corr_last = new_df_pro_transposed_smooth.loc[(new_df_pro_transposed_smooth['Frame'] >= fit_last_x) & (new_df_pro_transposed_smooth['Frame'] <= raw_img_ani_pg_2.shape[0]-1), f'smooth cell {i}']
                    exp_df_2 = pd.DataFrame({f'Bleach intensity {i}': column_corr_last})
                    exp_df_2['Frames'] = new_df_pro_transposed_smooth[fit_last_x:raw_img_ani_pg_2.shape[0]]['Frame']                   
                    exp_df = pd.concat([exp_df_1, exp_df_2], axis=0)
                    #st.write(exp_df)
                    popt_exp, pcov_exp = curve_fit(mono_exp_decay, exp_df['Frames'], exp_df[f'Bleach intensity {i}'], p0 = [np.max(exp_df['Frames']), find_b_est_decay(np.array(exp_df['Frames']), np.array(exp_df[f'Bleach intensity {i}']))])
                    photobleach_curve_exp = mono_exp_decay(new_df_pro_transposed_smooth['Frame'], *popt_exp)           
                    fit_exp_df = pd.DataFrame()
                    fit_exp_df['Frame'] = new_df_pro_transposed_smooth['Frame']
                    fit_exp_df['Photobleach Corr'] = photobleach_curve_exp
                    
                    plot_df_corr_intensity = new_df_pro_transposed_smooth[f'smooth cell {i}']-photobleach_curve_exp
                    plot_df_corr_intensity_min = min(plot_df_corr_intensity)                    
                    plot_df_corr_value = pd.DataFrame(np.round((plot_df_corr_intensity + abs(plot_df_corr_intensity_min)),3), columns = [f'smooth cell {i}'])
                    plot_df_corr = pd.concat([plot_df_corr.reset_index(drop=True), plot_df_corr_value] ,axis=1)
                    
                    #plot_df_corr['Smoothed Mean Intensity'][plot_df_corr['Smoothed Mean Intensity']<0] = 0
                    baseline_corr_each = plot_df_corr.loc[(plot_df_corr['Frame'] >= 0) & (plot_df_corr['Frame'] <= baseline_smooth_x), f'smooth cell {i}'].mean()
                    delta = np.round((plot_df_corr[f'smooth cell {i}']-baseline_corr_each)/baseline_corr_each,3)
                    plot_df_corr_value_delta = pd.DataFrame(list(delta), columns = [f'delta_f/f_0_{i}'])
                    plot_df_corr = pd.concat([plot_df_corr.reset_index(drop=True), plot_df_corr_value_delta],axis=1)
                    #plot_df_corr = pd.concat([plot_df_corr.reset_index(drop=True), (np.round(plot_df_corr[f'smooth cell {i}'],3)).reset_index(drop=True)],axis=1)
                    #plot_df_corr = pd.concat([plot_df_corr.reset_index(drop=True), (np.round(plot_df_corr[f'delta_f/f_0_{i}'],3)).reset_index(drop=True)],axis=1)
                    
                    keyval = {}
                    amp_keyval = {}
                    prev_intensity = 0
                    flag = 1
                    for frame_key, intensity_val in enumerate(plot_df_corr[f'smooth cell {i}']):
                        if prev_intensity == 0 and intensity_val > baseline_corr_each:
                            continue
                        elif intensity_val >= baseline_corr_each:
                            keyval[frame_key] = intensity_val
                            break
                        else:
                            if frame_key==len(plot_df_corr.index)-1:
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
                    for frame_key_2, intensity_val_2 in enumerate(plot_df_corr[f'smooth cell {i}']):                               
                        if frame_key_2 <= frame_key:
                            continue
                        elif frame_key_2 > frame_key:
                            if intensity_val_2 >= prev_intensity:
                                if intensity_val_2 < baseline_corr_each:
                                    frame_key = frame_key_2
                                    continue
                                elif intensity_val_2 >= baseline_corr_each:
                                    if prev_intensity < baseline_corr_each:
                                        first_key = frame_key_2
                                        first_intensity = intensity_val_2
                                        keyval[first_key] = first_intensity
                                        frame_key = frame_key_2
                                        prev_intensity = intensity_val_2
                                    else:
                                        frame_key = frame_key_2
                                        prev_intensity = intensity_val_2
                                    
            
                            elif intensity_val_2 < prev_intensity:
                                if intensity_val_2 > baseline_corr_each:
                                    frame_key = frame_key_2                                         
                                elif intensity_val_2 <= baseline_corr_each:
                                    if prev_intensity <= baseline_corr_each:
                                        frame_key = frame_key_2
                                        continue
                                    else:
                                        keyval[frame_key_2] = intensity_val_2
                                        frame_key = frame_key_2
                                        #start_key = plot_df.query(f'"Smoothed Mean Intensity" == {prev_intensity}')['Frame']
                                        amp_key_vals = plot_df_corr[plot_df_corr[f'smooth cell {i}']==prev_intensity]['Frame']
                                        amp_key_vals = amp_key_vals[amp_key_vals>=first_key].iloc[0]
                                        amp_key = str(amp_key_vals)
                                        #amp_key = str(new_df_pro_transposed_smooth[new_df_pro_transposed_smooth[f'smooth cell {i}']==prev_intensity]['Frame'].iloc[0])
                                        amplitude = prev_intensity - baseline_corr_each
                                        keyval[amp_key] = prev_intensity
                                        prev_intensity = intensity_val_2
                                        if (first_key == int(amp_key)): #or (int(amp_key) == frame_key): 
                                            first_key = int(amp_key)-1
                                            amp_keyval[f"{first_key}-{amp_key}-{frame_key}"] = amplitude
                                        else:
                                            amp_keyval[f"{first_key}-{amp_key}-{frame_key}"] = amplitude                
                    
                    
                    max_df_value = plot_df_corr[f'smooth cell {i}'].max()
                    #st.write(plot_df.dtypes)
                    #####test by setting a some equal high values#########plot_df.loc[plot_df['Frame'] == 39, 'Smoothed Mean Intensity'] = max_df_value ##plot_df.loc[plot_df['Frame'] == 69, 'Smoothed Mean Intensity'] = baseline_each
                    count_max = plot_df_corr[f'smooth cell {i}'].eq(max_df_value).sum()
                    max_frame = plot_df_corr.loc[plot_df_corr[f'smooth cell {i}'] == max_df_value, 'Frame']
                    decay_df = pd.DataFrame()
                    rise_df = pd.DataFrame()
                    if ~((plot_df_corr[f'smooth cell {i}'] <= baseline_corr_each) & (plot_df_corr['Frame'] > max(max_frame))).any(): ##trace crosses baseline but never comes back
                        #nested_dict_pro = {'Label':[], "Number of Events":[], "Rise time":[], "Rise Rate":[], "Decay time":[], "Decay Rate":[], "Duration":[], "Amplitude":[]}
                        continue
                        
                    else:
                        
                        if count_max == 1:
                            rise_df['Rise intensity'] = plot_df_corr.loc[(plot_df_corr[f'smooth cell {i}'] <= max_df_value) & (plot_df_corr[f'smooth cell {i}'] >= baseline_corr_each) & (plot_df_corr['Frame'] <= max(max_frame)) , f'smooth cell {i}']
                            decay_df['Decay intensity'] = plot_df_corr.loc[(plot_df_corr[f'smooth cell {i}'] <= max_df_value) & (plot_df_corr[f'smooth cell {i}'] >= baseline_corr_each) & (plot_df_corr['Frame'] >= max(max_frame)) , f'smooth cell {i}']
                            decay_df['Frame'] = decay_df.index
                            rise_df['Frame'] = rise_df.index
                            decay_df = decay_df[decay_df.columns[::-1]]
                            rise_df = rise_df[rise_df.columns[::-1]]
                            test_missing_value_df = next((decay_df['Frame'].iloc[i] + 1 for i in range(len(decay_df['Frame'])-1) if decay_df['Frame'].iloc[i+1] - decay_df['Frame'].iloc[i] > 1), None)
                            if test_missing_value_df is None:
                                missing_value_df = None
                            else:
                                missing_value_df = test_missing_value_df - 1 
                            missing_value_rise_df = (rise_df.loc[rise_df['Frame'].diff() > 1, 'Frame'].max())-1
                            #st.write(~rise_df['Rise intensity'].isin([baseline_each]).any())
                            # if ~decay_df['Decay intensity'].isin([baseline_each]).any():
                            #     new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_each}
                            #     decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_each]).any():
                            #     new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_each}
                            #     rise_df.loc[missing_value_rise_df] = new_row_rise
                            
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_df)
                            if not pd.isna(missing_value_df): #there is a missing value
                                #st.write('here')
                                decay_df = decay_df.loc[decay_df['Frame'] <= missing_value_df]
                             
                            else:
                                if (decay_df['Decay intensity'] == baseline_corr_each).any():
                                    baseline_frame = max(decay_df.loc[decay_df['Decay intensity'] == baseline_corr_each, 'Frame'])
                                    decay_df = decay_df.loc[(decay_df['Decay intensity'] >= baseline_corr_each) & (decay_df['Frame'] <= baseline_frame)]  
                                else:
                                    decay_df = decay_df
                                
                            if not pd.isna(missing_value_rise_df):
                                #st.write('here')
                                rise_df = rise_df.loc[rise_df['Frame'] >= missing_value_rise_df]
                            else:
                                if (rise_df['Rise intensity'] == baseline_corr_each).any():
                                    #st.write(rise_df)
                                    baseline_frame = max(rise_df.loc[rise_df['Rise intensity'] == baseline_corr_each, 'Frame'])
                                    rise_df = rise_df.loc[(rise_df['Rise intensity'] >= baseline_corr_each) & (rise_df['Frame'] >= baseline_frame)]  
                                else:
                                    rise_df = rise_df
                            #st.write(rise_df)
                            #st.write(missing_value_df)
                            
                        if count_max > 1: 
                            decay_df['Decay intensity'] = plot_df_corr.loc[(plot_df_corr[f'smooth cell {i}'] <= max_df_value) & (plot_df_corr[f'smooth cell {i}'] >= baseline_corr_each) & (plot_df_corr['Frame'] >= max(max_frame)) , f'smooth cell {i}']
                            last_index = decay_df.loc[decay_df['Decay intensity'] == max_df_value].index[-1]
                            rise_df['Rise intensity'] = plot_df_corr.loc[(new_df_pro_transposed_smooth[f'smooth cell {i}'] <= max_df_value) & (plot_df_corr[f'smooth cell {i}'] >= baseline_corr_each) & (plot_df_corr['Frame'] <= max(max_frame)) , f'smooth cell {i}']
                            first_index = decay_df.loc[decay_df['Decay intensity'] == max_df_value].index[0]                    
                            decay_df.loc[last_index, 'Decay intensity'] *= 1.01
                            rise_df.loc[first_index, 'Rise intensity'] *= 1.01
                            decay_df['Frame'] = decay_df.index
                            rise_df['Frame'] = rise_df.index
                            decay_df = decay_df[decay_df.columns[::-1]]
                            rise_df = rise_df[rise_df.columns[::-1]]
                            #st.write(decay_df)
                            test_missing_value_df = next((decay_df['Frame'].iloc[i] + 1 for i in range(len(decay_df['Frame'])-1) if decay_df['Frame'].iloc[i+1] - decay_df['Frame'].iloc[i] > 1), None)
                            if test_missing_value_df is None:
                                missing_value_df = None
                            else:
                                missing_value_df = test_missing_value_df - 1                         
                            missing_value_rise_df = (rise_df.loc[rise_df['Frame'].diff() > 1, 'Frame'].max())-1
                            #st.write(np.any(rise_df['Rise intensity']) != baseline_each)
                            # if ~decay_df['Decay intensity'].isin([baseline_corr_each]).any():
                            #     new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_corr_each}
                            #     decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_corr_each]).any():
                            #     new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_corr_each}
                            #     rise_df.loc[missing_value_rise_df] = new_row_rise
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_rise_df)
                            if not pd.isna(missing_value_df):
                                #st.write('here')
                                decay_df = decay_df.loc[decay_df['Frame'] <= missing_value_df]
                            else:
                                if (decay_df['Decay intensity'] == baseline_corr_each).any():
                                    baseline_frame = max(decay_df.loc[decay_df['Decay intensity'] == baseline_corr_each, 'Frame'])
                                    decay_df = decay_df.loc[(decay_df['Decay intensity'] >= baseline_corr_each) & (decay_df['Frame'] <= baseline_frame)]  
                                else:
                                    decay_df = decay_df
                                
                            if not pd.isna(missing_value_rise_df):
                                #st.write('here')
                                rise_df = rise_df.loc[rise_df['Frame'] >= missing_value_rise_df]
                            else:
                                if (rise_df['Rise intensity'] == baseline_corr_each).any():
                                    #st.write(rise_df)
                                    baseline_frame = max(rise_df.loc[rise_df['Rise intensity'] == baseline_corr_each, 'Frame'])
                                    rise_df = rise_df.loc[(rise_df['Rise intensity'] >= baseline_corr_each) & (rise_df['Frame'] >= baseline_frame)]  
                                else:
                                    rise_df = rise_df             
                            #st.write(rise_df)
                        
                        a_est = decay_df['Decay intensity'].iloc[0]
                        b_est = find_b_est_decay(np.array(decay_df['Frame']), np.array(decay_df['Decay intensity']))
                        a_est_rise = rise_df['Rise intensity'].iloc[-1]
                        b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                        #bounds = ([0, 0], [100, 100])
                        #st.write(a_est)
                        try:
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                # Handle the error and continue to the next iteration
                                continue
                            #st.write("here")
                            # Replace the error with a warning message
                            else:
                                warning_message = "Fitting cannot be performed"
                                warnings.warn(warning_message, category=UserWarning)
                                popt_decay, pcov_decay = None, None
                        else: 
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            decay_curve_exp = np.round((mono_exp_decay(decay_df['Frame'], *popt_decay)),3)

                            #st.write(popt_decay)
                        try:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                continue
                            # Replace the error with a warning message
                            else:                           
                                warning_message = "Fitting cannot be performed"
                                warnings.warn(warning_message, category=UserWarning)
                                popt_rise, pcov_rise = None, None
                                #bounds = ([0, 0], [100, 100])
                                #st.write(a_est)
                        else:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            rise_curve_exp = np.round((mono_exp_rise(rise_df['Frame'], *popt_rise)),3)                
                            #st.write(popt_decay)
                            #st.write(popt_rise)
                            
                            
                            # st.write(i)
                            # st.write(nested_dict_final)
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
    
                                if (amplitude_each > 0.1*baseline_corr_each) and (popt_rise is not None) and (popt_decay is not None):
                                    nested_dict_pro["Label"].append(i)
                                    nested_dict_pro["Number of Events"].append(event_num)
                                    nested_dict_pro["Rise time"].append(signal_rise)
                                    nested_dict_pro["Decay time"].append(signal_decay)
                                    nested_dict_pro["Duration"].append(signal_duration)
                                    nested_dict_pro["Amplitude"].append(amplitude_each) 
                                    rise_rate = np.round(popt_rise[1],4)
                                    nested_dict_pro["Rise Rate"].append(rise_rate)
                                    decay_rate = np.round(popt_decay[1],4)
                                    nested_dict_pro["Decay Rate"].append(decay_rate)
                                
                                nested_dict_final = nested_dict_pro.copy()
                
                st.write('*_The original intensity data_*')  
                st.dataframe(new_df_pro_transposed_smooth, 1000,200)
                multi_csv_bleach = convert_df(new_df_pro_transposed_smooth)           
                st.download_button("Press to Download",  multi_csv_bleach, 'multi_cell_data.csv', "text/csv", key='download_multi_-csv_bleach')   
                st.write('*_The Photobleaching-corrected data_*')
                st.dataframe(plot_df_corr, 1000,200)
                multi_csv_corr = convert_df(plot_df_corr)  
                st.download_button("Press to Download",  multi_csv_corr, 'multi_cell_data_corr.csv', "text/csv", key='download_multi_-csv_bleach_corr')   
                #st.write(nested_dict_final)
                nested_dict_final = (pd.DataFrame.from_dict(nested_dict_final))
                
                if nested_dict_final.empty:
                    pass
                else:                              
                    st.subheader("**_Parameters for selected labels_**")
                    col_7, col_8 = st.columns(2)
                    
                    with col_7: 
                        nested_dict_final = nested_dict_final[nested_dict_final.groupby('Label')['Amplitude'].transform(max) == nested_dict_final['Amplitude']]
                        nested_dict_final['Number of Events'] = nested_dict_final.groupby('Label')['Number of Events'].transform('count')
                        #nested_dict_final = nested_dict_final[(nested_dict_final['Amplitude']) == max((nested_dict_final['Amplitude']))]
                        #nested_dict_final["Number of Events"] = nested_dict_final.shape[0]
                        nested_dict_final = nested_dict_final.reset_index(drop=True)
                        st.write(nested_dict_final)  
                        all_csv_bleach = convert_df(nested_dict_final)           
                        st.download_button("Press to Download", all_csv_bleach, 'all_data.csv', "text/csv", key='all_download-csv_bleach')
                    with col_8:
                        average_rise_time = np.round(nested_dict_final['Rise time'].mean(),4)
                        st.write(f"The average rise time based on selected labels across all frames is {average_rise_time}")
                        average_rise_rate = np.round(nested_dict_final['Rise Rate'].mean(),4)
                        st.write(f"The average rise rate based on selected labels across all frames is {average_rise_rate}")
                        average_decay_time = np.round(nested_dict_final['Decay time'].mean(),4)
                        st.write(f"The average decay time based on selected labels across all frames is {average_decay_time}")
                        average_decay_rate = np.round(nested_dict_final['Decay Rate'].mean(),4)
                        st.write(f"The average decay rate based on selected labels across all frames is {average_decay_rate}")
                        average_duration = np.round(nested_dict_final['Duration'].mean(),4)
                        st.write(f"The average duration based on selected labels across all frames is {average_duration}")
                        average_amplitude = np.round(nested_dict_final['Amplitude'].mean(),4)
                        st.write(f"The average amplitude based on selected labels across all frames is {average_amplitude}")
                        
                    st.subheader("Distribution plots based on selected labels")    
                    col_9, col_10 = st.columns(2)  
                    col_11, col_12 = st.columns(2) 
                    with col_9:
                        sns.displot(data = nested_dict_final, x="Rise time")
                        st.pyplot(plt.gcf())
                    with col_10:
                        sns.displot(data = nested_dict_final, x="Decay time")
                        st.pyplot(plt.gcf())
                    with col_11:
                        sns.displot(data = nested_dict_final, x="Duration")
                        st.pyplot(plt.gcf())
                    with col_12:    
                        sns.displot(data = nested_dict_final, x="Amplitude")
                        st.pyplot(plt.gcf())
                    st.warning('Navigating to another page from the sidebar will remove all selections from the current page')
                    # if st.button("**_Go to Single Intensity Traces_**", help = 'Clicking on this switches to a new page and all selection in the current page will be lost'):
                    #     switch_page('Single_Intensity_Trace')  
                    
                    