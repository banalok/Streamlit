import streamlit as st
from utils import * 
from streamlit_extras.switch_page_button import switch_page 
import plotly.io
import plotly.graph_objs as go               
import math
from PIL import Image
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
from scipy.optimize import curve_fit, Bounds
from scipy import interpolate
from skimage import measure, color, io
import plotly.express as px
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

if 'all_param_table' not in st.session_state:
    st.session_state.all_param_table = False
    
if "button_clicked_sing_para" not in st.session_state:
    st.session_state.button_clicked_sing_para = False
    
def callback_sing():
    #Button was clicked
    st.session_state.button_clicked_sing_para = True
    
def callback_all_param_table():
   st.session_state.all_param_table = True    

if "button_clicked_movav" not in st.session_state:
    st.session_state.button_clicked_movav = False
    
def callback_movav():
    #Button was clicked
    st.session_state.button_clicked_movav = True
    
st.header('**_Intensity trace for single label_**')
     
if 'Collapsed_Image' not in st.session_state:
    pass
else:
    collapsed = st.session_state['Collapsed_Image']
    st.write('*_The Selected Image_*')
    st.image(collapsed,use_column_width=True,clamp = True)
    
if 'label_list_pg_2' not in st.session_state:
    pass
else:
    label_list_pg_2 = st.session_state['label_list_pg_2'] 
    
if 'raw_img_ani_pg_2' not in st.session_state:
    pass
else:
    raw_img_ani_pg_2 = st.session_state['raw_img_ani_pg_2']
    
if 'background_corr_pg_2' not in st.session_state:
    pass
else:
    background_corr_pg_2 = st.session_state['background_corr_pg_2']
    
if 'super_im_rgb_pg_2' not in st.session_state:
    pass
else:
    super_im_pg_2 = st.session_state['super_im_rgb_pg_2']
    st.write('*_Automatically labeled objects on the selected image_*')
    st.image(super_im_pg_2,use_column_width=True,clamp = True)
    
if 'final_label_rgb_pg_2' not in st.session_state:
    st.warning("Please generate the segmented and labeled image from the 'Preprocessing and Segmentation' page, and click on 'Single-cell Analysis' before proceeding")
else:
    label = st.session_state['final_label_rgb_pg_2']
    st.write('*_Automatically (or automatically plus manually) segmented and labeled objects on a black background_*')
    st.image(label,use_column_width=True,clamp = True)
    
if 'final_label_pg_2' not in st.session_state:
    pass
else:
    if 'area_thres_x' not in st.session_state:
        st.session_state['area_thres_x'] = st.number_input("*_Choose the area threshold percentage_*", min_value=0.00, max_value=1.00, value=0.30,step = 0.01, format="%0.2f", help = f"Default is 0.3. Pixels below 30% of the maximum ({np.amax(raw_img_ani_pg_2)}) are not counted to get the bright area of labels", key='area_thres_1')
    if 'df_pro' not in st.session_state:
        label_fin = st.session_state['final_label_pg_2']            
        data = list(np.unique(label_list_pg_2))        
        st.session_state['df_pro'] = pd.DataFrame(data, columns=['label'])
        col_arr = []
        props_pro = get_intensity(background_corr_pg_2[:, :, :, 0], [label_fin] * raw_img_ani_pg_2.shape[0]) 
        max_intensity = get_max_intensity(background_corr_pg_2[:, :, :, 0], [label_fin] * raw_img_ani_pg_2.shape[0]) 
        max_intensity_df = pd.DataFrame(max_intensity).T              
        props_pro = pd.DataFrame(props_pro).T
        props_pro['label'] = data
        
        for frames_pro in range(0,raw_img_ani_pg_2.shape[0]):
            col = []
            label_array = props_pro['label']
            intensity_im = background_corr_pg_2[frames_pro][:,:,0]                                  
            for lab in label_array:
                mask_label = label_fin == lab                
                intensity_values = intensity_im[mask_label]
                col.append(intensity_values)
            col_arr.append(np.array(col, dtype=object))
            
            df_single = props_pro            
            df_single.rename(columns = {frames_pro : f'intensity_mean_{frames_pro}'}, inplace=True)
            df_single[f'intensity_mean_{frames_pro}'] = np.round(df_single[f'intensity_mean_{frames_pro}'],3)

        st.session_state['df_pro'] = pd.merge(st.session_state['df_pro'], df_single, on = 'label', how = 'outer')                                                 
        
                
        ######## #################  ################# ###############Interactive table################################################################
        
        for frame_col in range(0, raw_img_ani_pg_2.shape[0]):            
            pixel_counts = []
            for label_val in st.session_state['df_pro']['label']:
                intensity_image = col_arr[frame_col][label_val-1]                
                count = np.sum(np.greater(intensity_image, st.session_state['area_thres_x'] * max_intensity_df.iloc[label_val - 1][frame_col])) #np.amax(raw_img_ani_pg_2[frame_col]))) #df_pro[f'intensity_mean_{frames_pro}'].mean()))
                pixel_counts.append(np.float64(count))           
            pixel_var = f'Bright_pixel_area_{frame_col}'            
            pixel_counts_df = pd.DataFrame(pixel_counts,columns = [pixel_var],dtype = np.float64)
            st.session_state['df_pro'] = pd.concat((st.session_state['df_pro'], pixel_counts_df),axis=1)  

        st.dataframe(st.session_state['df_pro'], 1000, 200)
        dataframe_df = st.session_state['df_pro']
        get_data_indi = convert_df(st.session_state['df_pro'])
        st.download_button("Press to Download", get_data_indi, 'label_intensity_data.csv', "text/csv", key='label_download-get_data') 
    else:
        label_fin = st.session_state['final_label_pg_2']
        max_intensity = get_max_intensity(background_corr_pg_2[:, :, :, 0], [label_fin] * raw_img_ani_pg_2.shape[0]) 
        max_intensity_df = pd.DataFrame(max_intensity).T        
        area_thres_x = st.number_input("*_Choose the area threshold percentage_*", min_value=0.00, max_value=1.00, value=0.3,step = 0.01, format="%0.2f", help = f"Default is 0.3. Pixels below 30% of the maximum ({np.amax(raw_img_ani_pg_2)}) are not counted to get the bright area of labels", key='area_thres')
        if area_thres_x == st.session_state['area_thres_x']:
            st.dataframe(st.session_state['df_pro'], 1000, 200)            
            dataframe_df = st.session_state['df_pro']
            get_data_indi = convert_df(st.session_state['df_pro'])
            st.download_button("Press to Download", get_data_indi, 'label_intensity_data.csv', "text/csv", key='label_download-get_data') 
            st.session_state['area_thres_x'] = area_thres_x
        else:
            st.session_state['area_thres_x'] = area_thres_x
                 
            data = list(np.unique(label_list_pg_2))        
            st.session_state['df_pro'] = pd.DataFrame(data, columns=['label'])
            col_arr = []
            props_pro = get_intensity(background_corr_pg_2[:, :, :, 0], [label_fin] * raw_img_ani_pg_2.shape[0])  
            props_pro = pd.DataFrame(props_pro).T
            props_pro['label'] = data            

            for frames_pro in range(0,raw_img_ani_pg_2.shape[0]):
                
                col = []
                label_array = props_pro['label']
                intensity_im = background_corr_pg_2[frames_pro][:,:,0]                                        
                for lab in label_array:
                    mask_label = label_fin == lab
                    
                    intensity_values = intensity_im[mask_label]
                    col.append(intensity_values)
                col_arr.append(np.array(col, dtype=object))
                df_single = props_pro                
                df_single.rename(columns = {frames_pro : f'intensity_mean_{frames_pro}'}, inplace=True)
                df_single[f'intensity_mean_{frames_pro}'] = np.round(df_single[f'intensity_mean_{frames_pro}'],3)
    
            st.session_state['df_pro'] = pd.merge(st.session_state['df_pro'], df_single, on = 'label', how = 'outer')                                               
           
            ######## #################  ################# ###############Interactive table################################################################
            
            for frame_col in range(0, raw_img_ani_pg_2.shape[0]):
                
                pixel_counts = []
                for label_val in st.session_state['df_pro']['label']:
                    intensity_image = col_arr[frame_col][label_val-1]
                    count = np.sum(np.greater(intensity_image, st.session_state['area_thres_x']* max_intensity_df.iloc[label_val - 1][frame_col])) #np.amax(raw_img_ani_pg_2[frame_col]))) #df_pro[f'intensity_mean_{frames_pro}'].mean()))
                    pixel_counts.append(np.float64(count))                
                pixel_var = f'Bright_pixel_area_{frame_col}'                
                pixel_counts_df = pd.DataFrame(pixel_counts,columns = [pixel_var],dtype = np.float64)
                st.session_state['df_pro'] = pd.concat((st.session_state['df_pro'], pixel_counts_df),axis=1)   

            st.dataframe(st.session_state['df_pro'], 1000, 200)
            dataframe_df = st.session_state['df_pro']
            get_data_indi = convert_df(st.session_state['df_pro'])
            st.download_button("Press to Download", get_data_indi, 'label_intensity_data.csv', "text/csv", key='label_download-get_data_st')                       
    st.write('*_Select a label to explore_*')  
    area_columns_to_drop = dataframe_df.columns[dataframe_df.columns.str.contains('Bright_pixel_area')]
    dataframe_df_pro = dataframe_df.drop(columns=area_columns_to_drop)   
  
    if "selected_aggrid" not in st.session_state:
        st.session_state["selected_aggrid"] = []
    gb = GridOptionsBuilder.from_dataframe(dataframe_df_pro)                       
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection(selection_mode="single", use_checkbox=True, groupSelectsChildren="Group checkbox select children", pre_selected_rows=st.session_state["selected_aggrid"]) #list(range(0, len(df_pro))))  #[str(st.session_state.selected_row)]
    gb.configure_columns(dataframe_df_pro.columns, editable=False)           
    gridOptions = gb.build()
    
    grid_response = AgGrid(
        dataframe_df_pro,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode=GridUpdateMode.MODEL_CHANGED,    #'MODEL_CHANGED',
        update_on='MANUAL',
        #data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
        theme='alpine', #Add theme color to the table
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        #reload_data=False,
        key='table_key'
    )
    
    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    st.session_state["selected_aggrid"] = selected
        
    df_selected = pd.DataFrame(selected)
    labels_rgb = np.expand_dims(label_fin, axis=2)
    labels_rgbb = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)
    
    if selected:         
        for i in df_selected['label']:
            # Extract the coordinates of the region boundary
            coords = np.argwhere(label_fin==i)
        
            # Create a polygon from the coordinates
            poly = polygon(coords[:, 0], coords[:, 1])
            #st.write(poly)
            # Set the color of the polygon to red
            color_poly = (255, 0, 0)
        
            # Color the polygon in the color image
            labels_rgbb[poly] = color_poly
        
        # Display the color image with the selected regions highlighted
        
        st.image(labels_rgbb,use_column_width=True,clamp = True)
        df_selected = df_selected.drop(columns = ['_selectedRowNodeInfo'])
    
        nested_dict = {'Label':[], "Number of Events":[], "Rise time":[], "Decay time":[], "Duration":[], "Amplitude":[]}
        
        frame_rate = st.number_input("Frame Rate (frames per second/fps)", min_value = 0.1, max_value = 100.0, value = 1.0, step = 0.1, format = "%.1f", help = "Type the values between 0.1 and 100.0 (inclusive). Takes values in steps of 0.1. Default is 1.0")
        bleach_corr_check = st.radio("Select one", ('No bleaching correction', 'Bleaching correction'), help='Analyze the trace as is (No bleaching correction) or fit mono-exponential curves and interpolate to correct for bleaching (Bleaching correction)')
        area_df = area(dataframe_df, df_selected, raw_img_ani_pg_2)       
        
        if bleach_corr_check == 'No bleaching correction':
        
            st.subheader("**_Data for intensity of selected label_**")  
            smooth_plot_x = st.number_input("*_Moving Average Window_*", min_value=1, max_value=5, help = "Adjust to smooth the mean intensity trace below. Moving average of 1 would mean the original 'Mean Intensity' trace")
            plot_df = intensity(df_selected, raw_img_ani_pg_2, smooth_plot_x)                      
            missing_values = pd.isna(plot_df["Smoothed Mean Intensity"])
            plot_df.loc[missing_values, "Smoothed Mean Intensity"] = plot_df.loc[missing_values, "Mean Intensity"]  
            plot_df["Smoothed Mean Intensity"] = np.round(plot_df["Smoothed Mean Intensity"],3)                            
            baseline_peak_selection = st.radio("Select one", ('Static', 'Dynamic'), help='Select "Static" to manually select single values for the baseline, peak and recovery frames; otherwise, select "Dynamic"')
            if baseline_peak_selection == "Dynamic":
                baseline_smooth_x = st.slider("*_Choose frame number(s) to average their corresponding intensity values for baseline calculation_*", min_value = 0, max_value = raw_img_ani_pg_2.shape[0]-1, value = 10,  key='smooth')
                #baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the 'Smoothed Mean Intensity' trace below. Baseline is calculated as: **_mode + n(S.D.)._**",  key='smooth')
                
                baseline_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Smoothed Mean Intensity'].mean()
                baseline_mean_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Mean Intensity'].mean()
                plot_df['Mean Intensity'] = plot_df['Mean Intensity']/baseline_mean_each
                
                #st.write(baseline_each)
                plot_df['Smoothed Mean Intensity'] = plot_df['Smoothed Mean Intensity']/baseline_each
                baseline_each = baseline_each/baseline_each                   
                plot_df['delta_f/f_0'] = (plot_df['Smoothed Mean Intensity'] - baseline_each)/baseline_each 
                plot_df['Time'] = plot_df['Frame']/frame_rate
                area_df['Time']  =  area_df["Frame"]/frame_rate
                
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
                                        x="Time",
                                        y="Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                                                
                    smoothed_figure =  px.line(
                                        plot_df,
                                        x="Time",
                                        y='Smoothed Mean Intensity'
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                    smoothed_figure.add_shape(type='line',
                                        x0=0,
                                        y0=baseline_each,
                                        x1=(raw_img_ani_pg_2.shape[0])/frame_rate,
                                        y1=baseline_each,
                                        line=dict(color='Red',),
                                        xref='x',
                                        yref='y') 
                    unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                
            
            
                    csv = convert_df(plot_df)           
                    st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_1')                    
                    st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                    st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)         
                    st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)  
                   
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
                    
                    count_items = 0
                    for item in amp_keyval.items():                        
                        if len(item[0].split('-'))==3:
                            count_items += 1
                            signal_start_frame = item[0].split('-')[0]                            
                            peak_frame = item[0].split('-')[1]                            
                            signal_decay_frame = item[0].split('-')[2]                            
                            event_num = count_items
                            amplitude_each = item[1]
                            signal_rise = (int(peak_frame)-int(signal_start_frame))/frame_rate
                            signal_decay = (int(signal_decay_frame)-int(peak_frame))/frame_rate
                            signal_duration = (int(signal_decay_frame)-int(signal_start_frame))/frame_rate
                            nested_dict["Label"].append(i)
                            nested_dict["Number of Events"].append(event_num)
                            nested_dict["Rise time"].append(signal_rise)
                            nested_dict["Decay time"].append(signal_decay)
                            nested_dict["Duration"].append(signal_duration)
                            nested_dict["Amplitude"].append(amplitude_each)
                    
                    
                    st.write("**_Table with normalized intensity values_**")
                    st.dataframe(plot_df, 1000,200)
                    
                    max_df_value = plot_df['Smoothed Mean Intensity'].max()
                    #st.write(plot_df.dtypes)
                    #####test by setting a some equal high values#########plot_df.loc[plot_df['Frame'] == 39, 'Smoothed Mean Intensity'] = max_df_value ##plot_df.loc[plot_df['Frame'] == 69, 'Smoothed Mean Intensity'] = baseline_each
                    count_max = plot_df['Smoothed Mean Intensity'].eq(max_df_value).sum()
                    max_frame = plot_df.loc[plot_df['Smoothed Mean Intensity'] == max_df_value, 'Frame']
                    decay_df = pd.DataFrame()
                    rise_df = pd.DataFrame()
    
                    if ((plot_df.loc[plot_df["Frame"].gt(max(max_frame)), "Smoothed Mean Intensity"]).gt(baseline_each)).all(): ##trace crosses baseline but never comes back
                        
                        if count_max == 1:
                            rise_df['Rise intensity'] = plot_df.loc[(plot_df['Smoothed Mean Intensity'] <= max_df_value) & (plot_df['Smoothed Mean Intensity'] >= baseline_each) & (plot_df['Frame'] <= max(max_frame)) , 'Smoothed Mean Intensity']
                            rise_df['Frame'] = rise_df.index
                            rise_df = rise_df[rise_df.columns[::-1]]
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
                            
                            #st.write(missing_value_df)
                            
                        if count_max > 1: 
                                                    
                            rise_df['Rise intensity'] = plot_df.loc[(plot_df['Smoothed Mean Intensity'] <= max_df_value) & (plot_df['Smoothed Mean Intensity'] >= baseline_each) & (plot_df['Frame'] <= min(max_frame)) , 'Smoothed Mean Intensity']
                            first_index = rise_df.loc[rise_df['Rise intensity'] == max_df_value].index[-1]                    
                            #rise_df.loc[first_index, 'Rise intensity'] *= 1.01                        
                            rise_df['Frame'] = rise_df.index
                            rise_df = rise_df[rise_df.columns[::-1]]
                            #st.write(decay_df)
                            
                            missing_value_rise_df = (rise_df.loc[rise_df['Frame'].diff() > 1, 'Frame'].max())-1
                            #st.write(np.any(rise_df['Rise intensity']) != baseline_each)
                            # if ~decay_df['Decay intensity'].isin([baseline_each]).any():
                            #     if missing_value_df is not None:
                            #         new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_each}
                            #         decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_each]).any():
                            #     if missing_value_rise_df is not None:
                            #         new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_each}
                            #         rise_df.loc[missing_value_rise_df] = new_row_rise
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_rise_df)
                                
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
                                       
                        a_est_rise = rise_df['Rise intensity'].iloc[-1]
                        b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                        #bounds = ([0, 0], [100, 100])
                        
                        popt_decay, pcov_decay = None, None                        
                        
                        try:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                pass
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
                            
                        signal_rise = (max(max_frame) - rise_df['Frame'].iloc[0])/frame_rate
                        amplitude_each = max_df_value - baseline_each
                        nested_dict["Label"].append(i)
                        nested_dict["Number of Events"].append(None)
                        nested_dict["Rise time"].append(signal_rise)
                        nested_dict["Decay time"].append(None)
                        nested_dict["Duration"].append(None)
                        nested_dict["Amplitude"].append(amplitude_each)
                        #st.write("here")
                        unsmoothed_figure =  px.line(
                                            plot_df,
                                            x="Time",
                                            y="Mean Intensity"
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )
                                                    
                        smoothed_figure =  px.line(
                                            plot_df,
                                            x="Time",
                                            y='Smoothed Mean Intensity'
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )
                        smoothed_figure.add_shape(type='line',
                                            x0=0,
                                            y0=baseline_each,
                                            x1=(raw_img_ani_pg_2.shape[0])/frame_rate,
                                            y1=baseline_each,
                                            line=dict(color='Red',),
                                            xref='x',
                                            yref='y') 
                        unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                    
                            
                        csv = convert_df(plot_df)           
                        st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_2')
                        #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
                        #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
                        
                        st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                        st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)         
                        st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)  
                        
                    else:
                        if count_max == 1:
                            rise_df['Rise intensity'] = plot_df.loc[(plot_df['Smoothed Mean Intensity'] <= max_df_value) & (plot_df['Smoothed Mean Intensity'] >= baseline_each) & (plot_df['Frame'] <= max(max_frame)) , 'Smoothed Mean Intensity']
                            decay_df['Decay intensity'] = plot_df.loc[(plot_df['Smoothed Mean Intensity'] <= max_df_value) & (plot_df['Smoothed Mean Intensity'] >= baseline_each) & (plot_df['Frame'] >= max(max_frame)) , 'Smoothed Mean Intensity']
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
                            
                            #st.write(missing_value_df)
                            
                        if count_max > 1: 
                            decay_df['Decay intensity'] = plot_df.loc[(plot_df['Smoothed Mean Intensity'] <= max_df_value) & (plot_df['Smoothed Mean Intensity'] >= baseline_each) & (plot_df['Frame'] >= max(max_frame)) , 'Smoothed Mean Intensity']
                            last_index = decay_df.loc[decay_df['Decay intensity'] == max_df_value].index[-1]
                            rise_df['Rise intensity'] = plot_df.loc[(plot_df['Smoothed Mean Intensity'] <= max_df_value) & (plot_df['Smoothed Mean Intensity'] >= baseline_each) & (plot_df['Frame'] <= min(max_frame)) , 'Smoothed Mean Intensity']
                            first_index = rise_df.loc[rise_df['Rise intensity'] == max_df_value].index[0]                  
                            #decay_df.loc[last_index, 'Decay intensity'] *= 1.01
                            #rise_df.loc[first_index, 'Rise intensity'] *= 1.01
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
                            #     if missing_value_df is not None:
                            #         new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_each}
                            #         decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_each]).any():
                            #     if missing_value_rise_df is not None:
                            #         new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_each}
                            #         rise_df.loc[missing_value_rise_df] = new_row_rise
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
                        
                        
                        a_est = decay_df['Decay intensity'].iloc[0]
                        b_est = find_b_est_decay(np.array(decay_df['Frame']), np.array(decay_df['Decay intensity']))
                        a_est_rise = rise_df['Rise intensity'].iloc[-1]
                        b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                        #bounds = ([0, 0], [100, 100])
                        
                        try:
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                # Handle the error and continue to the next iteration
                                pass
                            #st.write("here")
                            # Replace the error with a warning message
                            else:
                                warning_message = "Fitting cannot be performed"
                                warnings.warn(warning_message, category=UserWarning)
                                popt_decay, pcov_decay = None, None
                        else: 
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            decay_curve_exp = mono_exp_decay(decay_df['Frame'], *popt_decay)
                            
                        
                        try:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                pass
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
                        
                        unsmoothed_figure =  px.line(
                                            plot_df,
                                            x="Time",
                                            y="Mean Intensity"
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )
                                                    
                        smoothed_figure =  px.line(
                                            plot_df,
                                            x="Time",
                                            y='Smoothed Mean Intensity'
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )          
                       
                        if popt_decay is not None and pcov_decay is not None:
                            smoothed_figure.add_trace(go.Scatter(x = decay_df['Frame']/frame_rate, y = decay_curve_exp, mode="markers", name='Decay Fit'))

                        smoothed_figure.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_each, baseline_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                                                    
                       
                        unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                
                        
                        csv = convert_df(plot_df)           
                        st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_3')
                        #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
                        #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
                        st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                        st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)         
                        st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)    
                        #st.plotly_chart(fitted, theme="streamlit", use_container_width=True)                
    
            if baseline_peak_selection == "Static":
                baseline_recovery_frame_input = st.radio("Select one", ('Single Frame Value', 'Average Frame Value'), help='Baseline value based on a single frame, or on multiple frames')
                if baseline_recovery_frame_input ==   'Single Frame Value':                                     
                    baseline__frame_static = st.number_input("Baseline Intensity Frame number",  min_value=0, max_value=raw_img_ani_pg_2.shape[0]-1)
                    filtered_baseline_each = plot_df.query("Frame == @baseline__frame_static")
                    baseline_each = filtered_baseline_each['Smoothed Mean Intensity'].iloc[0]
                    #st.write(baseline__frame_static)
                    #baseline_each = plot_df.loc[plot_df['Frame'] == 5, 'Smoothed Mean Intensity'][0]
                    #st.write(baseline_each)
                    filtered_baseline_mean_each = plot_df.query("Frame == @baseline__frame_static")
                    baseline_mean_each = filtered_baseline_mean_each['Mean Intensity'].iloc[0]  
                    
                elif baseline_recovery_frame_input ==  'Average Frame Value':
                    baseline_smooth_x = st.slider("*_Choose frame number(s) to average their corresponding intensity values for baseline calculation_*", min_value = 0, max_value = raw_img_ani_pg_2.shape[0]-1, value = 10,  key='smooth')
                    #baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the 'Smoothed Mean Intensity' trace below. Baseline is calculated as: **_mode + n(S.D.)._**",  key='smooth')
                    
                    baseline_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Smoothed Mean Intensity'].mean()
                    baseline_mean_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Mean Intensity'].mean()                    
                    baseline__frame_static = int(sum(range(baseline_smooth_x + 1)) / (baseline_smooth_x + 1))
                    
                plot_df['Mean Intensity'] = plot_df['Mean Intensity']/baseline_mean_each
                
                #st.write(baseline_each)
                plot_df['Smoothed Mean Intensity'] = plot_df['Smoothed Mean Intensity']/baseline_each
                baseline_each = baseline_each/baseline_each
                #baseline_each = plot_df_smooth_mode + baseline_smooth_x*plot_df_smooth_sd   
                plot_df['delta_f/f_0'] = (plot_df['Smoothed Mean Intensity'] - baseline_each)/baseline_each 
                plot_df['Time'] = plot_df['Frame']/frame_rate
                area_df['Time']  =  area_df["Frame"]/frame_rate
                max_df_value = plot_df['Smoothed Mean Intensity'].max()
                peak_default_value = max(plot_df.loc[plot_df['Smoothed Mean Intensity'] == max_df_value, 'Frame'])
                
                peak__frame_static = st.number_input("Peak Intensity Frame number",  min_value=0, max_value=raw_img_ani_pg_2.shape[0]-1, value = int(peak_default_value)) 
                filtered_peak_each = plot_df.query("Frame == @peak__frame_static")
                max_df_value = filtered_peak_each['Smoothed Mean Intensity'].iloc[0]
                recovery_baseline__frame_static = st.number_input("Recovery Intensity Frame number",  min_value=0, max_value=raw_img_ani_pg_2.shape[0]-1, value = int(raw_img_ani_pg_2.shape[0])-1)                 
                
                nested_dict = {'Label':[], "Number of Events":[], "Rise time":[], "Decay time":[], "Duration":[], "Amplitude":[]}
                rise_df = plot_df[(plot_df['Frame'] >= baseline__frame_static) & (plot_df['Frame'] <= peak__frame_static)]
                decay_df = plot_df[(plot_df['Frame'] >= peak__frame_static) & (plot_df['Frame'] <= recovery_baseline__frame_static)]
                decay_df = decay_df[['Frame', 'Smoothed Mean Intensity']]
                decay_df.rename(columns={'Smoothed Mean Intensity': 'Decay intensity'}, inplace=True)
                rise_df = rise_df[['Frame', 'Smoothed Mean Intensity']]
                rise_df.rename(columns={'Smoothed Mean Intensity': 'Rise intensity'}, inplace=True)   
                
                amplitude_each = max_df_value - baseline_each
                signal_rise = (int(peak__frame_static)-int(baseline__frame_static))/frame_rate
                signal_decay = (int(recovery_baseline__frame_static)-int(peak__frame_static))/frame_rate
                signal_duration = (int(recovery_baseline__frame_static)-int(baseline__frame_static))/frame_rate
                nested_dict["Label"].append(i)
                nested_dict["Number of Events"].append(1)
                nested_dict["Rise time"].append(signal_rise)
                nested_dict["Decay time"].append(signal_decay)
                nested_dict["Duration"].append(signal_duration)
                nested_dict["Amplitude"].append(amplitude_each)
                
                st.write("**_Table with normalized intensity values_**")
                st.dataframe(plot_df, 1000,200)
                
                a_est_rise = rise_df['Rise intensity'].iloc[-1]
                b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                a_est = decay_df['Decay intensity'].iloc[0]
                b_est = find_b_est_decay(np.array(decay_df['Frame']), np.array(decay_df['Decay intensity'])) 
                               
                try:
                    popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                    
                except (TypeError, RuntimeError) as e:
                    error_message = str(e)
                    if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                        # Handle the error and continue to the next iteration
                        pass
                    #st.write("here")
                    # Replace the error with a warning message
                    else:
                        warning_message = "Fitting cannot be performed"
                        warnings.warn(warning_message, category=UserWarning)
                        popt_decay, pcov_decay = None, None
                else: 
                    popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                    decay_curve_exp = mono_exp_decay(decay_df['Frame'], *popt_decay)
                    
                try:
                    popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                    
                except (TypeError, RuntimeError) as e:
                    error_message = str(e)
                    if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                        pass
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
                
                unsmoothed_fig = px.line(
                                        plot_df,
                                        x="Time",
                                        y="Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                smoothed_fig = px.line(
                                        plot_df,
                                        x="Time",
                                        y="Smoothed Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                unsmoothed_area_fig =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                                   
                csv = convert_df(plot_df)           
                st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_2')                
                if popt_decay is not None and pcov_decay is not None:
                    smoothed_fig.add_trace(go.Scatter(x = decay_df['Frame']/frame_rate, y = decay_curve_exp, mode="markers", name='Decay Fit'))
                    #smoothed_figure.add_trace(go.Scatter(x = rise_df['Frame'], y = rise_curve_exp, mode="markers", name='Rise Fit'))
                    #smoothed_figure.add_trace(go.Scatter(x = plot_df['Frame'], y = photobleach_interpol, mode="lines", name='Fitted and Interpolated',fillcolor='green'))
                smoothed_fig.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_each, baseline_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                st.plotly_chart(unsmoothed_fig, theme="streamlit", use_container_width=True)
                st.plotly_chart(smoothed_fig, theme="streamlit", use_container_width=True)         
                st.plotly_chart(unsmoothed_area_fig, theme="streamlit", use_container_width=True)                     
                
                
            if st.button("Obtain the parameters for selected label",on_click=callback_sing) or st.session_state.button_clicked_sing_para:
                nested_dict = (pd.DataFrame.from_dict(nested_dict)) 
                if nested_dict.empty:
                    st.write("No parameter information for the selected label can be found based on the trace")
                else:
                    nested_dict = nested_dict[nested_dict['Amplitude']==(nested_dict['Amplitude']).max()]                          
                    #nested_dict_new = nested_dict[nested_dict['Amplitude']>0.1*baseline_each]
                    nested_dict_new = nested_dict
                    #st.write(nested_dict['Amplitude'])
                    if nested_dict_new.empty:
                        st.write("No parameter information for the selected label can be found based on the trace")
                    else:
                        st.subheader("**_Parameters for selected label across all frames_**")
                        col_1, col_2 = st.columns(2)
                        with col_1:
                            
                            nested_dict_new["Number of Events"] = nested_dict_new.shape[0]                        
                            st.write(nested_dict_new)
                            individual_csv = convert_df(nested_dict_new)           
                            st.download_button("Press to Download", individual_csv, 'individual_para_data.csv', "text/csv", key='individual_download-csv')
                            
                        with col_2:
                            average_rise_time = np.round(nested_dict_new['Rise time'].mean(),4)
                            st.write(f"The average rise time based on the selected label across all frames is {average_rise_time} s")
                            if popt_rise is not None:
                                rise_rate = np.round(popt_rise[1],4)
                                st.write(f"The average rise rate based on the selected label across all frames is {rise_rate} per s")
                            average_decay_time = np.round(nested_dict_new['Decay time'].mean(),4)
                            st.write(f"The average decay time based on the selected label across all frames is {average_decay_time} s")
                            if popt_decay is not None:
                                decay_rate = np.round(popt_decay[1],4)
                                st.write(f"The average decay rate based on the selected label across all frames is {decay_rate} per s")                        
                            average_duration = np.round(nested_dict_new['Duration'].mean(),4)
                            st.write(f"The average duration based on the selected label across all frames is {average_duration} s")
                            average_amplitude = np.round(nested_dict_new['Amplitude'].mean(),4)
                            st.write(f"The average amplitude based on the selected label across all frames is {average_amplitude}")

                        st.warning('Navigating to another page from the sidebar will remove all selections from the current page')  
                    
                    
        if bleach_corr_check == 'Bleaching correction':
            baseline_peak_selection = st.radio("Select one", ('Static', 'Dynamic'), help='Select "Static" to manually select single values for the baseline, peak and recovery frames; otherwise, select "Dynamic"')
            if baseline_peak_selection == "Dynamic":       
                st.subheader("**_Data for intensity of selected label_**")  
                smooth_plot_x = st.number_input("*_Moving Average Window_*", min_value=1, max_value=5, help = "Adjust to smooth the mean intensity trace below. Moving average of 1 would mean the original 'Mean Intensity' trace")
                plot_df = intensity(df_selected, raw_img_ani_pg_2, smooth_plot_x)                       
                missing_values = pd.isna(plot_df["Smoothed Mean Intensity"])
                plot_df.loc[missing_values, "Smoothed Mean Intensity"] = plot_df.loc[missing_values, "Mean Intensity"]  
                plot_df["Smoothed Mean Intensity"] = np.round(plot_df["Smoothed Mean Intensity"],3)                          
                baseline_smooth_x = st.slider("*_Choose frame number(s) to average their corresponding intensity values for baseline calculation_*", min_value = 0, max_value = raw_img_ani_pg_2.shape[0]-1, value = 10,  key='smooth')
                #baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the 'Smoothed Mean Intensity' trace below. Baseline is calculated as: **_mode + n(S.D.)._**",  key='smooth')
                
                baseline_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Smoothed Mean Intensity'].mean()
                plot_df['Smoothed Mean Intensity'] = plot_df['Smoothed Mean Intensity']/baseline_each
                baseline_each = baseline_each/baseline_each
                st.write(plot_df)

                fit_first_x = st.slider("*_Choose the number of first few frame number(s) to fit a mono-exponential decay_*", min_value = 1, max_value = int(np.floor(raw_img_ani_pg_2.shape[0]/2)), value = 30,  key='smooth_fit_first')
                fit_last_x = st.slider("*_Choose the number of last few frame number(s) to fit a mono-exponential decay_*", 1, int(np.floor(raw_img_ani_pg_2.shape[0]/2)), value = 30, key='smooth_fit_last')
                fit_last_x = raw_img_ani_pg_2.shape[0] - 1 - fit_last_x
                
                exp_df_1 = pd.DataFrame()
                exp_df_2 = pd.DataFrame()
                exp_df_1['Frames'] = plot_df[0:fit_first_x+1]['Frame']
                exp_df_1['Bleach intensity'] = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= fit_first_x), 'Smoothed Mean Intensity']
                exp_df_2['Frames'] = plot_df[fit_last_x:raw_img_ani_pg_2.shape[0]]['Frame']
                exp_df_2['Bleach intensity'] = plot_df.loc[(plot_df['Frame'] >= fit_last_x) & (plot_df['Frame'] <= raw_img_ani_pg_2.shape[0]-1), 'Smoothed Mean Intensity']
                
                exp_df = pd.concat([exp_df_1, exp_df_2], axis=0)
                
                popt_exp, pcov_exp = curve_fit(mono_exp_decay, exp_df['Frames'], exp_df['Bleach intensity'], p0 = [np.max(exp_df['Frames']), find_b_est_decay(np.array(exp_df['Frames']), np.array(exp_df['Bleach intensity']))])  #p0=[40,0.002])
                photobleach_curve_exp = mono_exp_decay(plot_df['Frame'], *popt_exp)      
                          
                fit_exp_df = pd.DataFrame()
                #fit_last_df = pd.DataFrame()
                fit_exp_df['Frame'] = plot_df['Frame']
                fit_exp_df['Photobleach Corr'] = photobleach_curve_exp

                plot_df_corr = pd.DataFrame()
                plot_df_corr_intensity = plot_df['Smoothed Mean Intensity']-photobleach_curve_exp
                plot_df_corr_intensity_min = min(plot_df_corr_intensity)
                #st.write(plot_df_corr_intensity_min)
                plot_df_corr['Frame'] = plot_df['Frame']
                plot_df_corr['Smoothed Mean Intensity'] = plot_df_corr_intensity + abs(plot_df_corr_intensity_min)
                plot_df_corr.loc[plot_df_corr['Smoothed Mean Intensity'] == 0, 'Smoothed Mean Intensity'] = plot_df_corr['Smoothed Mean Intensity'].replace(0, plot_df_corr['Smoothed Mean Intensity'][plot_df_corr['Smoothed Mean Intensity'] != 0].min())
                #plot_df_corr['Smoothed Mean Intensity'][plot_df_corr['Smoothed Mean Intensity']<0] = 0
                #plot_df_corr['Bright Pixel Area'] = plot_df['Bright Pixel Area']                
                baseline_corr_each = plot_df_corr.loc[(plot_df_corr['Frame'] >= 0) & (plot_df_corr['Frame'] <= baseline_smooth_x), 'Smoothed Mean Intensity'].mean()
                
                if baseline_corr_each == 0:
                    st.warning("Error: The chosen frame number pixel value (corrected) to determine the baseline is 0.")
                #plot_df_corr['Smoothed Mean Intensity'] = plot_df_corr['Smoothed Mean Intensity']/baseline_corr_each
                #baseline_corr_each = baseline_corr_each/baseline_corr_each
                
                plot_df_corr['delta_f/f_0'] = (plot_df_corr['Smoothed Mean Intensity']-baseline_corr_each)/baseline_corr_each
                plot_df_corr['Time'] = plot_df_corr['Frame']/frame_rate
                plot_df['Time'] = plot_df['Frame']/frame_rate
                area_df['Time']  =  area_df["Frame"]/frame_rate

                keyval = {}
                amp_keyval = {}
                prev_intensity = 0
                flag = 1
                
                for frame_key, intensity_val in enumerate(plot_df_corr['Smoothed Mean Intensity']):
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
                    st.write("Error! The trace doesn't cross the baseline. No parameters can be found")
                    
                    st.dataframe(plot_df, 1000,200)
            
                    unsmoothed_figure =  px.line(
                                        plot_df,
                                        x="Time",
                                        y="Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                                                
                    smoothed_figure =  px.line(
                                        plot_df,
                                        x="Time",
                                        y='Smoothed Mean Intensity'
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                    smoothed_figure.add_shape(type='line',
                                        x0=0,
                                        y0=baseline_corr_each,
                                        x1=(raw_img_ani_pg_2.shape[0])/frame_rate,
                                        y1=baseline_corr_each,
                                        line=dict(color='Red',),
                                        xref='x',
                                        yref='y') 
                    unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
    
                    phot_corr_figure =  px.line(
                                        plot_df_corr,
                                        x="Time",
                                        y="Smoothed Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )            
            
            
                    csv = convert_df(plot_df)           
                    st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_4')

                    smoothed_figure.add_trace(go.Scatter(x = plot_df['Frame']/frame_rate, y = photobleach_curve_exp, mode="lines", name='Fitted and Interpolated',fillcolor='green'))
                    #smoothed_figure.add_trace(go.Scatter(x=[0, raw_img_ani_pg_2.shape[0]], y=[baseline_each, baseline_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                    phot_corr_figure.add_trace(go.Scatter(x = plot_df_corr['Frame']/frame_rate, y = plot_df_corr['Smoothed Mean Intensity'], mode="lines", name='Corrected Intensity'))                 
                    
                    st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                    st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)   
                    st.write('*_The normalized Photobleaching-corrected data_*')
                    st.dataframe(plot_df_corr, 1000,200)
                    phot_corr_figure.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_corr_each, baseline_corr_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                    st.plotly_chart(phot_corr_figure, theme="streamlit", use_container_width=True)                  
                    st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)  
                  
                else:
                    first_key = frame_key
                    first_intensity = keyval[frame_key]
                    
                    prev_intensity = keyval[frame_key]
                    for frame_key_2, intensity_val_2 in enumerate(plot_df_corr['Smoothed Mean Intensity']):                               
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
                                        amp_key_vals = plot_df_corr[plot_df_corr['Smoothed Mean Intensity']==prev_intensity]['Frame']
                                        amp_key_vals = amp_key_vals[amp_key_vals>=first_key].iloc[0]
                                        amp_key = str(amp_key_vals)
                                        amplitude = prev_intensity - baseline_corr_each
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
                            signal_rise = (int(peak_frame)-int(signal_start_frame))/frame_rate
                            signal_decay = (int(signal_decay_frame)-int(peak_frame))/frame_rate
                            signal_duration = (int(signal_decay_frame)-int(signal_start_frame))/frame_rate
                            nested_dict["Label"].append(i)
                            nested_dict["Number of Events"].append(event_num)
                            nested_dict["Rise time"].append(signal_rise)
                            nested_dict["Decay time"].append(signal_decay)
                            nested_dict["Duration"].append(signal_duration)
                            nested_dict["Amplitude"].append(amplitude_each)
                    
                    
                    st.write('*_The original intensity data_*')
                    st.dataframe(plot_df, 1000,200)
            
                    unsmoothed_figure =  px.line(
                                        plot_df,
                                        x="Time",
                                        y="Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                                                
                    smoothed_figure =  px.line(
                                        plot_df,
                                        x="Time",
                                        y='Smoothed Mean Intensity'
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
        
                    phot_corr_figure =  px.line(
                                        plot_df_corr,
                                        x="Time",
                                        y="Smoothed Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )                  
 
                    smoothed_figure.add_trace(go.Scatter(x = plot_df['Frame']/frame_rate, y = photobleach_curve_exp, mode="lines", name='Fitted and Interpolated',fillcolor='green'))
                    smoothed_figure.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_each, baseline_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                    phot_corr_figure.add_trace(go.Scatter(x = plot_df_corr['Frame']/frame_rate, y = plot_df_corr['Smoothed Mean Intensity'], mode="lines", name='Corrected Intensity', line=dict(color='Blue', width=1))) 
                    
                    unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
            
                    
                    csv = convert_df(plot_df)           
                    st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_5')

                    st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                    st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)  
                    st.write('*_The normalized Photobleaching-corrected data_*')
                    st.dataframe(plot_df_corr, 1000,200)
    
                    max_df_value = plot_df_corr['Smoothed Mean Intensity'].max()
                    #st.write(plot_df.dtypes)
                    #####test by setting a some equal high values#########plot_df.loc[plot_df['Frame'] == 39, 'Smoothed Mean Intensity'] = max_df_value ##plot_df.loc[plot_df['Frame'] == 69, 'Smoothed Mean Intensity'] = baseline_each
                    count_max = plot_df_corr['Smoothed Mean Intensity'].eq(max_df_value).sum()
                    #st.write(count_max)
                    max_frame = plot_df_corr.loc[plot_df_corr['Smoothed Mean Intensity'] == max_df_value, 'Frame']
                    decay_df = pd.DataFrame()
                    rise_df = pd.DataFrame()
                    if ((plot_df_corr.loc[plot_df_corr["Frame"].gt(max(max_frame)), "Smoothed Mean Intensity"]).gt(baseline_corr_each)).all():
     
                        if count_max == 1:
                            rise_df['Rise intensity'] = plot_df_corr.loc[(plot_df_corr['Smoothed Mean Intensity'] <= max_df_value) & (plot_df_corr['Smoothed Mean Intensity'] >= baseline_corr_each) & (plot_df_corr['Frame'] <= max(max_frame)) , 'Smoothed Mean Intensity']
                            rise_df['Frame'] = rise_df.index
                            rise_df = rise_df[rise_df.columns[::-1]]
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
                            
                            #st.write(missing_value_df)
                            
                        if count_max > 1: 
                                                    
                            rise_df['Rise intensity'] = plot_df_corr.loc[(plot_df_corr['Smoothed Mean Intensity'] <= max_df_value) & (plot_df_corr['Smoothed Mean Intensity'] >= baseline_corr_each) & (plot_df_corr['Frame'] <= min(max_frame)) , 'Smoothed Mean Intensity']
                            first_index = rise_df.loc[rise_df['Rise intensity'] == max_df_value].index[-1]                    
                            #rise_df.loc[first_index, 'Rise intensity'] *= 1.01                        
                            rise_df['Frame'] = rise_df.index
                            rise_df = rise_df[rise_df.columns[::-1]]
                            #st.write(decay_df)
                            
                            missing_value_rise_df = (rise_df.loc[rise_df['Frame'].diff() > 1, 'Frame'].max())-1
                            #st.write(np.any(rise_df['Rise intensity']) != baseline_each)
                            # if ~decay_df['Decay intensity'].isin([baseline_each]).any():
                            #     if missing_value_df is not None:
                            #         new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_each}
                            #         decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_each]).any():
                            #     if missing_value_rise_df is not None:
                            #         new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_each}
                            #         rise_df.loc[missing_value_rise_df] = new_row_rise
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_rise_df)
                                
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
                                       
                        a_est_rise = rise_df['Rise intensity'].iloc[-1]
                        b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                        #bounds = ([0, 0], [100, 100])
                        
                        popt_decay, pcov_decay = None, None                        
                        
                        try:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                pass
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
                            
                        signal_rise = (max(max_frame) - rise_df['Frame'].iloc[0])/frame_rate
                        amplitude_each = max_df_value - baseline_corr_each
                        nested_dict["Label"].append(i)
                        nested_dict["Number of Events"].append(None)
                        nested_dict["Rise time"].append(signal_rise)
                        nested_dict["Decay time"].append(None)
                        nested_dict["Duration"].append(None)
                        nested_dict["Amplitude"].append(amplitude_each)                    
                        unsmoothed_figure =  px.line(
                                            plot_df,
                                            x="Time",
                                            y="Mean Intensity"
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )
                                                     
                        smoothed_figure =  px.line(
                                            plot_df,
                                            x="Time",
                                            y='Smoothed Mean Intensity'
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )
            
 
                        phot_corr_figure =  px.line(
                                            plot_df_corr,
                                            x="Time",
                                            y="Smoothed Mean Intensity"
                                            #color="sepal_length",
                                            #color=plot_df['Mean Intensity'],
                                        )
                        
                        smoothed_figure.add_trace(go.Scatter(x = plot_df['Frame']/frame_rate, y = photobleach_curve_exp, mode="lines", name='Fitted and Interpolated',fillcolor='green'))
                        smoothed_figure.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_each, baseline_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                        phot_corr_figure.add_trace(go.Scatter(x = plot_df_corr['Frame']/frame_rate, y = plot_df_corr['Smoothed Mean Intensity'], mode="lines", name='Corrected Intensity', line=dict(color='Blue', width=1))) 

                        unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )
                
                        
                        csv = convert_df(plot_df)           
                        st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_6')
                        #st.plotly_chart(figure, theme="streamlit", use_container_width=True)
                        #st.plotly_chart(figure_2, theme="streamlit", use_container_width=True)
                        
                        st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                        st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)         
                        st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)  
                    else:
                        if count_max == 1:
                            rise_df['Rise intensity'] = plot_df_corr.loc[(plot_df_corr['Smoothed Mean Intensity'] <= max_df_value) & (plot_df_corr['Smoothed Mean Intensity'] >= baseline_corr_each) & (plot_df_corr['Frame'] <= max(max_frame)) , 'Smoothed Mean Intensity']
                            decay_df['Decay intensity'] = plot_df_corr.loc[(plot_df_corr['Smoothed Mean Intensity'] <= max_df_value) & (plot_df_corr['Smoothed Mean Intensity'] >= baseline_corr_each) & (plot_df_corr['Frame'] >= max(max_frame)) , 'Smoothed Mean Intensity']
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
                            # if ~decay_df['Decay intensity'].isin([baseline_corr_each]).any():
                                
                            #     if missing_value_df is not None:
                            #         new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_corr_each}
                            #         decay_df.loc[len(decay_df)] = new_row_decay
                                    
                            # if ~rise_df['Rise intensity'].isin([baseline_corr_each]).any():
                            #     if missing_value_rise_df is not None:
                            #         new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_corr_each}
                            #         rise_df.loc[missing_value_rise_df] = new_row_rise
                            
                            #decay_df.loc[decay_df['Frame'] == missing_value_df, 'Decay intensity'] == baseline_each
                            #st.write(missing_value_rise_df)
                            if not pd.isna(missing_value_df):  #there is a missing value
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
                            decay_df['Decay intensity'] = plot_df_corr.loc[(plot_df_corr['Smoothed Mean Intensity'] <= max_df_value) & (plot_df_corr['Smoothed Mean Intensity'] >= baseline_corr_each) & (plot_df_corr['Frame'] >= max(max_frame)) , 'Smoothed Mean Intensity']
                            last_index = decay_df.loc[decay_df['Decay intensity'] == max_df_value].index[-1]
                            rise_df['Rise intensity'] = plot_df_corr.loc[(plot_df_corr['Smoothed Mean Intensity'] <= max_df_value) & (plot_df_corr['Smoothed Mean Intensity'] >= baseline_corr_each) & (plot_df_corr['Frame'] <= min(max_frame)) , 'Smoothed Mean Intensity']
                            first_index = rise_df.loc[rise_df['Rise intensity'] == max_df_value].index[0]
                            #decay_df.loc[last_index, 'Decay intensity'] *= 1.01
                            #rise_df.loc[first_index, 'Rise intensity'] *= 1.01
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
                            #     if missing_value_df is not None:
                            #         new_row_decay = {'Frame':  missing_value_df, 'Decay intensity': baseline_corr_each}
                            #         decay_df.loc[len(decay_df)] = new_row_decay
                            # if ~rise_df['Rise intensity'].isin([baseline_corr_each]).any():
                            #     if missing_value_rise_df is not None:
                            #         new_row_rise = {'Frame':  missing_value_rise_df, 'Rise intensity': baseline_corr_each}
                            #         rise_df.loc[missing_value_rise_df] = new_row_rise
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
                        
                        try:
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                # Handle the error and continue to the next iteration
                                pass
                            #st.write("here")
                            # Replace the error with a warning message
                            else:
                                warning_message = "Fitting cannot be performed"
                                warnings.warn(warning_message, category=UserWarning)
                                popt_decay, pcov_decay = None, None
                        else: 
                            popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                            decay_curve_exp = mono_exp_decay(decay_df['Frame'], *popt_decay)
                        try:
                            popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                            
                        except (TypeError, RuntimeError) as e:
                            error_message = str(e)
                            if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                                pass
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
       
                        phot_corr_figure.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_corr_each, baseline_corr_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                           
                        if popt_decay is not None and pcov_decay is not None:
                            phot_corr_figure.add_trace(go.Scatter(x = decay_df['Frame']/frame_rate, y = decay_curve_exp, mode="markers", name='Decay Fit'))
                        #phot_corr_figure.add_trace(go.Scatter(x = rise_df['Frame'], y = rise_curve_exp, mode="markers", name='Rise Fit'))                
                        st.plotly_chart(phot_corr_figure, theme="streamlit", use_container_width=True)  
                        st.plotly_chart(unsmoothed_area_figure, theme="streamlit", use_container_width=True)    
                    #st.plotly_chart(fitted, theme="streamlit", use_container_width=True)                

            if baseline_peak_selection == "Static":
                st.subheader("**_Data for intensity of selected label_**")  
                smooth_plot_x = st.number_input("*_Moving Average Window_*", min_value=1, max_value=5, help = "Adjust to smooth the mean intensity trace below. Moving average of 1 would mean the original 'Mean Intensity' trace")
                plot_df = intensity(df_selected, raw_img_ani_pg_2, smooth_plot_x)
                #st.write(plot_df)                                                      
                #smoothed_plot_df = plot_df['Smoothed Mean Intensity']                           
                missing_values = pd.isna(plot_df["Smoothed Mean Intensity"])
                plot_df.loc[missing_values, "Smoothed Mean Intensity"] = plot_df.loc[missing_values, "Mean Intensity"]  
                plot_df["Smoothed Mean Intensity"] = np.round(plot_df["Smoothed Mean Intensity"],3)                            

                baseline_recovery_frame_input = st.radio("Select one", ('Single Frame Value', 'Average Frame Value'), help='Baseline value based on a single frame, or on multiple frames')
                if baseline_recovery_frame_input ==   'Single Frame Value':                                     
                    baseline__frame_static = st.number_input("Baseline Intensity Frame number",  min_value=0, max_value=raw_img_ani_pg_2.shape[0]-1)
                    filtered_baseline_each = plot_df.query("Frame == @baseline__frame_static")
                    baseline_each = filtered_baseline_each['Smoothed Mean Intensity'].iloc[0]
                    
                    #baseline_corr_each = plot_df_corr.loc[plot_df_corr['Frame'] == baseline__frame_static, 'Smoothed Mean Intensity'][0]                              
                    plot_df['Smoothed Mean Intensity'] = plot_df['Smoothed Mean Intensity']/baseline_each
                    baseline_each = baseline_each/baseline_each
                    #st.write(baseline_corr_each)
                    plot_df['delta_f/f_0'] = (plot_df['Smoothed Mean Intensity']-baseline_each)/baseline_each
                    plot_df['Time'] = plot_df['Frame']/frame_rate
                    area_df['Time'] = area_df['Frame']/frame_rate
                    
                    fit_first_x = st.slider("*_Choose the number of first few frame number(s) to fit a mono-exponential decay_*", min_value = 1, max_value = int(np.floor(raw_img_ani_pg_2.shape[0]/2)), value = 30,  key='smooth_fit_first')
                    fit_last_x = st.slider("*_Choose the number of last few frame number(s) to fit a mono-exponential decay_*", 1, int(np.floor(raw_img_ani_pg_2.shape[0]/2)), value = 30, key='smooth_fit_last')
                    fit_last_x = raw_img_ani_pg_2.shape[0] - 1 - fit_last_x
                    
                    exp_df_1 = pd.DataFrame()
                    exp_df_2 = pd.DataFrame()
                    exp_df_1['Frames'] = plot_df[0:fit_first_x+1]['Frame']
                    exp_df_1['Bleach intensity'] = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= fit_first_x), 'Smoothed Mean Intensity']
                    exp_df_2['Frames'] = plot_df[fit_last_x:raw_img_ani_pg_2.shape[0]]['Frame']
                    exp_df_2['Bleach intensity'] = plot_df.loc[(plot_df['Frame'] >= fit_last_x) & (plot_df['Frame'] <= raw_img_ani_pg_2.shape[0]-1), 'Smoothed Mean Intensity']
                    
                    exp_df = pd.concat([exp_df_1, exp_df_2], axis=0)
                    
                    popt_exp, pcov_exp = curve_fit(mono_exp_decay, exp_df['Frames'], exp_df['Bleach intensity'], p0 = [np.max(exp_df['Frames']), find_b_est_decay(np.array(exp_df['Frames']), np.array(exp_df['Bleach intensity']))])  #p0=[40,0.002])
                    photobleach_curve_exp = mono_exp_decay(plot_df['Frame'], *popt_exp)           

                    fit_exp_df = pd.DataFrame()
                    #fit_last_df = pd.DataFrame()
                    fit_exp_df['Frame'] = plot_df['Frame']
                    fit_exp_df['Photobleach Corr'] = photobleach_curve_exp
                    plot_df_corr = pd.DataFrame()
                    plot_df_corr_intensity = plot_df['Smoothed Mean Intensity']-photobleach_curve_exp
                    plot_df_corr_intensity_min = min(plot_df_corr_intensity)
                    #st.write(plot_df_corr_intensity)
                    plot_df_corr['Frame'] = plot_df['Frame']
                    plot_df_corr['Smoothed Mean Intensity'] = plot_df_corr_intensity + abs(plot_df_corr_intensity_min)
                    plot_df_corr.loc[plot_df_corr['Smoothed Mean Intensity'] == 0, 'Smoothed Mean Intensity'] = plot_df_corr['Smoothed Mean Intensity'].replace(0, plot_df_corr['Smoothed Mean Intensity'][plot_df_corr['Smoothed Mean Intensity'] != 0].min())
                    baseline_mean_each = filtered_baseline_each['Mean Intensity'].iloc[0]                      
                    filtered_baseline_corr_each = plot_df_corr.query("Frame == @baseline__frame_static")
                    baseline_corr_each = filtered_baseline_corr_each['Smoothed Mean Intensity'].iloc[0]
                    plot_df_corr['Time'] = plot_df_corr['Frame']/frame_rate
                    #st.write(baseline_corr_each)
                    
                elif baseline_recovery_frame_input ==  'Average Frame Value':
                    baseline_smooth_x = st.slider("*_Choose frame number(s) to average their corresponding intensity values for baseline calculation_*", min_value = 0, max_value = raw_img_ani_pg_2.shape[0]-1, value = 10,  key='smooth')
                    #baseline_smooth_x = st.slider("*_Choose 'n' in n(S.D.) for Smoothed Intensity trace_*", min_value = 0.0, max_value = 3.0, step = 0.1, format="%.1f", value = 1.0,help = "Slide to adjust the baseline on the 'Smoothed Mean Intensity' trace below. Baseline is calculated as: **_mode + n(S.D.)._**",  key='smooth')
                    baseline_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Smoothed Mean Intensity'].mean()
                    #baseline_corr_each = plot_df_corr.loc[plot_df_corr['Frame'] == baseline__frame_static, 'Smoothed Mean Intensity'][0]                              
                    plot_df['Smoothed Mean Intensity'] = plot_df['Smoothed Mean Intensity']/baseline_each
                    baseline_each = baseline_each/baseline_each
                    #st.write(baseline_corr_each)
                    plot_df['delta_f/f_0'] = (plot_df['Smoothed Mean Intensity']-baseline_each)/baseline_each
                    plot_df['Time'] = plot_df['Frame']/frame_rate
                    area_df['Time'] = area_df['Frame']/frame_rate
                    
                    fit_first_x = st.slider("*_Choose the number of first few frame number(s) to fit a mono-exponential decay_*", min_value = 1, max_value = int(np.floor(raw_img_ani_pg_2.shape[0]/2)), value = 30,  key='smooth_fit_first')
                    fit_last_x = st.slider("*_Choose the number of last few frame number(s) to fit a mono-exponential decay_*", 1, int(np.floor(raw_img_ani_pg_2.shape[0]/2)), value = 30, key='smooth_fit_last')
                    fit_last_x = raw_img_ani_pg_2.shape[0] - 1 - fit_last_x
                    
                    exp_df_1 = pd.DataFrame()
                    exp_df_2 = pd.DataFrame()
                    exp_df_1['Frames'] = plot_df[0:fit_first_x+1]['Frame']
                    exp_df_1['Bleach intensity'] = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= fit_first_x), 'Smoothed Mean Intensity']
                    exp_df_2['Frames'] = plot_df[fit_last_x:raw_img_ani_pg_2.shape[0]]['Frame']
                    exp_df_2['Bleach intensity'] = plot_df.loc[(plot_df['Frame'] >= fit_last_x) & (plot_df['Frame'] <= raw_img_ani_pg_2.shape[0]-1), 'Smoothed Mean Intensity']
                    
                    exp_df = pd.concat([exp_df_1, exp_df_2], axis=0)
                    
                    popt_exp, pcov_exp = curve_fit(mono_exp_decay, exp_df['Frames'], exp_df['Bleach intensity'], p0 = [np.max(exp_df['Frames']), find_b_est_decay(np.array(exp_df['Frames']), np.array(exp_df['Bleach intensity']))])  #p0=[40,0.002])
                    photobleach_curve_exp = mono_exp_decay(plot_df['Frame'], *popt_exp)           

                    fit_exp_df = pd.DataFrame()
                    #fit_last_df = pd.DataFrame()
                    fit_exp_df['Frame'] = plot_df['Frame']
                    fit_exp_df['Photobleach Corr'] = photobleach_curve_exp
                    plot_df_corr = pd.DataFrame()
                    plot_df_corr_intensity = plot_df['Smoothed Mean Intensity']-photobleach_curve_exp
                    plot_df_corr_intensity_min = min(plot_df_corr_intensity)
                    #st.write(plot_df_corr_intensity)
                    plot_df_corr['Frame'] = plot_df['Frame']
                    plot_df_corr['Smoothed Mean Intensity'] = plot_df_corr_intensity + abs(plot_df_corr_intensity_min)
                    plot_df_corr.loc[plot_df_corr['Smoothed Mean Intensity'] == 0, 'Smoothed Mean Intensity'] = plot_df_corr['Smoothed Mean Intensity'].replace(0, plot_df_corr['Smoothed Mean Intensity'][plot_df_corr['Smoothed Mean Intensity'] != 0].min())
                    plot_df_corr['Time'] = plot_df_corr['Frame']/frame_rate
                    baseline_corr_each = plot_df_corr.loc[(plot_df_corr['Frame'] >= 0) & (plot_df_corr['Frame'] <= baseline_smooth_x), 'Smoothed Mean Intensity'].mean()
                    baseline_mean_each = plot_df.loc[(plot_df['Frame'] >= 0) & (plot_df['Frame'] <= baseline_smooth_x), 'Mean Intensity'].mean()                   
                    baseline__frame_static = int(sum(range(baseline_smooth_x + 1)) / (baseline_smooth_x + 1))
           
                st.write('*_The original intensity data_*')
                st.dataframe(plot_df, 1000,200)
                
                unsmoothed_figure =  px.line(
                                plot_df,
                                x="Time",
                                y="Mean Intensity"
                                #color="sepal_length",
                                #color=plot_df['Mean Intensity'],
                                )

                smoothed_figure =  px.line(
                                plot_df,
                                x="Time",
                                y='Smoothed Mean Intensity'
                                #color="sepal_length",
                                #color=plot_df['Mean Intensity'],
                                )
                smoothed_figure.add_trace(go.Scatter(x = plot_df['Frame']/frame_rate, y = photobleach_curve_exp, mode="lines", name='Fitted and Interpolated',fillcolor='green'))
                smoothed_figure.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_each, baseline_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                unsmoothed_area_figure =  px.line(
                                        area_df,
                                        x="Time",
                                        y="Bright Pixel Area"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )         
                st.plotly_chart(unsmoothed_figure, theme="streamlit", use_container_width=True)
                st.plotly_chart(smoothed_figure, theme="streamlit", use_container_width=True)  
                
                st.write('*_The normalized Photobleaching-corrected data_*')
                st.dataframe(plot_df_corr, 1000,200)
                
                peak_default_value = max(plot_df_corr.loc[plot_df_corr['Smoothed Mean Intensity'] == (plot_df_corr['Smoothed Mean Intensity']).max(), "Frame"])               
                peak__frame_static = st.number_input("Peak Intensity Frame number",  min_value=0, max_value=raw_img_ani_pg_2.shape[0]-1, value = int(peak_default_value)) 
                filtered_peak_each = plot_df_corr.query("Frame == @peak__frame_static")
                max_df_value = filtered_peak_each['Smoothed Mean Intensity'].iloc[0]
                recovery_baseline__frame_static = st.number_input("Recovery Intensity Frame number",  min_value=0, max_value=raw_img_ani_pg_2.shape[0]-1, value = int(raw_img_ani_pg_2.shape[0])-1)            
            
                nested_dict = {'Label':[], "Number of Events":[], "Rise time":[], "Decay time":[], "Duration":[], "Amplitude":[]}
                rise_df = plot_df_corr[(plot_df_corr['Frame'] >= baseline__frame_static) & (plot_df_corr['Frame'] <= peak__frame_static)]
                decay_df = plot_df_corr[(plot_df_corr['Frame'] >= peak__frame_static) & (plot_df_corr['Frame'] <= recovery_baseline__frame_static)]
                decay_df = decay_df[['Frame', 'Smoothed Mean Intensity']]
                decay_df.rename(columns={'Smoothed Mean Intensity': 'Decay intensity'}, inplace=True)
                rise_df = rise_df[['Frame', 'Smoothed Mean Intensity']]
                rise_df.rename(columns={'Smoothed Mean Intensity': 'Rise intensity'}, inplace=True)   
                amplitude_each = max_df_value - baseline_corr_each
                signal_rise = (int(peak__frame_static)-int(baseline__frame_static))/frame_rate
                signal_decay = (int(recovery_baseline__frame_static)-int(peak__frame_static))/frame_rate
                signal_duration = (int(recovery_baseline__frame_static)-int(baseline__frame_static))/frame_rate
                nested_dict["Label"].append(i)
                nested_dict["Number of Events"].append(1)
                nested_dict["Rise time"].append(signal_rise)
                nested_dict["Decay time"].append(signal_decay)
                nested_dict["Duration"].append(signal_duration)
                nested_dict["Amplitude"].append(amplitude_each)
                #st.write(decay_df)
                #st.write(rise_df)
                a_est_rise = rise_df['Rise intensity'].iloc[-1]
                b_est_rise = find_b_est_rise(np.array(rise_df['Frame']), np.array(rise_df['Rise intensity']))
                a_est = decay_df['Decay intensity'].iloc[0]
                b_est = find_b_est_decay(np.array(decay_df['Frame']), np.array(decay_df['Decay intensity'])) 
                   
                try:
                    popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                    
                except (TypeError, RuntimeError) as e:
                    error_message = str(e)
                    if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                        # Handle the error and continue to the next iteration
                        pass
                    #st.write("here")
                    # Replace the error with a warning message
                    else:
                        warning_message = "Fitting cannot be performed"
                        warnings.warn(warning_message, category=UserWarning)
                        popt_decay, pcov_decay = None, None
                else: 
                    popt_decay, pcov_decay = curve_fit(mono_exp_decay, decay_df['Frame'], decay_df['Decay intensity'], p0=[a_est,b_est])
                    decay_curve_exp = mono_exp_decay(decay_df['Frame'], *popt_decay)
                try:
                    popt_rise, pcov_rise = curve_fit(mono_exp_rise, rise_df['Frame'], rise_df['Rise intensity'], p0=[a_est_rise,b_est_rise])
                    
                except (TypeError, RuntimeError) as e:
                    error_message = str(e)
                    if error_message == "Optimal parameters not found: Number of calls to function has reached maxfev = 600":
                        pass
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
                                 
                smoothed_fig = px.line(
                                        plot_df_corr,
                                        x="Time",
                                        y="Smoothed Mean Intensity"
                                        #color="sepal_length",
                                        #color=plot_df['Mean Intensity'],
                                    )            
                csv = convert_df(plot_df_corr)           
                st.download_button("Press to Download", csv, 'intensity_data.csv', "text/csv", key='download-csv_2_corr')            
                
                if popt_decay is not None and pcov_decay is not None:
                     smoothed_fig.add_trace(go.Scatter(x = decay_df['Frame']/frame_rate, y = decay_curve_exp, mode="markers", name='Decay Fit'))

                smoothed_fig.add_trace(go.Scatter(x=[0, (raw_img_ani_pg_2.shape[0])/frame_rate], y=[baseline_corr_each, baseline_corr_each], mode='lines', name='Baseline', line=dict(color='Green', width=2)))
                smoothed_fig.add_trace(go.Scatter(x = plot_df_corr['Frame']/frame_rate, y = plot_df_corr['Smoothed Mean Intensity'], mode="lines", name='Corrected Intensity', line=dict(color='Blue', width=1)))
                #st.plotly_chart(unsmoothed_fig, theme="streamlit", use_container_width=True)
                st.plotly_chart(smoothed_fig, theme="streamlit", use_container_width=True)         
                #st.plotly_chart(unsmoothed_area_fig, theme="streamlit", use_container_width=True)               
            

            if st.button("Obtain the parameters for selected label",on_click=callback_sing) or st.session_state.button_clicked_sing_para:
                nested_dict = (pd.DataFrame.from_dict(nested_dict)) 
                if nested_dict.empty:
                    st.write("No parameter information for the selected label can be found based on the trace")
                else:
            
                    nested_dict_new = nested_dict
                    #else:
                    st.subheader("**_Parameters for selected label across all frames_**")
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        nested_dict_new = nested_dict[(nested_dict['Amplitude']) == max((nested_dict['Amplitude']))]
                        
                        #st.write(nested_dict_new.shape[0])
                        nested_dict_new["Number of Events"] = nested_dict_new.shape[0]
                        st.write(nested_dict_new)
                        individual_csv = convert_df(nested_dict_new)           
                        st.download_button("Press to Download", individual_csv, 'individual_para_data.csv', "text/csv", key='individual_download-csv')
                        
                    with col_2:
                        average_rise_time = np.round(nested_dict_new['Rise time'].mean(),4)
                        st.write(f"The average rise time based on the selected labels across all frames is {average_rise_time} s")
                        if popt_rise is not None:
                            rise_rate = np.round(popt_rise[1],4)
                            st.write(f"The average rise rate based on the selected label across all frames is {rise_rate} per s")                        
                        average_decay_time = np.round(nested_dict_new['Decay time'].mean(),4)
                        st.write(f"The average decay time based on the selected labels across all frames is {average_decay_time} s")
                        if popt_decay is not None:
                            decay_rate = np.round(popt_decay[1],4)
                            st.write(f"The average decay rate based on the selected label across all frames is {decay_rate} per s")                           
                        average_duration = np.round(nested_dict_new['Duration'].mean(),4)
                        st.write(f"The average duration based on the selected labels across all frames is {average_duration} s")
                        average_amplitude = np.round(nested_dict_new['Amplitude'].mean(),4)
                        st.write(f"The average amplitude based on the selected labels across all frames is {average_amplitude}")
   
                    st.warning('Navigating to another page from the sidebar will remove all selections from the current page')
       

            
