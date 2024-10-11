import streamlit as st
from streamlit_extras.switch_page_button import switch_page
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

from st_pages import show_pages_from_config

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
def load_single_image(images):
     img = io.imread(images)
     # re, img = cv2.imreadmulti(images, flags=cv2.IMREAD_UNCHANGED)
     # img = np.array(img)
     return img

@st.cache_data(max_entries=1, show_spinner=False, ttl = 2*60)
def stardist_seg(im,_model):
    img_labels, img_det = _model.predict_instances(normalize(im), prob_thresh=0.6)
    return img_labels

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

def area(df_sel_orig, df_sel, multi_tif_img):
    img_frames_list = list(range(0,multi_tif_img.shape[0]))
    selected_row = df_sel_orig[df_sel_orig['label'] ==df_sel['label'][0]]
    bright_pixel_cols = [col for col in df_sel_orig.columns if 'Bright_pixel' in col]
    bright_pixel_values = selected_row[bright_pixel_cols].values
    area_dataframe = pd.DataFrame(img_frames_list, columns = ['Frame'])
    area_dataframe['Bright Pixel Area'] = bright_pixel_values.T 
    return area_dataframe


def intensity(df_1, multi_tif_img, window):
    img_frames_list = list(range(0,multi_tif_img.shape[0]))
    img_frames = pd.DataFrame(img_frames_list, columns = ['Frame'])
    mean_intensity = []
    #p_count = []
    #change_in_F = []
    for frames_pro in range(0,multi_tif_img.shape[0]):
            #new_df = pd.DataFrame(frames_pro, df_pro[f'intensity_mean_{frames_pro}'].mean(),  columns = ['Frames', 'Mean Intensity'])
        mean_intensity.append(df_1[f'intensity_mean_{frames_pro}'].mean()) #[0]
        #p_count.append(df_1[f'Bright_pixel_area_{frames_pro}'].mean()) #[0]
            # new_df = pd.DataFrame.from_dict({'Frame': frames_pro, 'Mean Intensity': df_pro[f'intensity_mean_{frames_pro}'].mean()})
            # img_frames = pd.merge(img_frames, new_df, on = "Frame")
        #st.write(df_1[f'pixel_count_{frames_pro}'])
        #change_f = fluo_change(mean_intensity[frames_pro], baseline)
        #change_in_F.append(change_f)
        
    #change_F_df = pd.DataFrame(change_in_F, columns = ['delta_F/F'])
    smooth_F_df = pd.DataFrame(smooth_plot(mean_intensity, window), columns = ['Smoothed Mean Intensity'] ) #pd.DataFrame(smooth_df, columns = ['smoothed mean intensity'])
    mean_inten_df = pd.DataFrame(mean_intensity)
    #pixel_count_df = pd.DataFrame(p_count, columns = ['Bright Pixel Area'])
    new_d = pd.concat([img_frames, mean_inten_df, smooth_F_df],axis=1) #, pixel_count_df
    new_d.rename(columns = {0 : 'Mean Intensity'}, inplace=True)
    #new_d.rename(columns = {1 : 'Bright Pixel Number'}, inplace=True)
    return new_d 


def get_intensity_for_timepoint(intensity_image, label_layer):
    stats = measure.regionprops_table(label_layer, intensity_image=intensity_image, properties=['intensity_mean'])
    return stats['intensity_mean']

def get_centroid_for_timepoint(intensity_image, label_layer):
    stats = measure.regionprops_table(label_layer, intensity_image=intensity_image, properties=['centroid'])
    return stats

def get_max_intensity_for_timepoint(intensity_image, label_layer):
    stats = measure.regionprops_table(label_layer, intensity_image=intensity_image, properties=['intensity_max'])
    return stats['intensity_max']

def get_intensity(intensity_image_stack, labels_layer_stack):
    return [get_intensity_for_timepoint(intensity_image, label_layer) 
            for intensity_image, label_layer 
            in zip(intensity_image_stack, labels_layer_stack)]

def get_centroid(intensity_image_stack, labels_layer_stack):
    return [get_centroid_for_timepoint(intensity_image, label_layer) 
            for intensity_image, label_layer 
            in zip(intensity_image_stack, labels_layer_stack)]

def get_max_intensity(intensity_image_stack, labels_layer_stack):
    return [get_max_intensity_for_timepoint(intensity_image, label_layer) 
            for intensity_image, label_layer 
            in zip(intensity_image_stack, labels_layer_stack)]

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
        try:
            if y1 > 0 and y2 > 0 and x2 != x1:
                k = -math.log(y2/y1) / (x2-x1)
                decay_constants.append(k)
        except ValueError as e:
            decay_constants.append(0)
    
    # Calculate average decay rate
    avg_decay_rate = np.mean(decay_constants)
    
    return avg_decay_rate

def find_b_est_rise(x_data, y_data):
    
    rise_constants = []
    for i in range(len(x_data)-1):
        x1, y1 = x_data[i], y_data[i]
        x2, y2 = x_data[i+1], y_data[i+1]
        try:
            if y1 > 0 and y2 > 0 and x2 != x1:
                k = math.log(y2/y1) / (x2-x1)
                rise_constants.append(k)
        except ValueError as e:
            rise_constants.append(0)
    
    # Calculate average decay rate
    avg_rise_rate = np.mean(rise_constants)
    
    return avg_rise_rate

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