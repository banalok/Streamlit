import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import imageio
import tifffile
import os
num_frames = 150
frame_height = 512
frame_width = 512
image_size = (512, 512)
initial_num_objects = 1
noise_mean = 50
noise_stddev = 20
#object_diameter_range = (20, 31)
mean_intensities = np.array([196])
#last_frame_objects = [211, 236]
shuffled_mean_intensities_int = np.array([83.398, 82.455, 83.735, 83.189, 82.67, 83.477, 83.557, 81.489,	82.83,	82.14,	82.47,	82.307,	82.913,	82.947,	83.019,	82.439,	82.405,	82.799,	81.189,	82.485,	81.371,	82.462,	82.015,	81.273,	81.261,	82.561,	81.659,	86.53,	88.773,	89.008,	89.845,	87.761,	91.527,	95.78,	97.299,	97.909,	99.47,	99.265,	99.621,	100.136, 100.239,	99.371,	99.447,	98.466,	98.792,	98.792,	98.375,	97.705,	98.894,	97.898,	98.527,	97.879,	97.216,	95.909,	97.072,	95.837,	97.136,	96.095,	95.803,	92.621,	95.03,	95.784,	94.822,	94.03,	94.28,	93.455,	94.117,	93.549,	93.436,	93.159,	93.64,	92.39,	92.341,	92.197,	92.746,	91.489,	92.14,	91.045,	91.405,	91.462,	91.913,	91.087,	91.557,	91.167,	90.439,	90.595,	90.826,	90.977,	90.197,	89.189,	89.841,	90.39,	89.068,	90.064,	88.39,	88.875,	89.61,	88.837,	88.322,	88.78,	88.379,	88.678,	89.246,	88.898,	88.36,	88.087,	88.898,	88.405,	88.837,	88.83,	87.655,	87.633,	86.917,	86.833,	86.89,	87.299,	87.549,	86.773,	86.477,	86.432,	86.89,	87.568,	85.481,	87.284,	86.557,	86.985,	85.341,	86.527,	86.148,	86.341,	86.587,	86.386,	85.735,	86.58,	85.083,	85.379,	86,	86.186,	85.447,	85.292,	84.72,	86.337,	85.167,	86.167,	85.564,	85.727,	85.53,	86.102,	86.36,	86.458])+100
data = []
all_frames = []
total_noise = []
# Generate random positions for objects
object_positions = []
for _ in range(initial_num_objects):
    obj_x = np.random.randint(0, frame_width - 30) #object_diameter_range[1])
    obj_y = np.random.randint(0, frame_height - 30) #object_diameter_range[1])
    object_positions.append((obj_x, obj_y))

with imageio.get_writer('synthetic_cells.tif', format='TIFF', mode='I') as writer:
    for frame_number in range(num_frames):
        frame = np.random.randint(100, 200, size=(frame_height, frame_width), dtype=np.uint8)  # Less noisy background
        std = np.std(frame)
        #print(std)
        num_objects = initial_num_objects

        # if frame_number >= num_frames - 2:
        #     #num_objects += 1
        #     mean_intensities = np.append(mean_intensities, last_frame_objects[frame_number - (num_frames - 2)])

        shuffled_mean_intensities = np.array([shuffled_mean_intensities_int[frame_number]])#np.random.permutation(mean_intensities)

        frame_data = [frame_number]

        for obj_num in range(num_objects):
            obj_mean_intensity = shuffled_mean_intensities[obj_num]
            obj_diameter = 50 #np.random.randint(*object_diameter_range)

            obj_x, obj_y = object_positions[obj_num]  
            obj_img = Image.new('L', (obj_diameter, obj_diameter), 0)
            draw = ImageDraw.Draw(obj_img)
            
            fill_color = int(obj_mean_intensity)

            draw.ellipse((0, 0, obj_diameter, obj_diameter), fill=fill_color)            
            obj_array = np.array(obj_img)            
            frame[obj_y:obj_y + obj_diameter, obj_x:obj_x + obj_diameter][obj_array > 0] = obj_array[obj_array > 0]

            frame_data.append(obj_mean_intensity)

        data.append(frame_data)
        noise = np.random.normal(noise_mean, noise_stddev, size=image_size).astype(np.uint8)
        #noise = np.clip(noise, 50, 100)
        total_noise.append(np.var(noise))
        frame = frame + noise
        frame = frame.astype(np.int32)
        frame[frame>255] = 255
        frame = frame.astype(np.uint8)
        all_frames.append(frame)        
        writer.append_data(frame)
     
final_image_cov = np.std(all_frames)**2
#total_noise_std = np.std(total_noise)
total_noise_cov =  noise_stddev**2

snr = np.sqrt((final_image_cov/total_noise_cov)-1)
#psnr = 10*np.log10((255**2)/total_noise_cov) 
all_frames_array = np.array(all_frames)
# Save the NumPy array as a separate TIFF file
tifffile.imwrite('./single_ref.tif', all_frames_array)
#tifffile.imwrite(f'1_obj_synthetic_cells_array_{snr}.tif', all_frames_array)
os.remove('synthetic_cells.tif')
# num_columns = num_objects + 1 if frame_number >= num_frames - 2 else initial_num_objects + 1
# columns = ['Frame Number'] + [f'Object {i+1} Mean Intensity' for i in range(num_columns - 1)]
# df = pd.DataFrame(data, columns=columns)
# df.to_csv(f'1_object_mean_intensity_data_{snr}.csv', index=False)




#%%
######################################SECONDDDDDD############################################################

import numpy as np
import imageio
import tifffile
from skimage.draw import ellipse
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_ubyte, io
# Parameters
num_frames = 150
image_size = (512, 512)
object_size_range = (20, 30)
num_objects = 5
noise_mean = 45
noise_stddev = 5
background_gray_mean = 5
background_gray_stddev = 5

frame = list(range(num_frames))
# Create static base frame with identical objects
base_frame = np.zeros(image_size, dtype=np.uint8)
for _ in range(num_objects):
    diameter = np.random.randint(object_size_range[0], object_size_range[1] + 1)
    x = np.random.randint(0, image_size[1] - diameter)
    y = np.random.randint(0, image_size[0] // 2 - diameter)
    shape = np.random.choice(['circle', 'ellipse', 'oval'])
    obj = np.zeros((diameter, diameter), dtype=np.uint8)
    object_mask = np.zeros(obj.shape, dtype=bool)
    
    if shape == 'circle':
        rr, cc = np.indices((diameter, diameter))
        center = diameter // 2
        circle_mask = (rr - center)**2 + (cc - center)**2 <= (diameter // 2)**2
        intensity_range = (20, 100)
        intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
        #print(np.mean(intensities))
        #mean_intensity_base = np.mean(intensities)
        #mean_intensity_array.append(mean_intensity_base)
        obj[circle_mask] = intensities[circle_mask]
        #coordinates.append(np.argwhere(circle_mask))
        
        #object_m.append(circle_mask)
        
    elif shape == 'ellipse' or shape == 'oval':
        rr, cc = ellipse(diameter // 2, diameter // 2, diameter // 2, diameter // 3, shape=obj.shape)
        object_mask[rr, cc] = True
        intensity_range = (50, 200)
        intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
        #print(np.mean(intensities))
        #mean_intensity_base = np.mean(intensities)
        #mean_intensity_array.append(mean_intensity_base)        
        obj[object_mask] = intensities[rr, cc]
        #coordinates.append(np.argwhere(object_mask))
        
        #object_m.append(object_mask)
        #obj[rr, cc] = 255
        
    base_frame[y:y + diameter, x:x + diameter] = obj
noise_profile = np.random.normal(noise_mean, noise_stddev, size=image_size)  
clipped_noise = np.clip(noise_profile, 0, 255).astype(np.uint8)
noise_var = np.var(clipped_noise) 
# Generate a single consistent background noise
background_noise = np.random.normal(background_gray_mean, background_gray_stddev, size=image_size)
clipped_background_noise = np.clip(background_noise, 0, 255).astype(np.uint8)
background_noise_var = np.var(clipped_background_noise)
# Create frames with consistent background and noise profile
tif_stack = []
for _ in range(num_frames):
    noisy_frame = base_frame + np.random.normal(noise_mean, noise_stddev, size=image_size)
    noisy_frame += background_noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    tif_stack.append(noisy_frame)

#Add two shapes to frames 148 and 149
#=============================================================================
for i in range(148, 150):
    new_frame = np.copy(base_frame)
    diameter = np.random.randint(object_size_range[0], object_size_range[1] + 1)
    x = np.random.randint(0, image_size[1] - diameter)
    y = np.random.randint(0, image_size[0] // 2 - diameter)
    shape = np.random.choice(['circle', 'ellipse', 'oval'])
    obj = np.zeros((diameter, diameter), dtype=np.uint8)
    
    if shape == 'circle':
        rr, cc = np.indices((diameter, diameter))
        center = diameter // 2
        circle_mask = (rr - center)**2 + (cc - center)**2 <= (diameter // 2)**2
        intensity_range = (20, 100)
        intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
        #print(np.mean(intensities))
        #mean_intensity_rest = np.mean(intensities)
        #mean_intensity_array.append(mean_intensity_rest) 
        obj[circle_mask] = intensities[circle_mask]
        #coordinates.append(np.argwhere(circle_mask))
        #print(np.mean(obj[circle_mask]))
    elif shape == 'ellipse' or shape == 'oval':
        rr, cc = ellipse(diameter // 2, diameter // 2, diameter // 2, diameter // 3)
        intensity_range = (50, 200)
        intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
        #print(intensities)
        #mean_intensity_rest = np.mean(intensities)
        #mean_intensity_array.append(mean_intensity_rest)        
        obj[rr, cc] = intensities[rr, cc]
        #co = np.column_stack((rr, cc)) # Convert to a NumPy array
        #coordinates.append(co)  # Store the coordinates of the ellipse or oval
        
    new_frame[y:y + diameter, x:x + diameter] = obj
    n_frame = new_frame + np.random.normal(noise_mean, noise_stddev, size=image_size)
    n_frame += background_noise  # Add consistent background noise to the added object
    n_frame = np.clip(n_frame, 0, 255).astype(np.uint8)
    tif_stack[i] = n_frame

# Convert the list of frames to a numpy array
tiff_stack = np.array(tif_stack)
final_var = (np.std(tiff_stack))**2
total_noise_var = background_noise_var + noise_var
snr = np.sqrt((final_var/total_noise_var)-1)
# Save TIFF stack
tifffile.imsave('./multi_synthetic_neurons_1.tif', tiff_stack)


######################################THIRDDDDDDDDDDDDDDDDDDDDDD############################################################
#%%
import numpy as np
import imageio
import tifffile
from skimage.draw import ellipse
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_ubyte, io
# Parameters
num_frames = 150
image_size = (512, 512)
object_size_range = (20, 30)
num_objects = 1
noise_mean = 50
noise_stddev = 150
background_gray_mean = 150
background_gray_stddev = 10
total_intensity = []
total_noise = []

frame = list(range(num_frames))
# Create static base frame with identical objects
base_frame = np.zeros(image_size, dtype=np.uint8)

for _ in range(num_objects):
    diameter = np.random.randint(object_size_range[0], object_size_range[1] + 1)
    x = np.random.randint(0, image_size[1] - diameter)
    y = np.random.randint(0, image_size[0] // 2 - diameter)
    shape = np.random.choice(['circle', 'ellipse', 'oval'])
    obj = np.zeros((diameter, diameter), dtype=np.uint8)
    object_mask = np.zeros(obj.shape, dtype=bool)
    
    if shape == 'circle':
        rr, cc = np.indices((diameter, diameter))
        center = diameter // 2
        circle_mask = (rr - center)**2 + (cc - center)**2 <= (diameter // 2)**2
        intensity_range = (20, 100)
        intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
        #noise = np.random.normal(noise_mean, noise_stddev, size=(diameter, diameter))
        #noise = np.clip(noise, 50, 100)
        #final_obj_intensities = intensities + noise
        #final_obj_intensities[final_obj_intensities>255] = 255
        #final_obj_intensities = final_obj_intensities.astype(np.uint8)
        total_intensity.append(intensities)
        #total_noise.append(noise)
        #print(np.mean(intensities))
        #mean_intensity_base = np.mean(intensities)
        #mean_intensity_array.append(mean_intensity_base)
        obj[circle_mask] = intensities[circle_mask]
        #coordinates.append(np.argwhere(circle_mask))
        
        #object_m.append(circle_mask)
        
    elif shape == 'ellipse' or shape == 'oval':
        rr, cc = ellipse(diameter // 2, diameter // 2, diameter // 2, diameter // 3, shape=obj.shape)
        object_mask[rr, cc] = True
        intensity_range = (50, 200)
        intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
        #noise = np.random.normal(noise_mean, noise_stddev, size=(diameter, diameter))
        #noise = np.clip(noise, 50, 100)
        #final_obj_intensities = intensities + noise
        #final_obj_intensities[final_obj_intensities>255] = 255
        #final_obj_intensities = final_obj_intensities.astype(np.uint8)
        total_intensity.append(intensities)
        #total_noise.append(noise)
        #print(np.mean(intensities))
        #mean_intensity_base = np.mean(intensities)
        #mean_intensity_array.append(mean_intensity_base)        
        obj[object_mask] = intensities[rr, cc]
        #coordinates.append(np.argwhere(object_mask))
        
        #object_m.append(object_mask)
        #obj[rr, cc] = 255
        
    base_frame[y:y + diameter, x:x + diameter] = obj
    
# Generate a single consistent background noise
#background_noise = np.random.normal(background_gray_mean, background_gray_stddev, size=image_size)

# Create frames with consistent background and noise profile
tif_stack = []
for _ in range(num_frames):
    noise = np.random.normal(noise_mean, noise_stddev, size=image_size).astype(np.uint8)
    #noise = np.clip(noise, 50, 100)
    total_noise.append(noise)
    b_frame = base_frame + noise
    b_frame = b_frame.astype(np.int32)
    b_frame[b_frame>255] = 255
    b_frame = b_frame.astype(np.uint8)
#     noisy_frame = base_frame + np.random.normal(noise_mean, noise_stddev, size=image_size)
#     noisy_frame += background_noise
#     noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    tif_stack.append(b_frame)

#Add two shapes to frames 148 and 149
#=============================================================================
# for i in range(148, 150):
#     new_frame = np.copy(base_frame)
#     diameter = np.random.randint(object_size_range[0], object_size_range[1] + 1)
#     x = np.random.randint(0, image_size[1] - diameter)
#     y = np.random.randint(0, image_size[0] // 2 - diameter)
#     shape = np.random.choice(['circle', 'ellipse', 'oval'])
#     obj = np.zeros((diameter, diameter), dtype=np.uint8)
    
#     if shape == 'circle':
#         rr, cc = np.indices((diameter, diameter))
#         center = diameter // 2
#         circle_mask = (rr - center)**2 + (cc - center)**2 <= (diameter // 2)**2
#         intensity_range = (20, 100)
#         intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
#         #noise = np.random.normal(noise_mean, noise_stddev, size=(diameter, diameter))
#         #noise = np.clip(noise, 50, 100)
#         #final_obj_intensities = intensities + noise
#         #final_obj_intensities[final_obj_intensities>255] = 255
#         #final_obj_intensities = final_obj_intensities.astype(np.uint8)
#         total_intensity.append(intensities)
#         #total_noise.append(noise)
#         #print(np.mean(intensities))
#         #mean_intensity_rest = np.mean(intensities)
#         #mean_intensity_array.append(mean_intensity_rest) 
#         obj[circle_mask] = intensities[circle_mask]
#         #coordinates.append(np.argwhere(circle_mask))
#         #print(np.mean(obj[circle_mask]))
#     elif shape == 'ellipse' or shape == 'oval':
#         rr, cc = ellipse(diameter // 2, diameter // 2, diameter // 2, diameter // 3)
#         intensity_range = (50, 200)
#         intensities = np.random.randint(intensity_range[0], intensity_range[1] + 1, size=(diameter, diameter))
#         #noise = np.random.normal(noise_mean, noise_stddev, size=(diameter, diameter))
#         #noise = np.clip(noise, 50, 100)
#         #final_obj_intensities = intensities + noise
#         #final_obj_intensities[final_obj_intensities>255] = 255
#         #final_obj_intensities = final_obj_intensities.astype(np.uint8)
#         total_intensity.append(intensities)
#         #total_noise.append(noise)
#         #print(intensities)
#         #mean_intensity_rest = np.mean(intensities)
#         #mean_intensity_array.append(mean_intensity_rest)        
#         obj[rr, cc] = intensities[rr, cc]
#         #co = np.column_stack((rr, cc)) # Convert to a NumPy array
#         #coordinates.append(co)  # Store the coordinates of the ellipse or oval
        
#     new_frame[y:y + diameter, x:x + diameter] = obj
#     noise = (np.random.normal(noise_mean, noise_stddev, size=image_size)).astype(np.uint8)
#     #noise = np.clip(noise, 50, 100)
#     total_noise.append(noise)
#     n_frame = new_frame + noise
#     n_frame = n_frame.astype(np.int32)
#     n_frame[n_frame>255] = 255
#     n_frame = n_frame.astype(np.uint8)
#     #n_frame += background_noise  # Add consistent background noise to the added object
#     #n_frame = np.clip(n_frame, 0, 255).astype(np.uint8)
#     tif_stack[i] = n_frame
tiff_stack = np.array(tif_stack)
final_image_cov = np.std(tiff_stack)**2
total_noise_std = np.std(total_noise)
total_noise_cov = total_noise_std**2

snr = np.sqrt((final_image_cov/total_noise_cov))
#snr = 10*np.log10((255**2)/total_noise_cov)

print(snr)
# Convert the list of frames to a numpy array


# Save TIFF stack
#tifffile.imsave(f'1_obj_synthetic_neurons_{total_noise_std}_STD_{snr} db.tif', tiff_stack)
tifffile.imsave(f"./test_{snr}.png")

# #=========================================EXTRACT OBJECT MEAN INTENSITY IN EACH FRAME

#%%

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io

def calculate_psnr(original_img, processed_img, max_pixel_value):
    mse = np.mean((original_img - processed_img) ** 2)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

max_pixel_value = 255
#single_ref_img = io.imread("F:/Intensity measurement/Tool paper/snr test images/single_ref.tif")
multi_ref_img = io.imread("F:/Intensity measurement/Tool paper/multiple objects/multi_ref.tif")

pro_img = io.imread("F:/Intensity measurement/Tool paper/multiple objects/synthetic_neurons_74.89481263328304_STD_10.641768836200399 db.tif")

psnr = calculate_psnr(multi_ref_img, pro_img, max_pixel_value)
print(psnr)

#%%

import numpy as np
from PIL import Image

# Parameters
width, height = 512, 512  # Image dimensions
num_frames = 2  # Total number of frames
start_diameter = 150 # Starting diameter of the circle
max_diameter = 50  # Maximum diameter of the circle, now set to 20
background_noise_level = 10  # Noise level for the background
object_noise_level = 145  # Noise level for the object (circle)

# Function to generate a frame with a circle of given diameter
def generate_frame(diameter):
    # Create an image with random noise for the background
    background = np.random.randint(0, background_noise_level, (height, width), dtype=np.uint8)
    
    # Create an image with random noise for the object
    object_noise = np.random.randint(0, object_noise_level, (height, width), dtype=np.uint8)
    
    # Creating a circle in the center
    center_x, center_y = width // 3, height // 5
    radius = diameter // 2
    
    # Drawing the circle with noise on the background
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    # Apply noise within the circle area
    background[mask] = np.clip(background[mask] + object_noise[mask], 0, 255)
    
    return Image.fromarray(background)

# Generate diameters: Start small, grow to max, then shrink back
increase = np.linspace(start_diameter, max_diameter, num_frames // 2, endpoint=False)
decrease = np.linspace(max_diameter, start_diameter, num_frames - num_frames // 2)
diameters = np.concatenate([increase, decrease])

# Generate the frames based on calculated diameters
frames = [generate_frame(int(diameter)) for diameter in diameters]

# Save the frames to a TIFF file
frames[0].save(r'.\circle_sequence_512x512_with_noise_2.tif', save_all=True, append_images=frames[1:], compression="tiff_deflate")

print("Done!")


