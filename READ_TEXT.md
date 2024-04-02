DL-SCAN, which effectively segments and analyzes cells in images from fluorescent microscopy (in TIFF format), is developed using Streamlit library (version 1.21.0) in Python 3.8.8. The user-friendly graphical user interface (GUI) displayed immediately after the tool is launched.

1. Uploading Files

A microscopy image stack in TIFF format can be uploaded by clicking “Browse” on the homepage of the program. The original frame of the uploaded file is displayed. This is followed by the option to incorporate background correction. In case the user wishes to discard the current file and upload a new one, a simple page refresh will let them do that.


2. Preprocessing options

Background Correction

When background correction is required, users can select “Background Correction” option. This action will display the first frame, allowing them to draw a rectangle on it. The average intensity of the selected pixels is then calculated and subtracted from the original image in each frame. The background corrected frames are then displayed. Users can input the frame number or click “+/-” to view the resulting background-corrected images.


Gaussian Blurring

The first preprocessing option provided in the tool is Gaussian Blurring. The Gaussian square matrix generated as a result of the selected kernel size convolves the original image, calculating the weighted average to replace its center pixel value. This weighted averaging smooths the image, as the pixels closer to the center of the kernel have higher weights as compared to those that are far away.


Median Blurring

Users have the option to use median filtering by providing the Median Blurring Kernel Size. This process generates a square matrix of the selected size that slides on top of the image, computing median of the window and replacing the center pixel value. This works for reducing noise in an image, preserving edges and sharp features of the objects. More specifically, salt-and-pepper noise can be well-reduced using this technique.


Brightness and Contrast

Image brightness and contrast can also be adjusted in the program as one of the preprocessing options. 


Contrast Limited Adaptive Histogram Equalization (CLAHE)

To prevent over-amplification of noise, Contrast Limited Histogram Equalization (CLAHE) is provided as one of the options in the program. Unlike traditional histogram equalization, CLAHE adapts the contrast enhancement locally within smaller regions of the image (8 X 8 grid). Adjusting the provided clip limit factor value changes the amount of contrast enhancement. Although the higher value allows for more enhancement, one must be cautious using this option, as it can also lead to a noisier result.

Users are encouraged to experiment with individual options, examining the outcome before applying all options simultaneously. This approach ensures effective image preprocessing.

Processed Frames

The preprocessed frames, as a result of any of the applied preprocessing options or the combination of them is displayed as “Processed Frames”. Users can input the frame number or click “+/-” to view the resulting processed images.

Collapsed Image

To account for cells appearing in subsequent frames, the entire stack of images is collapsed into a single image. In this process, pixels with higher values are retained during pixel-by-pixel comparison between frames. Therefore, it is crucial to upload a TIFF stack where cells are represented by higher pixel values and the background by lower pixel values.


3. Segmentation

The collapsed image is now ready for segmentation. Clicking “Segment and generate labels” outputs a segmented image and labeled image and is ready for analysis. 

At times, the pixel distribution in uploaded images can be uneven due to differences in the experimental setup, resulting in overexposure in some areas and underexposure in others. To address this, we offer a Rolling Ball Background Correction (RBBC) option, which helps mitigate this problem. A ball of a user-selected radius rolls across the image, fitting into its valleys and crevices. The minimum pixel value within the region covered by the ball is subtracted from the pixel value at the center of the ball, effectively smoothing out the background when unevenly illuminated. 

Additionally, users also have the choice to segment based on the first image rather than the collapsed image.
In case needed, users have an option to manually draw regions of interest (ROI) that get added to the list of cells to be analyzed.  Once segmented, a dye-positive image can be uploaded and overlaid on top of the segmented image for efficient cell selection as needed.

4. Single-cell Analysis

This page can be accessed by clicking “Single-cell Analysis” in the sidebar. In this section, users can perform analysis of individual cells. To access the content of the page, the program assumes that the Preprocessing and Segmentation steps have been already completed. After displaying the collapsed and labeled images for reference at the top, a table is presented, followed by the same table, but now interactive. The first table can be downloaded as a CSV file and has the following format. 
 
The interactive table allows users to select a single cell and perform analysis on it. As soon as the cell is selected, it is isolated and shown in the image highlighted by red color. This is followed by the option to input frame rate and the bleaching correction option with “No bleaching correction” option as the default. 

No bleaching correction

When this option is selected, users will get further options to
1.	Adjust the moving average window for trace smoothing, that ranges from 1 to 5 (where 1 would mean the original trace). 
2.	Select “Static” or “Dynamic” analysis to be performed.

Static Analysis lets users select a single baseline intensity frame number (“Single Frame Value”) or number of consecutive frames to average their intensity values for baseline intensity calculation (“Average Frame Value”), peak intensity frame number and recovery intensity frame number.
Dynamic Analysis asks users to select number of consecutive frames to average their intensity values for baseline intensity calculation, while the peak and recovery are automatically computed.
The normalized intensity table is then displayed, followed by the original, smoothed, and bright pixel area traces. When the signal is present (the trace rises and decays crossing the baseline), clicking “Obtain the parameters for selected label” computes and outputs various parameters linked to the selected cell.

Bleaching Correction

When this option is selected, users will get further options to
1.	Select “Static” or “Dynamic” analysis to be performed.
2.	Adjust the moving average window for trace smoothing, that ranges from 1 to 5 (where 1 would mean the original trace). 
3.	Choose the number of the first and last few frames to fit a mono-exponential curve to correct for photobleaching.

All the other processes and outputs are similar to the “No bleaching correction” option.


5. Multi-cell Analysis

In this section, users can collectively analyze multiple (all or fewer) cells by selecting multiple cells at once. As soon as the cells are selected, their corresponding traces are displayed, which can be isolated by double clicking their legends.  All the other options remain unchanged, just as they were for single-cell analysis previously. Clicking “Obtain the parameters for selected labels” displays the normalized traces and computes the parameters based on the selected options.

Finally, DL-SCAN generates and displays distribution plots for the computed parameters



