#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import os
# import six.moves.urllib as urllib
import sys
# import tensorflow as tf
import collections
import statistics
import math
import tarfile
import os.path
import csv
import cv2
import time


# In[9]:


# ZED imports
import pyzed.sl as sl

# ## Object detection imports
# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util


# In[10]:


def load_image_into_numpy_array(image):
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_depth_into_numpy_array(depth):
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)




width = 704
height = 416
confidence = 0.75



def get_coordinates_and_rbox(image_np, depth_np, box):
    
    research_distance_box = 20 #30 

    box = tuple(box.tolist())

    # Find object distance
    ymin, xmin, ymax, xmax = box
    x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
    y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
    x_vect = []
    y_vect = []
    z_vect = []
    
    ymin_n = int(ymin * height)
    xmin_n = int(xmin * width)
    ymax_n = int(ymax * height)
    xmax_n = int(xmax * width)

    min_y_r = max(int(ymin * height), int(y_center - research_distance_box))
    min_x_r = max(int(xmin * width), int(x_center - research_distance_box))
    max_y_r = min(int(ymax * height), int(y_center + research_distance_box))
    max_x_r = min(int(xmax * width), int(x_center + research_distance_box))

    if min_y_r < 0: min_y_r = 0
    if min_x_r < 0: min_x_r = 0
    if max_y_r > height: max_y_r = height
    if max_x_r > width: max_x_r = width

    #Refering to depth array with coordiates of image array
    for j_ in range(min_y_r, max_y_r):
        for i_ in range(min_x_r, max_x_r):
            z = depth_np[j_, i_, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth_np[j_, i_, 0])
                y_vect.append(depth_np[j_, i_, 1])
                z_vect.append(z)

    if len(x_vect) > 0:
        x = np.percentile(x_vect, 50)
        y = np.percentile(y_vect, 75) 
        z = np.percentile(z_vect, 25)

        #distance = math.sqrt(x * x + y * y + z * z)
    else:
        x = 0
        y = 0
        z = 0
        
    return [x, y, z, min_y_r, min_x_r, max_y_r, max_x_r]


# In[76]:


def export_frame_xyz(frame, cur_frame, output_folder):
    
    data_per_frame = []
    width = sl.Mat.get_width(frame)
    height = sl.Mat.get_height(frame)
    for col in range(width):
        for row in range(height):
            point3D = frame.get_value(col,row)
            x = point3D[1][0]
            y = point3D[1][1]
            z = point3D[1][2]

            data_per_frame.append([col, row, x, y, z])
    
    np_data_per_frame = np.array(data_per_frame)
    output_csv = "{}/frame_{}.csv".format(output_folder, cur_frame)
    np.savetxt(output_csv, np_data_per_frame, delimiter=',', header='col,row,x,y,z')
    


# In[81]:


def main(args):
    # args = ['python3', svo_filepath, max_frames, frames_offset, output_folder]
    # args = ['python', svo_filepath, output_folder, frame_list]

    
    svo_filepath = None
    if len(args) > 1:
        svo_filepath = args[1]
        print(svo_filepath)
    else:
        raise(BaseException("Please specify an input SVO file!"))
    
    output_folder = './output'
    if len(args) > 2:
        output_folder = args[2]

    frames_list = []
    if len(args) > 3:
        frames_list = args[3]
    
    export_file_preffix = ''
    if len(args) > 4:
        export_file_preffix = args[4]

    frames_list.sort()
    print('Frames: {}'.format(frames_list))


    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
        
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_filepath))
    # init_params.camera_resolution = sl.RESOLUTION.HD720
    # init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.svo_real_time_mode = False

    # Open camera / input
    result = zed.open(init_params) 
    if result != sl.ERROR_CODE.SUCCESS:
        raise(BaseException(f"Error opening the camera : {result}"))
        zed.close()
        return

    left_image = sl.Mat()
    # right_image = sl.Mat()
    depth_image = sl.Mat()
    point_cloud = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()

    # CSV file with same path / filename as input file
    f = open(os.path.splitext(svo_filepath)[0] + '.csv', 'wb')
    cur_frame = 1 # frames start from 1

    # Main loop to iterate frames in the video
    # --------------------------------------------------------------------------------------------
    
    while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        
        # skip frames until we reach the offset, if any
        # if frames_offset is not None:
        #     if cur_frame <= frames_offset:
        #         # skip frames until offset is reached
        #         cur_frame += 1
        #         continue

        if cur_frame < frame_list[0]:
            cur_frame += 1
            continue
        
        
        print("Current Frame {}".format(cur_frame))
        
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
        # zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
        
        output_left_img = "{}/{}{}_left.png".format(output_folder, export_file_preffix, cur_frame)
        left_image.write(output_left_img)
        
        # output_right_img = "{}/{}{}_right.png".format(output_folder, export_file_preffix, cur_frame)
        # right_image.write(output_right_img)
        
        output_depth_img = "{}/{}{}_depth.png".format(output_folder, export_file_preffix, cur_frame)
        depth_image.write(output_depth_img)
        # cv2.imwrite(output_depth_img, depth_image.get_data().astype(np.uint16))
           
        export_frame_xyz(point_cloud, cur_frame, output_folder)

        cur_frame += 1
        frame_list.pop(0)
        if len(frame_list) == 0:
            break

    f.close()
 
    # Close camera / input
    zed.close()
    return frame_list




# ------------------------- Setup Information: start -------------------------

svo_filepath= '../../svo/HD720_SN24793_11-36-37.svo'
export_file_preffix = 'Explorer_HD720_SN24793_frame'
output_folder = '../../output'
frame_list = [15,30,45,60,75,90,105,120,135,150,165,180]

# ------------------------- Setup Information: end ==-------------------------





current_args = ['python', svo_filepath, output_folder, frame_list, export_file_preffix]

# max_frames = '2'
# frames_offset = '1500'
# current_args = ['python', svo_filepath, max_frames, frames_offset, output_folder]


if __name__ == '__main__':
#     main(sys.argv)
    start_time = time.time()
    main(current_args)
    
    seconds = time.time() - start_time
    hours = seconds/3600
    print("running time: {} seconds, {} hours".format(seconds, hours))

