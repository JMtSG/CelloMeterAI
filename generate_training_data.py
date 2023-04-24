## Using the images from the input folder, generate tiles and labels using the current model

import numpy as np
import cv2
import os
import glob
import random
import yaml

from image_processor import ImageProcessor
from CNN import CNN_Detector

# Load settings
with open('settings.yaml', 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
        COLOUR_REGULAR_CELL = tuple(settings['COLOURS']['REGULAR_CELL'][::-1])
        COLOUR_ABNORMAL_CELL = tuple(settings['COLOURS']['ABNORMAL_CELL'][::-1])
        COLOUR_BLOB_CLUSTER = tuple(settings['COLOURS']['BLOB_CLUSTER'][::-1])
    except yaml.YAMLError as e:
        print(e)


OUTPUT_DIR = '../data/autogenerated_data'
CONF_THRESHOLD = 0.4
TRAINING_TILES_SUBSAMPLE_PERC = 0.05     # only grab a random fraction of tiles from each image
STATUS_BAR_LEN = 50

def generate_training_data(input_img_fnames, image_processor, cnn_detector):
    # Load each image one at a time and process
    cell_count_summary = {}
    detection_id = 0    # unique serial ID for each detection across the whole image
    for file_idx,input_fname in enumerate(input_img_fnames):
        base_fname = os.path.basename(input_fname)
        print(f"\nProcessing image ({file_idx+1}/{len(input_img_fnames)}): \"{base_fname}\"")

        orig_img = cv2.imread(input_fname)
        orig_img_shape = orig_img.shape[:2]
        # Scale up the image
        img = cv2.resize(orig_img, (int(orig_img_shape[1] * settings['ZOOM_FACTOR']), 
                            int(orig_img_shape[0] * settings['ZOOM_FACTOR'])))
        # Generate tiles then take a random sample of them
        img_tiles = image_processor.generate_tiles(img,
                                                    settings['CNN_INPUT_DIM'],
                                                    settings['TILE_OVERLAP_FACTOR'])
        sampled_img_tiles = random.sample(img_tiles, int(len(img_tiles)*TRAINING_TILES_SUBSAMPLE_PERC))
        # Discard any tiles that aren't square
        sampled_img_tiles = [t for t in sampled_img_tiles if t[0].shape[0]==t[0].shape[1]]

        # Run them through the CNN\
        for tile_idx,(tile,(tile_x,tile_y), adjacent_tile_idxs) in enumerate(sampled_img_tiles):
            # Generate status bar and percentage indicator for displaying progress
            # In this loop, maximum progress caps out at 90%
            completion_pc = (tile_idx+1)/len(sampled_img_tiles)
            hash_symbol_cnt = round(completion_pc*STATUS_BAR_LEN)
            status_bar = '[' + '#'*hash_symbol_cnt + '-'*(STATUS_BAR_LEN-hash_symbol_cnt) + ']'
            print(f"{status_bar} %d%%"%round(completion_pc*100.0), end='\r')
            
            ### Run the CNN ###
            detections = cnn_detector.detect(tile)
            # Go through the detected objects
            label_data = []
            for detection in detections:
                x1 = int(detection[2])
                y1 = int(detection[3])
                x2 = int(detection[4])
                y2 = int(detection[5])
                # Get the centre coordinate of the bounding box, as a fraction of the image size
                cX = ((x1+x2)/2.0) / settings['CNN_INPUT_DIM']
                cY = ((y1+y2)/2.0) / settings['CNN_INPUT_DIM']
                # Get the width and height of the bounding box as a fraction of the image size
                bbW = (x2-x1) / settings['CNN_INPUT_DIM']
                bbH = (y2-y1) / settings['CNN_INPUT_DIM']
                label_data.append([detection[0], cX, cY, bbW, bbH])
            
            # Write labels to file, along with the tile image
            cv2.imwrite(f"{OUTPUT_DIR}/images/{os.path.splitext(os.path.basename(base_fname))[0]}_%03d.jpg"%tile_idx, tile)
            with open(f"{OUTPUT_DIR}/labels/{os.path.splitext(os.path.basename(base_fname))[0]}_%03d.txt"%tile_idx, 'w') as f:
                for label in label_data:
                    f.write(f"%d %.4f %.4f %.4f %.4f\n"%tuple(label))
        print('\n')



if __name__=='__main__':
    image_processor = ImageProcessor()  # For tiling & un-tiling images
    cnn_detector = CNN_Detector(f"./{settings['MODEL_NAME']}",
                                settings['CNN_INPUT_DIM'],
                                confidence_thresh=CONF_THRESHOLD)

    # Get filepaths of input images
    input_img_fnames = []
    for extension in settings['VALID_INPUT_FILETYPES']:
        input_img_fnames.extend(glob.glob(os.path.join(settings['INPUT_FOLDER'], extension)))
    input_img_fnames = sorted(list(set(input_img_fnames)))


    # Run object detection on all the images
    generate_training_data(input_img_fnames, image_processor, cnn_detector)