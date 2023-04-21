import numpy as np
import cv2
import os
import glob
import copy
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter

from settings import *
from image_processor import ImageProcessor
from CNN import CNN_Detector

import warnings
warnings.filterwarnings("ignore")

STATUS_BAR_LEN = 50

def get_IDs_to_discard(detections_tile_A, detections_tile_B):
    # Return a list of detection IDs from tile A that overlap with detections from tile B
    # Each detections_tile_* array is [[tile index, class enum, confidence score, bounding box coordinates]]
    detection_IDs_to_discard_A = []
    for detection_A in detections_tile_A:
        for detection_B in detections_tile_B:
            percent_overlap = calc_rect_overlap(detection_A[4:8], detection_B[4:8])
            if percent_overlap>MAX_OVERLAP_PERCENTAGE:
                detection_IDs_to_discard_A.append(detection_A[1])
    return detection_IDs_to_discard_A

def calc_rect_overlap(coords_A, coords_B):
    # Calculate the percentage overlap between two rectangles
    # Each rectangle is defined by [x1, y1, x2, y2], coordinates of two diagonal points.
    x_left = max(coords_A[0], coords_B[0])
    y_top = max(coords_A[1], coords_B[1])
    x_right = min(coords_A[2], coords_B[2])
    y_bottom = min(coords_A[3], coords_B[3])
    
    # Calculate the width and height of the intersection rectangle
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    
    # Calculate the area of the intersection rectangle
    intersection_area = intersection_width * intersection_height
    
    # Calculate the areas of both bounding boxes
    bb1_area = (coords_B[2] - coords_B[0]) * (coords_B[3] - coords_B[1])
    bb2_area = (coords_B[2] - coords_B[0]) * (coords_B[3] - coords_B[1])
    
    # Calculate the percentage overlap (overlap_area / total area)
    # overlap_percent = (intersection_area / float(bb1_area + bb2_area - intersection_area))
    # Calculate the percentage overlap (100% if any bounding box is enveloped by another)
    overlap_percent = max(intersection_area/float(bb1_area),
                          intersection_area/float(bb2_area))
    
    return overlap_percent


def generate_report(input_data, output_fname):
    data = copy.deepcopy(input_data)

    # Create a new workbook and select the active worksheet
    wb = Workbook()
    ws = wb.active

    # Define the header row
    header = ['File name', 'Normal Cells', 'Abnormal Cells', 'Total Cells']

    # Set the background color and font style for the header row
    fill = PatternFill(start_color='C0C0C0', end_color='C0C0C0', fill_type='solid')
    font_reg = Font(name='Arial', bold=False)
    font_bold = Font(name='Arial', bold=True)

    # Write the header row to the worksheet
    for col_num, value in enumerate(header, 1):
        cell = ws.cell(row=1, column=col_num, value=value)
        cell.fill = fill
        cell.font = font_bold

    # Calculate the total number of cells in each image
    for key in data:
        data[key].append(sum(data[key][:2]))

    # Write the data to the worksheet
    for row_num, key in enumerate(data, 2):
        row_data = [key] + data[key]
        for col_num, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_num, column=col_num, value=value)
            cell.font = font_reg

    # Calculate the totals for each column
    normal_cells_total = sum([data[key][0] for key in data])
    abnormal_cells_total = sum([data[key][1] for key in data])
    total_cells_total = sum([data[key][2] for key in data])

    # Add a total row at the bottom of the worksheet
    total_row = ['Final Totals', normal_cells_total, abnormal_cells_total, total_cells_total]
    for col_num, value in enumerate(total_row, 1):
        cell = ws.cell(row=len(data)+2, column=col_num, value=value)
        cell.font = font_bold

    # Auto-adjust the column widths
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_length = 0
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        ws.column_dimensions[col_letter].width = adjusted_width

    # Save the workbook
    wb.save(output_fname)


if __name__=='__main__':
    image_processor = ImageProcessor()  # For tiling & un-tiling images
    cnn_detector = CNN_Detector('./model_weights.pt', CNN_INPUT_DIM, confidence_thresh=CONFIDENCE_THRESHOLD)

    # Get filepaths of input images
    input_img_fnames = []
    for extension in VALID_INPUT_FILETYPES:
        input_img_fnames.extend(glob.glob(os.path.join(INPUT_FOLDER, extension)))
    input_img_fnames = sorted(input_img_fnames)

    # Create/Clear the output directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    else:
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path) and '.jpg' in file_path:
                os.remove(file_path)
        
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,0,255), (255,255,0), (0,255,255)]

    # Load each image one at a time and process
    cell_count_summary = {}
    detection_id = 0    # unique serial ID for each detection across the whole image
    for file_idx,input_fname in enumerate(input_img_fnames):
        print(f"Processing image ({file_idx+1}/{len(input_img_fnames)}): \"{input_fname.split('/')[-1]}\"")
        try:
            img = cv2.imread(input_fname)
            # Scale up the image
            img = cv2.resize(img, (int(img.shape[1] * ZOOM_FACTOR), 
                                int(img.shape[0] * ZOOM_FACTOR)))
            img_tiles = image_processor.generate_tiles(img, CNN_INPUT_DIM, TILE_OVERLAP_FACTOR)

            cell_detections = []    # a list of tuples, (tile index, detection serial ID, class, confidence, x1, y1, x2, y2, overlapping_tiles)
            for tile_idx,(tile,(tile_x,tile_y), adjacent_tile_idxs) in enumerate(img_tiles):
                # Generate status bar and percentage indicator for displaying progress
                # In this loop, maximum progress caps out at 90%
                completion_pc = (tile_idx+1)/len(img_tiles) * 0.9
                hash_symbol_cnt = round(completion_pc*STATUS_BAR_LEN)
                status_bar = '[' + '#'*hash_symbol_cnt + '-'*(STATUS_BAR_LEN-hash_symbol_cnt) + ']'
                print(f"{status_bar} %d%%"%round(completion_pc*100.0), end='\r')
                
                ### Run the CNN ###
                detections = cnn_detector.detect(tile)
                # Go through the detected objects
                for detection in detections:
                    # Transform bounding box coordinates from tile frame to full image frame
                    x1 = int(detection[2]) + tile_x
                    y1 = int(detection[3]) + tile_y
                    x2 = int(detection[4]) + tile_x
                    y2 = int(detection[5]) + tile_y
                    # (tile index, class enum, confidence score, bounding box coordinates)
                    cell_detections.append((tile_idx, detection_id, detection[0], detection[1], x1, y1, x2, y2))
                    detection_id += 1
            cell_detections = np.array(cell_detections)
            # print(f"{len(cell_detections)} cell_detections")
            
            if cell_detections.shape[0]>0:
                # Create an array that contains entries of bounding box coordinates, the tile index, and
                # For each tile, check if any detection is too close to a detection in any tile to the right or below the tile
                detection_IDs_to_discard = []
                for tile_idx,(tile,(tile_x,tile_y), adjacent_tile_idxs) in enumerate(img_tiles):
                    for adj_tile_idx in adjacent_tile_idxs:
                        # Complete the rest of the progress bar here
                        completion_pc = ((tile_idx+1)/len(img_tiles) * 0.1) + 0.9
                        hash_symbol_cnt = round(completion_pc*STATUS_BAR_LEN)
                        status_bar = '[' + '#'*hash_symbol_cnt + '-'*(STATUS_BAR_LEN-hash_symbol_cnt) + ']'
                        print(f"{status_bar} %d%%"%round(completion_pc*100.0), end='\r')
                        
                        # Discard the tuple of adjacent tile indexes
                        curr_tile_detections = cell_detections[cell_detections[:,0]==tile_idx]
                        adj_tile_detections = cell_detections[cell_detections[:,0]==adj_tile_idx]
                        # We now have two arrays of detections, where some detections may overlap with another
                        # Take the detections from the current tile, and discard any that overlap with the adjacent tile
                        discard_IDs = get_IDs_to_discard(curr_tile_detections, adj_tile_detections)
                        detection_IDs_to_discard.extend(discard_IDs)
                # Discard repeated cell detections
                # unique_cell_detections = np.unique(unique_cell_detections, axis=0)
                # Create a boolean mask for rows to delete
                mask = np.isin(cell_detections[:, 1].astype('int'), np.array(detection_IDs_to_discard).astype('int'))
                unique_cell_detections = np.delete(cell_detections, np.where(mask), axis=0)
                # print(f"{len(unique_cell_detections)} unique_cell_detections")

                # Annotate main image with the cell detections, and count the total numbers of cells
                cell_count = {'normal': 0, 'abnormal': 0}
                for detection in unique_cell_detections:
                    x1, y1, x2, y2 = detection[4:8].astype('int')
                    if detection[2]==0:     # normal cell
                        cell_count['normal'] += 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR_REGULAR_CELL, ANNOTATION_THICKNESS)
                        # cv2.rectangle(img, (x1, y1), (x2, y2), colors[tile_idx%6], ANNOTATION_THICKNESS)
                    elif detection[2]==1:
                        cell_count['abnormal'] += 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR_ABNORMAL_CELL, ANNOTATION_THICKNESS)
                    # cv2.circle(tile, (d[2],d[3]), radius, color, thickness)
                print(f"\nCell count: {cell_count['normal']} normal, {cell_count['abnormal']} abnormal")
                cell_count_summary[f"{input_fname.split('/')[-1]}"] = [cell_count['normal'], cell_count['abnormal']]
            else:
                print(f"[{'#'*STATUS_BAR_LEN}] 100%", end='\r')
                cell_count_summary[f"{input_fname.split('/')[-1]}"] = [0, 0]
                print(f"\nNo cells detected")

            # Output the an image regardless of whether there are detections
            output_img = cv2.resize(img, (OUTPUT_IMG_W, int(OUTPUT_IMG_W/img.shape[1] * img.shape[0])))
            cv2.imwrite(f"{OUTPUT_FOLDER}/{input_fname.split('/')[-1]}.jpg", output_img)
        except:
            print(f"Error on image {input_fname.split('/')[-1]}")
        print('')
    
        # Generate an excel spreadsheet report for the data we have so far
        generate_report(cell_count_summary, f"{OUTPUT_FOLDER}/{REPORT_FILE_NAME}.xlsx")