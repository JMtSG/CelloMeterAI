# CelloMeterAI

CelloMeterAI is a program that counts cells in microscope images. Specifically, the program applies a [YOLOv8 deep convolutional neural network (CNN)](https://ultralytics.com/yolov8) that has been fine-tuned to recognise protoplasts.

<p align="center">
  <img src="graphics/CelloMeterAI.gif" />
</p>


Deep convolutional neural networks offer an automated and robust approach to object detection, and publicly available general models such as YOLO can be leveraged for counting cells. These large pre-trained models can be fine tuned using a small custom dataset of labelled images to learn the features that distinguish cells from the background and other artifacts. In theory once properly trained, the CNN can generalise to recognise cells across a wide range of images, even if the images have varying resolutions, magnification levels, lighting conditions, etc.  
This approach differs from classical image processing techniques such as Watershed segmentation which require significant image pre-processing manipulation & morphological operations, and manually tuning parameters (e.g. pixel value and size thresholds) specific to the conditions of each image set to extract the cells from the image. While these methods can still be effective, they are often time-consuming as they don't generalise as well to varying photograph conditions.


This program offers an easy to use, cross-platform interface to a deep CNN; simply place images in the input folder, run the program, and you'll find the processed & annotated images in the output folder along with a spreadsheet report containing cell counts and totals. The program supports multithreaded CPU operation as well as CUDA GPU. The program also offers a live view to show annotation of cells as each tile is processed (enabled by the `LIVE_DISPLAY` parameter).  
The program development and CNN fine tuning were based around 640x480 input images, though other image resolutions are supported. You may need to adjust the `ZOOM_FACTOR` parameter appropriately in the settings file to suit your input image resolution. While the CNN can ingest images of this resolution directly, I found that for higher accuracy it was best to divide input images into smaller tiles and run them all through the CNN individually. In order to minimise the chances of a cell not being counted due to being split across multiple tile images, the program employs a sliding-window tiling method, generating tiles which overlap each other by some amount (see the `TILE_OVERLAP_FACTOR` parameter). In order to avoid counting the same cell twice, the program also runs a post-processing step on each image, rejecting detections if they overlap another detection in an adjacent tile greater than a specified percentage (`MAX_OVERLAP_PERCENTAGE`). The model distinguishes between clean cell detections, cells with an abnormal appearance due to damage or otherwise, and blurry regions which are likely cells but can not be reliably counted individually. Bounding boxes are on annotated output images are coloured magenta, red, and orange for each of these cases respectively, and the spreadsheet report will outline the count for each class separately.


## Usage (from pre-built Windows release)
- Download a .zip of the program from [the releases page](https://github.com/JMtSG/CelloMeterAI/releases) and unzip it on your computer
- Copy your images into the `input_images` folder
- Configure the program in `settings.yaml` (optional)
- Double click `CelloMeterAI` to run
- You'll find annotated images in the `results` directory. There will also be a .xlsx spreadsheet summary of the results included.

## Usage (from source code)
- Clone this repo to your local machine
- Copy your images into the `input_images` directory.
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Configure the program in `settings.yaml` (optional)
- Run `python3 CelloMeterAI.py`
- You'll find annotated images in the `results` directory. There will also be a .xlsx spreadsheet summary of the results included.


## Configuration
The `settings.yaml` file contains configuration parameters for this program.
- `INPUT_FOLDER`: The folder containing the input images to be processed.
- `MODEL_NAME`: The filename of the PyTorch model used for cell detection.
- `OUTPUT_FOLDER`: The folder where the output images and report file will be saved. This folder is cleared at the start of each run, so no files should be stored here.
- `OUTPUT_IMG_W`: The width of the output images in pixels. The height is automatically determined to maintain the aspect ratio.
- `REPORT_FILE_NAME`: The filename of the report file saved to the `OUTPUT_FOLDER`.
- `VALID_INPUT_FILETYPES`: A list of valid file extensions for input images.
- `CONFIDENCE_THRESHOLD`: The minimum confidence level required for a detection to be considered valid.
- `LIVE_DISPLAY`: If `True`, the input images will be displayed during the counting process.
- `DISPLAY_DOWNSCALE_FACTOR`: Downscale the input image by this factor when displaying it on-screen.
- `PROGRESS_IMG_DISP_STRIDE`: To speed up processing, only every N-th frame is displayed on-screen.
- `MAX_OVERLAP_PERCENTAGE`: The maximum allowed overlap percentage between cell detections of adjacent tiles. Too high and cells in overlapping tile regions may be counted twice; too low and cells that are squished together may be missed.
- `ZOOM_FACTOR`: The factor by which the input image is scaled up before passing through the CNN.
- `TILE_OVERLAP_FACTOR`: The percentage overlap between tiles.
- `COLOURS`: A dictionary of colours used to annotate the bounding boxes around cells. The keys are `REGULAR_CELL`, `ABNORMAL_CELL`, and `BLOB_CLUSTER`, and the values are lists of `[R, G, B]` values.
- `ANNOTATION_THICKNESS`: The thickness of the annotation bounding boxes.
- `CNN_INPUT_DIM`: The dimensions of the input to the CNN. The input is square, so only one dimension is required.
- `INFERENCE_DEVICE`: The device used for inference. Either a string representation of a device (e.g., '0', '1', etc.) or 'cpu' for CPU.
- `NUM_WORKERS`: The number of concurrent processing streams used.


 ## Developer Notes

For deployment on Windows systems without needing to install dependencies, you can build this project with its dependencies using [PyInstaller](https://pyinstaller.org/en/stable/).

- First setup a conda environment on a windows machine and ensure that the dependencies are all installed and `CelloMeterAI.py` runs correctly.
- With PyInstaller installed (`pip install pyinstaller`), run `pyinstaller -name Program_Files CelloMeterAI.py` from within the conda environment.
    - The reason --onefile isn't used is because it causes the Windows builtin antivirus to scan the executable each time it is started, which takes a significant amount of time.
- You'll find the built program under `dist\Program_Files`.
- If you attempt to run `Program_Files.exe`, you may get an error about a missing `default.yaml` file.
    - You can download it from [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml)
    - Place it in `Program_Files\ultralytics\yolo\cfg\default.yaml`

For ease of use, the following steps can also be taken:
- Find the `Program_Files` directory inside and (optional) set it to be hidden within right-click -> "Properties".
- Rename `Program_Files.exe` inside `Program_Files` to "CelloMeterAI.exe" (optional)
- Right-click -> create shortcut to CelloMeterAI.exe
- Copy it to the directory above
- Rename the shortcut to "CelloMeterAI"
- Right-click -> Properties, change Target to "%COMSPEC% /C .\Program_Files\CelloMeterAI.exe"
- Clear the "Start in" text box
- Copy settings.yaml and model_weights.pt to the same folder as the `CelloMeterAI` shortcut

You can also compress the directory contents for ease of deployment:
- Open the 7-zip File Manager
- Highlight all the files and folders (Program_Files, input_images, settings.yaml, CelloMeterAI.lnk, model_weights.pt)
- Click "Add"
    - Archive format: zip
    - Update mode: Add and replace files
    - Path mode: Relative pathnames


## Training

This model used in this program was fine-tuned to maximise accuracy on a very specific case of protoplast images. You can fine tune your own model by using the training scripts provided in this repo. `generate_training_data.py` will generate tiles for you and generate labels using the existing model in YOLO label format as a starting point, which you can modify to fix any mistakes. Once you have training images, you can use the `train.py` script to train your new model. The `cell_img_dataset_metadata.yaml` file is formatted as follows:
```
# train and val data as either:
# - directory: path/images/
# - file: path/images.txt
# - list: [path1/images/, path2/images/]
path: /path/to/your/images
train: ./train
val: ./val

# Number of classes
nc: 3

# Class names
names: [ 'normal_cell', 'abnormal_cell', 'blob_cluster' ]
```

Also consider that the optimal `ZOOM_FACTOR` and `TILE_OVERLAP_FACTOR` parameters may differ depending on your application.

## Acknowledgments

This project uses the YOLOv8 Deep CNN object detection model by Ultralytics: ([GitHub repo](https://github.com/ultralytics/ultralytics)), ([Docs](https://docs.ultralytics.com/))
