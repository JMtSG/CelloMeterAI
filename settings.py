
# Put the images to be processed in here
INPUT_FOLDER = './input_images'

# This output folder gets cleared each run, don't store anything in here
OUTPUT_FOLDER = './results'
OUTPUT_IMG_W = 1920
REPORT_FILE_NAME = 'cell_count_summary'

VALID_INPUT_FILETYPES = ('*.jpg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

# Minimum confidence level for CNN to mark a detection
CONFIDENCE_THRESHOLD = 0.4

# Maximum allowed overlap percentage between cell detections of adjacent tiles
# Too high and cells in overlapping tile regions may be counted twice
# Too low and cells that are squished together may be missed
MAX_OVERLAP_PERCENTAGE = 0.5

# Scale up the input image by this factor before passing through CNN
ZOOM_FACTOR = 7

# Percentage overlap between tiles
TILE_OVERLAP_FACTOR = 0.3

# Colours for the annotating box; (B,G,R)
COLOUR_REGULAR_CELL = (255, 0, 255)
COLOUR_ABNORMAL_CELL = (0, 50, 255)
ANNOTATION_THICKNESS = 2

# Frame size for inputs to CNN
CNN_INPUT_DIM = 640     # square image

INFERENCE_DEVICE = 'cpu'  # '0', '1', etc. for GPU, 'cpu' for CPU