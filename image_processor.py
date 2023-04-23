
class ImageProcessor:
    def __init__(self):
        """
        Initializes an ImageProcessor object
        """
        pass

    def generate_tiles(self, img, tile_size, overlap_factor):
        """
        Generates tiles from an input image with a specified tile size and overlap factor.

        Args:
        - img: A numpy array representing an image.
        - tile_size: An integer representing the size of the tile.
        - overlap_factor: A float representing the overlap factor.

        Returns:
        - A list of tuples, each containing a tile, its location, and a list of indices of adjacent tiles.
        """
        height, width = img.shape[:2]
        overlap = int(tile_size * overlap_factor)
        stride = tile_size - overlap
        tiles_per_row = int(width//stride)
        tiles = []
        tile_num = 0
        y = 0
        while y < height:
            x = 0
            while x < width:
                tile = img[y:y+tile_size, x:x+tile_size]
                # Create a list of indexes for all the tiles that this tile will overlap with
                adjacent_tile_idxs = []
                # if x > 0:
                #     adjacent_tile_idxs.append(tile_num - 1)  # Left tile
                if (x+stride) < width:
                    adjacent_tile_idxs.append(tile_num + 1)  # Right tile
                # if y > 0:
                #     adjacent_tile_idxs.append(tile_num - tiles_per_row)  # Tile above
                if (y+stride) < height:
                    adjacent_tile_idxs.append(tile_num + tiles_per_row)     # Tile below
                    adjacent_tile_idxs.append(tile_num + tiles_per_row+1)   # Tile diagonally below
                # Append all data to the main list to be returned
                tiles.append((tile, (x,y), adjacent_tile_idxs))
                x += stride
                tile_num += 1
            y += stride
        return tiles


if __name__=='__main__':
    import sys
    import cv2
    import yaml

    # Load settings
    with open('settings.yaml', 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    img_fpath = sys.argv[1]
    img = cv2.imread(img_fpath)
    img = cv2.resize(img, (int(img.shape[1] * settings['ZOOM_FACTOR']), 
                        int(img.shape[0] * settings['ZOOM_FACTOR'])))

    image_processor = ImageProcessor()
    img_tiles = image_processor.generate_tiles(img,
                                                settings['CNN_INPUT_DIM'],
                                                settings['TILE_OVERLAP_FACTOR'])

    for tile_idx,(tile,(tile_x,tile_y), adjacent_tile_idxs) in enumerate(img_tiles):
        cv2.imwrite(f"{sys.argv[2]}/%03d.jpg"%tile_idx, tile)
    
    print("Generated %d tiles"%len(img_tiles))