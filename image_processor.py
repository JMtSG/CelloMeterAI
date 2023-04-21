import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def generate_tiles(self, img, tile_size, overlap_factor):
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