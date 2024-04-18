import os

def make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def str_to_tup(s):
    return tuple([int(v) for v in s[1:-1].split(',')])

def get_areas_overlap(a1, a2):
    x1, y1, w1, h1 = a1
    x2, y2, w2, h2 = a2

    # Determine the (x, y) coordinates of the overlap rectangle's bottom-left corner
    overlap_x1 = max(x1, x2) 
    overlap_y1 = max(y1, y2) 
    
    # Determine the (x, y) coordinates of the overlap rectangle's top-right corner
    overlap_x2 = min(x1 + w1, x2 + w2) 
    overlap_y2 = min(y1 + h1, y2 + h2) 
    
    # Calculate the dimensions of the overlap rectangle
    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)

    # Compute the area of the overlap rectangle
    overlap_area = overlap_width * overlap_height
    if overlap_area == 0:
        return None
    return (overlap_x1, overlap_y1, overlap_width, overlap_height)
