import struct
import numpy as np
import cv2

def compute_endpoint_error(flow_gt, flow_pred):
    """
    Compute the endpoint error (EPE) between ground truth and predicted flow.
    
    @param flow_gt: Ground truth optical flow (height, width, 2)
    @param flow_pred: Predicted optical flow (height, width, 2)
    
    @return: Mean endpoint error
    """
    error = np.sqrt(np.sum((flow_gt - flow_pred) ** 2, axis=-1))
    return np.mean(error)


def read_flo_file(filename):
    """Reads a .flo file and returns the flow as a numpy array."""
    with open(filename, 'rb') as f:
        magic = struct.unpack('4s', f.read(4))[0]
        if magic != b'PIEH':
            raise ValueError("This is not a valid .flo file")
        
        width = struct.unpack('I', f.read(4))[0]
        height = struct.unpack('I', f.read(4))[0]
        data = np.fromfile(f, np.float32)
        flow = np.resize(data, (height, width, 2))  # (height, width, 2) -> (y, x, flow_components)
        
        return flow


def load_images(image1_path, image2_path):
    """Load a pair of images."""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Error reading images {image1_path} or {image2_path}")
    
    return img1, img2