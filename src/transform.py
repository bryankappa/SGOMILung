import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage import exposure
import cv2
import os


class BaseTransform:
    def __call__(self, img):
        raise NotImplementedError("Implement a __call__ method")
    

class HistogramEqual(BaseTransform):
    def __call__(self, image):
        image_np = np.array(image).astype(np.float32) / 255
        equalized_image = exposure.equalize_hist(image_np)
        return Image.fromarray((equalized_image * 255).astype(np.uint8))




