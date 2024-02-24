import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
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


class DenoiseImage(BaseTransform):
    def __call__(self, image):
        image_np = np.array(image)
        denoised_image = cv2.fastNlMeansDenoising(image_np, None, 10, 7, 21)
        return Image.fromarray(denoised_image)


class GrayScaleImage(BaseTransform):
    def __call__(self, image):
        image_np = np.array(image)
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(grayscale_image)


class LungDatasetLoader:
    def __init__(self, dataset_path, batch_size, num_workers):
        # Ensure the dataset path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset path {dataset_path} does not exist.")
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            GrayScaleImage(),
            transforms.Resize((256, 256)),
            HistogramEqual(),
            DenoiseImage(),
            transforms.ToTensor(),
            # Normalize images (values should be dataset-specific)
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def get_data_loader(self):
        # Create a dataset using the ImageFolder class and apply transformations
        dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
        # Create a DataLoader to load the images in batches
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return data_loader


if __name__ == "__main__":
    # Test the LungDatasetLoader class
    dataset_path = '../Data'
    batch_size = 32
    num_workers = 4
    lung_loader = LungDatasetLoader(dataset_path, batch_size, num_workers)
    data_loader = lung_loader.get_data_loader()