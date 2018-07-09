from PIL import Image
import numpy as np

class AffineWarp:

    def __init__(self, width, height):
        self.width = width
        self.height = height


    def __call__(self, images, labels):
        images_warped, labels_warped = [], []

        for image, image_labels in zip(images, labels):
            
            # Convert to PIL image and calculate warp factors
            image = Image.fromarray(image)
            width, height = image.size

            width_factor = width / self.width
            height_factor = height / self.height

            # Warp images
            image_warped = np.array(image.resize((self.width, self.height)))
            
            # Warp labels
            warp_factors = np.array([
                width_factor, 
                height_factor, 
                width_factor,
                height_factor
            ])

            image_labels_warped = (image_labels / warp_factors).astype(np.int)
            
            images_warped.append(image_warped)
            labels_warped.append(image_labels_warped)

        return images_warped, labels_warped
