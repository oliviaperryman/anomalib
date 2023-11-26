import math
import os
import sys
import tempfile
from typing import Sequence

import numpy as np

sys.path.append("/local/omp/eagle_eyes_hackathon/")

from artemis.general.custom_types import BGRImageArray
from hackathon.model_utils.interfaces import Detection, IDetectionModel
from PIL import Image, ImageDraw

from model import Model


class ModelEE(IDetectionModel):
    def __init__(self):
        self.model = Model()

    def get_name(self) -> str:
        """Return the name of this detector"""
        return "Olivia's anomalib model"

    def detect_patch(self, image_path: str, offset: tuple):
        _, bounding_boxes = self.model.predict(image_path)

        detections = []
        for box in bounding_boxes:
            x, y, w, h = box
            i = y + h / 2
            j = x + w / 2
            i += offset[0]
            j += offset[1]
            detections.append(Detection((i, j, h, w), 1, "", ""))

        print(detections)
        return detections

    def visualize_patches(self, paths):
        # Initialize a list to store the images
        images = [Image.open(path) for path in paths]

        # Calculate the size of the grid
        grid_size = math.ceil(math.sqrt(len(images)))
        width, height = images[0].size
        grid_img = Image.new("RGB", (width * grid_size, height * grid_size))

        # For each image and its position in the grid
        for i, img in enumerate(images):
            # Calculate the position in the grid
            row = i // grid_size
            col = i % grid_size

            # Paste the image into the appropriate position in the grid
            grid_img.paste(img, (col * width, row * height))

        # Save the grid image
        grid_img.save("image_grid.png")

    def detect(self, image: BGRImageArray) -> Sequence[Detection]:
        """Detect objects in an image.  Return a list of detections."""
        # Define the size of the patches and the overlap
        patch_size = 256
        overlap = 20
        image_np = np.array(image)
        # Calculate the number of patches in the x and y directions
        num_patches_x = (image_np.shape[1] - overlap) // (patch_size - overlap)
        num_patches_y = (image_np.shape[0] - overlap) // (patch_size - overlap)

        img_paths = []
        offsets = []
        # For each patch
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Calculate the start and end coordinates of the patch
                start_y = i * (patch_size - overlap)
                end_y = start_y + patch_size
                start_x = j * (patch_size - overlap)
                end_x = start_x + patch_size

                # Extract the patch from the image
                patch = image_np[start_y:end_y, start_x:end_x]

                # Convert the patch to a PIL Image object
                patch_img = Image.fromarray(patch)

                # Save the patch to a temporary path
                temp_path = os.path.join(tempfile.gettempdir(), f"patch_{i}_{j}.png")
                patch_img.save(temp_path)
                img_paths.append(temp_path)
                offsets.append((start_y, start_x))
                print(temp_path)

        self.visualize_patches(img_paths)

        all_detections = []
        for img_path, offset in zip(img_paths, offsets):
            detections = self.detect_patch(img_path, offset)
            all_detections.extend(detections)

        return all_detections


if __name__ == "__main__":
    # test_img = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/images/BREW_HUT_SNOW_SHADOWS_PERSON_WALKING-0.png"
    test_img = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/images/SQUAMISH_SNOWY_FIELD_WALKER_CLOSE-2.png"
    img_array = np.array(Image.open(test_img))
    model = ModelEE()
    detections = model.detect(img_array)
    print(detections)

    # save image
    img = Image.open(test_img)
    img1 = ImageDraw.Draw(img)

    for detection in detections:
        i, j, h, w = detection.ijhw_box
        x = int(j - w / 2)
        y = int(i - h / 2)
        w = int(w)
        h = int(h)
        img1.rectangle((x, y, x + w, y + h), outline="red")
    img.save("output.png")
