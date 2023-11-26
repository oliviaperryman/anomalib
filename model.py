import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from pytorch_lightning import Trainer
from skimage import measure
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


class Model:
    def __init__(self, config_path="config.yaml", weights="results/padim/eagleeyes/run/weights/lightning/model-v1.ckpt"):
        self.config_path = config_path
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = get_configurable_parameters(config_path=self.config_path)
        config.trainer.resume_from_checkpoint = str(self.weights)
        config.visualization.show_images = None
        config.visualization.mode = "full"

        self.config = config

        # create model and trainer
        self.model = get_model(self.config)
        callbacks = get_callbacks(self.config)
        self.trainer = Trainer(callbacks=callbacks, **config.trainer)

        transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
        image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
        center_crop = config.dataset.get("center_crop")
        if center_crop is not None:
            center_crop = tuple(center_crop)
        normalization = InputNormalizationMethod(config.dataset.normalization)
        self.transform = get_transforms(
            config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
        )

    def predict(self, input):
        # create the dataset
        dataset = InferenceDataset(
            input, image_size=tuple(self.config.dataset.image_size), transform=self.transform  # type: ignore
        )
        dataloader = DataLoader(dataset)

        # generate predictions
        preds = self.trainer.predict(model=self.model, dataloaders=[dataloader])

        pred_masks = preds[0]["pred_masks"][0]
        anomaly_maps = preds[0]["anomaly_maps"][0]

        # Convert prediction segmentation masks to bounding boxes
        labels = measure.label(pred_masks)
        bounding_boxes = []
        # For each unique label (excluding 0, which is the background)
        for label in np.unique(labels):
            if label == 0:
                continue

            # Get the x, y coordinates of the pixels that have this label
            _, y, x = np.where(labels == label)

            # Find the minimum and maximum x and y coordinates
            min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)

            # Append the bounding box to the list
            bounding_boxes.append((min_x, min_y, max_x, max_y))

        return preds, bounding_boxes


if __name__ == "__main__":
    # Test it is working
    # test_img = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/patches/positive/BOMBAY_BEACH_RUNNING-0.png"
    test_img = "/local/omp/eagle_eyes_hackathon/data/eagle_eyes_hackathon_dataset/patches/positive/SQUAMISH_SNOWY_FIELD_WALKER_CLOSE-2.png"
    model = Model()
    preds, bounding_boxes = model.predict(test_img)

    # save image
    img = Image.open(test_img)
    img1 = ImageDraw.Draw(img)
    for box in bounding_boxes:
        img1.rectangle(box, outline="red")
    img.save("output.png")

    anomaly_map_np = preds[0]["anomaly_maps"][0].cpu().numpy()
    anomaly_map_np = (anomaly_map_np - np.min(anomaly_map_np)) / (np.max(anomaly_map_np) - np.min(anomaly_map_np)) * 255
    anomaly_map_np = anomaly_map_np.astype(np.uint8)
    anomaly_map_img = Image.fromarray(anomaly_map_np[0])
    anomaly_map_img.save("anomaly_map.png")

    mask_np = preds[0]["pred_masks"][0].cpu().numpy()
    mask_np = mask_np * 255
    mask_np = mask_np.astype(np.uint8)
    mask_img = Image.fromarray(mask_np[0])
    mask_img.save("pred_mask.png")
