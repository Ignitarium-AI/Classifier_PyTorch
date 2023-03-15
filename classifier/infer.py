# --------------------------------------------------------------------------
#                         Copyright © by
#           Ignitarium Technology Solutions Pvt. Ltd.
#                         All rights reserved.
#  This file contains confidential information that is proprietary to
#  Ignitarium Technology Solutions Pvt. Ltd. Distribution,
#  disclosure or reproduction of this file in part or whole is strictly
#  prohibited without prior written consent from Ignitarium.
# --------------------------------------------------------------------------
#  Filename    : infer.py
# --------------------------------------------------------------------------
#  Description :
# --------------------------------------------------------------------------
"""
Perform inference on given images
RUN:
For inference on folder:
python -m classifier.infer --model_name "resnet18" --num_classes 2 \
    -ws "workspace" -n "neg_damage_classifier" --folder_path <PATH OF FOLDER>

For inference on image
python -m classifier.infer --model_name "resnet18" --num_classes 2 \
    -ws "workspace" -n "neg_damage_classifier" --img_path <PATH OF IMAGE>
"""
import os
import argparse
import random
import torch
import cv2
import numpy as np

from classifier.model import GetModel
from classifier import dataloader

class InferClass:
    """
    Class to perform the inference for model
    """
    def __init__(self,
                 model_name,
                 num_classes,
                 work_dir,
                 node_name,
                 classes,
                 image_size,
                 img_path,
                 folder_path,
                 weight_path):
        """
        Args:
            model_name: name of the model which needs to be loaded
            num_classes: number of classes (required for model loading)
            work_dir: path of workspace directory
            node_name: name of node (required for weights loading)
            classes: list of labels for model prediction
            image_size: size of image
            img_path: query image path
            folder_path: query images directory path
            weight_path: path where model is saved
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.workdir_path = work_dir
        self.node_name = node_name
        self.classes = classes
        self.image_size = image_size
        self.img_path = img_path
        self.folder_path = folder_path
        self.model_path = weight_path
        if self.model_path is None:
            self.model_path = os.path.join(self.workdir_path, "Weights")
        self.network = GetModel(model_name=self.model_name, num_classes=self.num_classes, labels=self.classes)
        self.model = self.network.build()

        # send model to device (GPU/CPU)
        self.select_device()
        self.model.to(self.device)

        # load the weights
        self.load_model()
        self.labels = sorted(self.classes)

    def check_validity(self, image=None, folder_path=None):
        """
        Helper function to validate if given image or directoruy is valid
        for performing inference.
        """
        if image is not None:
            assert os.path.isfile(image), f"\
                provided image path {image} is not valid image path, \
                if image path is relative, please try to run again with full image path"

            # Open the image file check if it is valid image
            img = cv2.imread(image)
            assert img is not None, f"Unable to read image {image}, \
                please check if file is not corrupted and has one of the \
                supported image format of 'jpeg', 'jpg', 'png', or 'bmp'"

        if folder_path is not None:
            assert os.path.isdir(folder_path), f"\
                provided directory path {folder_path} is not valid directory path, \
                if directory path is relative, please try to run again with \
                    full directory path"
            count = 0
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Check if the file is an image file
                if file_path.endswith(('.jpeg', '.JPEG', '.jpg', '.JPG',
                                       '.png', '.PNG', '.bmp', '.BMP')):
                    count += 1
                    image = cv2.imread(file_path)
                    assert image is not None, f"Unable to read image {file_path}, \
                        please check if file is not corrupted and has one of the \
                            supported image format of 'jpeg', 'jpg', 'png', or 'bmp'"
            assert count != 0, f"No images found with valid format of 'jpeg', 'jpg', \
                'png', or 'bmp' in given directory path {folder_path}. "

    def select_device(self):
        """
        Helper function to pick GPU if available, else CPU
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def create_safe_directory(self, path):
        """
        Helper function to check if directory exists
        if not exists, will create directory at given path
        Args:
            path: location of the directory
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def seed_everything(self, seed=42):
        """
        Helper function to seed all random parameters
        Describes the common system setting needed for reproducible training
        Args:
            seed: seed number for reproducibility
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_model(self):
        """
        Helper function to load saved model
        """
        if os.path.isfile(os.path.join(self.model_path, "best.pt")):
            try:
                checkpoint = torch.load(os.path.join(self.model_path, "best.pt"),
                                        map_location=self.device)
                self.model.load_state_dict(checkpoint["state_dict"])
                if hasattr(self, 'optimizer'):
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("[Info] Loaded weights successfully")
            except:
                print("[Info] No weights found")
        else:
            print("[Info] No weights found")

    def save_model(self):
        """
        Helper function to save model
        """
        if not os.path.exists(self.model_path):
            self.create_safe_directory(self.model_path)
        self.model.to("cpu")
        if hasattr(self, 'optimizer'):
            checkpoint = {"state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()}
        else:
            checkpoint = {"state_dict": self.model.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_path, "best.pt"))
        self.model.to(self.device)

    def predict(self, img):
        """
        predict the label for given image path
        Args:
            image_path: path of query image for prediction
        """
        # change model mode to eval before inference
        self.model.eval()
        transforms = dataloader.get_image_preprocess_transforms(self.image_size)
        img = transforms(img)
        img = img.unsqueeze_(0)
        img = img.to(self.device)
        img = torch.autograd.Variable(img)
        output = self.model(img)
        if hasattr(output, "logits"):
            output = output.logits
        index = output.detach().cpu().numpy().argmax()

        print(f"[Info] Predicted class is {self.labels[index]}\n")
        return self.labels[index]

    def inference(self):
        """
        perform inference on query image data
        """
        assert (self.folder_path != "None") or (self.img_path != "None"), "Either the \
            image path using '--img' or directory path using '--folder_path'  \
                must needs to be specified to perform the inference."
        assert not ((self.folder_path != "None") and (self.img_path != "None")), f"both \
            image path: {self.img_path} and directory path: {self.folder_path} can \
                not be used together"

        if self.img_path != "None":
            self.check_validity(image = self.img_path)
            img = cv2.imread(self.img_path)
            self.predict(img)

        elif args.folder_path != "None":
            self.check_validity(folder_path = self.folder_path)
            # following line will create list of images in given folder_path with
            # supported image format of "png", "jpg", "jpeg", and "bmp"
            images = [os.path.join(self.folder_path, img) for img in os.
                      listdir(self.folder_path) if img.endswith((".png", ".PNG", ".jpg",
                                                         ".JPG", ".jpeg", ".JPEG",
                                                         ".bmp", ".BMP"))]
            for image in images:
                img = cv2.imread(image)
                self.predict(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default="resnet18",
        help="name of the model")
    parser.add_argument(
        "-nc",
        "--num_classes",
        type=int,
        default=2,
        help="number of classes")
    parser.add_argument(
        "-ws",
        "--work_dir",
        default="workspace",
        help="workdirectory",
    )
    parser.add_argument(
        "-n",
        "--node_name",
        default=None,
        help="Provide nodes list, Eg; A1_scopito_wt, scopito_wt_body, scopito_cowling",
    )
    parser.add_argument(
        "-cl",
        "--classes",
        nargs="+",
        default=["cat", "dog"],
        help="list of classes which are used during training. \
            NOTE: Do not bother about order of classes",
    )
    parser.add_argument(
        "-is",
        "--image_size",
        default=(224, 224),
        help="Image size",
    )
    parser.add_argument(
        "-i",
        "--img_path",
        type=str,
        default="None",
        help="path to single image for inference \
            NOTE: Either specify this argument or next argument 'folder_path', \
                both can not be used together.")
    parser.add_argument(
        "-d",
        "--folder_path",
        type=str,
        default="None",
        help="path to directory containing images file\
            NOTE: Either specify this argument or previous argument 'img', \
                both can not be used together.")
    parser.add_argument(
        "-wt",
        "--weight_path",
        default=None,
        help="path to directory containing best.pt"
    )
    args = parser.parse_args()
    infer_obj = InferClass(model_name=args.model_name, num_classes=args.num_classes,
                           work_dir=args.work_dir, node_name=args.node_name,
                           classes=args.classes, image_size=args.image_size,
                           img_path=args.img_path, folder_path=args.folder_path,
                           weight_path=args.weight_path)
    infer_obj.inference()
