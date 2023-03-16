
# --------------------------------------------------------------------------
#                         Copyright © by
#           Ignitarium Technology Solutions Pvt. Ltd.
#                         All rights reserved.
#  This file contains confidential information that is proprietary to
#  Ignitarium Technology Solutions Pvt. Ltd. Distribution,
#  disclosure or reproduction of this file in part or whole is strictly
#  prohibited without prior written consent from Ignitarium.
# --------------------------------------------------------------------------
#  Filename    : test.py
# --------------------------------------------------------------------------
#  Description :
# --------------------------------------------------------------------------
"""
Evaluate the test dataset with accuracy, precision, recall,
f1 score and confusion matrix

RUN: python test.py -t 0.8
"""
import os
import argparse
from glob import glob
from sklearn.metrics import (precision_score, recall_score, confusion_matrix,
    accuracy_score, f1_score)
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from classifier.infer import InferClass
from classifier import dataloader

class TestClassifier(InferClass):
    """
    Class to evaluate the metrics
    """
    def __init__(self,
                 model_name,
                 num_classes,
                 work_dir,
                 node_name,
                 classes,
                 seed,
                 full_db_flag,
                 image_size,
                 learning_rate=0.0001,
                 threshold=0.8,
                 batch_size=1,
                 img_dir=None,
                 data_path=None,
                 weight_path=None):
        """
        Args:
            model_name: name of the model which needs to be loaded
            num_classes: number of classes (required for model loading)
            work_dir: path of workspace directory
            node_name: name of node (required for weights loading)
            classes: list of labels for model prediction
            seed: seed value for random value generation
            full_db_flag: if full_db training or dummy_db
            image_size: size of image
            class_imbalance_method: class imbalance method, one of ['upsampler',
                                    'class_weight']
            learning_rate: learning_rate used for updating weights in backpropogation
            batch_size: batch size used for training and testing
            img_dir: image directory used for testing
        """
        super().__init__(model_name, num_classes, work_dir, node_name,
                         classes, image_size, img_path="None", folder_path="None",
                         weight_path=weight_path)

        # seed everything
        self.seed = seed
        self.threshold = threshold
        self.data_path = data_path
        self.db_folder_name = "git_DB_full" if full_db_flag else "git_DB_dummy"
        if self.data_path is None:
            self.data_path = os.path.join(self.workdir_path, "DB", self.db_folder_name,
                                      node_name)
        self.img_dir = img_dir
        self.image_size = image_size

        self.seed_everything(self.seed)

        # check if data already available
        assert os.path.exists(self.data_path), f"Data is not available at \
            {self.data_path} path, please download data and continue"

        # get data loader for training, validation and testing
        _, _, self.test_ds, _ = dataloader.get_data_generators(
            self.data_path,
            image_size,
            class_imbalance_method="class_weight",
            batch_size=batch_size
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def test_classifier(self):
        """
        test evaluation metrics on test data
        """
        self.model.eval()

        # to get batch accuracy
        true_outputs = []
        pred_outputs = []

        if self.img_dir is not None:
            images = glob(os.path.join(self.img_dir, "*/*.png"))
            images += glob(os.path.join(self.img_dir, "*/*.PNG"))
            images += glob(os.path.join(self.img_dir, "*/*.jpg"))
            images += glob(os.path.join(self.img_dir, "*/*.JPG"))
            images += glob(os.path.join(self.img_dir, "*/*.jpeg"))
            images += glob(os.path.join(self.img_dir, "*/*.JPEG"))

            for image in images:
                self.model.eval()
                img = cv2.imread(image)
                indx_target = image.split(os.path.sep)[-2]
                transforms = dataloader.get_image_preprocess_transforms(self.image_size)
                img = transforms(img)
                img = img.unsqueeze_(0)
                img = img.to(self.device)
                img = torch.autograd.Variable(img)
                output = self.model(img)
                if hasattr(output, "logits"):
                    output = output.logits
                index = output.detach().cpu().numpy().argmax()

                true_outputs.append(indx_target)
                pred_outputs.append(self.labels[index])
        else:
            loop = tqdm(self.test_ds, leave=True)
            for data, target in loop:
                indx_target = target.clone()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                if hasattr(output, "logits"):
                    output = output.logits
                # Score to probability using softmax
                prob = F.softmax(output, dim=1)
                pred = prob.data.max(dim=1)[1]

                true_outputs.extend(indx_target.cpu().tolist())
                pred_outputs.extend(pred.cpu().tolist())

        acc = accuracy_score(pred_outputs, true_outputs)
        prec = precision_score(pred_outputs, true_outputs, pos_label=true_outputs[0])
        recall = recall_score(pred_outputs, true_outputs, pos_label=true_outputs[0])
        f1score = f1_score(pred_outputs, true_outputs, pos_label=true_outputs[0])
        confusion_mat = confusion_matrix(y_true=true_outputs, y_pred=pred_outputs)

        assert acc > float(self.threshold), f"Current accuracy of {acc} \
        is less than threshold accuracy of {float(self.threshold)}"

        print(f"Accuracy : {acc}")
        print(f"Precision : {prec:2.3f}")
        print(f"Recall : {recall}")
        print(f"F1-Score : {f1score}")
        print("Confusion Matrix: \n", confusion_mat)

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
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "-f",
        "--full_db_flag",
        default=False,
        help="full db or dummy db",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed value for random variables",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning_rate used for updating weights in backpropogation",
    )
    parser.add_argument(
        "-id",
        "--image_dir",
        default=None,
        help="image directory used for testing",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for default accuracy")
    parser.add_argument(
        "-data",
        "--data_path",
        default=None,
        help="path to directory containing train/val/test data"
    )
    parser.add_argument(
        "-wt",
        "--weight_path",
        default=None,
        help="path to directory containing best.pt"
    )
    args = parser.parse_args()
    test_obj = TestClassifier(model_name=args.model_name, num_classes=args.num_classes,
                           work_dir=args.work_dir, node_name=args.node_name,
                           classes=args.classes, seed=args.seed,
                           full_db_flag=args.full_db_flag, image_size=args.image_size,
                           learning_rate=args.learning_rate, threshold=args.threshold,
                           batch_size=args.batch_size, img_dir=args.image_dir,
                           data_path=args.data_path, weight_path=args.weight_path)
    test_obj.test_classifier()
