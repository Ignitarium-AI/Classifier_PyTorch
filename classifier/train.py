# --------------------------------------------------------------------------
#                         Copyright © by
#           Ignitarium Technology Solutions Pvt. Ltd.
#                         All rights reserved.
#  This file contains confidential information that is proprietary to
#  Ignitarium Technology Solutions Pvt. Ltd. Distribution,
#  disclosure or reproduction of this file in part or whole is strictly
#  prohibited without prior written consent from Ignitarium.
# --------------------------------------------------------------------------
#  Filename    : train.py
# --------------------------------------------------------------------------
#  Description :
# --------------------------------------------------------------------------
"""
Train and validate the model on training and validation dataset

RUN: python -m classifier.train --model_name "resnet18" --num_classes 2 --work_dir "workspace" --node_name "neg_damage_classifier" --seed 42 -cim "upsamplr" -lr 0.0001
"""

import os
import argparse
import unittest
from sklearn.metrics import (precision_score, recall_score,
    accuracy_score)
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from classifier.infer import InferClass
from classifier import dataloader

class TrainClassifier(InferClass):
    """
    Class to train, val and test the model
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
                 class_imbalance_method,
                 learning_rate,
                 batch_size,
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
        """
        super().__init__(model_name, num_classes, work_dir, node_name,
                         classes, image_size, img_path="None", folder_path="None",
                         weight_path=weight_path)
        self.train_step = 0
        self.val_step = 0
        self.best_accuracy = 0
        self.seed = seed
        self.data_path = data_path
        self.train_logs_dir = os.path.join(self.workdir_path, "logs", "train")
        self.val_logs_dir = os.path.join(self.workdir_path, "logs", "val")
        self.db_folder_name = "git_DB_full" if full_db_flag else "git_DB_dummy"
        if self.data_path is None:
            self.data_path = os.path.join(self.workdir_path, "DB", self.db_folder_name,
                                      node_name)

        # seed everything
        self.seed_everything(self.seed)

        # tensorboard logs
        self.train_writer = SummaryWriter(self.train_logs_dir)
        self.val_writer = SummaryWriter(self.val_logs_dir)

        # check if data already available
        assert os.path.exists(self.data_path), f"Data is not available at \
        {self.data_path} path, please download data and continue"

        # get data loader for training, validation and testing
        self.train_ds, self.val_ds, self.test_ds, self.class_wts = dataloader.get_data_generators(
            self.data_path,
            image_size,
            class_imbalance_method,
            batch_size
        )

        # initialize loss and optimizer with given learing rate
        if class_imbalance_method  == "class_weight" and len(self.class_wts) > 0:
            self.criterion = torch.nn.CrossEntropyLoss(torch.tensor(self.class_wts)).to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def validate_classifier(self, y_true=None, y_pred=None):
        """
        Validate model accuracy and loss on validation data
        Args:
            y_true: list used in unit_test
            y_pred: list used in unit_test
        """
        # change model in eval mode
        self.model.eval()

        # to get batch loss
        batch_loss = np.array([])
        # to get batch accuracy
        batch_acc = np.array([])

        loop = tqdm(self.test_ds, leave=True)
        with torch.no_grad():
            for _, (data, target) in enumerate(loop):
                indx_target = target.clone()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                if hasattr(output, "logits"):
                    output = output.logits

                # add loss for each mini batch
                loss = self.criterion(output, target)

                batch_loss = np.append(batch_loss, [loss.item()])

                # Score to probability using softmax
                prob = F.softmax(output, dim=1)
                pred = prob.data.max(dim=1)[1]
                if y_true is not None and y_pred is not None:
                    y_true.extend(indx_target.cpu().tolist())
                    y_pred.extend(pred.cpu().tolist())
                correct = pred.cpu().eq(indx_target).sum()
                acc = float(correct) / float(len(data))
                batch_acc = np.append(batch_acc, [acc])

                loop.set_postfix(
                    Accuracy=batch_acc.mean().item(),
                    Loss=batch_loss.mean().item(),
                )
            epoch_loss = batch_loss.mean()
            epoch_acc = batch_acc.mean()

        self.val_writer.add_scalar("Validation loss", epoch_loss, global_step=self.val_step)
        self.val_writer.add_scalar("Validation accuracy", epoch_acc, global_step=self.val_step)
        self.val_step += 1
        return epoch_acc

    def train_classifier(self):
        """
        Train the model for corresponding epoch and calculate accuracy and loss
        """
        # change model in training mode
        self.model.train()
        # to get batch loss
        batch_loss = np.array([])
        # to get batch accuracy
        batch_acc = np.array([])

        loop = tqdm(self.train_ds, leave=True)
        for _, (data, target) in enumerate(loop):
            indx_target = target.clone()
            # send data to device (its is medatory if GPU has to be used)
            data = data.to(self.device)
            # send target to device
            target = target.to(self.device)

            # reset parameters gradient to zero
            self.optimizer.zero_grad()

            # forward pass to the model
            output = self.model(data)
            if hasattr(output, "logits"):
                output = output.logits

            # cross entropy loss
            loss = self.criterion(output, target)

            # find gradients w.r.t training parameters
            loss.backward()
            # Update parameters using gardients
            self.optimizer.step()

            batch_loss = np.append(batch_loss, [loss.item()])

            # Score to probability using softmax
            prob = F.softmax(output, dim=1)
            # get the index of the max probability
            pred = prob.data.max(dim=1)[1]
            # correct prediction
            correct = pred.cpu().eq(indx_target).sum()
            # accuracy
            acc = float(correct) / float(len(data))
            batch_acc = np.append(batch_acc, [acc])

            loop.set_postfix(
                Accuracy=batch_acc.mean().item(),
                Loss=batch_loss.mean().item(),
            )
        epoch_loss = batch_loss.mean()
        epoch_acc = batch_acc.mean()

        self.train_writer.add_scalar("Training loss", epoch_loss, global_step=self.train_step)
        self.train_writer.add_scalar("Training accuracy", epoch_acc, global_step=self.train_step)
        self.train_step += 1

    def run_traininig(self, epochs=100):
        """
        run training and validation for given number of epochs
        """
        # Calculate Initial Test Loss and Test Accuracy
        init_test_accuracy = self.validate_classifier()
        print(f"[Info] Initial Test Accuracy : {init_test_accuracy*100}%\n")

        # training for given number of epochs
        for epoch in range(epochs):
            print(f"[Info] Epoch {epoch+1}/{epochs}\n")
            self.train_classifier()
            val_accuracy = self.validate_classifier()

            if val_accuracy >= self.best_accuracy:
                print(f"[Info] Best accuracy improved from {self.best_accuracy} to \
                      {val_accuracy}, so Saving the Model...\n")
                self.best_accuracy = val_accuracy
                self.save_model()

        print("[Info] Training finished, Testing the model")

        # Calculate final Test Loss and Test Accuracy
        final_test_accuracy = self.validate_classifier()
        print(f"[Info] Final Test Accuracy : {final_test_accuracy*100}%")
        print(f"[Info] Accuracy is improved from initial accuracy of \
              {init_test_accuracy} to final accuracy of {final_test_accuracy}.")

class TestAccuracy(unittest.TestCase):
    """Unit test for classwise accuracy testing"""

    def test_accuracy(self):
        """
        Train the model for 50 epochs and test accuracy
        """
        train_obj = TrainClassifier(model_name="resnet18", num_classes=2,
                           work_dir="workspace", node_name="neg_damage_classifier",
                           classes=["non_damage", "scopito_damage"], seed=42,
                           full_db_flag=False, image_size=(128, 64),
                           class_imbalance_method="upsampler",
                           learning_rate=0.0001, batch_size=16)

        for epoch in range(50):
            print(f"[Info] Epoch {epoch+1}/{50}\n")
            train_obj.train_classifier()
            val_accuracy = train_obj.validate_classifier()
            best_accuracy = torch.tensor(-np.inf)

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy

                print("[Info] Model Improved. Saving the Model...\n")
                train_obj.save_model()
        # loading best model
        train_obj.load_model()
        y_true, y_pred = [], []
        _ = train_obj.validate_classifier(
                y_true,
                y_pred
            )
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        assert acc > 0.95, f"Found Testing accuracy of {acc} \
            less than 95%"
        assert prec > 0.95, f"Found Testing accuracy of {prec} \
            less than 95%"
        assert recall > 0.95, f"Found Testing accuracy of {recall} \
            less than 95%"

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
        "-cim",
        "--class_imbalance_method",
        type=str,
        default="upsampler",
        help="class imbalance method, one of ['upsampler', 'class_weight']",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0001,
        help="learning_rate used for updating weights in backpropogation",
    )
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
    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        default=10,
        help="number of epochs for training"
    )
    args = parser.parse_args()
    train_obj = TrainClassifier(model_name=args.model_name, num_classes=args.num_classes,
                           work_dir=args.work_dir, node_name=args.node_name,
                           classes=args.classes, seed=args.seed,
                           full_db_flag=args.full_db_flag, image_size=args.image_size,
                           class_imbalance_method=args.class_imbalance_method,
                           learning_rate=args.learning_rate, batch_size=args.batch_size,
                           data_path=args.data_path, weight_path=args.weight_path)
    train_obj.run_traininig(args.epochs)
