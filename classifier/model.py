# --------------------------------------------------------------------------
#                         Copyright © by
#           Ignitarium Technology Solutions Pvt. Ltd.
#                         All rights reserved.
#  This file contains confidential information that is proprietary to
#  Ignitarium Technology Solutions Pvt. Ltd. Distribution,
#  disclosure or reproduction of this file in part or whole is strictly
#  prohibited without prior written consent from Ignitarium.
# --------------------------------------------------------------------------
#  Filename    : model.py
# --------------------------------------------------------------------------
#  Description :
# --------------------------------------------------------------------------
"""
source code of model architecture
"""
import sys
from torchvision import models
from torch import nn
from classifier.custom_model import CustomModel
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

class GetModel: # pylint: disable=too-few-public-methods
    """
    Class to create model from model name and number of classes
    """
    def __init__(self, model_name, num_classes, labels = None, use_pretrained=True):
        """
        given model name and number of classes, this function return the required model
        """
        self.model_ft = None
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        if self.model_name == "custom_model":
            self.model_ft = CustomModel(num_classes=self.num_classes)
        elif self.model_name == "vit":
            checkpoint = "google/vit-base-patch16-224-in21k"
            label2id, id2label = dict(), dict()
            for i, label in enumerate(labels):
                label2id[label] = str(i)
                id2label[str(i)] = label
            self.model_ft = AutoModelForImageClassification.from_pretrained(
                checkpoint,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
            )
        else:
            self.model_ft = getattr(models, self.model_name)(pretrained=
                self.use_pretrained)

    def build(self):
        """
        build model from given model name
        Returns:
            self.model_ft: model architecture
            self.input_size: size of image recommended to train the model
        """
        if self.model_name in ("alexnet", "vgg16", "vgg19"):
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name in ("convnext_tiny", "convnext_small", "convnext_large"):
            num_ftrs = self.model_ft.classifier[2].in_features
            self.model_ft.classifier[2] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name in ("densenet121", "densenet161", "densenet169",
            "densenet201"):
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"
            "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
            "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2"):
            num_ftrs = self.model_ft.classifier[1].in_features
            self.model_ft.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "inception_v3":
            # Handle the auxilary net
            num_ftrs = self.model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name in ("mobilenet_v3_large", "mobilenet_v3_small"):
            num_ftrs = self.model_ft.classifier[3].in_features
            self.model_ft.classifier[3] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name in ("resnet18", "resnet50", "resnet101",
            "resnet152", "resnext50_32x4d", "resnext101_32x8d", "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"):
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name in ("squeezenet1_0", "squeezenet1_1"):
            num_ftrs = self.model_ft.classifier[1].in_channels
            self.model_ft.classifier[1] = nn.Conv2d(num_ftrs, self.num_classes,
                kernel_size=(1, 1), stride=(1, 1))
            self.model_ft.self.num_classes = self.num_classes

        elif self.model_name in ("custom_model", "vit"):
            pass

        else:
            print("Invalid model name, exiting...")
            sys.exit()

        return self.model_ft
