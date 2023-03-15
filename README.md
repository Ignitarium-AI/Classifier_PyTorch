# Classifier (pytorch)

This repo contains the classifier code in pytorch for Training, Testing and Inferecing.

## Get started:

### Clone

`git clone https://gitlab.ignitarium.in/tyqi-platform/tyqi-model/experiments/classifier_pytorch.git`

### Project Organization

------------

    ├── dataloader                    <- Generating dataloader for train, val, and test
    ├── model.py                      <- Model Architecture source code
    ├── train.py                      <- Training code for given number of epochs
    ├── test.py            <- Testing code with accuracy, precision, recall, f1-score and confusion matrix
    ├── infer.py                      <- Inference code for images given folder name
    ├── README.md                     <- The top-level README for developers using this project.
    ├── requirements.txt              <- The requirements file for reproducing the analysis environment.
------------

## Dependencies

- `pip install -e .`

## (Optional) Virtual Environment Configuration

- `python3.8 -m venv venv3`
- `source venv3/bin/activate`
- `pip install -U pip`
- `pip install -e .`

### Data

Clone the dataset repo in `workspace/DB/git_DB_dummy/` folder. 

For example following command will clone the `cat_dog_classifier` repo at given path.

`git clone https://gitlab.ignitarium.in/tyqi-platform/tyqi-model/experiments/db.git -b cat_dog_classifier --depth=1 workspace/DB/cat_dog_classifier`

### Model

For training classifier model, following are the list of supported models

```["alexnet", "convnext_tiny", "convnext_small", "convnext_large", "densenet121", "densenet161", "densenet169", "densenet201", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "inception_v3", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "squeezenet1_0", "squeezenet1_1", "vgg16", "vgg19", "custom_model", "vit"]```

For training custom_model, modify `custom_model.py` and select custom_model from above list
### Training

To train the classifier, Run following command.

`python -m classifier.train --model_name "resnet18" --num_classes 2 --classes cat dog --work_dir "workspace" --seed 42 -lr 0.0001 -ep 50 --data_path <PATH OF FOLDER WHERE TRAIN/VAL/TEST DATASET IS PRESENT>`

The best checkpoint will be saved in `checkpoints` directory with `best.py` name.

*To handle the class imbalance problem, change `-cim`. To use classweight select `-cim=class_weight`, To upsample the imbalance data, select `-class_weight=upsampler` as. To disable the class imbalance issue, select `-cim=disable`*

### Testing

To test and evaluate the classifier, Run following command.

`python -m classifier.test --model_name "resnet18" --num_classes 2 --classes cat dog --work_dir "workspace" --data_path <PATH OF FOLDER WHERE TRAIN/VAL/TEST DATASET IS PRESENT> -t 0.8`

Above command will evaluate test dataloader with Precision, Recall, F1-score, and Confusion matrics. Here 0.8 is testing threshold, assertion is raised if achieved testing accuracy is less than 0.8.

### Inference

To perform the infernce on folder of images run following command.

`python -m classifier.infer --model_name "resnet18" --num_classes 2 --classes cat dog -ws "workspace" --folder_path <PATH OF FOLDER>`

To perform the infernce on single image run following command.

`python -m classifier.infer --model_name "resnet18" --num_classes 2 -ws "workspace" --img_path <PATH OF IMAGE>`