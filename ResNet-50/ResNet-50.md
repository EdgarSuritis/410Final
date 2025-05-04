---
title: Training the ResNet-50 + Binary Classifier Model

---

# Training the ResNet-50 + Binary Classifier Model


1. Install all required packages. This can be done manually, or via a Conda environment with the `environment.yaml` file.
2. Replace the directory path under the "Dataloader And Pretrained Model Import" section with your general project folder. This folder should contain a "JPEGImage" folder with all of the image files, as well as "TrainTestSplits" folder with the `train.csv` and `test.csv` files which denote which images will be used for training and testing.
3. Run all cells in the "Dataloader And Pretrained Model Import" section.
4. Adjust the hyperparameters to training section within the main training loop. Run the main training loop. To re-create the model described in the results section the following hyperparameters were used.
> Train with lr = 0.0001 for 9 epochs
> Train with lr = 0.0005 for 20 epochs
> Unfreeze backbone weights
> Train with lr = 0.0001 for 5 epochs

5. Run the first cell in the Testing section to get the model's predictions for the testing dataset, and print out model performance metrics.
6. If a confusion matrix for the model is desired, enter the FP, FN, TP, and TN values into the Visualization cell and run it.

---

# Results:

This model was able to achieve an F1 score of 0.80.

> Precision: 0.9715909090909091
> Recall: 0.6778029445073612
> F1: 0.7985323549032688

