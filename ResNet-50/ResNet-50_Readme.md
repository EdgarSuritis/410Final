# Training the ResNet-50 + Binary Classifier Model


1. Ensure you have followed all instructions in the general project readme file (including downloading and unzipping the dataset) before continuing with these directions.
2. Download the ResNet-50 folder that contains this readme file.
3.  Install all required packages. This can be done manually, or via a Conda environment with the `environment.yaml` file.
4. Replace the directory path under the "Dataloader And Pretrained Model Import" section with your ResNet-50 project folder that you downloaded above. This folder should contain a "JPEGImage" folder with all of the image files you unpacked in the general project directions, as well as the "TrainTestSplits" folder with the `train.csv` and `test.csv` files which denote which images will be used for training and testing.
5. Run all cells in the "Dataloader And Pretrained Model Import" section.
6. Adjust the hyperparameters in the training section within the main training loop. Run the main training loop as many times as desired. To re-create the model described in the results section the following hyperparameters were used.
> Train with lr = 0.0001 for 9 epochs
> Train with lr = 0.0005 for 20 epochs
> Unfreeze backbone weights
> Train with lr = 0.0001 for 5 epochs

5. Run the first cell in the Testing section to get the model's predictions for the testing dataset, and print out model performance metrics.
6. If a confusion matrix for the model is desired, enter the FP, FN, TP, and TN values into the Visualization cell and run it.

---

# Results:

This model was able to achieve an F1 score of 0.80 with the training regimine described above.

> Precision: 0.9715909090909091
> Recall: 0.6778029445073612
> F1: 0.7985323549032688

