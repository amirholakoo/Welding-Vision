# U-Net Segmentation Pipeline Guide

This guide explains how to use the provided Python script to perform interactive video-based data extraction, train a U-Net segmentation model with MobileNetV2 as the encoder, and evaluate its performance.

---

## Overview

The script implements the following key functionalities:

- *Interactive Data Extraction:* Using OpenCV, users can extract Regions of Interest (ROI) and create ROI/mask image pairs by clicking on target areas.

- *Data Loading with tf.data:* On-the-fly data augmentation and efficient data loading using TensorFlow datasets.

- *U-Net Model:* A U-Net model with MobileNetV2 as the encoder is defined and trained.

- *Custom Loss and Metrics:* A combination of Binary Crossentropy (BCE) and Dice Loss is used for training, with True Positive as a custom metric.

- *Evaluation:* The script calculates the confusion matrix and predicts segmentation masks on test images.

---

## Directory Structure

Ensure the following directories are correctly set up before running the script:

.

├── dataset

│   ├── images  # ROI images

│   └── masks   # Corresponding binary masks

├── train_videos  # Training videos for ROI extraction

├── test_dataset

│   ├── images   # Test images

│   ├── masks    # Ground-truth masks

│   └── outputs  # Predicted masks

└── test_videos

---

## Step 1: Interactive Data Extraction (Optional)

Uncomment the following lines in the main() function to extract ROI/mask pairs from training videos:

```

# extractor = DataExtractor()

# extractor.extract()

```

*How it works:*

1\. The script loads each video and displays the first frame. Click once to define the ROI.

2\. In the ROI window, click on up to 7 target areas to sample colors for segmentation.

3\. The script will then process all frames, create masks, and save the ROI/mask pairs in the dataset directory.

---

## Step 2: Training the U-Net Model

Uncomment the following lines in the main() function to train the U-Net model:

```

# if not os.path.exists("best_unet_segmentation.h5"):

#     model = train_save_unet_model()

# else:

#     model = load_trained_model()

```

*Training Process:*

- The dataset is split into training and validation sets (80/20 by default).

- The encoder (MobileNetV2) is initially frozen and later fine-tuned.

- The model is trained with the Combined Loss (BCE + Dice Loss) and True Positive as a metric.

- Training progress and metrics are logged using TensorBoard.

---

## Step 3: Predicting Segmentation Masks

To predict segmentation masks on test images, call the evaluation_save() function:

```

# evaluation_save(model)

```

This function will save predicted masks in the test_dataset/outputs directory.

---

## Step 4: Calculating Confusion Matrix

To compute and display the confusion matrix, uncomment the following line:

```

# calculate_confusion_matrix()

```

The confusion matrix compares predicted masks against ground-truth masks in the test_dataset/masks directory.

---

## Step 5: Calculating ROI Length

To calculate the ROI length for each predicted mask, the script reads the masks in test_dataset/outputs and prints the length:

```

pred_mask_dir = './test_dataset/outputs'

for pred_mask_path in os.listdir(pred_mask_dir):

    length = calculate_y_length(os.path.join(pred_mask_dir, pred_mask_path))

    print(f'Length of ROI {pred_mask_path}: {length}')

```

This function measures the number of non-zero pixels in a specific ROI region and scales it to a percentage.

---

## Notes

- Ensure OpenCV, TensorFlow, Matplotlib, and Seaborn are installed.

- Use TensorBoard for monitoring training progress: tensorboard --logdir=./logs

---

## Summary of Key Constants

- *TO_CLICK:* Number of clicks for color sampling (default: 7)

- *ROI_SIZE:* Size of ROI in pixels (width, height)

- *IMG_SIZE:* Image size after resizing (default: 224x224)

- *BATCH_SIZE:* Batch size for training (default: 32)

- *EPOCHS:* Number of training epochs (default: 10)

---

This guide should help you quickly understand and use the U-Net segmentation pipeline.
