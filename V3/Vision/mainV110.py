#!/usr/bin/env python
"""
Clean Segmentation Pipeline

This script implements:
  • An interactive video-based data extraction module (using OpenCV) for creating ROI/mask image pairs.
  • A tf.data-based training pipeline with on-the-fly augmentation.
  • A U-Net segmentation model that uses MobileNetV2 as the encoder.
  • Custom loss (BCE+Dice) and metric definitions.

Uncomment the data extraction section in main() if you need to create a new dataset.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

# =============================================================================
# Global Constants and Settings
# =============================================================================

# Extraction & ROI parameters
TO_CLICK = 7
ROI_SIZE = (120, 8)  # (width, height)
COLOR_TOLERANCE = 6
IMG_SIZE = 224

# Directories (ensure these exist or will be created)
IMAGE_DIR = "./dataset/images/"
MASK_DIR = "./dataset/masks/"
VIDEO_DIR = "./train_videos/"
TEST_DIR = "./test_dataset/"
TEST_VIDEO_DIR = "./test_videos/"
# Training parameters
BATCH_SIZE = 32
EPOCHS = 10

# =============================================================================
# Video and ROI Extraction Module
# =============================================================================

def find_video_paths(video_dir: str) -> list[str]:
    """Return a list of video file paths from the given directory."""
    video_exts = ('.mp4', '.avi', '.mov')
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(video_dir)
        for file in files if file.lower().endswith(video_exts)
    ]

def read_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file and return its capture object."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None
    return cap

def read_next_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray]:
    """Return the next frame from a video capture object."""
    ret, frame = cap.read()
    return (ret, frame) if ret else (False, None)

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to improve the contrast of the input frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def create_mask(roi: np.ndarray,
                lower_bounds: np.ndarray,
                upper_bounds: np.ndarray,
                min_area: int = 10,
                distance_threshold: int = 20) -> np.ndarray:
    """
    Create a mask for the region of interest (ROI) based on specified color intervals.
    Applies morphological operations and merges nearby contours.
    """
    mask = cv2.inRange(roi, lower_bounds[0], upper_bounds[0])
    for lower, upper in zip(lower_bounds[1:], upper_bounds[1:]):
        mask_ = cv2.inRange(roi, lower, upper)
        mask = cv2.bitwise_or(mask, mask_)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    merged_contours = []
    while filtered_contours:
        base = filtered_contours.pop(0)
        base_points = base.reshape(-1, 2)
        to_merge = []
        for cnt in filtered_contours:
            M1, M2 = cv2.moments(base), cv2.moments(cnt)
            if M1["m00"] and M2["m00"]:
                cx1, cy1 = M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]
                cx2, cy2 = M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]
                if np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2) < distance_threshold:
                    to_merge.append(cnt)
        if to_merge:
            all_points = np.vstack([base_points] + [cnt.reshape(-1, 2) for cnt in to_merge])
            merged_contours.append(cv2.convexHull(all_points))
            filtered_contours = [
                cnt for cnt in filtered_contours
                if not any(np.array_equal(cnt, merged) for merged in to_merge)
            ]
        else:
            merged_contours.append(base)
    
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, merged_contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask

def find_target_intervals(target_colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lower and upper color bounds based on target colors with an added tolerance.
    """
    lower_bounds, upper_bounds = [], []
    for color in target_colors:
        lower = np.clip(color - COLOR_TOLERANCE, 0, 255).astype(np.uint8)
        upper = np.clip(color + COLOR_TOLERANCE, 0, 255).astype(np.uint8)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    return np.array(lower_bounds), np.array(upper_bounds)

def save_data(video_dir: str, roi: np.ndarray, mask: np.ndarray, index: int) -> None:
    """Save the ROI and corresponding mask images to disk."""
    image_dir = os.path.join(video_dir, "images")
    mask_dir = os.path.join(video_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    filename = f"{index}.png"
    cv2.imwrite(os.path.join(image_dir, filename), roi)
    cv2.imwrite(os.path.join(mask_dir, filename), mask)

class DataExtractor:
    """
    Extracts ROI/mask pairs from video files via interactive clicks.
    
    The first click selects the ROI, and subsequent clicks within the ROI sample
    target colors to define the segmentation.
    """
    def __init__(self,
                 video_dir: str = VIDEO_DIR,
                 to_click: int = TO_CLICK,
                 roi_size: tuple = ROI_SIZE,
                 img_size: int = IMG_SIZE) -> None:
        self.video_dir = video_dir
        self.to_click = to_click
        self.roi_width, self.roi_height = roi_size
        self.img_size = img_size
        self.click_coordinates: list[tuple[int, int]] = []
        self.target_colors = np.empty((0, 3), dtype=np.uint8)
    
    def mouse_callback(self, event: int, x: int, y: int, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_coordinates.append((x, y))
    
    def extract(self) -> None:
        index = 0
        video_paths = find_video_paths(self.video_dir)
        if not video_paths:
            print("No videos found in the directory.")
            return
        
        for video_path in video_paths:
            cap = read_video(video_path)
            if cap is None:
                continue
            
            is_first_frame = True
            while True:
                ret, frame = read_next_frame(cap)
                if not ret:
                    break
                
                frame = apply_clahe(frame)
                frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_CUBIC)
                
                if is_first_frame:
                    # Select ROI (one click)
                    cv2.imshow("Select ROI", frame)
                    cv2.setMouseCallback("Select ROI", self.mouse_callback)
                    while len(self.click_coordinates) < 1:
                        cv2.waitKey(1)
                    cv2.destroyWindow("Select ROI")
                    x, y = self.click_coordinates[0]
                    roi = frame[max(0, y - self.roi_height): y + self.roi_height,
                                max(0, x - self.roi_width): x + self.roi_width]
                    roi = cv2.resize(roi, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                    
                    # Sample target colors within ROI.
                    cv2.imshow("ROI", roi)
                    cv2.setMouseCallback("ROI", self.mouse_callback)
                    while len(self.click_coordinates) < self.to_click:
                        cv2.waitKey(1)
                    cv2.destroyWindow("ROI")
                    
                    for (x_click, y_click) in self.click_coordinates:
                        patch = roi[max(0, y_click - 2): y_click + 2,
                                    max(0, x_click - 2): x_click + 2]
                        patch_colors = patch.reshape(-1, 3)
                        if self.target_colors.size == 0:
                            self.target_colors = patch_colors
                        else:
                            self.target_colors = np.concatenate([self.target_colors, patch_colors], axis=0)
                    
                    lower_bounds, upper_bounds = find_target_intervals(self.target_colors)
                    self.click_coordinates.clear()
                    is_first_frame = False
                
                # Process the current frame with the selected ROI and color intervals.
                mask = create_mask(roi, lower_bounds, upper_bounds)
                save_data(self.video_dir, roi, mask, index)
                index += 1
                
                # Recompute the ROI in the current frame using the same coordinates.
                roi = frame[max(0, y - self.roi_height): y + self.roi_height,
                            max(0, x - self.roi_width): x + self.roi_width]
                roi = cv2.resize(roi, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            cap.release()
        cv2.destroyAllWindows()

# =============================================================================
# tf.data-based Data Loading Module
# =============================================================================

def load_image(path: str,
               target_size: tuple = (IMG_SIZE, IMG_SIZE),
               is_mask: bool = False) -> tf.Tensor:
    """
    Load and preprocess an image from file.
    Returns a tensor with values in [0, 1].
    """
    image = tf.io.read_file(path)
    channels = 1 if is_mask else 3
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_image_mask(image_path: str, mask_path: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Load an image and its corresponding mask."""
    image = load_image(image_path, is_mask=False)
    mask = load_image(mask_path, is_mask=True)
    return image, mask

def augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply random augmentation (flips and 90° rotations) to image and mask."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    return image, mask

def get_train_val_datasets(image_dir: str,
                           mask_dir: str,
                           batch_size: int = BATCH_SIZE,
                           val_split: float = 0.2) -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Create training and validation datasets from the image and mask directories."""
    image_files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith(".png")
    ])
    mask_files = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir) if f.endswith(".png")
    ])
    
    total = len(image_files)
    split_index = int(total * (1 - val_split))
    train_img_files = image_files[:split_index]
    train_mask_files = mask_files[:split_index]
    val_img_files = image_files[split_index:]
    val_mask_files = mask_files[split_index:]
    
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_files, dtype=tf.string), 
                                               tf.constant(train_mask_files, dtype=tf.string)))
    train_ds = train_ds.map(lambda img, msk: load_image_mask(img, msk),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(100).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_img_files, dtype=tf.string), 
                                             tf.constant(val_mask_files, dtype=tf.string)))
    val_ds = val_ds.map(lambda img, msk: load_image_mask(img, msk),
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, len(train_img_files)

# =============================================================================
# U-Net Model Definition
# =============================================================================

def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
               l2_lambda=1e-4,
               dropout_rate=0.5) -> Model:
    """
    Build a U-Net model using MobileNetV2 as the encoder.
    
    Encoder layers used for skip connections:
      • block_1_expand_relu   (low-level features)
      • block_3_expand_relu   (mid-level features)
      • block_6_expand_relu   (higher-level features)
      • block_13_expand_relu  (deeper features)
      • block_16_project      (final encoder output)
    """
    inputs = layers.Input(shape=input_shape)
    # Specify a name for the base model so it can be retrieved later.
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        name="mobilenetv2_base"
    )
    base_model.trainable = False  # Freeze encoder initially

    # Extract the designated encoder layers as skip connections.
    skip1 = base_model.get_layer("block_1_expand_relu").output    # low-level features
    skip2 = base_model.get_layer("block_3_expand_relu").output    # mid-level features
    skip3 = base_model.get_layer("block_6_expand_relu").output    # higher-level features
    skip4 = base_model.get_layer("block_13_expand_relu").output   # deeper features
    encoder_output = base_model.get_layer("block_16_project").output  # final encoder output

    def upconv_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
        """Upsampling block with transposed convolution, skip connection, and refinement."""
        x = layers.Conv2DTranspose(filters, (3, 3), strides=2, padding="same",
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Build the decoder path using the skip connections.
    x = upconv_block(encoder_output, skip4, 192)
    x = upconv_block(x, skip3, 144)
    x = upconv_block(x, skip2, 96)
    x = upconv_block(x, skip1, 64)
    x = upconv_block(x, None, 32)  # Final upsampling without a skip connection

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid",
                            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
    model = Model(inputs, outputs)
    return model


# =============================================================================
# Loss and Metrics
# =============================================================================

class CombinedLoss(tf.keras.losses.Loss):
    """
    Custom loss combining binary crossentropy and Dice loss.
    """
    def __init__(self, smooth: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coeff = (2. * intersection + self.smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
        dice_loss = 1 - dice_coeff
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce + dice_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

def true_positive(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute the true positive rate."""
    y_true_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
    tp = tf.reduce_sum(y_true_pos * y_pred_pos)
    total_true = tf.reduce_sum(y_true_pos)
    return tp / (total_true + tf.keras.backend.epsilon())

# =============================================================================
# Training and Prediction Pipeline
# =============================================================================

def train_save_unet_model() -> Model:
    """Train the U-Net model and save the best and final versions."""
    best_model_path = 'best_unet_segmentation.h5'
    
    train_ds, val_ds, train_size = get_train_val_datasets(IMAGE_DIR, MASK_DIR,
                                                          batch_size=BATCH_SIZE, val_split=0.2)
    steps_per_epoch = train_size // BATCH_SIZE
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=2, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss",
                                                    save_best_only=True, verbose=1)
    tensorboard_cb = TensorBoard(log_dir='./logs', histogram_freq=1)

    if os.path.exists(best_model_path):
        print(f"Loading pretrained model from '{best_model_path}'...")
        custom_objects = {"CombinedLoss": CombinedLoss, "true_positive": true_positive}
        model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)
    else:
        model = build_unet()
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss=CombinedLoss(),
                      metrics=[true_positive])
        model.summary()
        print("Starting initial training...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_cb]
        )

    # Fine-tuning: unfreeze the MobileNetV2 encoder.
    print("Unfreezing encoder for fine-tuning...")
    model.trainable = True

    # Recompile with a lower learning rate for fine-tuning.
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss=CombinedLoss(),
                  metrics=[true_positive])
    print("Starting fine-tuning...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_cb]
    )

    model.save(best_model_path)
    print(f"Final model saved to '{best_model_path}'")
    return model


def load_trained_model(model_path: str = "best_unet_segmentation.h5") -> Model:
    """Load a trained U-Net model from the specified path."""
    custom_objects = {"CombinedLoss": CombinedLoss, "true_positive": true_positive}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model loaded from '{model_path}'")
    return model

def predict_mask(model: Model, image_path: str) -> np.ndarray:
    """Predict a segmentation mask for the given image and display it."""
    image_path = str(image_path)
    image = load_image(image_path, is_mask=False)
    image = tf.expand_dims(image, axis=0)
    pred_mask = model.predict(image)[0]

    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    return pred_mask

def evaluation_save(model: Model) -> None:
    for image_path in os.listdir(os.path.join(TEST_DIR, 'images')):
        read_path = os.path.join(TEST_DIR, 'images', image_path)
        pred_mask = predict_mask(model, read_path)
        cv2.imwrite(os.path.join(TEST_DIR, 'outputs', image_path), pred_mask)

def calculate_y_length(pred_mask_path: str) -> float:
    pred_mask = cv2.imread(pred_mask_path)
    
    roi = pred_mask[100:140, :]
    return np.count_nonzero(roi) * 100 / (224 * 40)

def calculate_confusion_matrix() -> None:
    """
    Compute and display the confusion matrix for all predicted masks in the test dataset.
    """
    y_true = []
    y_pred = []

    test_mask_dir = os.path.join(TEST_DIR, 'masks')
    test_output_dir = os.path.join(TEST_DIR, 'outputs')
    
    mask_paths = sorted([os.path.join(test_mask_dir, f) for f in os.listdir(test_mask_dir) if f.endswith(".png")])
    output_paths = sorted([os.path.join(test_output_dir, f) for f in os.listdir(test_output_dir) if f.endswith(".png")])

    
    for output_path, mask_path in zip(output_paths, mask_paths):
        gt_mask = load_image(mask_path, is_mask=True).numpy().flatten()
        pred_mask = load_image(output_path, is_mask=True).numpy().flatten()
        
        y_true.extend(gt_mask)
        y_pred.extend(pred_mask)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Background', 'Object'], yticklabels=['Background', 'Object'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Segmentation Model")
    plt.show()

# =============================================================================
# Main Function
# =============================================================================

def main():
    # Uncomment the following lines if you wish to perform data extraction.
    # extractor = DataExtractor()
    # extractor.extract()
    
    # if not os.path.exists("best_unet_segmentation.h5"):
    #     model = train_save_unet_model()
    # else:
    #     model = load_trained_model()

    pred_mask_dir = '/Users/sepehrmasoudizad/Desktop/Z/Mozaffar/V3/test_dataset/outputs'
    for pred_mask_path in os.listdir(pred_mask_dir):
        lenght = calculate_y_length(os.path.join(pred_mask_dir, pred_mask_path))
        print(f'Lenght of the roi {pred_mask_path}: {lenght}')

    # calculate_confusion_matrix()

if __name__ == "__main__":
    main()
