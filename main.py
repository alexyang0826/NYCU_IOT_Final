import cv2
import numpy as np
import os
import sys
import IOT.send as send

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras

pose_sample_rpi_path = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'raspberry_pi'

sys.path.append(pose_sample_rpi_path)

# Load MoveNet Thunder model
import utils
from data import BodyPart
from ml import Movenet

movenet = Movenet("./weights/movenet_thunder.tflite")


def detect(input_numpy, inference_count=3):
    # Detect pose using the full input image
    movenet.detect(input_numpy, reset_crop_region=True)

    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_numpy, reset_crop_region=False)

    return person


# Functions to visualize the pose estimation results.
def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True, keep_input_size=False
):
    # Draw the detection result on top of the image.
    image_np = utils.visualize(image, [person])

    # Plot the image with detection results.
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # im = ax.imshow(image_np)

    if close_figure:
        plt.close(fig)

    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

    return image_np


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""

    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """
    Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(
        landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
    )

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)

    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(
        pose_center_new, [tf.size(landmarks) // (17 * 2), 17, 2]
    )

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0, name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

    return pose_size


def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
    """
    # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, [tf.size(landmarks) // (17 * 2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size

    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)

    return embedding


class_names = ["crossleg", "forwardhead", "standard"]

# Define the model
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)
model.load_weights("./weights/weights.best.hdf5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)

valid_image_count = 0
current_type = -1
counter = 1
while True:
    ret, img = cap.read()

    # h, w, c = img.shape
    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    person = detect(img)

    # Save landmarks if all landmarks were detected
    min_landmark_score = min([keypoint.score for keypoint in person.keypoints])

    if min_landmark_score < 0.1:
        cv2.imshow("live", img)
        cv2.waitKey(1)
        continue

    # Draw the prediction result on top of the image for debugging later
    output_overlay = draw_prediction_on_image(
        img.astype(np.uint8), person, close_figure=True, keep_input_size=True
    )

    # Write detection result into an image file
    # output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)

    cv2.imshow("live", output_overlay)
    cv2.waitKey(1)

    # Get landmarks and scale it to the same size as the input image
    pose_landmarks = np.array(
        [
            [keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
            for keypoint in person.keypoints
        ],
        dtype=np.float32,
    )

    # Write the landmark coordinates to its per-class CSV file
    input_img = [pose_landmarks.flatten().astype("float32")]

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, input_img)
    interpreter.invoke()
    output = interpreter.tensor(output_index)
    predicted_label = np.argmax(output()[0])
    print(predicted_label)

    if current_type == predicted_label:
        valid_image_count += 1
    else:
        current_type = predicted_label
        valid_image_count = 0

    if valid_image_count > 30:
        print("Current type: " + class_names[current_type] + ' Counter:' , counter)
        if current_type < 2:
            counter += 1
        else:
            counter = 1

    # if counter > 500:
    #     counter = 1
    #     send.send_data(int(1))
    #     send.send_data(int(1))
    #     send.send_data(int(1))
    #     send.send_data(int(1))
    #     send.send_data(int(1))
    #     print("Send 1-----------------------------------")

    if counter % 150 == 0:
        send.send_data(int(0))
        print("Send 0------------------------------------")

