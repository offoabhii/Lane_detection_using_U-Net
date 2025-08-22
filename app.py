import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tempfile
import os

# ========== CONFIG ==========
# These must match the settings used during training
MODEL_PATH = "unet_road_lane_best.h5"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256

# ========== MODEL DEFINITION & LOADING ==========
# The model architecture must be redefined to load the weights.
# This entire block is copied from your training/prediction scripts.
def conv_block(input_tensor, num_filters):
    x = keras.layers.Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = keras.layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_features, num_filters):
    x = keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = keras.layers.concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = keras.layers.Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    return keras.Model(inputs, outputs, name="U-Net")

# Load the model ONCE when the script starts. This is a critical optimization.
try:
    model = build_unet((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded successfully from disk.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ========== CORE PREDICTION FUNCTION ==========
def detect_lanes_on_frame(bgr_image, threshold=0.5):
    """
    Takes a single BGR image (from OpenCV) and returns an image with the lane overlay.
    This is the core logic that will be used by all UI components.
    """
    if model is None:
        # If the model failed to load, return the original image with an error message
        cv2.putText(bgr_image, "MODEL FAILED TO LOAD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return bgr_image

    original_height, original_width, _ = bgr_image.shape
    
    # 1. Preprocess the image
    img_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # 2. Predict
    predicted_mask = model.predict(img_batch, verbose=0)[0]
    
    # 3. Post-process the mask
    binary_mask = (predicted_mask > threshold).astype(np.uint8)
    full_size_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # 4. Create overlay and combine
    lane_overlay = np.zeros_like(bgr_image)
    # The mask might have a channel dimension, so we handle that
    if full_size_mask.ndim == 3:
        full_size_mask = full_size_mask[:, :, 0]
    lane_overlay[full_size_mask == 1] = [0, 255, 0] # Green in BGR format
    
    result_image = cv2.addWeighted(bgr_image, 1, lane_overlay, 0.4, 0)
    
    return result_image

# ========== GRADIO PROCESSING FUNCTIONS ==========
# These functions act as wrappers between the Gradio UI and our core logic.

def process_image(input_image):
    """For the Image tab"""
    # Gradio provides images as RGB numpy arrays. Convert to BGR for OpenCV.
    bgr_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    output_bgr = detect_lanes_on_frame(bgr_image)
    # Convert back to RGB for Gradio display.
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    return output_rgb

def process_video(input_video_path):
    """For the Video tab"""
    # Create a temporary file to save the processed video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        output_video_path = temp_file.name

    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_lanes_on_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    return output_video_path

# ========== GRADIO UI LAYOUT ==========
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚘 Real-Time Lane Detection UI")
    gr.Markdown("An interactive application to detect road lanes using a U-Net model. Created by a World-Class Machine Learning Engineer.")

    with gr.Tabs():
        # --- Image Tab ---
        with gr.TabItem("🖼️ Image Detection"):
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Upload an Image")
                image_output = gr.Image(type="numpy", label="Detection Result")
            image_button = gr.Button("Detect Lanes")

        # --- Video Tab ---
        with gr.TabItem("🎬 Video File Detection"):
            with gr.Row():
                video_input = gr.Video(label="Upload a Video")
                video_output = gr.Video(label="Detection Result")
            video_button = gr.Button("Process Video")

        # --- Webcam Tab ---
        with gr.TabItem("📹 Live Webcam Feed"):
            with gr.Row():
                # The 'streaming=True' makes this a live feed
                # CORRECTED: Changed 'source' to 'sources' and the value to a list.
                webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input")
                webcam_output = gr.Image(label="Live Detection")

    # --- Linking Functions to UI Components ---
    image_button.click(fn=process_image, inputs=image_input, outputs=image_output)
    video_button.click(fn=process_video, inputs=video_input, outputs=video_output)
    
    # For the webcam, we use .stream() which runs continuously
    # The `input` to the function is the component that triggers it (webcam_input)
    webcam_input.stream(fn=process_image, inputs=webcam_input, outputs=webcam_output)

# ========== LAUNCH THE APP ==========
if __name__ == "__main__":
    # demo.launch(share=True) # Use share=True to create a public link
    demo.launch()
