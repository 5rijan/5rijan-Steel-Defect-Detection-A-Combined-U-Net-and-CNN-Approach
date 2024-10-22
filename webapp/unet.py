import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
from tensorflow.keras.models import load_model, Model


# Load model and define layers
model = load_model('unet_model_full.h5')
# Load the U-Net model
model_path = 'unet_model_full.h5'
model = load_model(model_path)

# Define layer names for extracting intermediate outputs
layer_names = ['conv2d', 'conv2d_1', 'max_pooling2d', 
               'conv2d_2', 'conv2d_3', 'conv2d_transpose', 
               'concatenate', 'conv2d_4', 'conv2d_5', 'conv2d_6']

# Create models for each intermediate layer
intermediate_layer_models = [Model(inputs=model.input, outputs=model.get_layer(name).output) 
                             for name in layer_names]

# Function to get intermediate outputs
def get_intermediate_outputs(image):
    outputs = [layer_model.predict(image) for layer_model in intermediate_layer_models]
    return outputs

# Function to convert image to grayscale and resize
def convert_to_grayscale_and_resize(image_path, new_size=(625, 100)):
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        img_resized = img_gray.resize(new_size, Image.Resampling.LANCZOS)
        image_array = np.array(img_resized)
        return image_array

# Function to generate a mask
def mask_generator(tags, image_path):
    en_pix = tags.split()
    rle = list(map(int, en_pix))
    pixel = [rle[i] for i in range(0, len(rle), 2)]
    pixel_count = [rle[i] for i in range(1, len(rle), 2)]

    rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(len(pixel))]
    rle_mask_pixels = sum(rle_pixels, [])

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    l, b, _ = img.shape
    max_index = l * b - 1
    rle_mask_pixels = [min(pixel, max_index) for pixel in rle_mask_pixels]
    mask_img = np.zeros((l * b,), dtype=np.uint8)

    mask_img[rle_mask_pixels] = 1
    mask = mask_img.reshape((b, l)).T

    # Scale mask to [0, 255] for better visualization
    mask = (mask * 255).astype(np.uint8)

    new_size = (625, 100)
    resized_array = cv2.resize(mask, new_size)
    return resized_array

# Load CSV data
CSV_PATH = "../Datasets/Severstal steel defect detection/train.csv"
data = pd.read_csv(CSV_PATH, header=None)

# Get image paths and labels
image_ids = data[0][1:].values
image_paths = [f"../Datasets/Severstal steel defect detection/train_images/{image_id}" for image_id in image_ids]

damage_data = data[2].values

# Streamlit app
st.title("Steel Defect Detection: U-Net Training Process")

# Model Details
st.sidebar.subheader("Model Details")
with st.sidebar.expander("Show Model Summary"):
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

# Explanation section
st.sidebar.subheader("U-Net")
st.sidebar.write("This application uses a U-Net model for semantic segmentation of steel defects. The model predicts defect locations in the input image.")
st.sidebar.write("Steps involved:")
st.sidebar.markdown("""
1. **Input Image Loading:** The selected image is loaded and preprocessed.
2. **Mask Generation:** A ground truth mask is generated based on the defect data.
3. **Model Prediction:** The preprocessed image is fed into the trained model to obtain predictions.
4. **Visualization:** The predicted and ground truth masks are visualized along with the intermediate U-Net outputs.
""")

# U-Net Details
st.sidebar.subheader("How U-Net Works")
st.sidebar.write("U-Net is a convolutional neural network architecture designed for semantic segmentation, primarily in biomedical image analysis but widely applicable in other domains such as industrial defect detection.")
st.sidebar.markdown("""
### Architecture Overview
U-Net consists of two main parts: a **contracting path** (encoder) that captures context and a **expansive path** (decoder) that enables precise localization.

### Contracting Path:
- **Convolutional Layers:** Each layer applies convolution operations to extract features from the input. The output of the convolution can be expressed mathematically as:
  
  \[
  \text{Output} = f(W * I + b)
  \]
  
  Where:
  - \( W \) is the filter (kernel),
  - \( I \) is the input image,
  - \( b \) is the bias,
  - \( f \) is the activation function, typically ReLU.

- **Max Pooling:** Downsample the feature maps to reduce spatial dimensions while retaining important features. The pooling operation can be defined as:
  
  \[
  \text{Output} = \max(\text{window})
  \]

### Expansive Path:
- **Transposed Convolutions:** Upsample the feature maps to increase spatial dimensions. This can be mathematically described as reversing the effect of a convolution operation:
  
  \[
  \text{Output}(i,j) = \sum_{m,n} \text{Input}(m,n) \cdot W(i-m,j-n)
  \]
  
  Where \( W \) is the filter and \( (i,j) \) are the output indices.

- **Concatenation:** Combine feature maps from the contracting path to retain spatial information lost during downsampling. This allows the network to leverage both high-level features and fine-grained details.

### Mathematical Details:
- **Convolution:** The output dimension after convolution can be calculated as:

  \[
  \text{Output Size} = \frac{(W - F + 2P)}{S} + 1
  \]

  Where:
  - \( W \): Input volume size,
  - \( F \): Filter size,
  - \( P \): Padding,
  - \( S \): Stride.

- **Pooling:** Reduces spatial dimensions by taking the maximum or average value in each window. For Max Pooling:
  
  \[
  \text{Output}(i,j) = \max_{k,l} \text{Input}(i+k, j+l)
  \]

- **Activation Function:** The ReLU activation function is defined as:
  
  \[
  f(x) = \max(0, x)
  \]

### Key Features of U-Net:
- **Skip Connections:** These are crucial in U-Net, as they allow gradients to flow easily during training, mitigating the vanishing gradient problem.
- **Symmetrical Architecture:** The encoder and decoder paths are symmetrical, which facilitates feature extraction and localization.

### References for Further Reading:
- Ronneberger, O., Fischer, P., & Becker, A. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.
- Isensee, F., et al. (2018). n-D U-Net: Deep Learning for Medical Image Segmentation. *arXiv preprint arXiv:1809.10486*.
- Çiçek, Ö., et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.

By leveraging the U-Net architecture, this application effectively identifies and segments defects in steel images, showcasing the model's powerful capabilities for pixel-wise classification.
""")
# Select an image from the list
selected_image_id = st.selectbox("Choose an image", image_ids)
selected_image_path = f"../Datasets/Severstal steel defect detection/train_images/{selected_image_id}"
selected_damage = data[data[0] == selected_image_id][2].values[0]

# Display the selected image
if os.path.exists(selected_image_path):
    st.image(selected_image_path, caption="Selected Image", use_column_width=True)
    input_image = convert_to_grayscale_and_resize(selected_image_path)
    
    # Normalize the input image for prediction
    sample_input = np.expand_dims(input_image / 255.0, axis=0)

    # Get the intermediate outputs
    intermediate_outputs = get_intermediate_outputs(sample_input)

    # Display the intermediate steps with a slider
    st.subheader("U-Net Intermediate Outputs")
    layer_index = st.slider("", min_value=0, max_value=len(layer_names) - 1, step=1)

    # Get the selected layer's output
    step_image = intermediate_outputs[layer_index].squeeze()

    # If the output has multiple channels, reduce it to a single channel
    if step_image.ndim == 3 and step_image.shape[2] > 1:
        # Option 1: Display a single channel (e.g., the first channel)
        step_image = step_image[:, :, 0]
        
        # Option 2: Alternatively, you can take the mean across all channels
        # step_image = np.mean(step_image, axis=-1)

    # Normalize and scale to [0, 255] for display
    step_image = (step_image - step_image.min()) / (step_image.max() - step_image.min())
    step_image_display = (step_image * 255).astype(np.uint8)

    # Display the processed image
    st.image(step_image_display, caption=f"Step {layer_index + 1}: {layer_names[layer_index]}", use_column_width=True)

    # Display details of the current layer
    st.markdown(f"**Layer {layer_index + 1} Details:**")
    st.markdown(f"- **Layer Name:** {layer_names[layer_index]}")
    st.markdown("- **Layer Role:**" + (" Contracting Path" if layer_index < 5 else " Expansive Path"))

    # Generate and display the ground truth mask
    mask = mask_generator(selected_damage, selected_image_path)
    st.subheader("Ground Truth vs Model Prediction")
    col1, col2 = st.columns(2)
    col1.image(mask, caption="Ground Truth Mask", use_column_width=True)

    # Make predictions using the model
    prediction = model.predict(sample_input)
    binary_prediction = (prediction > 0.3).astype(np.uint8).squeeze()

    # Resize prediction and mask to match the input image
    if binary_prediction.shape != input_image.shape:
        binary_prediction = cv2.resize(binary_prediction, (input_image.shape[1], input_image.shape[0]))

    col2.image(binary_prediction * 255, caption="Binary Prediction", use_column_width=True)

    # Overlay masks on the original image
    st.subheader("Overlay Masks on Original Image")
    original_image = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)
    original_image_resized = cv2.resize(original_image, (625, 100))

    # Create overlay images
    overlay_ground_truth = cv2.addWeighted(original_image_resized, 0.7, mask, 0.3, 0)
    overlay_prediction = cv2.addWeighted(original_image_resized, 0.7, binary_prediction * 255, 0.3, 0)

    col1, col2 = st.columns(2)
    col1.image(overlay_ground_truth, caption="Overlay Ground Truth", use_column_width=True)
    col2.image(overlay_prediction, caption="Overlay Prediction", use_column_width=True)



# Header and introduction
st.markdown("""
### Introduction
This app demonstrates the training process for a U-Net model to detect defects in steel. The workflow includes pre-processing the dataset, creating a U-Net model, and training the model. Below, we provide a detailed explanation of each step with relevant code snippets and visualizations.
""")

# Display and explain the code snippets
st.markdown("#### Step 1: Preprocessing the Dataset")
st.markdown("""
The first step involves converting images to grayscale and resizing them to a fixed size for consistent input to the model. This helps in normalizing the data and simplifying the input format.
""")

# Code snippet for converting to grayscale and resizing
code_snippet_1 = """
from PIL import Image
import numpy as np

def convert_to_grayscale_and_resize(image_path, new_size=(625, 100)):
    with Image.open(image_path) as img:
        
        # Convert the image to grayscale
        img_gray = img.convert("L")
        img_resized = img_gray.resize(new_size, Image.Resampling.LANCZOS)
        image_array = np.array(img_resized)
        
        return image_array
"""
st.code(code_snippet_1, language='python')

st.markdown("""
The `convert_to_grayscale_and_resize` function takes an image path and a target size as inputs, converts the image to grayscale, and resizes it. This prepares the images for the U-Net model, which expects inputs of uniform size.
""")

# Visualization of the preprocessed image
image_path_example = "../Datasets/Severstal steel defect detection/train_images/000a4bcdd.jpg"
img = convert_to_grayscale_and_resize(image_path_example, new_size=(625, 100))
st.image(img, caption="Grayscale and Resized Image", use_column_width=True)

# Step 2 explanation
st.markdown("#### Step 2: Mask Generation")
st.markdown("""
The next step involves creating a mask for each image based on run-length encoding (RLE) provided in the dataset. This is necessary for supervised training, where the model learns to predict the mask given an input image.
""")

# Code snippet for mask generation
code_snippet_2 = """
import cv2
import numpy as np

def mask_generator(tags, image_path):
    en_pix = tags.split()
    rle = list(map(int, en_pix))
    pixel = [rle[i] for i in range(0, len(rle), 2)]
    pixel_count = [rle[i] for i in range(1, len(rle), 2)]
    
    rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(len(pixel))]
    rle_mask_pixels = sum(rle_pixels, [])
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    l, b, _ = img.shape
    max_index = l * b - 1
    rle_mask_pixels = [min(pixel, max_index) for pixel in rle_mask_pixels]
    mask_img = np.zeros((l * b,), dtype=np.uint8)
    mask_img[rle_mask_pixels] = 1
    
    mask = mask_img.reshape((b, l)).T
    
    new_size = (625, 100)
    resized_array = cv2.resize(mask, new_size)
    
    return resized_array
"""
st.code(code_snippet_2, language='python')

st.markdown("""
The `mask_generator` function takes RLE-encoded mask data and the corresponding image path as inputs, decodes the RLE into pixel locations, and generates a binary mask. The mask is then resized to the target dimensions.
""")

# Visualization of the mask
mask_example = mask_generator("37607 3 37858 8 38108 14 38359 20 38610 25 38863 28 39119 28 39375 29 39631 29 39887 29 40143 29 40399 29 40655 30 40911 30 41167 30 41423 30 41679 31 41935 31 42191 31 42447 31 42703 31 42960 31 43216 31 43472 31 43728 31 43984 31 44240 32 44496 32 44752 32 45008 32 45264 33 45520 33 45776 33 46032 33 46288 33 46544 34 46803 31 47065 25 47327 19 47588 15 47850 9 48112 3 62667 12 62923 23 63179 23 63348 3 63435 23 63604 7 63691 23 63860 11 63947 23 64116 15 64203 23 64372 19 64459 23 64628 24 64715 23 64884 28 64971 23 65139 33 65227 23 65395 37 65483 23 65651 41 65740 22 65907 45 65996 22 66163 48 66252 22 66419 48 66508 22 66675 48 66764 22 66931 48 67020 22 67187 48 67276 20 67443 48 67532 16 67699 48 67788 13 67955 48 68044 9 68210 49 68300 6 68466 49 68556 2 68722 50 68978 50 69234 50 69490 50 69746 50 70009 43 70277 31 70545 19 70813 7 73363 5 73619 14 73875 23 74131 31 74387 40 74643 45 74899 46 75155 46 75411 47 75667 47 75923 48 76179 49 76435 49 76691 50 76947 50 77203 50 77459 50 77715 50 77971 50 78227 50 78483 50 78739 50 78995 50 79251 50 79507 50 79763 50 80019 50 80275 50 80531 50 80789 48 81048 45 81308 41 81567 38 81826 35 82085 32 82345 28 82604 23 82863 18 83122 13 83382 7 83641 2 86637 16 86893 31 87149 31 87405 31 87661 31 87917 32 88173 32 88429 32 88685 32 88941 32 89197 32 89453 32 89709 32 89965 33 90221 33 90306 35 90477 33 90562 35 90733 33 90818 35 90989 16 91074 35 91330 35 91586 35 91842 35 92098 35 92354 35 92610 35 92866 35 93122 35 93378 35 93635 34 93891 34 94147 34 94403 34 94659 34 94915 34 95171 34 95427 34 95623 9 95683 30 95879 18 95939 24 96135 18 96195 17 96390 19 96451 10 96646 19 96707 4 96902 20 97158 20 97414 20 97670 20 97925 21 98181 21 98437 22 98693 22 98949 22 99205 22 99460 23 99716 23 99972 24 100228 24 100484 24 100739 25 100995 25 101251 25 101507 26 101763 26 101820 5 102019 26 102076 13 102274 27 102332 18 102530 27 102587 19 102786 26 102843 20 103042 26 103099 20 103298 26 103355 20 103554 26 103611 20 103809 27 103866 22 104065 26 104122 22 104321 26 104378 22 104579 24 104634 21 104837 22 104889 22 105095 19 105145 22 105354 16 105401 21 105612 14 105657 21 105870 12 105913 20 106128 10 106168 21 106387 6 106424 15 106645 4 106680 5 106903 2 111614 3 111864 9 112114 15 112364 21 112440 8 112617 24 112696 22 112873 24 112952 36 113129 24 113208 43 113385 24 113464 43 113641 24 113720 43 113897 24 113976 43 114153 24 114232 43 114409 24 114488 43 114665 24 114744 43 114921 24 115000 43 115177 24 115256 43 115433 24 115512 43 115689 24 115768 43 115945 24 116024 43 116201 24 116280 43 116457 24 116536 43 116713 24 116792 43 116969 24 117048 43 117225 24 117310 37 117493 12 117576 27 117843 16 118109 6", "../Datasets/Severstal steel defect detection/train_images/000a4bcdd.jpg")
st.image(mask_example, caption="Generated Mask", use_column_width=True)

# Step 3 explanation
st.markdown("#### Step 3: U-Net Model Definition")
st.markdown("""
We use a U-Net model for semantic segmentation. The architecture consists of a contracting path for capturing context and a symmetric expanding path for precise localization.
""")

# Code snippet for U-Net model
code_snippet_3 = """
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Cropping2D

def unet_model(input_shape=(100, 625, 1)):
    inputs = Input(shape=input_shape)

    # Contracting path (Downsampling)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Bottleneck
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    # Expansive path (Upsampling)
    up1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2)

    # Cropping conv1 to match the shape of up1
    cropped_conv1 = Cropping2D(cropping=((0, 0), (0, 1)))(conv1)

    # Skip connection with cropped conv1
    up1 = concatenate([up1, cropped_conv1])

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)

    # Output layer (Sigmoid for binary segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv3)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
"""
st.code(code_snippet_3, language='python')

st.markdown("""
The U-Net model consists of three main parts:
1. **Contracting Path**: This part down-samples the input image and captures features.
2. **Bottleneck**: The deepest layer of the network that captures the abstract representation of the input.
3. **Expansive Path**: This part up-samples the features and concatenates them with features from the contracting path for precise segmentation.
""")

# Train the model section
st.markdown("#### Step 4: Training the Model")
st.markdown("""
We train the U-Net model using the prepared dataset, split into training and validation sets. The model is trained for 5 epochs using binary cross-entropy as the loss function and accuracy as the metric.
""")

# Code snippet for training
code_snippet_4 = """
# Convert lists to NumPy arrays
train_x1 = np.array(train_x)
train_y1 = np.array(train_y)

# Reshape data
train_x2 = train_x1.reshape(-1, 100, 625, 1)
train_y2 = train_y1.reshape(-1, 100, 625, 1)

train_x_subset = train_x2[:1000]
train_y_subset = train_y2[:1000, :, :624, :]

# Split dataset
split_index = int(0.8 * len(train_x_subset))
train_x, val_x = train_x_subset[:split_index], train_x_subset[split_index:]
train_y, val_y = train_y_subset[:split_index], train_y_subset[split_index:]

# Train model
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=30, epochs=5)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
"""
st.code(code_snippet_4, language='python')

st.markdown("""
During training, the model's accuracy and loss are plotted for both the training and validation sets to monitor progress and detect overfitting. The results indicate how well the model is learning to identify defects.
""")

image_path = '../resources/output2.png'

st.image(image_path, caption='This is the image of test coverage and shit', use_column_width=True)



