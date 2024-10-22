import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

# Load the pre-trained CNN model
model_path = 'cnn_model_mask.h5'
model = tf.keras.models.load_model(model_path)

# Function to generate masks using the given tags and image path
def mask_generator(tags, image_path):
    en_pix = tags.split()
    rle = list(map(int, en_pix))
    pixel = [rle[i] for i in range(0, len(rle), 2)]
    pixel_count = [rle[i] for i in range(1, len(rle), 2)]

    # Generate RLE pixels
    rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(len(pixel))]
    rle_mask_pixels = sum(rle_pixels, [])

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # Get image dimensions
    l, b, _ = img.shape
    max_index = l * b - 1
    rle_mask_pixels = [min(pixel, max_index) for pixel in rle_mask_pixels]
    mask_img = np.zeros((l * b,), dtype=np.uint8)

    # Set pixels of interest to 1
    mask_img[rle_mask_pixels] = 1

    mask = mask_img.reshape((b, l)).T

    # Resize the mask to the desired size
    new_size = (625, 100)
    resized_array = cv2.resize(mask, new_size)

    return resized_array

# Function to preprocess the image
def preprocess_image(image_path, tags):
    mask = mask_generator(tags, image_path)
    mask_array = np.expand_dims(mask, axis=0)
    mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension
    return mask_array

# Function to get intermediate layer outputs
def get_intermediate_outputs(image, model):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    activation_model = tf.keras.models.Model(inputs=model.layers[0].input, outputs=layer_outputs)
    activations = activation_model.predict(image)
    return activations

# Load CSV data
CSV_PATH = "../Datasets/Severstal steel defect detection/train.csv"
data = pd.read_csv(CSV_PATH)

# Rename columns for clarity
data = data.rename(columns={'ImageId': 'image_name', 'ClassId': 'grade', 'EncodedPixels': 'tags'})

# Get image paths and labels
image_ids = data['image_name'].values
image_paths = [f"../Datasets/Severstal steel defect detection/train_images/{image_id}" for image_id in image_ids]

# Streamlit app
st.title("Steel Defect Detection: CNN Classification Process")

# Select an image from the list
selected_image_id = st.selectbox("Choose an image", image_ids)
selected_image_path = f"../Datasets/Severstal steel defect detection/train_images/{selected_image_id}"
selected_damage = data[data['image_name'] == selected_image_id]['grade'].values[0]
selected_tags = data[data['image_name'] == selected_image_id]['tags'].values[0]

# Display the selected image
if os.path.exists(selected_image_path):
    st.image(selected_image_path, caption="Selected Image", use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(selected_image_path, selected_tags)
    
    # Ensure the preprocessed image has the correct shape
    if preprocessed_image.shape != (1, 100, 625, 1):
        preprocessed_image = np.transpose(preprocessed_image, (0, 2, 1, 3))
    
    # Get intermediate layer outputs
    intermediate_outputs = get_intermediate_outputs(preprocessed_image, model)
    
    # Display intermediate outputs
    st.subheader("CNN Intermediate Outputs")
    
    # Create a list of layer names for the slider
    layer_names = [layer.name for layer in model.layers[1:]]
    selected_layer = st.select_slider("", options=layer_names)
    
    for layer_index, layer_output in enumerate(intermediate_outputs):
        layer = model.layers[layer_index + 1]  # +1 to skip input layer
        
        if layer.name == selected_layer:
            st.markdown(f"### Layer: {layer.name} ({layer.__class__.__name__})")
            
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Convolutional layer
                fig, ax = plt.subplots(figsize=(10, 5))
                feature_map = np.mean(layer_output[0], axis=-1)
                ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f"Average Feature Map for {layer.name}")
                ax.axis('off')
                st.pyplot(fig)
                st.markdown("This visualization shows the average activation across all filters in this convolutional layer.")
            
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                # Max pooling layer
                fig, ax = plt.subplots(figsize=(10, 5))
                pooled_map = np.mean(layer_output[0], axis=-1)
                ax.imshow(pooled_map, cmap='viridis')
                ax.set_title(f"Pooled Feature Map for {layer.name}")
                ax.axis('off')
                st.pyplot(fig)
                st.markdown("This visualization shows how the max pooling layer reduces the spatial dimensions while preserving important features.")
            
            elif isinstance(layer, tf.keras.layers.Dense):
                # Dense layer
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(range(layer_output.shape[-1]), layer_output[0])
                ax.set_xlabel("Neuron")
                ax.set_ylabel("Activation")
                ax.set_title(f"Activations of {layer_output.shape[-1]} neurons in {layer.name}")
                st.pyplot(fig)
                st.markdown("This bar chart shows the activation of each neuron in the dense layer, indicating which features are most prominent for classification.")
            
            else:
                st.write(f"Layer output shape: {layer_output.shape}")
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction) + 1  # Add 1 to match the 1-4 class range
    
    # Display prediction results
    st.subheader("Prediction Results")
    st.table(pd.DataFrame({
        "Actual Damage Category": [selected_damage],
        "Predicted Damage Category": [predicted_class]
    }))

    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(10, 5))
    class_labels = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
    ax.bar(class_labels, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities for Each Damage Category")
    for i, v in enumerate(prediction[0]):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    st.pyplot(fig)
    st.markdown("This chart shows the model's confidence in each damage category prediction. The highest bar indicates the predicted category.")




# Model Details
st.sidebar.subheader("Model Details")
with st.sidebar.expander("Show Model Summary"):
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

    
# Detailed explanation of the CNN training process
st.markdown("""
### CNN Training Process

The Convolutional Neural Network (CNN) used for steel defect detection was trained using the following steps:

1. **Data Preparation:**
   ```python
   # Load and preprocess the data
   CSV_PATH = "../Datasets/Severstal steel defect detection/train.csv"
   data = pd.read_csv(CSV_PATH)
   
   # Rename columns for clarity
   data = data.rename(columns={'ImageId': 'image_name', 'ClassId': 'grade', 'EncodedPixels': 'tags'})
   
   # Create image paths
   data['image_path'] = data['image_name'].apply(lambda x: f"../Datasets/Severstal steel defect detection/train_images/{x}")
   
   # Convert grade to integer
   data['grade'] = data['grade'].astype(int)
   ```
   This step involves loading the CSV file containing image information and preprocessing it for further use.

2. **Data Balancing:**
   ```python
   # Separate the dataset by grades
   grade_1 = data[data['grade'] == 1]
   grade_2 = data[data['grade'] == 2]
   grade_3 = data[data['grade'] == 3]
   grade_4 = data[data['grade'] == 4]
   
   # Downsample grade 3 to 800 samples
   grade_3_downsampled = resample(grade_3, n_samples=800, random_state=42)
   
   # Combine the balanced dataset
   balanced_data = pd.concat([grade_1, grade_2, grade_3_downsampled, grade_4])
   
   # Shuffle the dataset
   balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
   ```
   To address class imbalance, we downsampled the overrepresented class (grade 3) and combined it with other classes to create a balanced dataset.

3. **Mask Generation:**
   ```python
   def mask_generator(tags, image_path):
       en_pix = tags.split()
       rle = list(map(int, en_pix))
       pixel = [rle[i] for i in range(0, len(rle), 2)]
       pixel_count = [rle[i] for i in range(1, len(rle), 2)]

       # Generate RLE pixels
       rle_pixels = [list(range(pixel[i], pixel[i] + pixel_count[i])) for i in range(len(pixel))]
       rle_mask_pixels = sum(rle_pixels, [])

       img = cv2.imread(image_path)
       if img is None:
           raise ValueError(f"Image not found at {image_path}")

       # Get image dimensions
       l, b, _ = img.shape
       max_index = l * b - 1
       rle_mask_pixels = [min(pixel, max_index) for pixel in rle_mask_pixels]
       mask_img = np.zeros((l * b,), dtype=np.uint8)

       # Set pixels of interest to 1
       mask_img[rle_mask_pixels] = 1

       mask = mask_img.reshape((b, l)).T

       # Resize the mask to the desired size
       new_size = (625, 100)
       resized_array = cv2.resize(mask, new_size)

       return resized_array

   # Create train_x and train_y
   train_x = np.array([
       mask_generator(tags, image_path) 
       for tags, image_path in zip(balanced_data['tags'], balanced_data['image_path'])
   ])
   train_y = np.array(balanced_data['grade'])

   # Reshape train_x to add the channel dimension (grayscale images)
   train_x = train_x.reshape(train_x.shape[0], 100, 625, 1)
   ```
   Instead of using the original images, we generate masks based on the encoded pixels information. These masks are then used as input to the CNN.

4. **Model Architecture:**
   ```python
   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(100, 625, 1)),
       MaxPooling2D((2, 2)),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D((2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dropout(0.5),
       Dense(4, activation='softmax')
   ])
   
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   ```
   The CNN architecture remains the same, but now it's trained on the mask data instead of the original images.

5. **Model Training:**
   ```python
   history = model.fit(train_x, train_y, epochs=10, batch_size=32, validation_split=0.2)
   ```
   The model is trained on the mask data for 10 epochs, using a batch size of 32 and a validation split of 20%.

6. **Model Evaluation and Saving:**
   The evaluation and saving process remains the same as before.

This modified training process results in a CNN model capable of classifying steel defects into four categories based on masks generated from the original steel surface images.
""")

image_path = '../resources/output4.png'

st.image(image_path, caption='This is the image of test coverage and shit', use_column_width=True)


image_path = '../resources/output5.png'

st.image(image_path, caption='This is the image of test coverage and shit', use_column_width=True)


# Detailed explanation of the CNN
st.sidebar.markdown("""
### Convolutional Neural Networks (CNNs)

CNNs are a class of deep learning models particularly effective for image processing tasks. They are designed to automatically and adaptively learn spatial hierarchies of features from input images.

#### Key Components of CNNs:

1. **Convolutional Layers:**
   - Function: Extract features from the input image
   - Mathematics: Convolution operation
     \[
     (f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau
     \]
   - In discrete form for 2D images:
     \[
     (I * K)(i,j) = \sum_m \sum_n I(m,n)K(i-m,j-n)
     \]
     where I is the input image and K is the kernel

2. **Activation Functions:**
   - Common choice: ReLU (Rectified Linear Unit)
   - Function: Introduce non-linearity
   - Mathematics:
     \[
     f(x) = \max(0, x)
     \]

3. **Pooling Layers:**
   - Function: Reduce spatial dimensions
   - Common types: Max Pooling, Average Pooling
   - Mathematics (Max Pooling):
     \[
     y_{ij} = \max_{(a,b) \in R_{ij}} x_{ab}
     \]
     where R_{ij} is the receptive field at position (i,j)

4. **Fully Connected Layers:**
   - Function: Perform high-level reasoning
   - Mathematics: Matrix multiplication and bias addition
     \[
     y = Wx + b
     \]

5. **Softmax Layer (for classification):**
   - Function: Convert outputs to probabilities
   - Mathematics:
     \[
     \sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
     \]

#### Training Process:

1. **Forward Propagation:** Compute the output of each layer
2. **Loss Calculation:** Measure the difference between predicted and actual outputs
3. **Backpropagation:** Compute gradients of the loss with respect to parameters
4. **Parameter Update:** Adjust weights and biases using an optimization algorithm (e.g., Stochastic Gradient Descent)

#### Key Concepts:

- **Receptive Field:** The region in the input space that a particular CNN feature is looking at
- **Stride:** The step size of the convolution operation
- **Padding:** Adding extra pixels to the input to control the output size

#### Advanced Topics:

- Transfer Learning
- Data Augmentation
- Regularization techniques (e.g., Dropout, L1/L2 regularization)

#### References:

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
""")
