
# Steel Defect Detection: A Combined U-Net and CNN Approach  

This repository contains the coding files and resources for the research project **Steel Defect Detection: A Combined U-Net and CNN Approach**. This research presents an advanced, two-stage deep learning framework for automated steel surface defect detection and classification.  

## Project Overview  

Steel manufacturing requires high-quality control standards. This study addresses the challenges of identifying and classifying steel surface anomalies using a novel approach that integrates:  
1. **U-Net Architecture**: For precise defect segmentation.  
2. **Convolutional Neural Network (CNN)**: For accurate defect classification.  

By combining these methodologies, the project enhances production efficiency and quality control processes in steel manufacturing.  

For a deeper dive into the science, methodology, and mathematics behind the project, as well as live visualizations, visit the web application:  
[Steel Defect Detection - Live Visualizations](https://steel-defect-detection.streamlit.app/)  

---

## Repository Structure  

- **`notebooks/`**: Jupyter notebooks containing code for model training and evaluation.  
- **`webapp/`**: Files related to the live web application hosted on Streamlit.  

---

## Installation  

To set up the project locally, ensure you have the following dependencies installed:  

```python
from PIL import Image  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report  
import streamlit  
```

You can install them using pip:  
```bash
pip install -r requirements.txt
```  

---

## Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/steel-defect-detection.git
   cd steel-defect-detection
   ```  

2. Run the Jupyter notebooks in the `notebooks/` folder to understand and replicate model training.  

3. To explore live visualizations:  
   - Navigate to the `webapp/` folder and follow the setup instructions.  
   - Alternatively, visit the [Streamlit app](https://steel-defect-detection.streamlit.app/).  

---

## License  

This project is licensed under the MIT License. See the `LICENSE` file for more details.  

---

## Acknowledgments  

- **Dataset**: We are grateful to Severstal for providing the dataset that was crucial for this research.  
- **Open-source Community**: A big thank you to the developers and maintainers of the tools and libraries used in this project.  

---

## Contributing  

Contributions are welcome! If you would like to suggest improvements, please open an issue or submit a pull request.  

---

## Contact  

For questions or collaboration opportunities, feel free to reach out or open an issue in this repository.  
