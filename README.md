# Water Quality Transformer Classifier

A PyTorch-based deep learning project that uses a **Transformer model** to predict **water potability** based on chemical features from the dataset.

---

## 📌 Features
- Loads and preprocesses the **Water Potability dataset** (`water_potability.csv`).
- Handles missing values by imputing with column means.
- Standardizes features using **scikit-learn**.
- Implements a **Transformer Encoder** architecture with:
  - Multi-head attention
  - Layer normalization
  - Dropout regularization
- Trains using **AdamW optimizer** with weight decay.
- Reports **loss and accuracy** after each epoch.

---

## 🛠 Prerequisites
Before running the project, make sure you have:

1. **Python 3.8+** installed → [Download here](https://www.python.org/downloads/)  
2. **VS Code** installed → [Download here](https://code.visualstudio.com/)  
3. **Required Python libraries** (installed later with `requirements.txt`).  

---

## 🚀 Instructions

Follow these numbered steps to set up and run the project in VS Code:

1. Clone the repository and enter the folder:  
   ```bash
   git clone https://github.com/your-username/water-quality-transformer.git && cd water-quality-transformer
Open a VS Code terminal:
Terminal → New Terminal

Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

On Windows (PowerShell):

bash
Copy code
.\venv\Scripts\Activate
On Windows (CMD):

bash
Copy code
venv\Scripts\activate.bat
On macOS/Linux:

bash
Copy code
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the main training script:

bash
Copy code
python src/water_quality_transformer.py
(Optional) Run the alternative model:

bash
Copy code
python src/import_torch_model.py
View the training progress and accuracy logs in your terminal.

📊 Dataset
Source: Kaggle – Water Potability Dataset

Features include pH, hardness, solids, chloramines, sulfate, etc.

Target column: Potability (0 = not potable, 1 = potable).

✅ Results
The model trains on 80% of the dataset and evaluates on the remaining 20%.

Accuracy is printed after each epoch for tracking performance.
