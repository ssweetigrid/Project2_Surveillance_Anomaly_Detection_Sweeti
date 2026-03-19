# :rotating_light: Surveillance Anomaly Detection using Transformers

A deep learning pipeline for detecting anomalous events in traffic surveillance videos using temporal transformer models and self-supervised learning.

The system learns patterns of normal traffic behavior and detects deviations as anomalies based on reconstruction error.

This project demonstrates how deep visual features + temporal modeling can be used to identify unusual events in surveillance footage.



---

# :pushpin: Project Objective

The goal of this project is to detect unusual events in traffic surveillance footage, such as:

- traffic accidents
- sudden abnormal motion
- unusual vehicle behavior
- unexpected scene activity

Instead of relying on labeled anomaly examples, the model learns normal patterns and identifies anomalies as reconstruction failures.

Model behaviour in anomaly frames:

![ans](https://github.com/griddynamics/ds_interns_project_2_2026/blob/sweeti_Surveillance_Anomaly_Detection/ui/Screenshot%202026-03-06%20at%206.57.06%E2%80%AFAM.png?raw=true)

**Detected anomaly:** A vehicle performs an abrupt lane cut while turning, which corresponds to an **illegal turn pattern** commonly associated with traffic violations.

## 🎥 Project Demo

Watch the demo video of the project here:

[▶️ Click here to watch the demo](https://drive.google.com/file/d/1VEAPv5jSxfPotMCU5i9UQV1onumQRLiu/view)

---

# :brain: Methodology

The system uses deep visual embeddings and temporal sequence modeling.

## Pipeline Architecture

```text
Video Frames (.jpg)
↓
ResNet50 Feature Extraction
(2048-dimensional embeddings)
↓
Feature Normalization
(Z-score using training statistics)
↓
Sliding Window Sequence Creation
(Window size = W, Stride = S)
↓
Temporal Model
• Transformer Autoencoder
• Transformer Variational Autoencoder
↓
Reconstruction Error
↓
Anomaly Score
↓
Threshold (2σ / 3σ)
↓
Anomaly Detection
```

---

## :open_file_folder: Repository Structure

```text
ds_interns_project_2_2026
│
├── experiments/
│   ├── exp1_transformer_w16_s5
│   ├── exp2_transformer_w8_s5
│   ├── exp3_vae_w8_s5
│   └── experiment outputs, plots and results
│
├── ui/
│
├── main.ipynb
├── surveillance_anomaly_detection.ipynb
│
├── best_transformer_model.pth
├── event_labels.txt
│
├── streamlit_app.py
├── requirements.txt
│
└── README.md
```

---

# :notebook: Notebooks

## :one: `main.ipynb`

This notebook contains the feature extraction pipeline.

### Steps performed

- Load surveillance video frames
- Use pretrained **ResNet50 (ImageNet)**
- Remove classification head
- Extract **2048-dimensional embeddings per frame**
- Store extracted features for model training



These embeddings are used as input to the temporal anomaly detection models.

---

## :two: `surveillance_anomaly_detection.ipynb`

This is the main notebook containing the full anomaly detection pipeline.

### It includes

#### Data Processing

- loading extracted features
- feature normalization
- sliding window sequence generation

#### Model Architectures

- Transformer Autoencoder
- Transformer Variational Autoencoder
- LSTM Autoencoder

#### Model Training

- temporal sequence training
- reconstruction loss optimization
- validation evaluation

#### Model Evaluation

- ROC curve analysis
- AUC computation
- anomaly threshold selection

#### Visualization

- training loss curves
- ROC comparisons
- anomaly timeline plots
- anomaly frame visualizations

---

# :gear: Experiments

Three different architectures were evaluated.

| Experiment | Architecture | Window | Stride | Description |
|-----------|-------------|--------|--------|-------------|
| Exp1 | Transformer Autoencoder | 16 | 5 | Larger temporal context |
| Exp2 | Transformer Autoencoder | 8 | 5 | Faster temporal modeling |
| Exp3 | Transformer VAE | 8 | 5 | Variational bottleneck |

---

# :bar_chart: Results

| Model | Validation AUC | Test AUC |
|------|---------------|---------|
| Transformer AE (window=16) | 0.8168 | 0.4743 |
| Transformer AE (window=8) | **0.8524** | 0.6490 |
| Transformer VAE (window=8) | 0.8095 | **0.8100** |

---

# :trophy: Best Model: Transformer VAE

- **Window Size:** 8
- **Stride:** 5
- **Test AUC:** 0.81

---

# :trophy: Final Model

The final best model checkpoint is stored at:
```  experiments/exp3_vae_w8_s5/model.pth ```

This model achieved the highest anomaly detection performance on the test dataset.

---

# :chart_with_upwards_trend: Key Findings

### :one: Transformer models outperform LSTM models

Self-attention mechanisms capture long-range temporal dependencies more effectively than recurrent architectures for high-dimensional visual features.

### :two: MAX anomaly scoring improves sensitivity

Anomalies are detected using the maximum reconstruction error within a sequence window:
``` anomaly_score = max(reconstruction_error) ```


This makes the system sensitive to short anomalous events.

### :three: Optimal sliding window configuration

Best performance was obtained using:
``` window = 8```
```    stride = 5 ```

This balances temporal coverage and training stability.

---

# :bar_chart: Experiment Outputs

All experiment artifacts are stored in:
``` experiments/ ```

This includes:

- trained model checkpoints
- ROC curves
- anomaly visualizations
- experiment result files
- training plots

---

# :movie_camera: Example Anomaly Detection

The model detects anomalies by identifying frames where reconstruction error increases significantly.

These correspond to unusual traffic behavior or unexpected scene dynamics.

---

# :computer: Installation

Clone the repository:

```bash
git clone <repository_url>
cd ds_interns_project_2_2026

```
Install dependencies:
``` pip install -r requirements.txt ```

## :arrow_forward: Run Feature Extraction

```bash
jupyter notebook main.ipynb
```

Feature files are not included in the repository.
You can upload the extracted features and directly load the saved features for training.

---

## :arrow_forward: Train and Evaluate Models

```bash
jupyter notebook surveillance_anomaly_detection.ipynb
```

This notebook performs:

- model training
- evaluation
- visualization
- experiment comparison

---

## :desktop_computer: Streamlit Interface

A simple web interface is provided for testing anomaly detection.

Run:

```bash
streamlit run streamlit_app.py
```

The interface allows users to:

- check the motion change in all testing frames
- run anomaly detection
- visualize anomaly frames
- adjust **window-size** and **stride** using sliders to observe anomaly variations

---

## :wrench: Technologies Used

- Python
- PyTorch
- Transformers
- Variational Autoencoders
- ResNet50
- OpenCV
- NumPy
- Matplotlib
- Streamlit

---

## :rocket: Future Improvements

### Multi-Camera & Multi-View Training

The current model is trained on data captured from a single camera angle.

Future work will focus on training the model using multiple street views and camera perspectives to improve view-invariant anomaly detection and generalization across different environments.

---

### Larger and More Diverse Datasets

Future training could incorporate:

- multiple cities
- different weather conditions
- day and night scenarios
- varying traffic densities

This would improve the robustness and generalization ability of the model.

---

### Vision Transformer Feature Extraction

Currently the pipeline uses **ResNet50 embeddings**.

Future work could explore:

- Vision Transformers (ViT)
- CLIP embeddings
- self-supervised video representations

### Spatio-Temporal Interaction Modeling

Future models could incorporate:

object detection

vehicle tracking

trajectory analysis

to better understand interactions between objects in traffic scenes.

---

## 📄 Project Report

👉 [View Full Report](https://docs.google.com/document/d/1XEXbaH-zKNdPiCg15tA-H2L46LQbFyuitB7RcqVnuIc/edit?usp=sharing)
##  Author

**Sweeti Swami**

Data Science Intern

Focus: AI Systems for Real-World Decision Intelligence
