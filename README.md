# Bangla Handwritten Word Recognition System

**Name:** Tarek Alam Bhuiyan  
**Student ID:** 1000129291  
**Course:** Skills for Industry Competitiveness and Innovation Program (SICIP) — BRAC University  

---

## Approach

1. **Dataset Subsampling (`BanglaLekha-Isolated`):**  
   Samples a balanced subset (default 200 images per class across all 84 Bangla character, numeral, and compound character classes) with an 80/20 stratified train/validation split.
2. **Transfer Learning with Hugging Face Models:**  
   Fine-tunes pretrained vision models from Hugging Face (`google/mobilenet_v2_1.0_224` as a frozen-backbone baseline and `microsoft/resnet-18` with full fine-tuning and affine data augmentation).
3. **Experiment Tracking (MLflow):**  
   Logs parameters, per-epoch training/validation loss, validation accuracy, and model artifacts to `artifacts/mlflow/` across multiple runs, saving the best-performing model to `models/hf_model/`.
4. **Whole-Word Prediction (`app.py`):**  
   Uses OpenCV vertical-projection segmentation below the top *matra* band to split a drawn Bangla word into left-to-right character crops, classifies each crop with the fine-tuned Hugging Face model, and joins the predicted characters into a normalized whole Bangla word (e.g., `জল`, `মানুষ`).

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (with MLflow Tracking)
```bash
python train.py --data-dir ../BanglaLekha-Isolated/Images --samples-per-class 200
```

### 3. View MLflow Experiments
```bash
mlflow ui --backend-store-uri ./artifacts/mlflow --host 0.0.0.0 --port 5000
```

### 4. Run the Streamlit App Locally
```bash
streamlit run app.py
```

### 5. Build and Run with Docker
```bash
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```
Open `http://localhost:8501` in your browser.

---

## Limitations

- **Vowel Signs (*Kar*) & Diacritics:** `BanglaLekha-Isolated` contains isolated basic characters, numerals, and 24 compound characters rather than standalone vowel signs (e.g., `া`, `ি`, `ু`), which can reduce accuracy on words with heavy vowel-sign attachments.
- **Segmentation Sensitivity:** Heavily cursive or overlapping strokes below the *matra* line may be grouped into a single segment or split incorrectly by projection-based segmentation.
- **Subset Training:** Training on a subset of 200 images per class trades a small amount of generalization for fast iteration compared to the full 166,105-image dataset.
