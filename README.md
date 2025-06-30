# 🧠 Intent Detection with Transformer Models

Identify user intents from short text inputs using traditional and transformer-based models (BERT, Pythia). This project combines data augmentation, model fine-tuning, and an interactive Gradio interface.

---

## 📁 Repository Structure

```bash
.
├── .gradio/                         # Hugging Face Spaces settings
├── intent_model/                   # Trained model directory (BERT or similar)
├── sof-intent-detection/intent_model/ # Additional model directory
├── .gitattributes                  # Git attributes
├── README.md                       # Project overview and documentation
├── app.py                          # Gradio app main file
├── app11.py                        # Renamed/variant app script
├── app_backup.py                   # Backup of older app version
├── config.json                     # Model/configuration file
├── label_classes.npy               # Encoded label classes
├── model.safetensors               # Model weights in safetensors format
├── prompt_training_examples.json   # Augmented prompt examples
├── requirements.txt                # Project dependencies
├── special_tokens_map.json         # Tokenizer metadata
├── temp_backup.py                  # Temporary backup script
├── tokenizer_config.json           # Tokenizer configuration
├── train_model.py                  # Training script
├── vocab.txt                       # Vocabulary file


```
---

## 🚀 Features

- Multi-class classification across 21 intent labels
- Fine-tuned BERT and Pythia-70M models
- Augmented dataset: 328 → 1004 examples
- Real-time Gradio interface
- Custom PyTorch training for flexibility

---

## 📦 Installation & Setup

1. Clone this repository:

git clone https://github.com/niha-bilal/Tifin.git


cd Tifin


2. Install dependencies:

3. pip install -r requirements.txt


4. Run the app:


python app.py
---
🧪 Model Accuracy
| Model        | Accuracy | Highlights                        |
| ------------ | -------- | --------------------------------- |
| Pythia-70M   | 92–93%   | Strong semantic performance       |
| BERT         | 90–92%   | Flexible, custom PyTorch training |
| SVM (TF-IDF) | 71–92%   | Good with augmentation            |
| Naive Bayes  | \~50%    | Lightweight but too shallow       |
---
🧰 How to Train
- Train BERT:python bert.py
- Train Pythia or other models:
- python train.py
Models are saved to:

./Tifin/trained_model/ (Pythia)

./intent_model/ (BERT)

-----
🌐 Gradio Web Interface


To test predictions:


python app.py

You’ll get:

Running on local URL: http://127.0.0.1:7860

Running on public URL: https://xxxx.gradio.live

---
📈 Recommendations

- Use weighted loss or SMOTE to balance class distribution

- Add feedback/error correction in Gradio interface

- Test with optimized models like DistilBERT

- Apply active learning and Ray Tune for fine-tuning
