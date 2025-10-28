# Investigating the Effectiveness of Data Mining Techniques in Identifying Early Signs of Mental Health Issues through Multimodal Social Media Data

This repository contains the code and analysis for my **MSc Dissertation in Data Science** at the **University of Birmingham (2025)**. The project explores how **multimodal features** — text, emojis, and images — from social media (Reddit) can be used to **detect early indicators of mental health concerns** using deep learning models.

---

## 🔍 Project Overview

### Problem
Mental health issues such as depression and anxiety continue to rise globally. Social media posts often reveal subtle emotional cues that, if analyzed responsibly, can support early detection and intervention.

### Approach
- Collected Reddit posts containing **text, emojis, and image links** from mental-health-related subreddits.  
- Preprocessed each modality using:
  - NLP techniques for text  
  - **VADER** for emoji sentiment scoring  
  - **CLIP embeddings** for image features  
  - Temporal activity features for behavioral patterns  
- Built a **multimodal deep learning architecture** integrating **BiLSTM, LSTM**, and **cross-attention fusion**.  
- Introduced **mismatch features** (valence gap, sign flip) to capture emotional incongruence (e.g., cheerful emoji with negative text).  
- Evaluated performance with **5-fold cross-validation**, focusing on **Recall** and **PR-AUC** metrics to address class imbalance.

---

## 💡 Key Contributions
- ✅ Developed a **multimodal deep learning pipeline** combining text, emoji, image, and temporal data.  
- ✳️ Proposed novel **mismatch features** to capture emotional inconsistency across modalities.  
- 🔄 Explored **label refinement** methods to reduce annotation noise.  
- 📈 Demonstrated that **mismatch-aware multimodal models** outperform unimodal baselines in Recall and PR-AUC.

---

## 📂 Repository Contents
```
├── Mental_health_code_UPDATED_with_test_set.ipynb   → Main Jupyter Notebook
├── Reddit_Data_Final_Raw.csv                        → Dataset (text, emoji, image links)
├── Das_dxd444.pdf                                   → Full dissertation document
├── requirements.txt                                 → Dependencies list
└── README.md                                        → Project documentation
```

---

## ⚙️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/debolina2908/multimodal-mental-health-detection.git
cd multimodal-mental-health-detection

# Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```
Then open the notebook:
```bash
jupyter lab
```

---

## 🚀 Usage
Run all cells in the Jupyter Notebook to:
- Preprocess textual, emoji, and image data.  
- Train **unimodal and multimodal** models.  
- Evaluate with metrics: Accuracy, Recall, PR-AUC, ROC curves.  
- Visualize results (word clouds, PCA, temporal activity, mismatch effects).

---

## 📊 Results (Highlights)
| Model | Description | Recall | PR-AUC |
|--------|--------------|--------|--------|
| Text-only | BiLSTM on textual posts | 0.73 | 0.79 |
| Image/Emoji-only | CLIP + Emoji sentiment | 0.65 | 0.70 |
| Multimodal (no mismatch) | Fused model without valence gap | 0.74 | 0.84 |
| **Multimodal (with mismatch)** | Full architecture | **0.77** | **0.907** |

🧩 **Insight:** Emotional incongruence across modalities is a strong indicator of early mental distress.

---

## 🔮 Future Scope
- Incorporate **Bayesian NNs** and **soft labels** for improved label refinement.  
- Extend to **audio/visual modalities** (voice, facial cues).  
- Real-world deployment in **mental health monitoring tools** with privacy-by-design frameworks.

---

## 🧠 Tech Stack
Python • TensorFlow • Keras • Pandas • NumPy • CLIP • NLTK • VADER • Matplotlib • Scikit-learn

---

## 📚 Citation
> Das, D. (2025). *Investigating the Effectiveness of Data Mining Techniques in Identifying Early Signs of Mental Health Issues through Multimodal Social Media Data*. MSc Dissertation, University of Birmingham.

---

## 🙏 Acknowledgements
Supervised by **Dr. Mubashir Ali**  
Grateful to my family, friends, and colleagues at the University of Birmingham for their encouragement and feedback.
