# Multimodal IMDB Genre Classification - CNN & LSTM

This project focuses on predicting **movie genres** using both **poster images** and **text overviews**, combining computer vision and natural language processing in one workflow.

---

## Objective

The goal is to classify movies into multiple genres by analyzing two types of inputs:

1. **Visual data:** Movie posters processed through a Convolutional Neural Network (CNN)
2. **Textual data:** Movie overviews processed through a Long Short-Term Memory (LSTM) model

By comparing these two deep learning models, the project identifies which modality performs better for genre prediction and explores how combining them can improve accuracy.

---

## Methodology

### 1. Image Model (CNN)

* **Input:** Poster images resized to 64×64 pixels.
* **Process:** Multiple convolution and pooling layers extract visual features, followed by fully connected layers for classification.
* **Output:** Predicted probabilities for each genre.

### 2. Text Model (LSTM)

* **Input:** Movie descriptions (text overviews).
* **Process:** Text is tokenized using TensorFlow’s `TextVectorization` layer. Sequences are embedded and passed through an LSTM to capture contextual meaning.
* **Output:** Multi-label predictions (each genre represented by a sigmoid activation output).

Both models are trained using binary cross-entropy loss since each movie can belong to multiple genres simultaneously.

---

## Results & Insights

**Evaluation criterion:** metrics reported at the checkpoint with the **lowest validation loss** (best generalization point).

| Model            | Best Val Loss | Val Precision | Val Recall |
| ---------------- | ------------: | ------------: | ---------: |
| **CNN (images)** |    **0.2353** |     **0.607** |  **0.261** |
| **LSTM (text)**  |    **0.2251** |     **0.622** |  **0.245** |

### What this means

* **Overall:** The **LSTM** achieved a *lower* validation loss (0.2251) and slightly *higher precision* (0.622), while the **CNN** offered *higher recall* (0.261 vs 0.245) at its best-val-loss point.
* **Trade-offs across epochs:**

  * Early CNN epochs reached **higher recall** (e.g., ~0.37) at the cost of precision and loss; later training balanced toward ~0.61/0.26 (P/R).
  * LSTM recall briefly peaked near **~0.287** (epoch 19) with precision ~0.598, but the best-val-loss checkpoint favors a more conservative operating point.
* **Takeaway:** If you prioritize **finding more true genres** (recall), the **CNN** checkpoint is slightly better. If you want **fewer false positives** and overall calibration (loss), **LSTM** edges it.

### Next steps to improve recall without losing precision

* **Threshold tuning per label:** move from 0.50 to per-genre thresholds based on PR curves.
* **Class weighting / focal loss:** emphasize underrepresented genres.
* **Multimodal fusion:** concatenate CNN + LSTM embeddings and train a joint head; often lifts recall while keeping precision stable.
* **Calibration:** temperature scaling or Platt scaling on a held-out set to stabilize probabilities.

## Technical Summary

* Framework: **TensorFlow / Keras**
* Models: CNN for image classification, LSTM for text classification
* Tasks: Multi-label classification (sigmoid activation)
* Data: Movie posters + overviews from IMDB-style dataset

---

## Conclusion

This project demonstrates how both **vision and language models** can be applied to predict complex attributes like movie genres. CNNs excel at visual cues (colors, faces, tones), while LSTMs interpret story semantics. Together, they provide complementary insights for accurate and interpretable predictions.

The study concludes that CNN slightly outperforms LSTM individually, but a **multimodal combination** could achieve the most robust performance in future work.

