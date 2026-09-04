DNA Sequence Classification (NusraGenX v2)

Multi-class classification of DNA sequences (human, chimpanzee, dog) using a custom deep learning model combining CNN + Bi-LSTM + Self-Attention + MoE head.

Pipeline: Data loading (Kaggle) → EDA → k-mer feature engineering → model training (with label smoothing, cosine LR warmup, SWA) → evaluation (accuracy, F1, ROC-AUC, confusion matrix, 5-fold CV) → ablation study → LIME explainability → cross-species generalization testing.

Tech stack:
Data: pandas, numpy, kagglehub
ML utils: scikit-learn (train/test split, cross-validation, metrics)
Deep learning: TensorFlow / Keras
Visualization: matplotlib, seaborn
Explainability: lime
