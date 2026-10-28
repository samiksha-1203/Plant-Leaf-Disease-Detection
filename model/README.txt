The locally trained classifier and matching labels are stored here:

    disease_model.keras
    labels.json

To retrain using the included extracted Kaggle data, run from the project root:

    python train_model.py --data-dir "data\Dataset"

The generated model is ignored by Git because it is a local binary artifact.
If either file is missing, the website shows:
"Disease classification model not available. Please train/load the model."
(and does NOT invent any disease result).
