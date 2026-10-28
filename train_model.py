"""
train_model.py - command-line training (optional).

You can ALSO train directly on the website: upload leaf images there and
press "Train Model". Both ways use the same code (trainer.py).

Class folders go in  training_data/<ClassName>/  e.g.
    training_data/Healthy/
    training_data/Grape_Black_Rot/
(The website creates this folder automatically when you upload images.)

Usage:
    python train_model.py
    python train_model.py --epochs 12 --fine-tune-epochs 6
    python train_model.py --no-pretrained        # offline / from scratch
"""
import argparse
import sys

import trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=trainer.DATA_DIR)
    ap.add_argument("--model-dir", default=trainer.MODEL_DIR)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    try:
        r = trainer.run_training(a.data_dir, a.model_dir, a.img_size, a.batch_size, a.epochs,
                                 a.fine_tune_epochs, not a.no_pretrained)
    except trainer.TrainingError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
    print(f"\nDone. Validation accuracy: {r['val_accuracy']}%  ->  now run: python app.py")


if __name__ == "__main__":
    main()
