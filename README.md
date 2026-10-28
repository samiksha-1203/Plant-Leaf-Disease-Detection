# LeafVision: Plant Leaf Disease Detection

Digital Image Processing semester mini project by Samiksha Chavan and Jagruti Thotam.

LeafVision runs a sequence of image-processing demonstrations and uses a trained
Keras image classifier to predict one of the classes it was trained on. The
classifier predicts a class for every image; it is not an expert diagnosis and
does not identify diseases outside its training classes.

## Project layout

```text
app.py                  Flask application and JSON endpoints
trainer.py              Shared model training and evaluation
train_model.py          Command-line training entry point
dip_pipeline.py         Image preprocessing, DIP stages and visual mask estimate
disease_info.py         General information for the 23 dataset classes
templates/index.html    Web page structure
static/app.js           Browser interactions and result rendering
static/style.css        Page styling
model/
  disease_model.keras   Trained classifier (local generated artifact)
  labels.json           Class order and training metadata
data/Dataset/            Extracted 23-class image folders
outputs/                 Generated image results and training reports
tests/                   Fast project regression tests
```

The extracted images are local data and should not be committed to
source control. `.gitignore` also excludes generated output, Python caches,
virtual environments, and locally trained model files.

## Requirements and installation

Use Python 3.10–3.12 for the most predictable TensorFlow compatibility.

From PowerShell, in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The installed TensorFlow build can use the CPU on native Windows. TensorFlow
2.11 and newer do not provide native-Windows CUDA GPU support. For NVIDIA GPU
training on Windows, use WSL2 with a current NVIDIA Windows driver and install
TensorFlow in the Linux environment; follow the official
[TensorFlow pip installation guide](https://www.tensorflow.org/install/pip).
The local Windows and WSL2 Python environments are separate; install the
requirements in the environment where you run the app or training.

## Run the website

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Choose **Upload Leaf Image**, select a JPG, JPEG,
or PNG image, then choose **Analyze Leaf**. The training panel is available
under **Manage training / Retrain** when a model is already loaded.

## Use the included dataset

The extracted Kaggle dataset is at `data/Dataset/` and contains one folder
per class. From the project root, train with:

```powershell
python train_model.py --data-dir "data\Dataset"
```

The default command-line data path is `training_data/`; the explicit
`--data-dir` above is needed to use the extracted Kaggle dataset. Training saves
`model/disease_model.keras` and `model/labels.json` only after successful
training and evaluation. It also writes plots under `outputs/training_report/`.
The website's upload workflow stores its own training images under
`training_data/`.

The command-line trainer accepts:

```text
--epochs N             Classifier training epochs (default 10)
--fine-tune-epochs N   Fine-tuning epochs (default 5)
--batch-size N         Batch size (default 16)
--img-size N           Square image size (default 224)
--no-pretrained        Train without downloading ImageNet weights
```

For GPU training on WSL2, use the same project files through their mounted
Windows path. For example, from Ubuntu:

```bash
cd "/mnt/c/Users/samiksha/Downloads/plant_leaf_disease_project (2)/plant_leaf_disease_project"
python train_model.py --data-dir "data/Dataset"
```

Keep the terminal open until the command reports completion. Do not start a
second training process against the same model directory at the same time.

## What the Processing Dashboard stages actually do

The website shows eight stages as a **DIP learning dashboard**. This does not
mean that all eight stages are chained together and used to classify a leaf.
The project uses the small set of preprocessing operations that feed the
classifier, while keeping other operators as visual demonstrations. This is
the distinction to make when presenting the project: **the dashboard teaches
DIP operators; the disease prediction comes from a trained CNN classifier.**

| Dashboard stage | Methods shown or used | Role in the project |
|---|---|---|
| 1. Image Acquisition | Validate and decode JPG/PNG; convert to BGR; limit very large images | Required input preparation. |
| 2. Enhancement | Contrast stretching (min-max normalization); negative and gamma correction are also displayed | Contrast stretching is used for model input. Negative and gamma images are demonstrations only. |
| 3. Filtering | Gaussian and median smoothing; Laplacian and high-boost examples | A small Gaussian blur is used for model input. Other filter outputs are demonstrations only. |
| 4. Edge Detection | Roberts, Prewitt, Sobel, and Laplacian operators | Visual demonstrations only; edge maps are not sent to the classifier. |
| 5. Thresholding | Fixed/global, Otsu, and adaptive thresholding | Dashboard grayscale examples are demonstrations. The active mask path uses Otsu thresholding on HSV saturation to estimate the leaf mask and local adaptive Gaussian thresholding on brightness to identify relatively dark candidate patches. |
| 6. Morphology | Erosion, dilation, opening, and closing on a threshold image | Dashboard examples are demonstrations. Closing/opening and erosion are also used in the separate approximate leaf/colour mask for the overlay. |
| 7. Disease Detection | MobileNetV2-based image classifier | Predicts one of the model's 23 trained classes from the preprocessed colour image. This is machine learning, not an edge or thresholding algorithm. |
| 8. Final Result | Prediction, confidence, and annotated image | Displays the classifier output. The optional red overlay is a heuristic colour-based estimate, not a classifier-confirmed lesion. |

### The selected path used for disease prediction

The prediction path is intentionally shorter than the dashboard:

1. Decode and resize oversized input images if needed.
2. Apply contrast stretching (min-max normalization).
3. Apply a 3 x 3 Gaussian blur.
4. Resize to 224 x 224 pixels and convert BGR to RGB.
5. Pass the image to the trained MobileNetV2 classifier.

The training and prediction code share this preprocessing so that the model
receives images in the same format in both cases. Edge maps and the dashboard's
grayscale threshold outputs are not classifier inputs. Leaf segmentation and
candidate-region detection run separately to produce the optional overlay:
HSV saturation and Otsu thresholding estimate the leaf, then local adaptive
Gaussian thresholding on the HSV brightness channel detects dark areas
relative to nearby pixels. Colour rules, morphology, and connected-component
area filtering further refine the mask. This active DIP segmentation does not
change the predicted class or require retraining the classifier. Its overlay
is approximate and can flag shadows or natural colour variation; it is not a
confirmed lesion detector.

## Model and accuracy

The disease classifier is a convolutional neural network built around
**MobileNetV2** and a project-specific 23-class output layer. The training code
uses transfer learning: it attempts to initialize MobileNetV2 with ImageNet
weights, first trains a new classification head with the base network frozen,
and then fine-tunes the last 30 base-network layers when pretrained weights
are available. If those weights cannot be downloaded, the trainer can fall
back to initializing the network without them. The classification head uses
global average pooling, dropout, and a 23-unit softmax layer. Random flips,
rotations, and zoom are training-time augmentation; they are not extra
prediction-time DIP stages.

The included model's saved **validation accuracy is 93.68%** (0.9368 in
`model/labels.json`) across the 23 supported classes. This is the accuracy
reported on the model's validation split, not the accuracy guaranteed for a
new uploaded photo. The included model predates the current grouped split and
used a random image-level split. Since the dataset contains transformed
versions of source images, related images may have ended up in both training
and validation sets; this can make the score optimistic. It is not an
independent test-set result. The current training code groups recognized
transformed siblings before splitting, so retraining with it provides a more
reliable validation estimate, though an independently collected test set is
still needed to measure real-world performance.

The percentage shown beside an individual prediction is that prediction's
softmax confidence, **not** the model's validation accuracy. Neither number is
a guarantee of correctness.

## Project presentation questions

### 1. What problem does your project solve?

LeafVision demonstrates how a photo of a plant leaf can be processed and
classified into one of the disease or healthy classes known to its model. It
provides an accessible first-look result and educational information, but it
does not replace diagnosis by an agricultural expert.

### 2. Why did you select this problem?

Plant diseases can reduce crop health and yield, and checking leaf images is a
clear, visual way to explore digital image processing and machine learning.
This project combines image enhancement, filtering, edge detection,
thresholding, morphology, and image classification in one demonstrable
workflow.

### 3. What dataset did you use?

We used the [Plant Disease Detection dataset on Kaggle](https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection).
The extracted `data/Dataset/` directory has 23 class folders and 35,726 image
files covering healthy leaves and diseases of apple, corn, pepper, potato, and
tomato. The number of images is not evenly distributed across classes.

### 4. Why did you select the particular preprocessing/enhancement technique?

The classifier preprocessing uses contrast stretching followed by a small
Gaussian blur, then resizes the image to 224 × 224 pixels and converts it to
RGB. Contrast stretching normalizes the image's intensity range; Gaussian
filtering reduces small pixel-level noise before classification. The same
preprocessing is used by the training and prediction pipelines. Other
enhancement examples, such as negative and gamma transformation, are shown as
DIP demonstrations and are not classifier inputs. We selected this pipeline
as a straightforward, consistent preprocessing choice; we have not run a
controlled experiment proving it is optimal.

### 5. How does your segmentation method work?

The app estimates the leaf foreground using HSV saturation and Otsu
thresholding, followed by morphological closing and opening. It keeps the
largest sufficiently sized contour and rejects masks that cover too little
or too much of the image. Within an accepted leaf mask, simple HSV colour and
brightness rules flag non-green, very dark, or pale pixels as candidate
regions; morphology and connected-component size filtering clean up the
estimate. This is a heuristic visual aid, not a trained segmentation model
and not proof that marked pixels are diseased tissue.

### 6. Why did you use erosion/dilation or other morphological operations?

Morphological operations use a small structuring element to refine binary
masks. Closing can fill small gaps, opening can remove small foreground
specks, and erosion/dilation shrink or expand a mask. In this project they
help clean threshold-derived leaf and colour candidate masks and demonstrate
standard DIP concepts. They do not determine the classifier's disease label.

### 7. Which method produced the best results and why?

The included classifier uses MobileNetV2 transfer learning, with class
weighting and fine-tuning, and its saved validation accuracy is **93.68%**.
That is the recorded result for this model, not proof that it is the best
possible architecture or preprocessing method. We did not perform a controlled
comparison (ablation) of alternative classifiers or preprocessing methods,
so we cannot claim that contrast stretching, Gaussian filtering, or a
particular edge detector was empirically the best. Also, this model's score
came from an older random image-level split; transformed sibling images may
have appeared across both splits, so the score may be optimistic.

### 8. How did you evaluate your results?

During training, the classifier is evaluated on a held-out validation split.
The saved report includes validation accuracy, per-class precision and
recall, and a confusion matrix. For the included model, the saved validation
accuracy is 93.68%. This is not an independent test-set score, and the saved
model predates the newer grouped splitting that keeps recognized transformed
copies together. The project also has software regression tests for image
preprocessing, mask behavior, dataset splitting, and disease information;
those tests verify code behavior, not real-world diagnostic accuracy.

### 9. What are the limitations of your system?

- It can predict only among its 23 trained classes and may misclassify
  unrelated plants, unseen diseases, or poor-quality images.
- The displayed confidence is a model probability, not a guarantee of
  correctness or overall accuracy.
- The included model's validation score may be optimistic because it used a
  random image-level split before grouped splitting was implemented.
- Dataset images may not represent different cameras, lighting, backgrounds,
  cultivars, or field conditions.
- The colour-based region overlay is approximate and may mark healthy
  discolouration or miss real lesions.
- The system is an academic demonstration, not an expert diagnosis or
  treatment recommendation.

### 10. What improvements can be made in the future?

- Retrain and evaluate with the current grouped split, then report results on
  a separate, independently collected test set.
- Collect more balanced, varied, field-captured images and evaluate across
  different cameras, lighting, and plant varieties.
- Compare preprocessing choices and model architectures using controlled
  experiments and per-class metrics.
- Add out-of-distribution detection so the app can flag images outside its
  supported classes instead of always choosing a known class.
- Replace colour heuristics with a properly labelled and evaluated lesion
  segmentation model if confirmed lesion localization is required.
- Improve calibration, accessibility, deployment speed, and expert-reviewed
  disease information.

## Training methodology and validation

The trainer uses MobileNetV2 transfer learning, class weighting, and a
validation set. Dataset files that appear to be transformed versions of the
same source image are grouped together before the train/validation split to
reduce duplicate-image leakage. This grouped split is used for new training
runs.

The currently included model reports a saved validation score of **93.68%**,
but that model was trained before grouped splitting was added. This dataset
contains transformed sibling images, so the score may be optimistic; it is not
an independent test result. Retrain with the current `trainer.py` for a more
reliable validation estimate. Results on new photos can still be lower,
especially when their crop, camera, background, or lighting differs from the
training images.

## What the result means

- The predicted class and displayed model score come from the classifier.
  The score is a softmax probability, **not overall model accuracy** and not a
  guarantee that the prediction is correct.
- The project displays common symptoms and general management information for
  the supplied dataset classes. This information is educational, not a
  treatment plan.
- The red region estimate is based on simple colour segmentation and
  morphology. It is not produced by the classifier and is not a confirmed
  lesion location. If the leaf cannot be isolated reliably, the app leaves the
  image unmarked.
- The model always selects among its trained classes, even if an uploaded
  image is unrelated or contains a different plant.

> **Academic project only. Verify results with an agricultural expert before
> taking action.**

## Run checks

From the project root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q app.py trainer.py train_model.py dip_pipeline.py disease_info.py
node --check static\app.js
```
