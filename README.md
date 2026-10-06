# COVID-19 Chest X-ray Classification

This notebook experiments with image classification for three chest X-ray classes: `COVID19`, `NORMAL`, and `PNEUMONIA`. It trains a custom convolutional neural network and compares it with a ResNet50 transfer-learning model.

The workflow covers image loading, augmentation, model training, learning curves, a confusion matrix, ROC plots, and examples the model classified incorrectly. The saved dataset run used 5,144 training images and 1,288 validation images.

## What the notebook shows

- A custom CNN trained for 5 epochs and then continued for 10 more epochs
- Data augmentation with Keras `ImageDataGenerator`
- Per-class precision, recall, and F1 output
- ResNet50 with ImageNet weights and a small classification head
- Training and validation plots for both approaches

The custom CNN reached about 94% validation accuracy during training, but its saved classification report later shows 49% accuracy. That mismatch can happen because prediction order does not match label order when a generator shuffles data. Fix the evaluation generator with `shuffle=False` before treating the classification report as valid.

## Run the notebook

The image dataset is not included. Create this directory structure and place your images inside it:

```text
data/
  train/
    COVID19/
    NORMAL/
    PNEUMONIA/
  test/
    COVID19/
    NORMAL/
    PNEUMONIA/
```

Then update the `train_data` and `test_data` variables near the top of the notebook.

```bash
git clone https://github.com/dilatedtime/Covid-Detection-using-CNN-and-Transfer-Learning-.git
cd Covid-Detection-using-CNN-and-Transfer-Learning-
python -m venv .venv
python -m pip install jupyter numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
jupyter notebook CovidDetection.ipynb
```

This notebook is a learning project. It has not been validated for clinical use and must not be used to diagnose patients.

