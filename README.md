# medical-marijuana

Train a Convolutional Neural Network (CNN) using Keras to automatically classify root health without having to physically touch the plants.

roots needed to be classified into two groups:
“Hairy” roots
“Non-hairy” roots


![Dataset Sample](https://github.com/walaeddine/medical-marijuana/blob/main/images/mm_dataset.png?raw=true)


The dataset of 1,524 root images includes:

Hairy: 748 images (left)
Non-hairy: 776 images (right)


##### Project structure

```
├── dataset
│   ├── hairy_root [748 images]
│   └── non_hairy_root [776 images]
├── pipeline
│   ├── __init__.py
│   └── simplenet.py
├── train_model.py
└── plot.png
```

Clone the repository:
```
git clone git@github.com:walaeddine/medical-marijuana.git
```

Requirments:
```
Tensorflow '2.2.0'
```

Go ahead and train SimpleNet on our hydroponics, and medical marijuana dataset.

```
python train_model.py --dataset dataset/
```
Results: 
```
[INFO] evaluating network...
                precision    recall  f1-score   support

    hairy_root       1.00      0.95      0.97       299
non_hairy_root       0.95      1.00      0.98       311

      accuracy                           0.98       610
     macro avg       0.98      0.97      0.98       610
  weighted avg       0.98      0.98      0.98       610
```

The network obtained 98% classification accuracy, and as the plot demonstrates, there is no overfitting.

![Dataset Sample](https://github.com/walaeddine/medical-marijuana/blob/main/plot.png?raw=true)


And here is a our prediction:

![Dataset Sample](https://github.com/walaeddine/medical-marijuana/blob/main/Output_screenshot_02.01.2021.png?raw=true)
