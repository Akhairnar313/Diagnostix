# Diagnostix
A Machine Learning/Deep Learning Web Application for multiple disease prediction

## About

Diagnostix is a web application that utilizes machine learning, deep learning, and image processing techniques to predict over 10 different diseases and conditions. The project incorporates various technologies such as Python, Flask, Keras, Numpy, TensorFlow, and Sci-kit. The models used to predict the diseases were trained on large Datasets. All the links for datasets used for model creation are mentioned below in this readme. The web app can predicts the following Diseases:
 
- Diabetes
- Heart Disease
- Kidney Disease
- Liver Disease
- Malaria
- Pneumonia
- Other Blood-Related Diseases/Conditions

## Models with their Accuracy of Prediction

| Disease        | Type of Model            | Accuracy |
| -------------- | ------------------------ | -------- |
| Diabetes       | Machine Learning Model   | 98.25%   |
| Heart Disease  | Machine Learning Model   | 85.25%   |
| Kidney Disease | Machine Learning Model   | 99%      |
| Liver Disease  | Machine Learning Model   | 78%      |
| Malaria        | Deep Learning Model(CNN) | 96%      |
| Pneumonia      | Deep Learning Model(CNN) | 95%      |


## Steps to run this application in your system

1. Clone or download the repo.
2. Open command prompt in the downloaded folder.
3. Create a virtual environment

```
mkvirtualenv environment_name
```

4. Install all the dependencies:

```
pip install -r requirements.txt
```

5. Run the application

```
python app.py
```

## Dataset Links

All the datasets were used from kaggle.

- [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- [Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- [Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)
- [Kidney Disease Dataset](https://www.kaggle.com/mansoordaku/ckdisease)
- [Liver Disease Dataset](https://www.kaggle.com/uciml/indian-liver-patient-records)
- [Malaria Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
- [Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
