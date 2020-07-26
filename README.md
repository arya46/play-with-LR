# Logistic Regression Playground
[Live version here!](https://play-with-lr.herokuapp.com/)

## Table of Contents
- [Overview](#overview)
- [What is Logistic Regression?](#what-is-logistic-regression)
- [Technical Aspect](#technical-aspect)
- [Usage](#usage)
- [Project Directory Structure](#project-directory-structure)
- [Technologies Used](#technologies-used)
- [Contributions / Bug](#contributions--bug)
- [License](#license)

## Overview
Logistic Regression Playground is an educational sandbox for those who want to understand Logistic Regression from a more intuitive perspective.

The Logistic Regression model used in this app is implemented from scratch in Python.

## What is Logistic Regression?
Logistic Regression is one of the most common machine learning algorithms for classification. It a statistical model that uses a logistic function to model a binary dependent variable. In essence, it predicts the probability of an observation belonging to a certain class or label. For instance, is this a cat photo or a dog photo?

## Technical Aspect
The mathematics and code implementation is covered in this [Medium](https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59) article.

### Parameters: 
The model accepts the following params:
- `learning_rate` : The tuning parameter for the optimization algorithm (here, Gradient Descent) that determines the step size at each iteration while moving toward a minimum of the cost function.
- `max_iter` : Maximum number of iterations taken for the optimization algorithm to converge
- `penalty` : None or 'l2', Option to perform L2 regularization.
- `C` : Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization. 
- `tolerance` : Value indicating the weight change between epochs in which gradient descent should terminated. 

### Methods:
The model has the following exposed methods:
- `predict()` : Predicts class labels for an observation
- `predict_proba()` : Predicts the estimate for a class
- `get_params()` : Returns the coefficients and intercept

## Usage
Get the code for my custom implementation of Logistic Regression from [here](https://gist.github.com/arya46/59536f6231ffd13c509aa1be59212cfa#file-log_reg_clf-py)
```
# fit the data
clf = LogisticRegression()
clf.fit(X,y)

# predict probabilities
probs = clf.predict_proba(x_test)

# predict class labels
preds = clf.predict(x_test)
```

__Results on a dummy dataset__:

<img src="https://miro.medium.com/max/518/1*cg5u-0iKthH82o0d7yu8uA.jpeg">

## Project Directory Structure
```
├── static                       # contains files for HTML
│   ├── bootstrap
│   ├── css
│   └── js
├── templates                    # the main web page
│   └── index.html 
├── utils 
│   ├── models
|   |   └── logreg_classifier.py # contains the classifier codes
│   ├── configs.py
│   └── helper_functions.py
├── LICENSE
├── Procfile
├── README.md
├── main.py
└── requirements.txt
```

## Technologies Used
- Programming Language: Python
- ML Tools/Libraries: Numpy, Scipy
- Web Tools/Libraries: Flask, HTML

## Contributions / Bug
If you want to contribute to this project, or want to report a bug, kindly open an issue [here](https://github.com/arya46/play-with-LR/issues/new).

## License
[LICENSE](https://github.com/arya46/play-with-LR/blob/master/LICENSE)