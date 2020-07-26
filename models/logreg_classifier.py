from numpy.random import rand
import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=10, iterations=2000, fit_intercept=True, verbose=False):
        """
        Logistic Regression Model

        Parameters
        ----------
        learning_rate : int or float, default=10
            The tuning parameter for the optimization algorithm (here, Gradient Descent) 
            that determines the step size at each iteration while moving toward a minimum 
            of the loss function.

        max_iter : int, default=2000
            Number of iterations to run the optimization algorithm.

        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be
            added to the decision function.

        verbose : bool, default=False
            Specifies if verbose should be printed during training.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        """

        if self.fit_intercept:
            X = self.__add_intercept(X)

        X = self.__normalize(X)
        
        self.theta = np.random.rand(X.shape[1])
        N = len(y)

        for _ in range(self.iterations):
            y_hat = self.__sigmoid(X @ self.theta)
            self.theta -= (self.learning_rate / N) * (X.T @ (y_hat - y))
            if((self.verbose == True) and ((_+1)%10 == 0)):
                print(f'Iter: {_ + 1}, Loss: {self.__loss(X, y, N)}')
        
        return self

    def predict_proba(self, X):
        """
        Probability estimates for samples in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        flag : bool
            To handle input mismatch on unseen data
        Returns
        -------
        T : array-like of shape (n_samples,)
            Returns the probability of each sample.
        """
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        return self.__sigmoid(self.__normalize(X) @ self.theta)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        return self.predict_proba(X).round()
        
    def __sigmoid(self, z):
        """
        The sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def __loss(self, X, y, N):
        """
        Calculates mean loss for the samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        N : integer
            Number of samples in X
        Returns
        -------
        loss : float
            Mean log loss for the samples
        """
        y_hat = self.__sigmoid(X @ self.theta)
        y = np.squeeze(y) 
        step1 = y * np.log(y_hat) 
        step2 = (1 - y) * np.log(1 - y_hat) 
        return np.mean(- step1 - step2)

    def __add_intercept(self, X):
        """
        Modify X to handle intercept (a.k.a. bias) adde to the decision function.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        X' : array, shape [n_samples, n_features + 1]
            Modified X after prepending each sample in X with "1".
        """
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)
    
    
    def __normalize(self, X): 
        """
        Normalize the samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        norm_X : array, shape [n_samples, n_features + 1]
            Normalized feature matrix of X.
        """
        mins = np.min(X, axis = 0) 
        maxs = np.max(X, axis = 0) 
        rng = maxs - mins 
        norm_X = 1 - ((maxs - X)/(rng + 1e-5)) 
        return norm_X

    def get_params(self):
        """
        Returns the models coeffients and/or intercept (if fit_intercept=True).
        Parameters
        ----------
        none
        Returns
        -------
        params : dict
        """
        
        try:

            params = dict()
            if self.fit_intercept:
                params['intercept'] = self.theta[0]
                params['coef'] = self.theta[1:]
            else :
                params['coef'] = self.theta
            return params

        except:
            raise Exception('Fit the model first!')