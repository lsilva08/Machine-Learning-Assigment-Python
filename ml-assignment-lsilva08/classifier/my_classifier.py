import numpy as np

class MyClassifier:

    def __init__(self):
        """
        Empty init function - does nothing yet..
        """

    def fit(self, x, y):
        """
        Empty training function - does nothing yet..
        :return: self
        """

        return self

    def predict(self, x):
        """
        Dummy test function
        :param x: feature matrix
        :return: list of classification labels
        """

        y_pred = []

        # iterates over all the instances and assigns a class ZERO to them
        for i in range(len(x)):
            y_pred.append(0)

        return y_pred


if __name__ == '__main__':
    my_classifier = MyClassifier()

    x_train = np.array([[10, 12, 11], [101, 121, 111], [102, 122, 112], [11, 13, 12]])
    y_train = np.array([0, 1, 2, 0])

    x_test = np.array([[10, 12, 11], [101, 121, 111], [102, 122, 112], [10, 12, 11], [101, 121, 111], [102, 122, 112]])

    my_classifier.fit(x_train, y_train)
    y_pred = my_classifier.predict(x_test)

    print(y_pred)
