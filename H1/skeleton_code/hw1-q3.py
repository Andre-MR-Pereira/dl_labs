#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
        pred = self.predict(x_i)
        #print(pred,y_i)
        if pred != y_i:
            self.W[y_i] += x_i
            self.W[pred] -= x_i
                

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        n_classes = np.size(self.W, 0)
        label_scores = np.dot(self.W, x_i.T)
        y_one_hot = np.zeros(n_classes)
        y_one_hot[y_i] = 1
        for i in range(n_classes):
            label_probability = np.exp(label_scores[i]) / np.sum(np.exp(label_scores))
            self.W[i] += learning_rate * (y_one_hot[i] - label_probability) * x_i


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        units = [n_features, hidden_size, n_classes]
        mu, sigma = 0.1, 0.1
        
        print("NN shape:\n ===============================")
        print(n_features," L0 : features")
        print(hidden_size," L1 : hidden units")
        print(n_classes," L2 : classes")
        print(" ===============================")
        
        self.W1 = np.random.normal(mu, sigma, size=(units[1], units[0]))
        self.b1 = np.zeros(units[1])
        self.W2 = np.random.normal(mu, sigma, size=(units[2], units[1]))
        self.b2 = np.zeros(units[2])
        

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        #FORWARD PROPAGATION
        y_hat=[]
        for x in X:
            z1 = self.W1.dot(x) + self.b1
            h1 = np.maximum(z1 , 0)

            z2 = self.W2.dot(h1) + self.b2
            
            z2 -= z2.max()
            probs = np.exp(z2) / np.sum(np.exp(z2))

            y_hat.append(np.argmax(probs))
        
        return y_hat

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        print(np.shape(y_hat))
        print(y_hat[0:7],"y_hat predicted")
        print(np.shape(y))
        print(y[0:7],"y true")
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print(n_correct / n_possible,"Ratio")
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        
        #FORWARD PROPAGATION
        #print(np.shape(X),"X")
        #print(np.shape(self.W1),"W1")
        #print(np.shape(self.b1),"b1")
        #print(np.shape(self.W2),"W2")
        #print(np.shape(self.b2),"b2")
        #print(np.shape(y))
        #print(y[0:5],"y true")
        
        grad_W1_save = []
        grad_b1_save = []
        grad_W2_save = []
        grad_b2_save = []
        
        for x_step,y_step in zip(X,y):
            z1 = self.W1.dot(x_step) + self.b1
            h1 = np.maximum(z1 , 0)
            #print(h1,"Teste")

            z2 = self.W2.dot(h1) + self.b2

            z2 -= z2.max()
            probs = np.exp(z2) / np.sum(np.exp(z2))
            
            #BACKWARD PROPAGATION: so atualizar os weights no fim da back prop TODA!

            grad_z2 = probs - y_step
            grad_W2 = grad_z2[:, None].dot(h1[:, None].T)
            grad_b2 = grad_z2
            grad_h1 = self.W2.T.dot(grad_z2)
            
            #print((grad_h1 > 0) * 1,"Teste")
            
            grad_z1 = grad_h1 * ((grad_h1 > 0) * 1)
            grad_W1 = np.dot(grad_z1[:, None],x_step[:, None].T)
            grad_b1 = grad_z1
            
            grad_W1_save.append(grad_W1)
            grad_b1_save.append(grad_b1)
            grad_W2_save.append(grad_W2)
            grad_b2_save.append(grad_b2)

        #WEIGHTS UPDATE
        for i in range(np.shape(X)[0]):
            self.W1 -= learning_rate*grad_W1_save[i]
            self.b1 -= learning_rate*grad_b1_save[i]
            self.W2 -= learning_rate*grad_W2_save[i]
            self.b2 -= learning_rate*grad_b2_save[i]
            
        #print(np.shape(grad_W1))
        #print(grad_W1[0:5],"grad_W1")
        #print(np.shape(self.W1))
        #print(self.W1[0:5],"self.W1")
        #print(np.shape(self.b1))
        #print(self.b1[0:5],"self.b1")
        

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
        one_hot = np.zeros((np.size(train_y, 0), n_classes))
        for i in range(np.size(train_y, 0)):
            one_hot[i, train_y[i]] = 1
        train_y = one_hot
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
