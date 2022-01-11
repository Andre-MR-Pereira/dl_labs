#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import random

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
        
        n_size = 1
       
        self.W1 = np.random.normal(mu, sigma, size=(units[1], units[0]))
        self.b1 = np.zeros((units[1],n_size))
        self.W2 = np.random.normal(mu, sigma, size=(units[2], units[1]))
        self.b2 = np.zeros((units[2],n_size))
        
        print("NN shape:\n ===============================")
        print(n_features," L0 : features")
        print(hidden_size," L1 : hidden units")
        print(n_classes," L2 : classes")
        print(self.W1.shape," W1 shape")
        print(self.b1.shape," b1 shape")
        print(self.W2.shape," W2 shape")
        print(self.b2.shape," b2 shape")
        print(" ===============================")
        

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        #FORWARD PROPAGATION
        y_hat=[]
        #print(np.shape(self.W1))
        #print(self.W1[0:5],"self.W1 no predict")
        teste=True
        
        for x in X:
            z1 = self.W1.dot(x[:, None]) + self.b1
            h1 = np.maximum(z1 , 0)

            z2 = self.W2.dot(h1) + self.b2

            y_hat.append(np.argmax(z2))
            
        if teste==True:
            print("Predict shape:\n ###########################")
            print(x[:, None].shape," x shape")
            print(z1.shape," z1 shape")
            print(h1.shape," h1 shape")
            print(z2.shape," z2 shape")
            print(" ###########################")
            
        
        return y_hat

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print("Evaluate shape:\n TTTTTTTTTTTTTTTTTTTTTTTTTT")
        print(n_correct," n_correct")
        print(n_possible," n_possible")
        print(n_correct / n_possible," Ratio")
        
        print("Results:")
        print(y_hat[0:7],"y_hat predicted")
        print(y[0:7],"y true")
        print(" TTTTTTTTTTTTTTTTTTTTTTTTTT")
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        
        n_classes = 10
        one_hot = np.zeros((np.size(y, 0), n_classes))
        for i in range(np.size(y, 0)):
            one_hot[i, y[i]] = 1
        y = one_hot
        
        teste=True
        
        for x_step,y_step in zip(X,y):
            z1 = self.W1.dot(x_step[:, None]) + self.b1
            h1 = np.maximum(z1 , 0)

            z2 = self.W2.dot(h1) + self.b2
            
            #if teste==True:
                #print(z2,"z2",z2.max())            
            z2 -= z2.max()
            #if teste==True:
                #teste=False
                #print(z2,"z2")
            
            probs = np.exp(z2) / np.sum(np.exp(z2))
            
            #BACKWARD PROPAGATION: so atualizar os weights no fim da back prop TODA!
            #print(probs,"probs")
            #print(y_step,"y_step")
            grad_z2 = probs - y_step[:, None]
            #print(grad_z2,"grad_z")
            grad_W2 = grad_z2.dot(h1.T)
            grad_b2 = grad_z2
            grad_h1 = self.W2.T.dot(grad_z2)
            
            
            grad_z1 = grad_h1 * ((h1 > 0) * 1)
            grad_W1 = np.dot(grad_z1,x_step[:, None].T)
            grad_b1 = grad_z1

            if teste == True:
                teste=False
                print("Evaluate shape:\n $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                print(X.shape," X shape")
                print(y.shape," y shape")
                print(x_step[:, None].shape," x_step shape")
                print(y_step[:, None].shape," y shape")
                print(z1.shape," z1 shape")
                print(h1.shape," h1 shape")
                print(z2.shape," z2 shape")
                print(probs.shape," probs shape")
                print(grad_z2.shape," grad_z2 shape")
                print(grad_W2.shape," grad_W2 shape")
                print(grad_b2.shape," grad_b2 shape")
                print(grad_h1.shape," grad_h1 shape")
                print(grad_z1.shape," grad_z1 shape")
                print(grad_W1.shape," grad_W1 shape")
                print(grad_b1.shape," grad_b1 shape")
                print(self.W1.shape," self.W1 shape")
                print(self.b1.shape," self.b1 shape")
                print(self.W2.shape," self.W2 shape")
                print(self.b2.shape," self.b2 shape")
                print(" $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                
            #WEIGHTS UPDATE
            self.W1 -= learning_rate*grad_W1
            self.b1 -= learning_rate*grad_b1
            self.W2 -= learning_rate*grad_W2
            self.b2 -= learning_rate*grad_b2

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
