import os
import sys
import numpy as np
from scipy.special import logsumexp
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
from lasso1 import lasso_evaluate
# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry,load_simulate_data,  generate_q4_data
# helpers to learn and traverse the tree over attributes
import matplotlib.pyplot as plt


# pseudocounts for uniform dirichlet prior
alpha = 0.1


#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------


class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    '''
    super(NBCPT, self).__init__()

    self.i = A_i
    # Given each value of c (ie, c = 0 and c = 1)
    self.pseudocounts = [alpha, alpha]
    self.c_count = [2 * alpha, 2 * alpha]

  def learn(self, A, C):
    '''
    TODO
    populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    for i in range(2):
      self.c_count[i] += len(C[C == i])
      self.pseudocounts[i] += len(C[(A[:, self.i] == 1) & (C == i)])


  def get_cond_prob(self, entry, c):
    ''' TODO
    return the conditional probability P(A_i|C=c) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables
            e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class
    '''
    entry_is_one_prob = self.pseudocounts[c] / float(self.c_count[c])
    return entry_is_one_prob if entry[self.i] == 1 else (1 - entry_is_one_prob)


class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    super(NBClassifier, self).__init__()
    assert (len(np.unique(C_train))) == 2
    n, m = A_train.shape
    self.cpts = [NBCPT(i) for i in range(m)]
    self.P_c = 0.0
    self._train(A_train, C_train)

  def _train(self, A_train, C_train):
    ''' TODO
    train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    self.P_c = len(C_train[C_train == 1]) / float(len(C_train))
    for cpt in self.cpts:
      cpt.learn(A_train, C_train)

  def classify(self, entry):
    ''' TODO
    return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)
    '''
    # Calculate the log probability to avoid underflow issues.
    # We DO NOT normalize these results.
    P_c_pred = [0, 0]
    # Find all unobserved so we can try all settings of them.
    unobserved_idx = [i for i, e in enumerate(entry) if e == -1]
    # Add empty set so loop is always executed even when we have full
    # assignment.
    unobserved_assigments = list(itertools.product(
      (0, 1), repeat=len(unobserved_idx))) + [[]]
    for unobserved_assigment in unobserved_assigments:
      probability = 0.0
      for i, value in enumerate(unobserved_assigment):
        entry[unobserved_idx[i]] = value

      # Calculate joint given the above
      full_P_c_pred = [1 - self.P_c, self.P_c]
      for cpt in self.cpts:
        for i in range(2):
          full_P_c_pred[i] *= cpt.get_cond_prob(entry, i)

      for i in range(2):
        P_c_pred[i] += full_P_c_pred[i]

    # Normalize the distributions.
    P_c_pred /= np.sum(P_c_pred)
    c_pred = np.argmax(P_c_pred)
    return (c_pred, np.log(P_c_pred[c_pred]))


  def predict_unobserved(self, entry, index):
    ''' TODO
    Predicts P(A_index  | mid entry)
    Return a tuple of probabilities for A_index=0  and  A_index = 1
    We only use the 2nd value (P(A_index =1 |entry)) in this assignment
    '''
    if entry[index] == 1 or entry[index] == 0:
      return [1 - entry[index], entry[index]]

      # Not observed, so use model to predict.
    P_index_pred = [0.0, 0.0]
    # Find all unobserved so we can try all settings of them except the one
    # we wish to predict.
    unobserved_idx = [i for i, e in enumerate(entry) if e == -1 and i != index]
    # Add empty set so loop is always executed even when we have full
    # assignment.
    unobserved_assigments = list(itertools.product(
      (0, 1), repeat=len(unobserved_idx))) + [[]]
    for p_value in range(2):
      entry[index] = p_value
      for unobserved_assigment in unobserved_assigments:
        probability = 0.0
        for i, value in enumerate(unobserved_assigment):
          entry[unobserved_idx[i]] = value

        # Calculate joint given the above
        full_P_c_pred = [1 - self.P_c, self.P_c]
        for cpt in self.cpts:
          for i in range(2):
            full_P_c_pred[i] *= cpt.get_cond_prob(entry, i)
        # Sum over c.
        P_index_pred[p_value] += np.sum(full_P_c_pred)

      # Normalize the distributions.
    P_index_pred /= np.sum(P_index_pred)
    return P_index_pred



# load data
A_data, C_data = load_vote_data()


def evaluate(classifier_cls, train_subset=False, subset_size = 0,get_bills = 0):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or other classifiers
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true
  NOTE you do *not* need to modify this function
  '''
  global A_data, C_data

  A, C = A_data, C_data


  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  train_correct = 0
  train_test = 0
  step = int(M / 10 + 1)
  for holdout_round, i in enumerate(range(0, M, step)):
    # print("Holdout round: %s." % (holdout_round + 1))
    A_train = np.vstack([A[0:i, :], A[i+step:, :]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i: i+step, :]
    C_test = C[i: i+step]
    if train_subset:
      A_train = A_train[: subset_size, :]
      C_train = C_train[: subset_size]
    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    train_results = get_classification_results(classifier, A_train, C_train)
    if (get_bills):
       pdmr = get_bill_importance(classifier, A_train, C_train)

    # print(
    #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)
    train_correct += sum(train_results)
    train_test += len(train_results)

  if get_bills:
    return 1.*tot_correct/tot_test, 1.*train_correct/train_test, pdmr
  else:
    return 1.*tot_correct/tot_test, 1.*train_correct/train_test


  # score classifier on specified attributes, A, against provided labels,
  # C
def get_classification_results(classifier, A, C):
  results = []
  pp = []
  for entry, c in zip(A, C):
    c_pred, unused = classifier.classify(entry)
    results.append((c_pred == c))
    pp.append(unused)
    # print('logprobs', np.array(pp))
  return results


def get_bill_importance(classifier, A, C):
  results = []
  dem = np.zeros(16)
  rep = np.zeros(16)
  for entry, c in zip(A, C):
    c_pred, unused = classifier.classify(entry)
    if(c_pred): # Democrat
      dem = dem + entry
    else:
      rep = rep+ entry
    results.append(c_pred)
    # print('logprobs', np.array(pp))

  pdem = dem/sum(results)
  prep = rep / (len(results)-sum(results))
  return sum(abs(pdem-prep)<0.1)/12



def evaluate_incomplete_entry(classifier_cls):

  global A_data, C_data

  # train a classifier on the full dataset
  classifier = classifier_cls(A_data, C_data)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)
  print("  P(C={}|A_observed) = {:2.4f}".format(c_pred, np.exp(logP_c_pred)))

  return


def predict_unobserved(classifier_cls, index=11):
  global A_data, C_data

  # train a classifier on the full dataset
  classifier = classifier_cls(A_data, C_data)
  # load incomplete entry 1
  entry = load_incomplete_entry()

  a_pred = classifier.predict_unobserved(entry, index)
  print("  P(A{}=1|A_observed) = {:2.4f}".format(index+1, a_pred[1]))

  return


def main():

  '''
  TODO modify or use the following code to evaluate your implemented
  classifiers
  Suggestions on how to use the starter code for Q2, Q3, and Q5:
  '''
  ##For Q1
  print('Naive Bayes (Q1)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test error {:2.4f}'.format(1 - accuracy))

  ##For Q3
  print("------------------------------------------")
  print('Naive Bayes (Small Data) (Q3)')
  train_error = np.zeros(10)
  test_error = np.zeros(10)
  for x in range(10):
    accuracy, train_accuracy = evaluate(NBClassifier, train_subset=True,subset_size = (x+1)*10)
    train_error[x] = 1-train_accuracy
    test_error[x] = 1- accuracy
    print('  10-fold cross validation total test error {:2.4f} total train error {:2.4f} on {} ''examples'.format(1 - accuracy, 1- train_accuracy  ,(x+1)*10))
  print("train error for different data size")
  print(train_error)
  print("test error for different data size")
  print(test_error)

  ## For plot
  # lasso
  train_error_lasso = np.zeros(10)
  test_error_lasso = np.zeros(10)
  for i in range(10):
    x, y = lasso_evaluate(train_subset=True, subset_size=i * 10 + 10)
    test_error_lasso[i] = y
    train_error_lasso[i] = x

  sample_size = range(10,110,10)
  plt.plot(sample_size,train_error,label = "train NB")
  plt.plot(sample_size, test_error, label = "test NB")
  plt.plot(sample_size, train_error_lasso, label = "train Lasso")
  plt.plot(sample_size, test_error_lasso, label = "test Lasso")
  plt.xlabel('sample size')
  plt.ylabel('Error')
  plt.legend()




  ##For Q5
  print("------------------------------------------")
  print('Naive Bayes Classifier on missing data (Q5)')
  evaluate_incomplete_entry(NBClassifier)
  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
       index + 1))
  predict_unobserved(NBClassifier, index)





  ##For Q4 TODO
  print("------------------------------------------")
  print('Q4')
  generate_q4_data(4000, "./data/synthetic.csv")
  global A_data, C_data
  A_data, C_data = load_simulate_data("./data/synthetic.csv")
  train_error = np.zeros(10)
  test_error = np.zeros(10)
  pdmr_list = np.zeros(10)
  for x in range(10):
    accuracy, train_accuracy, pdmr = evaluate(NBClassifier, train_subset=True, subset_size=(x + 1) * 400, get_bills = 1)
    pdmr_list[x] = pdmr
    train_error[x] = 1 - train_accuracy
    test_error[x] = 1 - accuracy
    print('  10-fold cross validation for synthetic data total test error {:2.4f} total train error {:2.4f} on {} ''examples'.format(
      1 - accuracy, 1 - train_accuracy, (x + 1) * 400))

  sample_size = range(400, 4400, 400)
  plt.figure()
  plt.plot(sample_size, pdmr_list)

  plt.show()









if __name__ == '__main__':
  main()