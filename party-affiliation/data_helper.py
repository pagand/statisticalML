#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import random
import pandas as pd

vote_data_path = './data/house-votes-84.complete.data'
incomplete_data_path = './data/house-votes-84.incomplete.data'

# You do not need to make any changes to this file
# Call generate_q4_data to generate data for Q4. Call load_simulate_data to
# load data for your NB classifier  


def generate_q4_data(n, file_name):
    '''
....generate voting dataset
....n = number of samples
....file_name = output file name
....'''

    samples = n
    c = np.zeros(n)
    for (index, num) in enumerate(c):
        if(index%2 == 0):
            c[index] = 1

    d = {}

    c_string = []
    for (index, num) in enumerate(c):
        if (c[index] == 1):
            c_string.append('democrat')
        else:
            c_string.append('republican')

    d['Class'] = pd.Series(c_string)

    for i in range(16):
        temp = np.random.random_sample(samples)
        result = []
        for (index, num) in enumerate(temp):
            if(i>3):
                y_or_n = ('y' if num > 0.5 else 'n')
                result.append(y_or_n)
            elif(c[index] == 1):
                y_or_n = ('y' if num > 0.05 else 'n')
                result.append(y_or_n)
            else:
                y_or_n = ('y' if num > 0.95 else 'n')
                result.append(y_or_n)
        name = 'A' + str(i + 1)
        d[name] = pd.Series(result)

    # creates Dataframe.

    dff = pd.DataFrame(d)
    dff.to_csv(file_name, encoding='utf-8', index=False)

def load_simulate_data(file_name):
    '''
  load q4 voting dataset
  '''

    A = []
    C = []
    with open(file_name) as fin:
        next(fin)
        for line in fin:
            entries = line.strip().split(',')
            A_i = [(1 if x == 'y' else (0 if x == 'n' else -1))
                   for x in entries[1:]]
            assert -1 not in A_i
            C_i = int(entries[0] == 'democrat')
            A.append(A_i)
            C.append(C_i)
    A = np.vstack(A)
    C = np.array(C)
    (M, N) = A.shape
    l = list(range(M))
    A = A[l, :]
    C = C[l]
    return (A, C)


def load_vote_data():
    '''
  load voting dataset
  '''

    A = []
    C = []
    with open(vote_data_path) as fin:
        for line in fin:
            entries = line.strip().split(',')
            A_i = [(1 if x == 'y' else (0 if x == 'n' else -1)) for x in entries[1:]]
            assert -1 not in A_i
            C_i = int(entries[0] == 'democrat')
            A.append(A_i)
            C.append(C_i)
    A = np.vstack(A)
    C = np.array(C)
    (M, N) = A.shape
    l = list(range(M))
    A = A[l, :]
    C = C[l]
    return (A, C)


def load_incomplete_entry():
    '''
  load incomplete entry 1
  '''

    with open(incomplete_data_path) as fin:
        for line in fin:
            entries = line.strip().split(',')
            A_i = [(1 if x == 'y' else (0 if x == 'n' else -1))
                   for x in entries[:]]
            return np.array(A_i)
