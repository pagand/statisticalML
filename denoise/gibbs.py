# CMPT 727 PA2
# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Heng Liu
# Final Edit: Pedram Agand
# Date : 3/1/2020

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

MAX_BURNS = 50
MAX_SAMPLES = 200
ETA = 1
BETA = 1



def denoise_image(filename, initialization='rand', logfile=None):
  '''
   TODO
   Do Gibbs sampling and compute the energy of each assignment for the image
   specified in filename.
   It should run MAX_BURNS iterations of burn in and then
   MAX_SAMPLES iterations for collecting samples.

   It is highly recommended to break the implementation in separate functions.
   And only use this function as a wrapper function.

   filename: file name of image in txt
   initialization: 'same' or 'neg' or 'rand'
   logfile: the file name that stores the energy log (will use for plotting
       later) look at the explanation of plot_energy to see detail

   For Q2:
    A log file with file name taken from the value of logfile should be created
   For Q3:
   return value: denoised
       denoised: a 2d-array with the same size of the image, the value of each entry
           should be either 0(black) or 1(white).
         This value is calculated based on the posterior probability of that being 1 (estimated by the Gibbs
           sampler).


  '''
  X = read_txt_file(filename)
  print(X)
  output_file = 'log'
  if initialization == 'rand':
    Y = np.random.randint(2, size=X.shape)
    Y = 2*Y - 1
    output_file += '_rand'
  elif initialization == 'same':
    Y= X.copy()
    output_file += '_same'
  elif initialization == 'neg':
    Y = (-1) * X.copy()
    output_file += '_neg'
  else:
    print("Wrong Initialization!")
    return

  
  Y_history = []
  energy_history = []
  C = np.zeros(Y.shape)
  edge_set = generate_ising_edge_set(X.shape)
  for iteration in range(MAX_BURNS+MAX_SAMPLES):
    if iteration % 10 == 0:
      print(iteration)  
    for i in range(Y.shape[0]):
      for j in range(Y.shape[1]):
        neighbours = []
        if i-1 >=0:
          neighbours.append((i-1,j))
        if i+1 <= Y.shape[0] - 1:
          neighbours.append((i+1,j))
        if j-1 >=0:
          neighbours.append((i, j-1))
        if j+1 <= Y.shape[1] - 1:
          neighbours.append((i,j+1))
        
        temp = (2 * BETA * sum([Y[m,n] for m,n in neighbours])) + (2*ETA*X[i,j])
        P_y_1 = math.exp(temp) / (1+math.exp(temp))
        Y[i,j] = 2*(np.random.binomial(1,P_y_1,1)[0]) - 1
        
        if (iteration > MAX_BURNS) and (Y[i,j] == 1): C[i,j] += 1 
        
    
    energy = (-1)* sum([Y[i,j]*X[i,j] for i in range(Y.shape[0]) for j in range(Y.shape[1])])
    energy -= BETA* sum([Y[i,j]*Y[i_prime, j_prime] for i,j,i_prime,j_prime in list(edge_set)])
    
    
    if iteration > MAX_BURNS:
      Y_history.append(Y)
      energy_history.append((iteration+1, energy, 'B'))
      
    else:
      energy_history.append((iteration+1, energy, 'S'))
  write_log_file(output_file, energy_history)

  C /= MAX_SAMPLES
  C[C<0.5] = 1
  C[C!=1] = 0
  return C

  


# added by me

def write_log_file(outputname, energy_history):
  with open(outputname, 'w') as file:
    for item in energy_history:
      file.write(str(item[0]) + " " + str(item[1]) + " " + str(item[2]) + "\n")
    

def generate_ising_edge_set(shape):
  edge_set = set()
  for i in range(shape[0]):
      for j in range(shape[1]):
        if i-1 >=0:
          if (i-1,j,i,j) not in edge_set:
            edge_set.add((i,j,i-1,j))
        if i+1 <= shape[0] - 1:
          if (i+1,j,i,j) not in edge_set:
            edge_set.add((i,j,i+1,j))
        if j-1 >=0:
          if (i,j-1,i,j) not in edge_set:
            edge_set.add((i,j,i,j-1))
        if j+1 <= shape[1] - 1:
          if (i,j+1,i,j) not in edge_set:
            edge_set.add((i,j,i,j+1))
  return edge_set

# end of added by me

def plot_energy(filename):
  '''
  filename: a file with energy log, each row should have three terms separated
    by a \t:
      iteration: iteration number
      energy: the energy at this iteration
      S or B: indicates whether it's burning in or a sample
  e.g.
      1   -202086.0   B
      2   -210446.0   S
      ...
  '''
  its_burn, energies_burn = [], []
  its_sample, energies_sample = [], []
  with open(filename, 'r') as f:
    for line in f:
      it, en, phase = line.strip().split()
      if phase == 'B':
        its_burn.append(float(it))
        energies_burn.append(float(en))
      elif phase == 'S':
        its_sample.append(float(it))
        energies_sample.append(float(en))
      else:
        print("bad phase: -%s-" % phase)

  p1, = plt.plot(its_burn, energies_burn, 'r')
  p2, = plt.plot(its_sample, energies_sample, 'b')
  plt.title(filename)
  plt.legend([p1, p2], ["burn in", "sampling"])
  plt.savefig(filename)
  plt.close()


def read_txt_file(filename):
  '''
  filename: image filename in txt
  return:   2-d array image
  '''
  f = open(filename, "r")
  lines = f.readlines()
  height = int(lines[0].split()[1].split("=")[1])
  width  = int(lines[0].split()[2].split("=")[1])
  Y = [[0]*(width+2) for i in range(height+2)]
  print(np.shape(Y))
  for line in lines[2:]:
    i, j, val = [int(entry) for entry in line.split()]
    Y[i+1][j+1] = val
  return np.array(Y)


def convert_to_png(denoised_image, title):
  '''
  save array as a png figure with given title.
  '''
  plt.imshow(denoised_image, cmap=plt.cm.gray)
  plt.title(title)
  plt.savefig(title + '.png')


def text_to_png(imagefile):
    imageData = read_txt_file(imagefile)
    imageData = .5 * (1 - imageData)
    convert_to_png(imageData,imagefile)

def get_error(img_a, img_b):
  '''
  compute the fraction of all pixels that differ between the two input images.
  '''
  N = len(img_b[0])*len(img_b)*1.0
  return sum([sum([1 if img_a[row][col] != img_b[row][col] else 0
                   for col in range(len(img_a[0]))])
              for row in range(len(img_a))]
             ) / N


def run_q2():
  '''
  Run denoise_image function with different initialization and plot out the
  energy functions.
  '''
  #Saving the denoised image for Q3
  global denoised_a

  denoise_image("a_noise10.png.txt", initialization='rand',
                logfile='log_rand')
  
  denoise_image("a_noise10.png.txt", initialization='neg',
                logfile='log_neg')
  denoised_a = denoise_image("a_noise10.png.txt",
                                             initialization='same',
                                               logfile='log_same')

  # plot out the energy functions
  plot_energy("log_rand")
  plot_energy("log_neg")
  plot_energy("log_same")


def run_q3():
  '''
  Run denoise_image function with two different pics, and
  report the errors between denoised images and original image
  '''
  global denoised_b
  denoised_b = denoise_image("b_noise10.png.txt",
                                             initialization='same',
                                             logfile=None)
  orig_img_a = read_txt_file("a.png.txt")
  orig_img_a = .5 * (1 - orig_img_a)
  orig_img_b = read_txt_file("b.png.txt")
  orig_img_b = .5 * (1 - orig_img_b)

  # save denoised images and original image to png figures
  convert_to_png(denoised_b, "denoised_b")
  convert_to_png(denoised_a, "denoised_a")
  convert_to_png(orig_img_b, "orig_img_b")
  convert_to_png(orig_img_a, "orig_img_a")

  N, M = orig_img_a.shape
  print("restoration error for image %s : %s" %
        ("a", np.sum((orig_img_a != denoised_a)[1:N-1, 1:M-1]) / float((N-1) * (M-1))))
  N, M = orig_img_b.shape
  print("restoration error for image %s : %s" %
        ("b", np.sum((orig_img_b != denoised_b)[1:N-1, 1:M-1]) / float((N-1) * (M-1))))



if __name__ == "__main__":
  run_q2()
  run_q3()
  
'''
    TODO modify or use run_q2() and run_q3() to
    run your implementation for this assignment.

 run_q2()
 run_q3()
'''

