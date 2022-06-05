import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV


# load data
dpa = pd.read_csv('./data/house-votes-84.complete.csv')
dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
for i in range(16):
	index = 'A'+ str(i+1)
	dpa[index] = dpa[index].map({'y': 1, 'n': 0})
#dpa.info()

pay = dpa.Class
paX = dpa.drop('Class', axis = 1)



'''
  10-cv with house-votes-84.complete.csv using LASSO
  - train_subset: train the classifier on a smaller subset of the training
    data
  -subset_size: the size of subset when train_subset is true
  NOTE you do *not* need to modify this function
  '''
def lasso_evaluate(train_subset=False, subset_size = 0,get_coeff = 0):
	global pay, paX
	sample_size = pay.shape[0]
	tot_incorrect=0
	tot_test=0
	tot_train_incorrect=0
	tot_train=0
	step = int( sample_size/ 10 + 1)
	for holdout_round, i in enumerate(range(0, sample_size, step)):
		#print("CV round: %s." % (holdout_round + 1))
		if(i==0):
			X_train = paX.iloc[i+step:sample_size]
			y_train = pay.iloc[i+step:sample_size]
		else:
			X_train =paX.iloc[0:i]
			X_train = X_train.append(paX.iloc[i+step:sample_size], ignore_index=True)
			y_train = pay.iloc[0:i]
			y_train = y_train.append(pay.iloc[i+step:sample_size], ignore_index=True)
		X_test = paX.iloc[i: i+step]
		y_test = pay.iloc[i: i+step]
		if(train_subset):
			X_train = X_train.iloc[0:subset_size]
			y_train = y_train.iloc[0:subset_size]
		#print(" Samples={} test = {}".format(y_train.shape[0],y_test.shape[0]))
		# train the classifiers
		lasso = Lasso(alpha = 0.001)
		lasso.fit(X_train, y_train)
		lasso_predit = lasso.predict(X_test)           # Use this model to predict the test data
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_test.values.tolist()[index] != num):
				error+=1
		tot_train_incorrect+= error
		tot_train += len(lasso_result)
		#print('Error rate {}'.format(1.0*error/len(lasso_result)))
		lasso_predit = lasso.predict(X_train)           # Use this model to get the training error

		# added by me
		if get_coeff:
			coeff = lasso.coef_
			pdmr = sum(abs(coeff) ==0)/12
		lasso_result = [1 if x>0.5 else 0 for x in lasso_predit]
		error = 0
		for (index, num) in enumerate(lasso_result):
			if(y_train.values.tolist()[index] != num):
				error+=1
		#print('Train Error rate {}'.format(1.0*error/len(lasso_result)))
		tot_incorrect += error
		tot_test += len(lasso_result)
	#print('10CV Error rate {}'.format(1.0*tot_incorrect/tot_test))
	#print('10CV train Error rate {}'.format(1.0*tot_train_incorrect/tot_train))

	if get_coeff:
		return 1.0 * tot_incorrect / tot_test, 1.0 * tot_train_incorrect / tot_train, pdmr
	else:
		return 1.0*tot_incorrect/tot_test, 1.0*tot_train_incorrect/tot_train

'''
 For Q5
 Run LASSO, using the commonly-used strategy of replacing unknown values with 0 (“no)
 We have modified the csv file for this purpose.
  NOTE you do *not* need to modify this function
  '''

def lasso_evaluate_incomplete_entry():
	# get incomplete data
	dpc = pd.read_csv('./data/house-votes-84.incomplete.csv')
	for i in range(16):
		index = 'A'+ str(i+1)
		dpc[index] = dpc[index].map({'y': 1, 'n': 0})
	lasso = Lasso(alpha = 0.001)
	lasso.fit(paX, pay)
	lasso_predit = lasso.predict(dpc)
	print(lasso_predit)


def main():
	'''
	TODO modify or use the following code to evaluate your implemented
	classifiers
	Suggestions on how to use the starter code for Q2, Q3, and Q5:
	 '''
	#For Q2
	print("Q2")
	error_rate, unused = lasso_evaluate(train_subset=False)
	print('10CV Error rate {}'.format(error_rate))

	#For Q3
	print("------------------------------------------")
	print("Q3")
	train_error = np.zeros(10)
	test_error = np.zeros(10)
	for i in range(10):
		x, y =lasso_evaluate(train_subset=True, subset_size=i*10+10)
		test_error[i] = y
		train_error[i] =x
	print("train error for different data size")
	print(train_error)
	print("test error for different data size")
	print(test_error)


	# Q5
	print("------------------------------------------")
	print("Q5")
	print('LASSO  P(C= 1|A_observed) ')
	lasso_evaluate_incomplete_entry()




	#Q4
	#TODO
	print("------------------------------------------")
	print("Q4")
	dpa = pd.read_csv('./data/synthetic.csv')
	dpa['Class'] = dpa['Class'].map({'republican': 0, 'democrat': 1})
	for i in range(16):
		index = 'A' + str(i + 1)
		dpa[index] = dpa[index].map({'y': 1, 'n': 0})
	# dpa.info()
	global pay, paX
	pay = dpa.Class
	paX = dpa.drop('Class', axis=1)

	train_error = np.zeros(10)
	test_error = np.zeros(10)
	pdmr_list = np.zeros(10)
	for i in range(10):
		x, y,pdmr = lasso_evaluate(train_subset=True, subset_size=i * 400 + 400, get_coeff = 1)
		pdmr_list[i] = pdmr
		test_error[i] = y
		train_error[i] = x
		print('  10-fold cross validation for synthetic data total test error {:2.4f} total train error {:2.4f} on {} ''examples'.format(
				y, x, (i + 1) * 400))
	sample_size = range(400, 4400, 400)
	plt.plot(sample_size,pdmr_list)
	plt.show()



	#You may find lasso.coef_ useful


if __name__ == "__main__":
	main()
