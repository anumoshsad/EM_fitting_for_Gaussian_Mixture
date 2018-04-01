#!/usr/bin/python

from __future__ import print_function, division
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

np.random.seed(1234)

def initialize(data, K):
	N = data.shape[0]
	dim = data.shape[1]

	idx = np.random.randint(N, size = K)
	means = data[idx, :]
	
	sigmas = np.zeros([K, dim, dim])
	coeffs = np.zeros([K])
	for i in range(K):
		sigmas[i] = np.identity(dim)
		coeffs[i] = np.random.random()
	coeffs = coeffs/coeffs.sum()
	
	return means, sigmas, coeffs

def E_step(data, means, sigmas, coeffs):
	N = data.shape[0]
	K = coeffs.shape[0]
	gamma = np.zeros([N,K])

	for k in range(K):
		gamma[:,k] = coeffs[k] * norm.pdf(data, means[k], sigmas[k])
	
	return gamma / gamma.sum(axis = 1, keepdims = True)




def M_step(data, gamma, means, sigmas, coeffs):
	N, K = gamma.shape
	for k in range(K):
		N_k = np.sum(gamma[:,k])
		means[k] = np.sum(gamma[:,k].reshape([N,1]) * data, axis = 0)/N_k
		temp = data - means[k]
		sigmas[k] = np.sum(gamma[:,k].reshape([N,1,1]) * np.einsum('bi, bj ->bij', temp,temp), axis = 0)/N_k
		coeffs[k] = N_k/N
	####tied covariance
	#sigmas[:] = sigmas.sum(axis = 0)/K
	return means, sigmas, coeffs


def log_likelihood(data, means, sigmas, coeffs):
	N = data.shape[0]
	K = coeffs.shape[0]
	l = np.zeros([N,K])
	for k in range(K):
		l[:, k] = coeffs[k] * norm.pdf(data, means[k], sigmas[k])
	return np.sum(np.log(l.sum(axis=1)),axis=0)

if __name__ == '__main__':
	data = np.loadtxt('points.dat')
	plt.scatter(data[:,0], data[:,1])
	plt.title('Scatterplot')
	plt.savefig('scatter.png')
	plt.show()
	
	train = data[0:900]
	dev = data[900::]

	maxIter = 100
	epsilon = 1e-8
	
	clusters = [2,3,4,5,6,7,8]
	Training_loglike = []
	Dev_loglike = []
	get_clusters = {}  # this is for finding the cluster the points belong to.

##############################################
######## Main algo for EM fitting ############
	for k in clusters:
		cur_train_loglike = []
		cur_dev_loglike = []

		prev_loglike = -float('inf')
		means, sigmas, coeffs = initialize(train, k)
		for _ in range(maxIter):
			gamma = E_step(train, means, sigmas, coeffs)
			means, sigmas, coeffs = M_step(train, gamma, means, sigmas, coeffs)
			
			train_loglike = log_likelihood(train, means, sigmas, coeffs)
			cur_train_loglike.append(train_loglike)
			dev_loglike = log_likelihood(dev, means, sigmas, coeffs)
			cur_dev_loglike.append(dev_loglike)

			loglike_diff = np.absolute(train_loglike - prev_loglike)
			if loglike_diff < epsilon:
				break
			prev_loglike = train_loglike

		Training_loglike.append(cur_train_loglike)
		Dev_loglike.append(cur_dev_loglike)
		get_clusters[k] = gamma.argmax(axis = 1)
 

		print('****Number of Clusters: ', k)
		print('Means: ', means)
		print('Mixing Coefficients: ', coeffs)
		print('======================================')


###########################################
####### plotting cluters for k = 4 ########
	colors = 'bgrcmyk'
	k = 4
	plt.scatter(data[:,0], data[:,1], c = [colors[x] for x in get_clusters[k]])	
	plt.title('Scatterplot ( 4 clusters)')
	plt.savefig('4_cluster_sep_cov.png')
	#plt.savefig('4_cluster_tied_cov.png')	
	plt.show()

###########################################
####### plotting cluters for k = 6 ########
	colors = 'bgrcmyk'
	k = 6
	plt.scatter(data[:,0], data[:,1], c = [colors[x] for x in get_clusters[k]])	
	plt.title('Scatterplot ( 6 clusters)')
	plt.savefig('6_cluster_sep_cov.png')
	#plt.savefig('6_cluster_cov.png')	
	plt.show()

############################################
###### Plotting the loglikelihood ##########
	legends = []
	marker = ['bo-', 'gv-', 'rs-', 'c*-', 'mh-', 'y+-', 'kx-']
	k=2
	for loglike, m in zip(Training_loglike, marker):
		plt.plot(loglike, m)
		legends.append('k = '+str(k))
		k+=1

	plt.ylabel('Training Log-likelihood')
	plt.xlabel('Number of Iterations')
	plt.legend(legends, loc = 'lower right')
	plt.title('Log-likelihood vs iterations')
	plt.savefig('Training_separate_cov.png')
	#plt.savefig('Training_tied_cov.png')
	plt.show()

	legends = []
	marker = ['bo-', 'gv-', 'rs-', 'c*-', 'mh-', 'y+-', 'kx-']
	k=2
	for loglike, m in zip(Dev_loglike, marker):
		plt.plot(loglike, m)
		legends.append('k = '+str(k))
		k+=1

	plt.ylabel('Dev Data Log-likelihood')
	plt.xlabel('Number of Iterations')
	plt.legend(legends, loc = 'lower right')
	plt.title('Log-likelihood vs iterations')
        plt.savefig('Dev_separate_cov.png')
	#plt.savefig('Dev_tied_cov.png')
	plt.show()
