import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_class= W.shape[1]
  num_train = X.shape[0]
  grad_i = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    score = W.transpose().dot(X[i])
    loss -= np.log(pow(np.e,score[y[i]])/sum(pow(np.e,score)))/num_train
    grad_i += (1/num_train)*(X[i][:,np.newaxis].dot(pow(np.e,score)[np.newaxis,:])/sum(pow(np.e,score)))
    grad_i.transpose()[y[i]] -= (1/num_train)*X[i]

  loss += reg*np.sum(W*W) 
  dW = grad_i + 2*reg*W    
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_class= W.shape[1]
  num_train = X.shape[0]
  grad_i = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    score = W.transpose().dot(X[i])
    loss -= np.log(pow(np.e,score[y[i]])/sum(pow(np.e,score)))/num_train
    grad_i += (1/num_train)*(X[i][:,np.newaxis].dot(pow(np.e,score)[np.newaxis,:])/sum(pow(np.e,score)))
    grad_i.transpose()[y[i]] -= (1/num_train)*X[i]

  loss += reg*np.sum(W*W) 
  dW = grad_i + 2*reg*W    
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

