import numpy as np
import scipy as sp

# We generate some arbitrary examples to test our scoring function and to show how we use it.
# Note that our true targets prediction have the same format, just the size change.

# First test example

# from PIL import Image                                                           
# pred1 = np.zeros((200, 200))
# sol1 = np.zeros((200, 200))	
# sol1[:100,:100] = 255
# pred1[0:100,:] = 255
# pred1[:,:100] = 255
# im_pred = Image.fromarray(pred1)
# im_sol = Image.fromarray(sol1)
# im_sol.show()
# im_pred.show()

# Second test example


# pred1 = np.array([[0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
# 				  [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

# sol2 = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
# 				 [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])

# score = iou_metric(sol2, pred1)
# print(score)

# score_iou = 0.33 (for both data example)

# @param -> solutions : list of all true target to be predicted.
# @param -> predictions : list of all predictions of the model.
# @return -> the iou score computed.
# solutions shape : (N, W*H)
# predictions shape : (N, W*H)
def iou_metric(solutions, predictions):
	intersection = 0
	union = 0
	for sol, pred in zip(solutions, predictions):
		intersection += np.sum(np.logical_and(sol, pred))
		union += np.sum(np.logical_or(sol, pred))
	iou_score = intersection / union
	return iou_score
