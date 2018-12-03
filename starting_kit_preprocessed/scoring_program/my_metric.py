import numpy as np
import scipy as sp

def accuracy(solution, prediction):
	error = 0
	for sol, pred in zip(solution, prediction):
		if sol != pred:
			error += 1
	return 1 - (error / len(solution))
