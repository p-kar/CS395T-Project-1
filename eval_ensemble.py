import numpy as np

resnet_fname = 'predictions.txt'
xception_fname = 'output.txt'
data_file = 'data/yearbook/valid.txt'

def readFile(fname):
	with open(fname, 'r') as fp:
		content = fp.readlines()

	content = [x.strip() for x in content]
	data = [x.split('\t') for x in content]

	ret = [float(d[1]) for d in data]
	return ret


with open(resnet_fname, 'r') as f:
	resnet_preds = [float(l.strip()) for l in f.readlines()]

with open(xception_fname, 'r') as f:
	xception_preds = [float(l.strip()) for l in f.readlines()]

preds = [(a * 5.0 + b) / 6.0 for a, b in zip(resnet_preds, xception_preds)]
gt = readFile(data_file)

norm = [abs(g - p) for g, p in zip(gt, preds)]
print (np.array(norm).mean())
