import numpy as np
import sys
import cv2
import os

########## Preparation ##########
class normalizer:
    def __init__(self, matX):
        self.origin = matX
        self.mean = np.mean(matX, axis=1).reshape(-1, 1)
        self.var = np.var(matX, axis=1, ddof=1).reshape(-1, 1)
    def norm(self, matX):
        #return (matX - self.mean) / self.var
        return matX - self.mean
    def denorm(self, matX):
        #return matX*self.var + self.mean
        return matX + self.mean

num_sample = 39
num_test = 5
os.makedirs('./2020193004', exist_ok=True)
f = open('./2020193004/output.txt', 'w')


########## STEP 1 ##########
# Get matrix X
for i in range(num_sample):
    img = cv2.imread(f'./faces_training/face{i+1:02d}.pgm', -1)
    if i == 0:
        size = img.shape
        X = np.array(img).reshape(-1, 1)
    else:
        img = np.array(img).reshape(-1, 1)
        X = np.hstack((X, img))

# Calculate PCA
norm_X = normalizer(X)
X = norm_X.norm(X)
threshold = float(sys.argv[1])
U, s, V = np.linalg.svd(X)

# Calculate Dimension
s = (s**2)
sum_lambda = np.sum(s)
summ = 0
dimension = 0
while True:
    summ += s[dimension]
    dimension += 1
    if summ / sum_lambda > threshold:
        break

# Write File
f.write("#"*10 + " STEP 1 " + "#"*10 + "\n")
f.write(f"Input Percentage: {threshold}\n")
f.write(f"Selected Dimension: {dimension}\n\n")


########## STEP 2 ##########
# Reconstruction
W = U[:dimension,:]
Y = np.matmul(W, X)
recon = np.matmul(np.transpose(W), Y)
recon = norm_X.denorm(recon)

# Calculate MSE / Write Image
mse = []
for i in range(num_sample):
    img = recon[:,i]
    mse.append(np.mean((img - norm_X.origin[:,i]) ** 2))
    cv2.imwrite(f'./2020193004/face{i+1:02d}.pgm', img.reshape(size))

# Write File
f.write("#"*10 + " STEP 2 " + "#"*10 + "\n")
f.write(f"Average: {np.mean(np.array(mse)):02.4f}\n")
for i in range(num_sample):
    f.write(f"{i+1:02d}: {mse[i]:02.4f}\n")


########## STEP 3 ##########
# Image Recognition
min_idx = []
for i in range(num_test):
    img = cv2.imread(f'./faces_test/test{i+1:02d}.pgm', -1).reshape(-1, 1)
    img = norm_X.norm(img)
    w_img = np.matmul(W, img)
    min_idx.append(np.argmin(np.sum((Y - w_img)**2, axis=0) ** (1/2)))

# Write File
f.write("\n" + "#"*10 + " STEP 3 " + "#"*10 + "\n")
for i in range(num_test):
    f.write(f"test{i+1:02d}.pgm ==> face{min_idx[i]+1:02d}.pgm\n")




