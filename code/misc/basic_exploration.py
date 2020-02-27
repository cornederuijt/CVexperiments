"""Some exploration with very basic operations"""
from PIL import Image
import pylab as pl
import numpy as np
from scipy.ndimage import filters
import cv2
from scipy.ndimage import measurements, morphology

# Exercise 1)
# Read image and convert to gray scale (convert 'L'), to set the values in the array to floating point, at 'f'
im = pl.array(Image.open('./data/raw_data/dogs-vs-cats/train/cat.128.jpg').convert('L'))

pl.imshow(im)

# some points
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]
# plot the points with red star-markers
pl.plot(x, y, 'r*')
# line plot connecting the first two points
pl.plot(x[:2], y[:2])
# add title and show the plot
pl.title("plot cat").show()

# Exercise 2)
# create a new figure
pl.figure()
# don’t use colors
pl.gray()
# show contours with origin upper left corner
pl.contour(im, origin='image').axis('equal')
pl.axis('off')

# Contour histogram?
pl.figure()
pl.hist(im.flatten(), 128)
pl.show()


# Exercise 3)
#
im2 = 255 - im  # invert image (apparently grey scales go between 0 and 255)
pl.imshow(im2)
im3 = (100.0/255) * im + 100 #  clamp to interval 100...200 im4 = 255.0 * (im/255.0)**2 #squared
pl.imshow(im3)

# pil_im = Image.fromarray(im) : convert array structure back to image
# pil_im = Image.fromarray(uint8(im)) : in case the data type was changed
# array(pil_im.resize(sz)): is used to resize the image (cannot conveniently be done with numpy)

# Exercise 4) PCA

# First full matrix
row_means = im.mean(axis=0)
m, n = im.shape
X = im - row_means  # Center over rows
U, S, VH = np.linalg.svd(X)  # Compute SVD
S_new = np.zeros((m, n))
S_new[:m, :m] = np.diag(S)
X_new = np.dot(U, np.dot(S_new, VH))

# new image:
pl.figure()
pl.gray()
pl.imshow(X_new)

# compressed matrix
K = 10
U_new = U[:, :K]
VH_new = VH[:K, :]
S_new = np.zeros((K, K))
S_new[:K, :K] = np.diag(S[:K])
X_new = np.dot(U_new, np.dot(S_new, VH_new))

# new image:
pl.figure()
pl.gray()
pl.imshow(X_new)

# Exercise 5) Blurring images
im2 = filters.gaussian_filter(im, 5)
pl.imshow(im2)

# Exercise 6) Image derivatives
#Sobel derivative filters
imx = np.zeros(im.shape)
filters.sobel(im,1,imx)
imy = np.zeros(im.shape)
filters.sobel(im,0,imy)
magnitude = np.sqrt(imx**2+imy**2)

pl.imshow(imx)
pl.imshow(imy)
pl.imshow(magnitude)

# Exercise 7) Some fun with SIFT
img = cv2.imread('./data/raw_data/dogs-vs-cats/train/cat.128.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img=cv2.drawKeypoints(gray, kp, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

res = sift.compute(gray, kp)

pl.imshow(img)

