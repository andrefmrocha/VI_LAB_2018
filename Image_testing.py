import numpy as np
import cv2
from extractRect import *

i = 0
img = cv2.imread('4.jpg')
b,g,r = cv2.split(img)
img_rgb = cv2.merge((r,g,b))
#Method to increase a certain color
import matplotlib.pyplot as plt
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# img = img.astype(float)
# img[:, :, 0] += 30
# img = np.clip(img, 0, 255).astype(np.uint8)
#plt.imshow(img)
# plt.show()
img = cv2.GaussianBlur(img_rgb, (5, 5), 0)


sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)

sobelx[sobelx < 0] = 0
sobely[sobely < 0] = 0
sobelxy[sobelxy < 0] = 0
sobelx[sobelx > 255] = 255
sobely[sobely > 255] = 255
sobelxy[sobelxy > 255] = 255

print(sobelxy)
print(sobelxy.shape)
# plt.subplot(131)
# plt.imshow(sobelx[:,:,0]/255., 'gray')
# plt.subplot(132)
# plt.imshow(sobely[:,:,0]/255., 'gray')
# plt.subplot(133)
# plt.imshow(sobelxy[:,:,0]/255., 'gray')
# plt.show()
from sklearn.cluster import KMeans
kernel = np.ones((7, 7), np.uint8)
lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
frows = np.arange(img.shape[0])[:,np.newaxis].repeat(img.shape[1], axis=1)
fcolumns = np.arange(img.shape[1])[:,np.newaxis].repeat(img.shape[0], axis=1).T
# plt.subplot(121)
# plt.imshow(frows)
# plt.subplot(122)
# plt.imshow(fcolumns)
# plt.show()

def norm(img):
    img = (img - min(img.ravel()))/(max(img.ravel())-min(img.ravel()))
    return img

feat = np.vstack((norm(img[:,:,0].ravel()), norm(img[:,:,1].ravel()), norm(img[:,:,2].ravel()), norm(frows.ravel()), norm(fcolumns.ravel()), norm(sobelxy[:,:,0].ravel()), norm(sobelxy[:,:,1].ravel()), norm(sobelxy[:,:,2].ravel()))).T
#feat = np.vstack((norm(img[:,:,0].ravel()), norm(img[:,:,1].ravel()), norm(img[:,:,2].ravel()), norm(frows.ravel()), norm(fcolumns.ravel()))).T
print(feat.shape)
#feat = feat[:, np.newaxis]
nclusters = 3
kmeans = KMeans(nclusters, 'k-means++').fit(feat)
labels = kmeans.labels_.reshape((img.shape[0],img.shape[1]))
plt.subplot(121)
plt.imshow(img_rgb)
plt.subplot(122)
plt.imshow(labels)
plt.show()

max_int = 0
for c in range(nclusters):
    bin = (labels == c)
    mean_int = np.mean(img[bin])

    if mean_int > max_int:
        c_card = c
        max_int = mean_int

    print(c, mean_int)


bin_card = (labels == c_card)*255

plt.imshow(bin_card,'gray')
plt.show()

bin_card = bin_card.astype(np.uint8)

# Copy the thresholded image.
im_floodfill = bin_card.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = bin_card.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = bin_card | im_floodfill_inv

plt.imshow(im_out,'gray')
plt.show()


idx_in  = np.where(im_out==255)
idx_out = np.where(im_out==0)
aa = np.ones_like(im_out)
aa[idx_in]  = 0

#get coordinate of biggest rectangle
#----------------
time_start = datetime.datetime.now()

rect_coord_ori, angle, coord_out_rot= findRotMaxRect(aa, flag_opt=True, nbre_angle=4, \
                                                         flag_parallel=False,         \
                                                         flag_out='rotation',         \
                                                         flag_enlarge_img=False,      \
                                                         limit_image_size=100         )

print('time elapsed =', (datetime.datetime.now()-time_start).total_seconds())
print('angle        =',  angle)
rect_coord_ori = np.array(rect_coord_ori)
print(rect_coord_ori)
plt.imshow(img)
plt.plot(rect_coord_ori[0,1],rect_coord_ori[0,0], 'rx')
plt.plot(rect_coord_ori[1,1],rect_coord_ori[1,0], 'rx')
plt.plot(rect_coord_ori[2,1],rect_coord_ori[2,0], 'rx')
plt.plot(rect_coord_ori[3,1],rect_coord_ori[3,0], 'rx')
plt.show()
#plot
#----------------
fig = plt.figure()
ax = fig.add_subplot(121, aspect='equal')
ax.imshow(aa.T,origin='lower',interpolation='nearest')
patch = patches.Polygon(rect_coord_ori, edgecolor='green', facecolor='None', linewidth=2)
ax.add_patch(patch)

center_rot = ( (aa.shape[1]-1)/2, (aa.shape[0]-1)/2 )
if max(center_rot)%2 == 0:
    center_rot = (center_rot[0]+1,center_rot[1]+1)
M = cv2.getRotationMatrix2D( center_rot, angle,1)
nx,ny = aa.shape
RotData = cv2.warpAffine(aa,M,(ny,nx),flags=cv2.INTER_NEAREST,borderValue=1)
ax = plt.subplot(122)
ax.imshow(RotData.T,origin='lower',interpolation='nearest')
patch = patches.Polygon(coord_out_rot, edgecolor='green', facecolor='None', linewidth=2)
ax.add_patch(patch)
plt.show()


# blur = cv2.blur(gray, (5, 5))
# ret, thresh = cv2.threshold(blur, 200, 255, 1)
# cv2.imshow('image',thresh)
# cv2.waitKey(0)
# #thresh = 255 - thresh
# erosion = cv2.erode(thresh, kernel, iterations=3)
# dilation = cv2.dilate(erosion, kernel, iterations=1)
# # dilation = cv2.dilate(gray, kernel, iterations =2)
# # while(i < 10):
# #      erosion = cv2.erode(dilation, kernel, iterations=3)
# #      dilation = cv2.dilate(erosion, kernel, iterations=4)
# #      i+=1
# # cv2.imshow('image',gray)
# # cv2.waitKey(0)
# # erosion = cv2.erode(dilation,kernel,iterations = 4)
# dilation = 255-dilation
#
# image, contour, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contour))
# max_cnt = contour[0]
# index = 0
# for idx,c in enumerate(contour):
#     if (cv2.contourArea(max_cnt) < cv2.contourArea(c)):
#         max_cnt = c
#         index = idx
#
# img2 = img
# epsilon = 0.1*cv2.arcLength(contour[index],True)
# approx = cv2.approxPolyDP(contour[index], epsilon, True)
# (x,y,w,h) = cv2.boundingRect(approx)
# cv2.drawContours(img, [approx],-1, (255, 255,60), 3)
# # cv2.approxPolyDP()
# cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('image', img)
# cv2.waitKey(0)
