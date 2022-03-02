from cgitb import small
import cv2
from cv2 import illuminationChange
import numpy as np

def reddify(image):
    img = cv2.imread(image)
    b, g, r = cv2.split(img)
    zeros_ch = np.zeros(img.shape[0:2], dtype="uint8")
    red_img = cv2.merge([zeros_ch, zeros_ch, r])
    cv2.imwrite("Red_Bozu.jpg", red_img)
    return "Red_Bozu.jpg"
# reddify('bozu.png')

def greenify(image):
    img = cv2.imread(image)
    b, g, r = cv2.split(img)
    zeros_ch = np.zeros(img.shape[0:2], dtype="uint8")
    blue_img = cv2.merge([zeros_ch, g, zeros_ch])
    cv2.imwrite("Green_Bozu.jpg", blue_img)
    return "Green_Bozu.jpg"
# greenify('Bozu.png')

def blueify(image):
    img = cv2.imread(image)
    b, g, r = cv2.split(img)
    zeros_ch = np.zeros(img.shape[0:2], dtype="uint8")
    green_img = cv2.merge([b, zeros_ch, zeros_ch])
    cv2.imwrite("Blue_Bozu.jpg", green_img)
    return "Blue_Bozu.jpg"
# blueify('bozu.png')

def grayify(image):
    img=cv2.imread(image,0)
    for i in img:
        i[0]*2989+i[1]*0.587+i[2]*0.114
    cv2.imwrite('Gray_Bozu.jpg',img)
grayify('bozu.png')

def negative_bozu(image):
    img=cv2.imread(image,0)
    img1=255-img
    cv2.imwrite('Negative_Bozu.jpg',img1)
    return 'Negative_Bozu.jpg'
# negative_bozu('bozu.png')

def horizontal_flip(image):
    img=cv2.imread(image)
    mylist=[]
    for i in img:
        fliphorizonatal=i[::-1]
        mylist.append(fliphorizonatal)
    arr=np.array(mylist)
    cv2.imwrite('Horizontal_Bozu.jpg',arr)
    return 'Horizontal_Bozu.jpg'
# horizontal_flip('Red_Bozu.jpg')

def vertical_flip(image):
    img=cv2.imread(image)
    flipVertical = img[::-1]
    cv2.imwrite('Vertical_Bozu.jpg',flipVertical)
    return 'Vertical_Bozu.jpg'
# vertical_flip('Red_Bozu.jpg')

def clip(broken_image):
    img=cv2.imread(broken_image)
    for i in img:
        for j in i:
            if int(j[0])>255:
                j[0]=255
            if int(j[0])<0:
                j[0]=0
            if int(j[1])>255:
                j[1]=255
            if int(j[1])<0:
                j[1]=0
            if int(j[2])>255:
                j[2]=255
            if int(j[2])<0:
                j[2]=0
# clip('bozu.png')

def contrast(image,alpha):
    img=cv2.imread(image)
    for i in img:
        for j in i:
            j[0]*=alpha
            j[1]*=alpha
            j[2]*=alpha
    cv2.imwrite('Contrast_Bozu.jpg',img)
    return 'Contrast_Bozu.jpg'
# contrast('Bozu.png',10)

def add_brightness(image,alpha):
    img=cv2.imread(image)
    for i in img:
        for j in i:
            j[0]+=alpha
            j[1]+=alpha
            j[2]+=alpha
    cv2.imwrite('Bright_Bozu.jpg',img)
    return 'Bright_Bozu.jpg'
# add_brightness('bozu.png',600)

def apply_threshold(image,threshold):
    img=cv2.imread(image)
    for i in img:
        for j in i:
            if int(j[0])>threshold:
                j[0]=1
            if int(j[0])<threshold:
                j[0]=0
            if int(j[1])>threshold:
                j[1]=1
            if int(j[1])<threshold:
                j[1]=0
            if int(j[2])>threshold:
                j[2]=1
            if int(j[2])<threshold:
                j[2]=0
            # print(j)
    cv2.imwrite('Silhouette_Bozu.jpg',img)
    return 'Silhouette_Bozu.jpg'
# apply_threshold('bozu.png',0)

def bozu_headshot(image,x,y,height,width):
    img = cv2.imread(image)
    bozu_headshot = img[y:y+height, x:x+width]
    cv2.imwrite("Bozu_headshot.jpg", bozu_headshot)
# bozu_headshot('blue_bozu.jpg',0,0, 450,750)

def bozu_frame_1(image):
    image = cv2.imread(image)
    border_size=200
    border_size2=200
    border_bozu = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, None, value = (0,0,128))

    border_bozu1 = cv2.copyMakeBorder(border_bozu, border_size2, border_size2, border_size2, border_size2, cv2.BORDER_CONSTANT, None, value = (0,255,255))

    cv2.imwrite("Bozu_frame_1.jpg", border_bozu1)
# bozu_frame_1('bozu.png')


def super_impose(picture1,picture2):
    img1=cv2.imread(picture1)
    img2=cv2.imread(picture2)
    large_img=cv2.resize(img1,(1080,720))
    small_img = cv2.resize(img2,(300,300))
    x = 400
    y = 170
    x1 = x + small_img.shape[1]
    y1 = y + small_img.shape[0]
    large_img[y:y1,x:x1] = small_img
    cv2.imwrite('Galaxy_Bozu.jpg',large_img)


# super_impose('andromeda_galaxy.jpg','Bozu.png')


def vintage_bozu(image):
    img1=cv2.imread(image)
    rows=img1.shape[0]
    columns=img1.shape[1]
    x = cv2.getGaussianKernel(columns,200)
    y = cv2.getGaussianKernel(rows,200)
    resultant_kernel = y * x.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    output = np.copy(img1)
    for i in range(3):
        output[:,:,i] = output[:,:,i] * mask
    cv2.imwrite('Vintage_Bozu.jpg',output)
# vintage_bozu('bozu.png')