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
    cv2.imwrite('New_Hipo.jpg',img)
# grayify('bozu.png')

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
    cv2.imwrite('New_Hipo2.jpg',img)
    return 'Bright_Bozu.jpg'
# add_brightness('bozu.png',600)

def apply_threshold(image,threshold):
    img=cv2.imread(image)
    for i in img:
        for j in i:
            # print(j)
            # if int(j[0])>=threshold:
            #     j[0]=255
            # if int(j[0])<threshold:
            #     j[0]=0
            if int(j[1])>=threshold:
                j[0]=255
                j[1]=255
                j[2]=255
            if int(j[1])<threshold:
                j[1]=0
                # j[2]=0
                # j[0]=0
            # if int(j[2])>=threshold:
            #     j[2]=255
            # if int(j[2])<threshold:
            #     j[2]=0
            # print(j)
    cv2.imwrite('New_Hipo2.jpg',img)
    return 'New_Hipo.jpg'
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
    cv2.imwrite('New_Hipo1.jpg',output)
# vintage_bozu('bozu.png')

def sepia(image):
    img=cv2.imread(image)
    img_sepia = np.array(img, dtype=np.float64) 
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]])) 
    img_sepia[np.where(img_sepia > 255)] = 255 
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    cv2.imwrite('sepia.jpg',img_sepia)
sepia('bozu.png')

def sharp_sepia(image):
    img=cv2.imread(image)
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imwrite('sharp_sepia.jpg',image_sharp)
sharp_sepia('sepia.jpg')


""" Fix this bug pronto"""
# grayify('real-hippo.jpeg')
# vintage_bozu('New_Hipo.jpg')
apply_threshold('New_Hipo1.jpg',120)

# -------------------------------------------


def myself(image):
    img1 = cv2.imread(image)
    # img1=cv2.resize(img,(300,400))
    Z = img1.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img1.shape))
    # cv2.imshow('res2',res2)
    cv2.imwrite('cartoon_myself.jpg',res2)
myself('Tanishq1.jpg')