import cv2
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
    image=cv2.imread(image,0)
    cv2.imwrite('Gray_Bozu.jpg',image)
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
contrast('Bozu.png',10)

def add_brightness(image,alpha):
    img=cv2.imread(image)
    for i in img:
        for j in i:
            j[0]+=alpha
            j[1]+=alpha
            j[2]+=alpha
    cv2.imwrite('Bright_Bozu.jpg',img)
    return 'Bright_Bozu.jpg'
add_brightness('bozu.png',600)