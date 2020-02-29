import cv2
import numpy as np


def main():
    color()


def hex_to_hsv():
    h = input('hex값 입력: ').lstrip('#')
    bgr = tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
    print(bgr)
    bgr_split = str(bgr).split(', ')

    # 값 쪼개기
    blue = bgr_split[0]
    blue = blue[1:]
    green = bgr_split[1]
    red = bgr_split[2]
    red = red[:-1]


    hsv = np.uint8([[[blue, green, red]]])
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    #print(hsv)
    return hsv

def color():

    hsv = hex_to_hsv()# hex값 넣을 곳
    print("h = " + str(hsv[0][0][0]))
    print("s = " + str(hsv[0][0][1]))
    print("v = " + str(hsv[0][0][2]))
    h = int(hsv[0][0][0])
    s = int(hsv[0][0][1])
    v = int(hsv[0][0][2])

    img = cv2.imread('color.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #lower = np.array([0, 40, 40]) # 직접 넣을 때 사용 할 코드
    #upper = np.array([30, 255, 255])

    lower = np.array([h-10, 40, 40]) # hsv값 뽑은 거 이용 할 코드
    upper = np.array([h+10, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    new_img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('mask', mask) # 색칠 된 자리 가 어딘지 알려주는 곳
    cv2.imshow("color", new_img) # 색칠까지 보여주는 곳
    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()