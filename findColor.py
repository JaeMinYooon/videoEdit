import cv2 as cv
import numpy as np

# 후에 조금더 수정이 필요함
# 범위 조정이 필요할듯 흰색을 못찾는 기분 + 빨강색과 피부색을 같이 섞어버림
hsv = 0
lower1 = 0
upper1 = 0
lower2 = 0
upper2 = 0
lower3 = 0
upper3 = 0


def mouse_callback(event, x, y, flags, param):
    global hsv, lower1, upper1, lower2, upper2, lower3, upper3

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print(color_select[y, x])
        color = color_select[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            lower1 = np.array([hsv[0]-10+180, 30, 30])
            upper1 = np.array([180, 255, 255])
            lower2 = np.array([0, 30, 30])
            upper2 = np.array([hsv[0], 255, 255])
            lower3 = np.array([hsv[0], 30, 30])
            upper3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower1 = np.array([hsv[0], 30, 30])
            upper1 = np.array([180, 255, 255])
            lower2 = np.array([0, 30, 30])
            upper2 = np.array([hsv[0]+10-180, 255, 255])
            lower3 = np.array([hsv[0]-10, 30, 30])
            upper3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower1 = np.array([hsv[0], 30, 30])
            upper1 = np.array([hsv[0]+10, 255, 255])
            lower2 = np.array([hsv[0]-10, 30, 30])
            upper2 = np.array([hsv[0], 255, 255])
            lower3 = np.array([hsv[0]-10, 30, 30])
            upper3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower1, "~", upper1)
        print("@2", lower2, "~", upper2)
        print("@3", lower3, "~", upper3)


cv.namedWindow('select_color')
cv.setMouseCallback('select_color', mouse_callback)



while(True):
    img_color = cv.imread('data/1.png')
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
    color_select = cv.imread('color.JPG')
    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower1, upper1)
    img_mask2 = cv.inRange(img_hsv, lower2, upper2)
    img_mask3 = cv.inRange(img_hsv, lower3, upper3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    # 이미지 resize해주기
    resize_img = cv.resize(img_color, dsize=(350, 600), interpolation=cv.INTER_AREA)
    resize_img_result = cv.resize(img_result, dsize=(350, 600), interpolation=cv.INTER_AREA)



    #cv.imshow('img_color', img_color)
    #cv.imshow('img_mask', img_mask) # 기존 것 어디 칠하는지 알려줌
    #cv.imshow('img_result', img_result)


    cv.imshow('resize', resize_img)
    cv.imshow('resize_result', resize_img_result)
    cv.imshow('select_color', color_select)



    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()