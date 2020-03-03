from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from videoMake.util import *
import argparse
import os
import os.path
from videoMake.darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from ex1 import *

#from Main import complete,lock

personNum = 1
classes = load_classes("videoMake/data/coco.names")
upperHexCode= ""
lowerHexCode= ""

def videoMake(exStr, model):
    # kind에따라 찾을 물체를 정함. (ex :  person=0, dog=16)
    kind = 0
    inputString = exStr.split('&')
    if inputString[0] == "person":
        kind = 0
    if inputString[0] == "dog":
        kind = 16

    global upperHexCode
    upperHexCode = inputString[3]
    global lowerHexCode
    lowerHexCode = inputString[4]

    videofile = inputString[len(inputString)-1]

    num_classes = 80
    bs = 1
    confidence = 0.5
    nms_thresh = 0.4
    reso = 416


    batch_size = int(bs)
    confidence = float(confidence)
    nms_thesh = float(nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    # Set up the neural network
    print("Loading network.....")
    # model = Darknet(cfgfile)
    # model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    # Detection phase

    print(videofile)

    cap = cv2.VideoCapture(videofile)
    # cap = cv2.VideoCapture(0) # for webcam

    assert cap.isOpened(), 'Cannot capture source'

    start = time.time()

    frameCount = 0
    framesForVideo = []
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameStepForSave = int(totalFrame / 5)
    frameBreakForSave = int(totalFrame / 5)

    print("step ", frameStepForSave)
    print("break ", frameBreakForSave)
    print("total ", totalFrame)

    yoloCounting = 0
    yoloCountingCheck = 0

    while cap.isOpened():
        ret, frame = cap.read()

        # 진행도에 따라 중간중간 저장하는 코드
        if frameCount == frameBreakForSave:
            # 분할 했을때 몇번째 시작점, 종료점 필요함.
            convert_start = frameBreakForSave - frameStepForSave
            convert_end = frameBreakForSave
            convert_step = int(frameBreakForSave/frameStepForSave)
            '''
            분할 하고 남은 프레임 그냥 버림.
            # 프레임이 딱 맞게 분할 안되면 프레임 몇조각 남으니까 막번째는 total 까지
            if convert_step == totalFrame/frameStepForSave:
                convert_end = totalFrame
            '''
            #print("convert_step ", convert_step)
            #print("convert_end ", convert_end)
            #print("convert_start ", convert_start)
            stepFrameToVideo(framesForVideo, convert_start, convert_end-1, convert_step)

            frameBreakForSave += frameStepForSave

        if ret:
            # 초당 30 -> 30이지만 욜로는 초당 6만 할거임.
            if yoloCounting == 0:
                yoloCounting += 1
                #print("** This is Yolo : ", yoloCountingCheck)
                #print("-- Do Yolo : ", yoloCounting)
                yoloCountingCheck+=1
                img = prep_image(frame, inp_dim)
                #        cv2.imshow("a", frame)
                im_dim = frame.shape[1], frame.shape[0]
                im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                with torch.no_grad():
                    output = model(Variable(img, volatile=True), CUDA)
                output = write_results(output, kind, confidence, num_classes, nms_conf=nms_thesh)

                if type(output) == int:
                    frameCount += 1
                    print("FPS of the video is {:5.4f}".format(frameCount / (time.time() - start)))
                    #cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    yoloCounting = 0
                    if key & 0xFF == ord('q'):
                        break
                    continue

                im_dim = im_dim.repeat(output.size(0), 1)
                scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

                output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                output[:, 1:5] /= scaling_factor

                for i in range(output.shape[0]):
                    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                classes = load_classes('videoMake/data/coco.names')
                colors = pkl.load(open("videoMake/pallete", "rb"))

                # 사람이미지 자르는데 이건 모든프레임 필요x 1/6 마다 추출
                findRes = list(map(lambda x: cutPerson(x, frame), output))


            else:
                yoloCountingCheck += 1
                yoloCounting += 1
                yoloCounting %= 5

            # 얘는 욜로 결과 그리는건데 모든 프레임에 결과 있어야함.
            for i, out in enumerate(output):
                if findRes[i]:
                    write(out, frame)



            # cv2.imshow("frame", frame)  # 화면 보여주는 곳
            framesForVideo.append(frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frameCount += 1
            #print(time.time() - start)
            #print("FPS of the video is {:5.2f}".format(frameCount / (time.time() - start)))
        else:
            break
    print("Total Time : ", (time.time() - start))
    frameToWholeVideo(framesForVideo)
    # frameToVideo(framesForVideo, 5)

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    #color = random.choice(colors)
    color = (255, 0, 0)
    # label1 은 클래스가 뭔지 label2는 확률나오는거
    label1 = "{0}".format(classes[cls])
    label2 = float("{0:0.4f}".format((x[5])))
    # label2 = torch.item(float((x[5])[cls]))

    #label = label1 + str(label2)
    label = label1

    cv2.rectangle(img, c1, c2, color, 1)  # 객체 네모칸 쳐주는 코드
    t_size = cv2.getTextSize(label, cv2.FONT_ITALIC, 0.3, 1)[0]  # 폰트 바꾸는 코드
    #c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # person 만 나오게 사이즈 조절함.
    c2 = c1[0] + t_size[0], c1[1] - t_size[1]

    cv2.rectangle(img, c1, c2, color, -1)  # 글자 네모칸 쳐주는 코드
    #cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_ITALIC, 1, [225, 255, 255], 1);
    cv2.putText(img, label, (c1[0], c1[1]), cv2.FONT_ITALIC, 0.3, [225, 255, 255], 1);
    return img

# 이미지에서 좌표로 박스 자름
def cutPerson(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    x, y = c1
    x_max, y_max = c2
    w = x_max - x
    h = y_max - y
    img_cut = img[y:y+h, x:x+w]

    global personNum
    path = "./cuttedperson/person" +str(personNum)+".jpg"
    upperPath = "./cuttedupper/upper" +str(personNum)+".jpg"
    lowerPath = "./cuttedlower/lower" +str(personNum)+".jpg"

    upperFind, cutToUpper = cutUpperBody(img_cut)

    if not bool(upperFind):
        return False
    lowerFind, cutToLower = cutLowerBody(img_cut)

    if not bool(lowerFind):
        return False

    cv2.imwrite(path, img_cut)
    cv2.imwrite(upperPath, cutToUpper)
    cv2.imwrite(lowerPath, cutToLower)
    personNum += 1

    return True

def cutUpperBody(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    cutToUpper = img[int((img_h/10)*1): int((img_h/10)*6), : img_w]
    global upperHexCode
    find = color(upperHexCode, cutToUpper)
    if not find:
        return int(0), ""
    #print("Upper True")
    return int(1) , cutToUpper

def cutLowerBody(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    cutToLower = img[int((img_h / 10) * 5): img_h, : img_w]

    global lowerHexCode
    find = color(lowerHexCode, cutToLower)
    if not find:
        return int(0), ""
    #print("Lower True")
    return int(1), cutToLower

# 영상 중간중간 yolo 결과로 저장 하는 메소드
def stepFrameToVideo(inputs, start, end, step, fps=25, pathDir="./yoloresult"):
    height, width, layers = inputs[0].shape
    size = (width, height)

    pathOut = pathDir + '/yoloed_' + str(step) + '.mp4'
    #pathOut = pathDir + '/yoloed_' + str(step) + '.avi'
    #out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    out = cv2.VideoWriter(pathOut, 0x00000021, fps, size)
    count = start
    while count < end:
        out.write(inputs[count])
        count += 1

    global complete
    lock.acquire()
    try:
        out.release()
        complete[0] = complete[0]+1
        print("++complete : ", complete[0])
    finally:
        lock.release()



    # 서버스레드 깨워

def frameToWholeVideo(inputs, fps=25, pathDir="./yoloresult"):
    height, width, layers = inputs[0].shape
    size = (width, height)

    pathOut = pathDir + '/yoloVideo.mp4'
    #pathOut = pathDir + '/yoloVideo.avi'
    #out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    out = cv2.VideoWriter(pathOut, 0x00000021, fps, size)
    for i in range(len(inputs)):
        # writing to a image array
        out.write(inputs[i])
    out.release()

def frameToVideo(inputs, cutNum, fps=25, pathDir="./yoloresult"):
    results = []
    count = 1

    height, width, layers = inputs[0].shape
    size = (width, height)

    curLen = int(float(len(inputs) / cutNum))
    cuttedLen = int(float(len(inputs) / cutNum))

    for i in range(cutNum):
        pathOut = pathDir + '/video' + str(i + 1) + '.avi'
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        while True:
            if count == curLen:
                break
            if count >= len(inputs):
                break
            # writing to a image array
            out.write(inputs[count])
            count += 1

            #print(count, curLen, cuttedLen)

        curLen += cuttedLen

    out.release()