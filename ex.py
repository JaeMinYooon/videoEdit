from __future__ import division
import time
from videoMake.util import *
from videoMake.darknet import Darknet
from ex1 import *
personNum = 1
classes = load_classes("videoMake/data/coco.names")
hexCode=""

def exVideoMake(exStr):

    # kind에따라 찾을 물체를 정함. (ex :  person=0, dog=16)
    kind = 0
    inputString = exStr.split(',')
    if inputString[0] == "person":
        kind = 0
    if inputString[0] == "dog":
        kind = 16
    global hexCode
    hexCode = inputString[3]
    videofile = inputString[len(inputString)-1]
    print("kind "+ str(kind) + "videofile : " +str(videofile))
    print(type(kind))
    num_classes = 80
    bs = 1
    confidence = 0.5
    nms_thresh = 0.4
    cfgfile = "videoMake/cfg/yolov3.cfg"
    weightsfile = "videoMake/cfg/yolov3.weights"
    reso = 416

    batch_size = int(bs)
    confidence = float(confidence)
    nms_thesh = float(nms_thresh)
    start = time.time()
    CUDA = torch.cuda.is_available()

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
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


    frame = cv2.imread(videofile)

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


    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    classes = load_classes('videoMake/data/coco.names')
    # 사람이미지 자르는데 이건 모든프레임 필요x 1/6 마다 추출
    what = list(map(lambda x: cutPerson(x, frame), output))
    print(what)
    # 얘는 욜로 결과 그리는건데 모든 프레임에 결과 있어야함.
    #list(map(lambda x: write(x, frame), output))
    for i, out in enumerate(output):
        if what[i]:
            write(out, frame)

    cv2.imshow("dd",frame)
    cv2.waitKey(0)
    print("Total Time : ", (time.time() - start))


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    # color = random.choice(colors)
    color = (255, 0, 0)
    # label1 은 클래스가 뭔지 label2는 확률나오는거
    label1 = "{0}".format(classes[cls])
    label2 = float("{0:0.4f}".format((x[5])))
    # label2 = torch.item(float((x[5])[cls]))

    # label = label1 + str(label2)
    label = label1

    cv2.rectangle(img, c1, c2, color, 1)  # 객체 네모칸 쳐주는 코드
    t_size = cv2.getTextSize(label, cv2.FONT_ITALIC, 0.3, 1)[0]  # 폰트 바꾸는 코드
    # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # person 만 나오게 사이즈 조절함.
    c2 = c1[0] + t_size[0], c1[1] - t_size[1]

    cv2.rectangle(img, c1, c2, color, -1)  # 글자 네모칸 쳐주는 코드
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_ITALIC, 1, [225, 255, 255], 1);
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
    b = cutUpperBody(img_cut,"./cuttedupper/upper" +str(personNum)+".jpg")
    cutLowerBody(img_cut,"./cuttedlower/lower" +str(personNum)+".jpg")
    cv2.imwrite(path,img_cut)
    personNum += 1
    return b

def cutUpperBody(img,path):
    img_h = img.shape[0]
    img_w = img.shape[1]
    cutToUpper = img[int((img_h/10)*1): int((img_h/10)*6), : img_w]
    global hexCode
    find = color(hexCode ,cutToUpper)
    if not find:
        return False
    cv2.imwrite(path, cutToUpper)
    print("True")
    return True

def cutLowerBody(img,path):
    img_h = img.shape[0]
    img_w = img.shape[1]
    cutToUpper = img[int((img_h / 10) * 5): img_h, : img_w]
    cv2.imwrite(path, cutToUpper)


# 영상 중간중간 yolo 결과로 저장 하는 메소드
def stepFrameToVideo(inputs, start, end, step, fps=25, pathDir="./yoloresult"):
    height, width, layers = inputs[0].shape
    size = (width, height)

    pathOut = pathDir + '/yoloed_' + str(step) + '.avi'
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    count = start
    while count < end:
        out.write(inputs[count])
        count += 1
    out.release()


def frameToWholeVideo(inputs, fps=25, pathDir="./yoloresult"):
    height, width, layers = inputs[0].shape
    size = (width, height)
    pathOut = pathDir + '/yoloVideo.avi'
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
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

            # print(count, curLen, cuttedLen)

        curLen += cuttedLen

    out.release()