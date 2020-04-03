from videoMake import *
import socket
import time
import threading
import ex

complete = [0]
lock = threading.Lock()
label = {
    'top': {'shortTopC': 0, 'longTopC': 1},
    'bottom': {'shortBottomC': 0, 'longBottomC': 1, 'skirtBottomC': 2}
}

top_classes = {v: k for k, v in label['top'].items()}
bottom_classes = {v: k for k, v in label['bottom'].items()}

top_class_num = len(top_classes.values())
bottom_class_num = len(bottom_classes.values())
models = []

# input 동영상(보류) /// 종류 (ex : 사람) , 상의 , 하의 , 상의색, 하의색
def noServer():
    # 맨밑에서 main 대신 noServer 부르면 됨
    # 좀 된 버전 V3인듯
    cfgfile = "videoMake/cfg/yolov3.cfg"
    weightsfile = "videoMake/cfg/yolov3.weights"
    # 가장 최신 V3
    #cfgfile = "videoMake/cfg/yolov3new.cfg"
    #weightsfile = "videoMake/cfg/yolov3new.weights"
    # EfficientNet Yolo
    #cfgfile = "videoMake/cfg/enetb0-coco_final.cfg"
    #weightsfile = "videoMake/cfg/enetb0-coco_final.weights"
    # tiny 욜로
    #cfgfile = "videoMake/cfg/yolov3-tiny.cfg"
    #weightsfile = "videoMake/cfg/yolov3-tiny.weights"
    # openimages
    #cfgfile = "videoMake/cfg/yolov3-openimages.cfg"
    #weightsfile = "videoMake/cfg/yolov3-openimages.weights"
    model = darknet.Darknet(cfgfile)
    model.load_weights(weightsfile)

    start = time.time()
    # 상하의 모델 분류는 보류.
    #top_model = clothClassification.create_model(top_class_num, "/videoMake/top/my_checkpoint_top")
    #bottom_model = clothClassification.create_model(bottom_class_num, "/videoMake/bottom/my_checkpoint_bottom")
    print("top bottom load time : ", time.time()-start)
    models.append(model)
    #models.append(top_model)
    #models.append(bottom_model)

    exStr = "person&long&long&0022ff&0022ff&airport"
    #exStr = "backpack&./inputvideo/backpack.mp4"
    VideoMake.videoMakeWithYolo(exStr, models)

    # 사진 테스트용 그냥 주석풀고 실행하면됨
    #exStr = "person&long&long&000000&000000&./park.jpg"
    #exStr = "dog&./sample.jpg"
    #ex.exVideoMake(exStr, model)

def main():

    # 모델 로드 해놓고 실행하는데 이거 좀 다듬어야할듯
    # 가장 최신 V3
    cfgfile = "videoMake/cfg/yolov3new.cfg"
    weightsfile = "videoMake/cfg/yolov3new.weights"
    model = darknet.Darknet(cfgfile)
    model.load_weights(weightsfile)
    models.append(model)

    PYPORT = 5803
    PYIP = "192.168.0.38"
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((PYIP, PYPORT))

    server_socket.listen(5)
    print("TCPServer Waiting for client on port " + str(PYPORT))
    global complete, lock
    while True:
        # 클라이언트 요청 대기중 .
        client_socket, address = server_socket.accept()
        # 클라이언트 호스트네임
        # 연결 요청 성공

        data = client_socket.recv(1024)

        recStartStr = data.decode()
        print(recStartStr)
        if recStartStr == "start":
            data = client_socket.recv(1024)
            strData = data.decode()
            # "person&long&long&0022ff&0022ff&airport"
            print("string : ", strData)
            mappingName = strData.split("&")
            mappingName = mappingName[len(mappingName)-1]

            thread = threading.Thread(target=VideoMake.videoMakeWithYolo,
                                      args=(strData, models, complete, lock))

            thread.start()

            i = 0
            while True:
                print("밀린 욜로 파일 : ", complete[0])
                if i >= 5 :
                    print("전체 파일 전송 종료")
                    break
                if complete[0] >= 1:
                    lock.acquire()
                    try:
                        complete[0] = complete[0] -1
                        print("보내고 남은 양 : ", complete[0])
                    finally:
                        lock.release()


                    start = time.time()

                    # ------------    시간Text파일 전송    ---------------
                    sendStartStr = "start"
                    client_socket.sendall(sendStartStr.encode())

                    #filename = "yoloed_" + str(i + 1) + ".txt"
                    filename = mappingName + "_"+ str(i+1)+ ".txt"
                    client_socket.sendall(filename.encode())

                    print("filename", filename)
                    data = client_socket.recv(1024)
                    if data.decode() == "go":
                        # yoloed_file = open(filename, 'rb')
                        with open("./yoloresult/" + filename, 'rb') as yoloed_file:
                            filePack = yoloed_file.read(1024)
                            print('sending....')
                            while filePack:
                                client_socket.sendall(filePack)
                                filePack = yoloed_file.read(1024)

                            data = client_socket.recv(1024)
                            if data.decode() == "recieve":
                                client_socket.sendall("fileend".encode())
                    # ------------    Yolo 동영상 전송    ---------------
                    data = client_socket.recv(1024)
                    if data.decode() == "mp4go":
                        print("mp4go")
                        sendStartStr = "start"
                        client_socket.sendall(sendStartStr.encode())
                        data = client_socket.recv(1024)
                        if data.decode() == "filenamego":
                            #filename = "yoloed_" + str(i + 1) + ".mp4"
                            filename = mappingName + "_" + str(i+1) + ".mp4"
                            client_socket.sendall(filename.encode())

                            print("filename", filename)
                        data = client_socket.recv(1024)
                        if data.decode() == "go":
                            # yoloed_file = open(filename, 'rb')
                            with open("./yoloresult/" + filename, 'rb') as yoloed_file:
                                filePack = yoloed_file.read(1024)
                                print('sending....')
                                while filePack:
                                    client_socket.sendall(filePack)
                                    filePack = yoloed_file.read(1024)

                                data = client_socket.recv(1024)
                                if data.decode() == "recieve":
                                    client_socket.sendall("fileend".encode())
                    print("********** : " ,time.time()-start)

                    i += 1
                else:
                    time.sleep(1)

            # exit 보내야함.
            client_socket.sendall("exit".encode())

if __name__ == '__main__':
    noServer()