from videoMake import *
from ex1 import complete, lock
import socket
import threading
import time

# input 동영상(보류) /// 종류 (ex : 사람) , 상의 , 하의 , 상의색, 하의색
def webSock():
    # sample2.jgp 는 원래 안써야하는데 그냥 인풋뭘로할지 테스트용임
    # 이부분 수정좀
    exStr = "person,long,long,000000,010000,./inputvideo/jaemin.mp4"

    #exStr = "person,long,long,2e2b3c,70051d,sample3.jpg"
    return exStr
def main():
    # 모델 로드 해놓고 실행하는데 이거 좀 다듬어야할듯
    cfgfile = "videoMake/cfg/yolov3.cfg"
    weightsfile = "videoMake/cfg/yolov3.weights"
    model = darknet.Darknet(cfgfile)
    model.load_weights(weightsfile)

    exStr = webSock()

    # 영상 yolo작업 후 실행 =======================================
    VideoMake.videoMake(exStr, model)
main()
'''
    PYPORT = 5803
    PYIP = "192.168.0.35"
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((PYIP, PYPORT))

    server_socket.listen(5)
    print("TCPServer Waiting for client on port " + str(PYPORT))

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
            print("string : ", strData)

            thread = threading.Thread(target=VideoMake.videoMake, args=(strData, model))

            thread.start()



            global complete
            i=0
            while True:
                print("밀린 욜로 파일 : ", complete[0])
                if i >= 5 :
                    print("전체 파일 전송 종료")
                    break
                if complete[0] >=1:
                    lock.acquire()
                    try:
                        complete[0] = complete[0] -1
                        print("보내고 남은 양 : ", complete[0])
                    finally:
                        lock.release()
                    filename = "yoloed_" + str(i+1) + ".mp4"

                    start = time.time()
                    sendStartStr = "start"
                    client_socket.send(sendStartStr.encode())

                    sendFileStr = filename
                    client_socket.send(sendFileStr.encode())

                    print("filename", filename)
                    data = client_socket.recv(1024)
                    if data.decode() == "go":
                        # yoloed_file = open(filename, 'rb')
                        with open("./yoloresult/" + filename, 'rb') as yoloed_file:
                            filePack = yoloed_file.read(1024)
                            print('sending....')
                            filesize = 0
                            while filePack:
                                filesize += client_socket.send(filePack)
                                filePack = yoloed_file.read(1024)

                            data = client_socket.recv(1024)
                            if data.decode() == "recieve":
                                client_socket.sendall("fileend".encode())
                    print("********** : " ,time.time()-start)

                    yoloed_file.close()
                    i += 1
                else:
                    time.sleep(1)

            # exit 보내야함.
            client_socket.send("exit".encode())
        
    
        # client_socket.send("ok".encode())
        excel_file = open('', 'rb')
        l = excel_file.read(1024)
        while (l):
            client_socket.send(l)
            print('sending....')
            l = excel_file.read(1024)
        excel_file.close()
        

    exStr = webSock()


    # 영상 yolo작업 후 실행 =======================================
    VideoMake.videoMake(exStr,weightsfile)


    #exVideoMake(exStr)

    # exVideoMake("sample2.jpg")
    '''


'''
# 서버 소켓 오픈
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("223.194.131.27", 5801))
    server_socket.listen(5)
    print("TCPServer Waiting for client on port 5000")

    while True:

        # 클라이언트 요청 대기중 .
        client_socket, address = server_socket.accept()
        # 클라이언트 호스트네임
        # 연결 요청 성공
        print("I got a connection from ", address)

        data = None
        # img_data = client_socket.recv(1024)

        excetionCtrl = False
        # Data 수신
        while True:
            img_data = client_socket.recv(1024)
            data = img_data
            if img_data:
                while img_data:
                    print("recving Img...")
                    img_data = client_socket.recv(1024)
                    data += img_data
                else:
                    break

        # 받은 데이터 저장
        # 이미지 파일 이름은 현재날짜/시간/분/초.jpg
        img_fileName, toMain = fileName()
        img_file = open(img_fileName, "wb")
        print("finish img recv")
        print(sys.getsizeof(data))
        img_file.write(data)
        img_file.close()

        print("Finish ")
        # print(toMain)
        # client_socket.send("ok".encode())
        try:
            Cmain(imagePath=toMain, model=model)
            excetionCtrl = True
            print("성공!")
        except:
            excetionCtrl = False
            print("실패!")
        client_socket.shutdown(socket.SHUT_RD)
        # client_socket.sendall(getFileData(toMain+'.jpg', src))
        if excetionCtrl == True:
            # client_socket.send("ok".encode())
            excel_file = open(toMain + '.xlsx', 'rb')
            l = excel_file.read(1024)
            while (l):
                client_socket.send(l)
                print('sending....')
                l = excel_file.read(1024)
            excel_file.close()
        else:
            client_socket.send("no".encode())
        client_socket.shutdown(socket.SHUT_WR)
        print("보냈는데?")

        # client_socket.shutdown(SHUT_RD)
    client_socket.close()
    print("SOCKET closed... END")

'''