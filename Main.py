from videoMake import *
from ex import *

# input 동영상(보류) /// 종류 (ex : 사람) , 상의 , 하의 , 상의색, 하의색
def webSock():
    # sample2.jgp 는 원래 안써야하는데 그냥 인풋뭘로할지 테스트용임
    exStr = "person,long,long,cb3013,red,./inputvideo/11.mp4"
    #exStr = "person,long,long,cb3013,red,sample3.jpg"
    return exStr

def main():

    # 영상 yolo작업 후 실행 =======================================

    exStr = webSock()
    VideoMake.videoMake(exStr)

    #exVideoMake(exStr)

    #exVideoMake("sample2.jpg")
main()