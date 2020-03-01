from videoMake import *
from ex import *

# input 동영상(보류) /// 종류 (ex : 사람) , 상의 , 하의 , 상의색, 하의색
def webSock():
    # sample2.jgp 는 원래 안써야하는데 그냥 인풋뭘로할지 테스트용임
    exStr = "person,long,long,00688B,red,sample2.jpg"
    return exStr

def main():

    # 영상 yolo작업 후 실행 =======================================
    #VideoMake.videoMake("./inputvideo/high.mp4")


    exStr = webSock()

    exVideoMake(exStr)

    #exVideoMake("sample2.jpg")
main()