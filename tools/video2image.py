import cv2
path = "/home/jeasonde/ori/ByteTrack/YOLOX_outputs/yolox_nano_mix_det/track_vis/mot17-07"
cap = cv2.VideoCapture('/home/jeasonde/ori/ByteTrack/YOLOX_outputs/yolox_nano_mix_det/track_vis/mot17-07/MOT17-07-DPM-raw.mp4') #读入视频文件
c=1
 
timeF = 1  #视频帧计数间隔频率
 
while True:
    ret, frame = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    cv2.imshow('oxxostudio', frame)     # 如果讀取成功，顯示該幀的畫面
    cv2.imwrite(path + str(c)+".jpg",frame)
    c += 1
    if cv2.waitKey(1) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
        break
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()   
 