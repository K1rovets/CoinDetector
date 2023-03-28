import numpy as np
import cv2
import sys

def detectCoins(image):
    # STEP 1: Preparation
    #import image, blur it, and convert to grayscale colorspace
    imgOriginal = cv2.imread(image, cv2.IMREAD_COLOR)
    if imgOriginal is None:
        sys.exit("Could not read the image.")    
    imgBlur = cv2.medianBlur(imgOriginal,11)  
    imgGray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
    #get binary lvl photo and contours
    ret, thresh = cv2.threshold(imgGray, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # STEP 2: Detect tray's countour
    #find contour of the TRAY - it is contour with the biggest area - using for loop and if condition
    imax=0
    areamax=0
    for i in range(len(contours)):
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > areamax:
            imax=i
            areamax=area
    tray = contours[imax]
    trayarea = cv2.contourArea(tray)
    cv2.drawContours(imgOriginal, [tray], 0, (0,255,0), 2)

    # STEP 3: Detect and count coins
    #find circles
    circles = cv2.HoughCircles(imgGray,method = cv2.HOUGH_GRADIENT_ALT,dp = 1.5,minDist=30,param1=100,param2=0.85,minRadius=10,maxRadius=100)
    circles = np.uint16(np.around(circles))
    #draw circles
    for i in circles[0,:]:
        cv2.circle(imgOriginal,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(imgOriginal,(i[0],i[1]),2,(0,0,255),3)
    #calculate coins
    BigCoinInTray=0
    SmallCoinInTray=0
    BigCoinOutTray=0
    SmallCoinOutTray=0
    mRadius = max(circles[0,:,-1])
    for i in circles[0,:]:
        if cv2.pointPolygonTest(tray, (i[0],i[1]), False)>-1:
            if i[2] > mRadius - 5:
                BigCoinInTray=BigCoinInTray+1
            else:
                SmallCoinInTray=SmallCoinInTray+1
        else:
            if i[2] > mRadius - 5:
                BigCoinOutTray=BigCoinOutTray+1
            else:
                SmallCoinOutTray=SmallCoinOutTray+1
    imgOriginal = cv2.putText(imgOriginal,"BigCoinInTray = "+str(BigCoinInTray), (50,50),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,255,255), thickness=1)
    imgOriginal = cv2.putText(imgOriginal,"BigCoinOutTray = "+str(BigCoinOutTray), (50,100),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,255,255), thickness=1)
    imgOriginal = cv2.putText(imgOriginal,"SmallCoinInTray = "+str(SmallCoinInTray), (50,150),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,255,255), thickness=1)
    imgOriginal = cv2.putText(imgOriginal,"SmallCoinOutTray = "+str(SmallCoinOutTray), (50,200),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,255,255), thickness=1)
    imgOriginal = cv2.putText(imgOriginal,"Area = "+str(trayarea), (50,250),fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(0,255,255), thickness=1)

    # STEP 4: Show final result
    imgRes = cv2.resize(imgOriginal, (0,0), fx=0.8, fy=0.8) 
    cv2.imshow(image,imgRes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#%%
#Main task
detectCoins('tray8.jpg')
detectCoins('tray3.jpg')
detectCoins('tray7.jpg')
#Additional 
detectCoins('tray1.jpg')
detectCoins('tray2.jpg')
detectCoins('tray4.jpg')
detectCoins('tray5.jpg')
detectCoins('tray6.jpg')