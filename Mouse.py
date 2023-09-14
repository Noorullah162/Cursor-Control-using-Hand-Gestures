import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui
import time

# from autopy.mouse import LEFT_BUTTON, RIGHT_BUTTON


# width, height = 1024, 7860

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

width_cam = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height_cam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width_cam, height_cam)
# width_screen, height_screen = autopy.screen.size()
width_screen, height_screen = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]
frame = 170
smoothening = 7
pTime = 10
plocX, plocY = 0, 0
clocX, clocY = 0, 0


def findHands(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return img


def findPostion(img, handNo=0, connect=True, circle=True, rectangle=True):
    xlist = []
    ylist = []
    bbox = []
    lmlist = []
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if connect:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            myhand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                lmlist.append([id, cx, cy])
                if circle:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)

            bbox = xmin, xmax, ymin, ymax
            if rectangle == True:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return lmlist, bbox


def fingersUp(l):
    fingers = []

    # Thumb
    if l[tipIds[0]][1] < l[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for id in range(1, 5):

        if l[tipIds[id]][2] < l[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    # totalFingers = fingers.count(1)

    return fingers


def f(l):
    fingers = []
    # Thumb
    if l[tipIds[0]][1] < l[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for id in range(1, 5):

        if l[tipIds[id]][2] < l[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    # totalFingers = fingers.count(1)

    return fingers


def findDistance(p1, p2, img, lmlist, draw=True, r=15, t=3):
    x1, y1 = lmlist[p1][1:]
    x2, y2 = lmlist[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]


while True:
    success, img = cap.read()
    # img1 = findHands(img)
    # img = findHands(img,draw= False)
    list, bbox = findPostion(img, circle=False)

    if len(list) != 0:
        x1, y1 = list[8][1:]
        x2, y2 = list[12][1:]
        # print(x1, y1, x2, y2) #tip coordinates of the index finger and midle finger

        # fingers = fingersUp(list)
        # print(fingers)

        # print(bbox)

        fingers = fingersUp(list)
        # print(fingers)

        cv2.rectangle(img, (frame, frame), (int(width_cam) - frame, int(height_cam) - frame), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frame, width_cam - frame + 20), (0, width_screen))
            y3 = np.interp(y1, (frame, height_cam - frame + 20), (0, height_screen))

            # clocX = plocX + (x3 - plocX) / smoothening
            # clocY = plocY + (y3 - plocY) / smoothening

            print(x3, y3)
            pyautogui.moveTo(x3, y3)
            cv2.circle(img, (x1, y1), 7, (0, 255, 0), cv2.FILLED)

            # plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:

            length, img, info = findDistance(8, 12, img, list)
            # print(length)
            if length < 40:
                cv2.circle(img, (info[4], info[5]), 7, (255, 255, 0), cv2.FILLED)
                pyautogui.click()

        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            length, img, info = findDistance(12, 16, img, list)
            if length < 30:
                cv2.circle(img, (info[4], info[5]), 7, (255, 255, 0), cv2.FILLED)

                pyautogui.rightClick()

        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 0:
            break

    '''
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
D:\sampelimages
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            '''
    cv2.imshow("Image", img)
    cv2.waitKey(1)