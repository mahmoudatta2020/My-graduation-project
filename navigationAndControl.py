import cv2
import numpy as np
import time
import autopy
import pyautogui
import mediapipe as mp
import urllib.request

def mousefunch (sl=0,kb=0):
   import HandTrackingModule as htm
   mp_hands = mp.solutions.hands#from sign********************************************************
   hands = mp_hands.Hands()#from sign********************************************************
   mp_draw = mp.solutions.drawing_utils#from sign********************************************************
   finger_tips = [8, 12, 16, 20]#from sign********************************************************
   thumb_tip = 4#from sign***********************************************************************
   kbB = 0 #***********************************************************************************
   slB = 0 #***********************************************************************************
   stopB=0 #***********************************************************************************
   ######################
   wCam, hCam = 640, 480
   frameR = 100  # Frame Reduction
   smoothening = 7  # random value
   ######################

   pTime = 0
   plocX, plocY = 0, 0
   clocX, clocY = 0, 0
   cap = cv2.VideoCapture(0)
   cap.set(3, wCam)
   cap.set(4, hCam)

   detector = htm.handDetector(maxHands=1)
   wScr, hScr = autopy.screen.size()

   # print(wScr, hScr)

   while True:
      # Step1: Find the landmarks
      success, img = cap.read()
      h, w, c = img.shape    #from sign********************************************************
      results = hands.process(img)#from sign********************************************************
      img = detector.findHands(img)
      lmList, bbox = detector.findPosition(img)

      # Step2: Get the tip of the index and middle finger
      if len(lmList) != 0:

         x2, y2 = lmList[12][1:]
         x1, y1 = lmList[8][1:]  # middleeee
         x4, y4 = lmList[4][1:]  # *******
         x5, y5 = lmList[16][1:]  # *
         x6, y6 = lmList[20][1:]  # *

         # Step3: Check which fingers are up
         fingers = detector.fingersUp()
         cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                       (255, 0, 255), 2)

         # Step4: Only Index Finger: Moving Mode
         if fingers[1] == 1 and fingers[3] == 0 and fingers[4] == 0:
            # Step5: Convert the coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Step6: Smooth Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Step7: Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

         # Step8: Both Index and middle are up: Clicking Mode
         if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:

            # Step9: Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Step10: Click mouse if distance short
            if length < 30:
               cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

               # autopy.mouse.click()
               pyautogui.click(button='left')
         # *********************************

         # Step10: scroll mouse if distance long
         # if length > 40:
         # cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

         # pyautogui.scroll(15, x=100, y=100)
         # ***************scrolling down******************





         # ******************************************************************************************************
         # ******************************************************************************************************

         if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            slB+=1

            if slB == 10:
               sl=1
               break
         if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
               lm_list = []
               for id, lm in enumerate(hand_landmark.landmark):
                  lm_list.append(lm)
               finger_fold_status = []
               for tip in finger_tips:
                  x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                  # print(id, ":", x, y)
                  # cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                  if lm_list[tip].x < lm_list[tip - 2].x:
                     # cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                     finger_fold_status.append(True)
                  else:
                     finger_fold_status.append(False)

               print(finger_fold_status)

               x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
               print(x, y)
            if lm_list[9].y < lm_list[1].y and lm_list[9].y > lm_list[0].y and \
                    lm_list[6].x > lm_list[8].x and lm_list[10].x > lm_list[12].x and lm_list[14].x > lm_list[
               16].x and \
                    lm_list[18].x > lm_list[20].x:
               print("DISLIKE")
               cv2.putText(img, "quit", (550,20 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               stopB+=1
               if stopB==8:
                  break
         #******************************************************************************************************
         #******************************************************************************************************





         if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # Step9: Find distance between fingers
            # length, img, lineInfo = detector.findDistance(8, 12, img)

            # Step10: scroll mouse if distance long
            # if length > 40:
            # cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

            pyautogui.scroll(-15, x=100, y=100)
         #*******************************************************************


         # *****************scrolling up****************
         if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            pyautogui.scroll(15, x=100, y=100)

            # tryyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
         if fingers[1] == 1 and fingers[0] == 1:

            # Step9: Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 4, img)

            # Step10: Click mouse if distance short
            if length < 25:
               cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

               # autopy.mouse.click()
               pyautogui.click(button='right')

               # Step11: Frame rate
      cTime = time.time()
      fps = 1 / (cTime - pTime)
      pTime = cTime
      cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)

      # Step12: Display
      cv2.imshow("Image", img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break
   cap.release()
   cv2.destroyAllWindows()
   return sl,kb

# B = tkinter.Button(top, text="Hello", command=helloCallBack)
def signlangandControl():
   mp_holistic = mp.solutions.holistic  # Holistic model
   mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

   def mediapipe_detection(image, model):
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
      image.flags.writeable = False  # Image is no longer writeable
      results = model.process(image)  # Make prediction
      image.flags.writeable = True  # Image is now writeable
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
      return image, results

   def draw_styled_landmarks(image, results):
      # Draw left hand connections
      mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                )
      # Draw right hand connections
      mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )

   url = "http://192.168.1.184"
   # for lamp image *******************************************
   onlamp = cv2.imread(r'lamp.png')
   onlamp = cv2.resize(onlamp, (50, 50))
   # for right hand *******************************************
   l1 = 0
   l2 = 0
   l3 = 0
   l1break = 0
   l2break = 0
   l3break = 0
   allbreak = 0
   alloffbreak = 0
   stopB=0
   mp_holistic = mp.solutions.holistic  # Holistic model
   mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

   finger_tips = [8, 12, 16, 20]
   thumb_tip = 4
   cap = cv2.VideoCapture(0)
   # Set mediapipe model
   with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():

         # Read feed
         ret, frame = cap.read()
         frame = cv2.flip(frame, 1)
         h, w, c = frame.shape
         # Make detections
         image, results = mediapipe_detection(frame, holistic)
         print(results)
         if results.right_hand_landmarks:
            # for hand_landmark in results.right_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(results.right_hand_landmarks.landmark):
               lm_list.append(lm)
            finger_fold_status = []
            for tip in finger_tips:
               x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
               # print(id, ":", x, y)
               # cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

               if lm_list[tip].x < lm_list[tip - 2].x:
                  # cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                  finger_fold_status.append(True)
               else:
                  finger_fold_status.append(False)

            print(finger_fold_status)

            x, y = int(lm_list[8].x * w), int(lm_list[8].y * h)
            print(x, y)

            # stop
            if lm_list[4].y < lm_list[2].y and lm_list[8].y < lm_list[6].y and lm_list[12].y < lm_list[10].y and \
                    lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y and lm_list[17].x < lm_list[0].x < \
                    lm_list[5].x:
               cv2.putText(image, "STOP", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               print("STOP")
               stopB+=1
               if stopB==10:
                  break
            # # Forward
            # if lm_list[3].x > lm_list[4].x and lm_list[8].y < lm_list[6].y and lm_list[12].y > lm_list[10].y and \
            #         lm_list[16].y > lm_list[14].y and lm_list[20].y > lm_list[18].y:
            #     cv2.putText(image, "FORWARD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     print("FORWARD")

            # # Backward
            # if lm_list[3].x > lm_list[4].x and lm_list[3].y < lm_list[4].y and lm_list[8].y > lm_list[6].y and lm_list[12].y < lm_list[10].y and \
            #         lm_list[16].y < lm_list[14].y and lm_list[20].y < lm_list[18].y:
            #     cv2.putText(image, "BACKWARD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     print("BACKWARD")

            # # Left
            # if lm_list[4].y < lm_list[2].y and lm_list[8].x < lm_list[6].x and lm_list[12].x > lm_list[10].x and \
            #         lm_list[16].x > lm_list[14].x and lm_list[20].x > lm_list[18].x and lm_list[5].x < lm_list[0].x:
            #     cv2.putText(image, "LEFT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     print("LEFT")

            # Right
            if lm_list[4].y < lm_list[2].y and lm_list[8].x > lm_list[6].x and lm_list[12].x < lm_list[10].x and \
                    lm_list[16].x < lm_list[14].x and lm_list[20].x < lm_list[18].x:
               cv2.putText(image, "number one :)-", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               print("RIGHT")

            # yes
            if lm_list[4].y > lm_list[8].y and lm_list[4].y > lm_list[12].y and lm_list[4].y > lm_list[16].y and \
                    lm_list[4].y > lm_list[20].y and \
                    lm_list[6].y > lm_list[17].y and lm_list[5].y < lm_list[18].y:
               print("LIKE")
               cv2.putText(image, "yes", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # i love you
            if lm_list[8].y < lm_list[7].y < lm_list[5].y and lm_list[20].y < lm_list[19].y and \
                    lm_list[6].y < lm_list[12].y and lm_list[6].y < lm_list[16].y and \
                    lm_list[6].y < lm_list[4].y:
               cv2.putText(image, "i love you ", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # like
            if lm_list[9].y > lm_list[1].y and lm_list[9].y < lm_list[0].y and \
                    lm_list[6].x > lm_list[8].x and lm_list[10].x > lm_list[12].x and lm_list[14].x > lm_list[16].x and \
                    lm_list[18].x > lm_list[20].x:
               print("LIKE")
               cv2.putText(image, "LIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
               # h, w, c = like_img.shape
               # img[35:h + 35, 30:w + 30] = like_img
            # Dislike
            if lm_list[9].y < lm_list[1].y and lm_list[9].y > lm_list[0].y and \
                    lm_list[6].x > lm_list[8].x and lm_list[10].x > lm_list[12].x and lm_list[14].x > lm_list[16].x and \
                    lm_list[18].x > lm_list[20].x:
               print("DISLIKE")
               cv2.putText(image, "dislike", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

               # h, w, c = dislike_img.shape
               # img[35:h + 35, 30:w + 30] = dislike_img

         if results.left_hand_landmarks:
            # for hand_landmark in results.right_hand_landmarks:
            llm_list = []
            for id, llm in enumerate(results.left_hand_landmarks.landmark):
               llm_list.append(llm)
            lfinger_fold_status = []
            for tip in finger_tips:
               x, y = int(llm_list[tip].x * w), int(llm_list[tip].y * h)
               # print(id, ":", x, y)
               # cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

               if llm_list[tip].x < llm_list[tip - 2].x:
                  # cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                  lfinger_fold_status.append(True)
               else:
                  lfinger_fold_status.append(False)

            print(lfinger_fold_status)

            x, y = int(llm_list[8].x * w), int(llm_list[8].y * h)
            print(x, y)

            # if llm_list[4].y < llm_list[2].y and llm_list[8].y < llm_list[6].y and llm_list[12].y < llm_list[10].y and \
            #         llm_list[16].y < llm_list[14].y and llm_list[20].y < llm_list[18].y and llm_list[17].x > llm_list[
            #     0].x > \
            #         llm_list[5].x:
            #     cv2.putText(image, "STOP", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     print("STOP")
            # turn on all
            if llm_list[9].y > llm_list[1].y and llm_list[9].y < llm_list[0].y and \
                    llm_list[6].x < llm_list[8].x and llm_list[10].x < llm_list[12].x and llm_list[14].x < llm_list[
               16].x and \
                    llm_list[18].x < llm_list[20].x:
               print("LIKE")
               cv2.putText(image, "turn on all ", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
               allbreak += 1
               if allbreak == 10:
                  l1 = 1
                  l2 = 1
                  l3 = 1
                  allbreak = 0

            if llm_list[9].y < llm_list[1].y and llm_list[9].y > llm_list[0].y and \
                    llm_list[6].x < llm_list[8].x and llm_list[10].x < llm_list[12].x and llm_list[14].x < llm_list[
               16].x and \
                    llm_list[18].x < llm_list[20].x:
               print("DISLIKE")
               cv2.putText(image, "turn off all", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               alloffbreak += 1
               if alloffbreak == 10:
                  l1 = 0
                  l2 = 0
                  l3 = 0
                  alloffbreak = 0
               # lamp1
            if llm_list[8].y < llm_list[7].y < llm_list[6].y and \
                    llm_list[6].y < llm_list[12].y and llm_list[6].y < llm_list[16].y and llm_list[6].y < llm_list[
               20].y and \
                    llm_list[6].y < llm_list[4].y:
               cv2.putText(image, "lamb1", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               print("lamb1")

               l1break += 1
               if l1break == 10:
                  if l1 == 0:
                     l1 = 1
                  else:
                     l1 = 0
                  l1break = 0

            # lamp2
            if llm_list[8].y < llm_list[7].y and llm_list[12].y < llm_list[11].y and \
                    llm_list[6].y < llm_list[16].y and llm_list[6].y < llm_list[20].y and llm_list[6].y < llm_list[4].y \
                    and llm_list[10].y < llm_list[16].y and llm_list[10].y < llm_list[20].y and llm_list[10].y < \
                    llm_list[4].y:
               cv2.putText(image, "lamb2", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               print("lamb2")
               l2break += 1
               if l2break == 10:
                  if l2 == 0:
                     l2 = 1
                  else:
                     l2 = 0
                  l2break = 0

            # lamp3
            if llm_list[16].y < llm_list[15].y and llm_list[12].y < llm_list[11].y and llm_list[20].y < llm_list[
               19].y and \
                    llm_list[10].y < llm_list[4].y and llm_list[10].y < llm_list[8].y \
                    and llm_list[14].y < llm_list[4].y and llm_list[14].y < llm_list[8].y \
                    and llm_list[18].y < llm_list[4].y and llm_list[18].y < llm_list[8].y \
                    and llm_list[6].y < llm_list[17].y:
               cv2.putText(image, "lamb3", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
               print("lamb3")
               l3break += 1
               if l3break == 10:
                  if l3 == 0:
                     l3 = 1
                  else:
                     l3 = 0
                  l3break = 0

         # ****************************************************************
         if l1 == 1:
            h, w, c = onlamp.shape
            image[20:h + 20, 580:w + 580] = onlamp
         if l2 == 1:
            h, w, c = onlamp.shape
            image[80:h + 80, 580:w + 580] = onlamp
         if l3 == 1:
            h, w, c = onlamp.shape
            image[140:h + 140, 580:w + 580] = onlamp

         # ****************************************************************
         # Draw landmarks
         draw_styled_landmarks(image, results)

         # Show to screen
         cv2.imshow('OpenCV Feed', image)
         """if l1 == 1:
            try:
               urllib.request.urlopen("http://192.168.1.184/led1on")
            except:
               print("")
         elif l1 == 0:
            try:
               urllib.request.urlopen("http://192.168.1.184/led1off")
            except:
               print("")
         if l2 == 1:
            try:
               urllib.request.urlopen("http://192.168.1.184/led2on")
            except:
               print("")
         elif l2 == 0:
            try:
               urllib.request.urlopen("http://192.168.1.184/led2off")
            except:
               print("")
         if l3 == 1:
            try:
               urllib.request.urlopen("http://192.168.1.184/led3on")
            except:
               print("")
         elif l3 == 0:
            try:
               urllib.request.urlopen("http://192.168.1.184/led3off")
            except:
               print("") 
         # Break gracefully
         if cv2.waitKey(10) & 0xFF == ord('q'):
            break """


      cap.release()
      cv2.destroyAllWindows()
# B.pack()
# top.mainloop()


while True:
   keyboard=0
   signlang=0
   signlang,keyboard=mousefunch()
   if signlang==1:
      signlangandControl()
   if signlang==0 and keyboard==0:
      break


