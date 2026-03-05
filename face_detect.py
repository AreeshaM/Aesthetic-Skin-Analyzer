import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Starting Aesthetic Analysis... Press 'q' to quit.")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    recommendation = "Analyzing skin..."
    disclaimer = "This analysis is AI generated. For better result concern your doctor."

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        roi_color = frame[y:y+h , x:x+w]
        roi_gray = gray[y:y+h , x:x+w]

        # -------- Texture Analysis --------
        laplacian = cv2.Laplacian(roi_gray,cv2.CV_64F)
        texture_value = laplacian.var()

        if texture_value > 150:
            texture = "Rough Texture"
        else:
            texture = "Smooth Skin"


        # -------- Spot / Acne Detection --------
        hsv = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)

        lower_spot = np.array([0,0,0])
        upper_spot = np.array([180,255,70])

        mask = cv2.inRange(hsv,lower_spot,upper_spot)

        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        spot_count = 0

        for cnt in contours:

            if cv2.contourArea(cnt) > 40:

                spot_count += 1

                sx,sy,sw,sh = cv2.boundingRect(cnt)

                cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),1)


        # -------- Skin Tone Analysis --------
        avg_color = np.mean(roi_color)

        if avg_color < 85:
            tone = "Dry/Dull Skin"
        elif avg_color < 170:
            tone = "Normal Skin"
        else:
            tone = "Oily/Bright Skin"


        # -------- Recommendation Logic --------
        if spot_count > 6:
            recommendation = "Use Salicylic Acid face wash and Vitamin C serum."
        elif texture == "Rough Texture":
            recommendation = "Use gentle exfoliator and hydrating moisturizer."
        else:
            recommendation = "Skin looks healthy. Maintain cleansing and sunscreen."


        # -------- Display Results --------
        cv2.putText(frame,f"Texture: {texture}",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame,f"Spots: {spot_count}",(20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame,f"Skin Type: {tone}",(20,140),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)


    cv2.putText(frame,recommendation,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    cv2.putText(frame,disclaimer,(20,460),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

    cv2.imshow("Aesthetic Skin Analyzer",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()