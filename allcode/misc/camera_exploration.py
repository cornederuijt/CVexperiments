import numpy as np
import cv2

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frame = cv2.flip(frame, 1)

        #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        winter = cv2.cvtColor(frame, cv2.COLORMAP_WINTER)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()