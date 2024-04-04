
import cv2
import os
import imageio

file = r'D:\Gil\kanferg_web\blog\temp\videos\smlm_example.avi'
cap = cv2.VideoCapture(file)
image_lst = []
 
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_lst.append(frame_rgb)
    
    cv2.imshow('a', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
 
 
# Convert to gif using the imageio.mimsave method
out = r'D:\Gil\kanferg_web\blog\_static\images\smlm.gif'
imageio.mimsave(out, image_lst, fps=10)