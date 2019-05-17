import cv2
import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd

drawing=False # true if mouse is pressed

pts = []

def trainAndDraw() :
    global pts
    model = LinearRegression()
    df = pd.DataFrame(data = pts, columns=['x', 'y'])
    model.fit(df.iloc[:, 0:1], df.iloc[:, 1])
    x1 = pts[0][0]
    x2 = pts[-1][0]
    y1 = model.predict([[x1]])
    y2 = model.predict([[x2]])
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    pts = []
    
# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global drawing, pts

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pts.append([x, y])

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(img, (x,y), 1, (0,0,255), -1)    
        pts.append([x, y])
               
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        cv2.circle(img, (x,y), 1, (0,0,255), -1)  
        pts.append([x, y])
        trainAndDraw()


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Window')
cv2.setMouseCallback('Window', mouse_callback)
while(True):
    cv2.imshow('Window',img)
    if cv2.waitKey(1) == 27: # ESC
        break
cv2.destroyAllWindows()
