import numpy as np
import cv2
import os

home=os.getenv("HOME")
path=home+"/aqualoc_dataset/images_sequence/frame00"


path0=path+"0000.png"

i0=cv2.imread(path0)

print(home)
i0_gray=cv2.cvtColor(i0,cv2.COLOR_BGR2GRAY)

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 5,
                       blockSize = 7 )
lkt_params=dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))               

features = cv2.goodFeaturesToTrack(i0_gray, mask = None, **feature_params)
mask1 = np.zeros_like(i0)
mask2 = np.zeros_like(i0)
prev_img=i0
prev_img_gray=i0_gray
prev_points=features
#print(np.shape(prev_points))
#t=int(input("Enter frame number"))
for i in range(1,3000):
    suf= "000" if i<=9 else "00" if i<=99 else "0" if i<=999 else ""
    suf=suf+str(i)+".png"
    pathi=path+suf

    next_img=cv2.imread(pathi)
    next_img_gray=cv2.cvtColor(next_img,cv2.COLOR_BGR2GRAY)

    next_points, st, err=cv2.calcOpticalFlowPyrLK(prev_img_gray, next_img_gray, prev_points, None, **lkt_params)
    filter_points, st, err=cv2.calcOpticalFlowPyrLK(next_img_gray, prev_img_gray, next_points, None, **lkt_params)

    
    n=next_points[st==1]
    p=prev_points[st==1]
    f=filter_points[st==1]

    error=np.sum(((f-p)*(f-p)), axis=1)
    er=error<1

    for i,(new,old,filt) in enumerate(zip(n,p,er)):
        if not filt :
            continue
        a,b = new.ravel()
        c,d = old.ravel()
        mask1 = cv2.line(mask1, (a,b),(c,d), [255,0,0], 2)
        rem=i%4
        if i>=2:
            mask2 = cv2.line(mask2, (a,b),(c,d), [255,0,0], 2)
        #i1 = cv2.circle(i1,(a,b),5,[255,0,0],-1)
    
    img = cv2.add(next_img,mask1)
    if i%4==0:
        mask1=mask2.copy()
        mask2 = np.zeros_like(i0)
        
    prev_img=next_img.copy()
    prev_img_gray=next_img_gray.copy()
    prev_points=np.reshape(n,(-1,1,2))
    if i%10==0:
        prev_points = cv2.goodFeaturesToTrack(prev_img_gray, mask = None, **feature_params)
    cv2.imshow("Optical Flow",img)
    cv2.waitKey(40)
cv2.waitKey(0)
cv2.destroyAllWindows()

