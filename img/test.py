import cv2

img_path = "./test/test/db2069113dbf51a739f8d4cac2d23e23146cdbe48f1facd704ac46f3e2f7038a.jpg"

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv = 255 - gray
w,h = inv.shape
with open("./gray.txt", 'w') as f:
    for i in range(w):
        for j in range(h):
            f.write(str(gray[i,j]) + " ")
        f.write("\n")
with open("./inv.txt", 'w') as f:
    for i in range(w):
        for j in range(h):
            f.write(str(inv[i,j]) + " ")
        f.write("\n")        
blur = cv2.GaussianBlur(inv, ksize=(15,15), sigmaX=50, sigmaY=50)
res = cv2.divide(gray, 255 - blur, scale=255)
cv2.imwrite("./ssss.png", res)
# cv2.imshow("ssss", gray)
# cv2.waitKey(0)