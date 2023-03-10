import cv2


# Read a sprite sheet and cut it into individual sprites.

image = cv2.imread('/home/fsuser/practice/stable-diffusion-webui-docker/fdata/c1e17c53f3b60ccb6d780ed8a3adcc83.png')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
dilate = cv2.dilate(close, kernel, iterations=1)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

sprite_number = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ROI = image[y:y+h, x:x+w]
    cv2.imwrite('sprites/sprite_{}.png'.format(sprite_number), ROI)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    sprite_number += 1