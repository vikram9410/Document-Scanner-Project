import cv2
import numpy as np
import imutils
#  this function is for joining images
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# Driver Code

args_image ="resources/img1.jpg"

image = cv2.imread(args_image)
imgContours=image.copy()
image=cv2.resize(image,(500,500))
orig = image.copy()
#Converting color image to grayscale image
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Converting grayscale image to blur image
grayImageBlur = cv2.blur(grayImage,(2,2))

#Converting blur image to canny image
#Canny image helps us in finding edges in image
edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)


# This below code helps to find largest size contour in image
# procedure #
# 1--> It find all contours present in image and return list of all contours.
# 2--> sort all contours according to their size in decreasing order.
# 3--> select 1 index contour from sorted contours list.

allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContours = imutils.grab_contours(allContours)
allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]

# detecting scanned document
perimeter = cv2.arcLength(allContours[0], True)
ROIdimensions =cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
cv2.drawContours(imgContours,ROIdimensions, -1, (0,255,0), 3)

# Apply warp perspective transformation
# warp perspective transformation is point sensitive so before applying wrap perspective we need to find
# exact ordered points of our contour
# This below code helps us to do reordering of contour point
# rect[0] --> top left point in contour (tl)
# rect[2] --> bottom right point in contour (br)
# rect[1] --> top right  point in contour (tr)
# rect[3] --> bottom left point in contour (bl)
# Detailed description about how to find tl,br,tr,bl using mathematics is provided in readme file.
ROIdimensions = ROIdimensions.reshape(4,2)
rect = np.zeros((4,2), dtype="float32")
s = np.sum(ROIdimensions, axis=1)
rect[0] = ROIdimensions[np.argmin(s)]
rect[2] = ROIdimensions[np.argmax(s)]
diff = np.diff(ROIdimensions, axis=1)
rect[1] = ROIdimensions[np.argmin(diff)]
rect[3] = ROIdimensions[np.argmax(diff)]

(tl, tr, br, bl) = rect
# this will find the width and height of our document using distance formula
widthA = np.sqrt((tl[0] -tr[0])**2 + (tl[1] - tr[1])**2 )
widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
maxHeight = max(int(heightA), int(heightB))
dst = np.array([
    [0,0],
    [maxWidth-1, 0],
    [maxWidth-1, maxHeight-1],
    [0, maxHeight-1]], dtype="float32")

# applying transformation
transformMatrix = cv2.getPerspectiveTransform(rect, dst)
scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))

# scanning document

scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
scanGray= cv2.adaptiveThreshold(scanGray,100,1,1,7,2)
scanGray= cv2.bitwise_not(scanGray)

# showing output result
# stackImage function join all images to single image
output=stackImages(0.8,[[image,grayImage,edgedImage],[imgContours,scan,scanGray]])
cv2.imshow("scanGray", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
