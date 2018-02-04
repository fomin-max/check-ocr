# USAGE
# python check_ocr.py --image test.png
# python check_ocr.py --image 4.png --reference test1.png

# import the necessary packages
import argparse

import cv2
import imutils
import numpy as np
from imutils import contours


def extract_digits_and_symbols(image, charCnts, minW=3, minH=3):
    # grab the internal Python iterator for the list of character
    # contours, then  initialize the character ROI and location
    # lists, respectively
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    # keep looping over the character contours until we reach the end
    # of the list
    while True:
        try:
            # grab the next character contour from the list, compute
            # its bounding box, and initialize the ROI
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            # check to see if the width and height are sufficiently
            # large, indicating that we have found a digit
            print(cW, cH)
            if cW >= minW and cH >= minH:
                # extract the ROI
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX - 1, cY - 1, cX + cW + 1, cY + cH + 1))

            # otherwise, we are examining one of the special symbols
            else:
                # MICR symbols include three separate parts, so we
                # need to grab the next two parts from our iterator,
                # followed by initializing the bounding box
                # coordinates for the symbol
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
                                        -np.inf)

                # loop over the parts
                for p in parts:
                    # compute the bounding box for the part, then
                    # update our bookkeeping variables
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                # extract the ROI
                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))

        # we have reached the end of the iterator; gracefully break
        # from the loop
        except StopIteration:
            break

    # return a tuple of the ROIs and locations
    return (rois, locs)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-r", "--reference", required=True,
                help="path to reference")
args = vars(ap.parse_args())

charNames = ['J', '1', 'S', 'A', '2', 'K', 'T', 'B', '3', 'U', 'L', 'C', '4', 'M', 'V', 'D', 'N', '5', 'W',
             'E', '6', '0', 'X', 'F', '7', 'Y', 'P', 'G', '8', 'Q', 'Z', 'H', '9', 'O', 'R', 'I', '<', '>']

# load the reference image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = imutils.resize(ref, width=650, height=250)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
                    cv2.THRESH_OTSU)[1]

# find contours in the MICR image (i.e,. the outlines of the
# characters) and sort them from left to right
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]


# extract the digits and symbols from the list of contours, then
# initialize a dictionary to map the character name to the ROI
(refROIs, refLocs) = extract_digits_and_symbols(ref, refCnts,
                                                minW=3, minH=3)
chars = {}

# loop over the reference ROIs
for (name, roi) in zip(charNames, refROIs):
    # resize the ROI to a fixed size, then update the characters
    # dictionary, mapping the character name to the ROI
    roi = cv2.resize(roi, (8, 8))
    chars[name] = roi
    # cv2.imshow(str(name), chars[str(name)])
# cv2.imshow("M", chars["M"])
# cv2.imshow("F", chars["F"])
# cv2.waitKey(0)

ref = cv2.imread(args["image"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = imutils.resize(ref, width=850, height=350)
ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV |
                    cv2.THRESH_OTSU)[1]

# find contours in the MICR image (i.e,. the outlines of the
# characters) and sort them from left to right
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]


# extract the digits and symbols from the list of contours, then
# initialize a dictionary to map the character name to the ROI
(refROIs, refLocs) = extract_digits_and_symbols(ref, refCnts,
                                                minW=3, minH=3)


groupOutput = []

# re-initialize the clone image so we can draw on it again
clone = np.dstack([ref.copy()] * 3)

# loop over the reference ROIs and locations
for (roi, loc) in zip(refROIs, refLocs):
    # draw a bounding box surrounding the character on the output
    # image
    (xA, yA, xB, yB) = loc
    cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 1)

    # resize the ROI to a fixed size, then update the characters
    # dictionary, mapping the character name to the ROI
    roi = cv2.resize(roi, (8, 8))
    scores = []

    for charName in charNames:
        # apply correlation-based template matching, take the
        # score, and update the scores list
        result = cv2.matchTemplate(roi, chars[charName],
                                   cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        scores.append(score)
    # chars[name] = roi
    groupOutput.append(charNames[np.argmax(scores)])
    # display the character ROI to our screen
# show the output

t = groupOutput.copy()
t1 = []
t2 = []

for i in range(len(t)):
    if i % 2 == 1:
        t1.append(t[i])
    else:
        t2.append(t[i])
print(t1)
print(t2)
# print(groupOutput)
cv2.imshow("Result", clone)
cv2.waitKey(0)
