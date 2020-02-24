import cv2
import numpy as np

from PIL import Image


def ORB_detector(new_image, image_template):
    # Function that compares input image to template
    # It then returns the number of ORB matches between them
    
    # img = cv2.fastNlMeansDenoising(new_image)
    # img_c = cv2.fastNlMeansDenoising(image_template)
    
    test = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    control= cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000)

    # Detect keypoints of original image
    kp1, des1 = orb.detectAndCompute(test, None)

    # Detect keypoints of rotated image
    kp2, des2 = orb.detectAndCompute(control, None)


    # Create matcher 
    # Note we're no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Do matching
    matches = bf.match(des1,des2)

    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)

    return len(matches)


# Load our image template, this is our reference image

# im = Image.open("./assets/sofaFive.webp").convert("RGB")
# im.save("./assets/SofaFive.jpg","jpeg")
image_template = cv2.imread('./assets/s2.jpg') 
# im2 = Image.open("./assets/sofaFive_wood.webp").convert("RGB")
# im2.save("./assets/sofaFive_wood.jpg","jpeg")
image = cv2.imread('./assets/s2.jpg') 
# image_template = cv2.imread('images/kitkat.jpg', 0) 

    
# Get number of ORB matches 
matches = ORB_detector(image, image_template)
    
# Display status string showing the current no. of matches 
output_string = "Matches = " + str(matches)

print(output_string)
cv2.destroyAllWindows()   