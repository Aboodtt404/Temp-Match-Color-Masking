import cv2
import numpy as np

def rotate(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

def tmri(template, image, threshold=0.5):  #0.93
    best_angle = 0
    best_score = -1
    rel_loc = 0

    for angle in range(0, 360, 5):
        rotated_template = rotate(template, angle)
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
        _ , max_val, _, maxLoc = cv2.minMaxLoc(result)

        if max_val > best_score and max_val > threshold:
            best_score = max_val
            best_angle = angle
            rel_loc = maxLoc

    return best_angle, best_score, rel_loc

template_path = "./Images/rotated_ball.png"
image_path = "./Images/soccer_practice.jpg"

template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

rotation_angle, best_score, max_loc = tmri(template, image)

if best_score > 0:
    h, w = template.shape
    rotated_template = rotate(template, rotation_angle)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)

    cv2.imshow("Image with Rotated Template", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No match found.")