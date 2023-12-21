import cv2
import numpy as np

def tmmo(template, image, threshold=0.9):
    h, w = template.shape
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    matches = []

    for loc in zip(*locations[::-1]):
        match = {
            "top_left": loc,
            "bottom_right": (loc[0] + w, loc[1] + h)
        }
        matches.append(match)

    return matches

template_path = "./test-images/mario.png"
image_path = "./test-images/mario.png"

template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

matches = tmmo(template, image)

for match in matches:
    top_left = match["top_left"]
    bottom_right = match["bottom_right"]
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)

cv2.imshow("Image with Template Matches", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
