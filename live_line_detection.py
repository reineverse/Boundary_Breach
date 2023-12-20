import cv2
import numpy as np
import pygame

pygame.mixer.init()
pygame.mixer.init()
sound = pygame.mixer.Sound("C://Users//Rekhapallamreddy//Downloads//police.mp3")

def detect_movement(frame, bg_subtractor, min_contour_area):
    # Apply the background subtractor to the frame
    mask = bg_subtractor.apply(frame)
    # Apply a binary threshold to the mask
    _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate over the contours and find the ones that are above the minimum area
    moving_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            moving_contours.append(contour)
    # Draw a green border around each moving contour
    output = frame.copy()
    for contour in moving_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Play the sound if there is at least one moving object
    if len(moving_contours) > 0 and not pygame.mixer.get_busy():
        sound.play()
    # Stop the sound if there are no moving objects
    elif len(moving_contours) == 0 and pygame.mixer.get_busy():
        sound.stop()
    return output

cap = cv2.VideoCapture(0)
# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
# Set the minimum contour area
min_contour_area = 500

while True:
    ret, frame = cap.read()
    if not ret:
        break
    output = detect_movement(frame, bg_subtractor, min_contour_area)
    cv2.imshow("Output", output)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
