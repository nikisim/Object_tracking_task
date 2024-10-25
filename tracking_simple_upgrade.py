import cv2
import numpy as np

def calculate_movement_direction(prev_bbox, curr_bbox, eps = 2):
    # Previous bounding box (x, y, w, h)
    prev_x, prev_y, prev_w, prev_h = prev_bbox
    # Current bounding box (x, y, w, h)
    curr_x, curr_y, curr_w, curr_h = curr_bbox

    prev_center_x = prev_x + prev_w // 2
    prev_center_y = prev_y + prev_h // 2

    curr_center_x = curr_x + curr_w // 2
    curr_center_y = curr_y + curr_h // 2

    # Calculate the movement delta
    dx = curr_center_x - prev_center_x
    dy = curr_center_y - prev_center_y

    if abs(dy) > eps:
        if dy > 0:
            print('Движение вниз')
        elif dy < 0:
            print('Движение вверх')

    if abs(dx) > eps:
        if dx > 0:
            print('Движение вправо')
        elif dx < 0:
            print('Движение влево')


if __name__ == "__main__":

    # video = cv2.VideoCapture('Moving Circle.mp4')
    # video = cv2.VideoCapture('Road_traffic.mp4')
    video = cv2.VideoCapture('Car passing.mp4')

    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=True)

    prev_rect = (0,0,0,0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Delete background
        fgmask = fgbg.apply(frame)
        cv2.imshow('Bg mask', fgmask)


        # # Применение операции ерозии
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        # mask_opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        mask_opening = cv2.erode(fgmask,kernel,iterations = 1)

        cv2.imshow('Opening mask', mask_opening)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Filter by size
            if cv2.contourArea(contour) > 2200:
                print("---"*10)
                print("Объект обнаружен")
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                curr_rect = (x, y, x + w, y + h)

                calculate_movement_direction(prev_rect, curr_rect)

                prev_rect = curr_rect

        cv2.imshow('Detected Objects', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()