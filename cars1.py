import cv2

def calculate_movement_direction(prev_bbox, curr_bbox, eps = 2):
    # Previous bounding box (x, y, w, h)
    prev_x, prev_y, prev_w, prev_h = prev_bbox
    # Current bounding box (x, y, w, h)
    curr_x, curr_y, curr_w, curr_h = curr_bbox

    # Calculate the centroids
    prev_center_x = prev_x + prev_w // 2
    prev_center_y = prev_y + prev_h // 2

    curr_center_x = curr_x + curr_w // 2
    curr_center_y = curr_y + curr_h // 2

    # Calculate the movement vector
    dx = curr_center_x - prev_center_x
    dy = curr_center_y - prev_center_y

    # Determine the direction
    if abs(dx) > eps:
        if dx > 0:
            print('Moving right →')
        elif dx < 0:
            print('Moving left ←')

    if abs(dy) > eps:
        if dy > 0:
            print('Moving down ↓')
        elif dy < 0:
            print('Moving up ↑')


if __name__ == "__main__":

    video = cv2.VideoCapture('Car passing.mp4')  # Replace with your video source

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

    prev_rect = (0,0,0,0)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Delete foreground
        fgmask = fgbg.apply(frame)

        cv2.imshow('Mask', fgmask)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter by size
            print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 4200:
                print("---"*10)
                print("Object detected")
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