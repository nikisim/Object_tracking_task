import cv2
from tracking_simple import calculate_movement_direction


if __name__ == "__main__":

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # Load the labels
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    video = cv2.VideoCapture('Car passing.mp4')

    prev_rect = (0,0,0,0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Prepare for YOLO
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Run the net
        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]

                # Filter detections
                if confidence > 0.5:
                    # Getting boxes
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Convert the coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indices) > 0:
            print("Объект обнаружен")
            # Draw
            for i in indices:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                curr_rect = (x, y, x + w, y + h)
                calculate_movement_direction(prev_rect, curr_rect)
                prev_rect = curr_rect

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                print('---'*10)
        else:
            print("Объекта в кадре нет")



        cv2.imshow('YOLO Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()