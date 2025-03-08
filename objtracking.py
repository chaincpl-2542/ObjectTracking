import cv2 as cv
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo12m.pt")

video_path = "vdo6.mp4"
cap = cv.VideoCapture(video_path)

target_width, target_height = 1920, 1080

fps = int(cap.get(cv.CAP_PROP_FPS))

out = cv.VideoWriter("output_resized.avi", cv.VideoWriter_fourcc(*'XVID'), fps, (target_width, target_height))

left_area = np.array([(0, 1080), (0, 540), (1020, 540), (1020, 1080)], np.int32) 
right_area = np.array([(1020, 1080), (1020, 540), (1920, 540), (1920, 1080)], np.int32)

tracked_vehicles = {}

cars_go = 0
cars_return = 0

MOVEMENT_THRESHOLD = 50
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv.resize(frame, (target_width, target_height))

    results = model.track(frame, conf=0.25)

    cv.polylines(frame, [left_area], isClosed=True, color=(0, 255, 255), thickness=2)
    cv.polylines(frame, [right_area], isClosed=True, color=(255, 0, 255), thickness=2)

    current_vehicles = {}

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = box.conf[0]  
            class_id = int(box.cls[0]) 

            # Only process cars
            if class_id != 2:
                continue

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            in_left = cv.pointPolygonTest(left_area, (center_x, center_y), False) >= 0
            in_right = cv.pointPolygonTest(right_area, (center_x, center_y), False) >= 0

            if in_left or in_right:
                vehicle_id = None
                for vid, (prev_x, prev_y) in tracked_vehicles.items():
                    if abs(center_x - prev_x) < MOVEMENT_THRESHOLD and abs(center_y - prev_y) < MOVEMENT_THRESHOLD:
                        vehicle_id = vid
                        break
                
                if vehicle_id is None:
                    vehicle_id = len(tracked_vehicles) + 1
                    tracked_vehicles[vehicle_id] = (center_x, center_y)

                current_vehicles[vehicle_id] = (center_x, center_y)

                if vehicle_id not in tracked_vehicles:
                    tracked_vehicles[vehicle_id] = (center_x, center_y)
                    if in_left:
                        cars_go += 1
                    if in_right:
                        cars_return += 1

                color = (0, 255, 0)
                label = f'Car {confidence:.2f} ID:{vehicle_id}'
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.putText(frame, f'Going: Cars {cars_go}', (20, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.putText(frame, f'Returning: Cars {cars_return}', (20, 100), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    out.write(frame)

    cv.imshow("YOLO Vehicle Detection with Real Counting", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()

