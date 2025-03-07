import cv2 as cv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo12n.pt")

video_path = "vdo6.mp4"
cap = cv.VideoCapture(video_path)

target_width, target_height = 1920, 1080

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv.CAP_PROP_FPS))

# Define output video writer
out = cv.VideoWriter("output_resized.avi", cv.VideoWriter_fourcc(*'XVID'), fps, (target_width, target_height))

detection_area = np.array([(0, 1080), (0, 540), (1920, 540), (1920, 1080)], np.int32)
detection_area = detection_area.reshape((-1, 1, 2)) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize frame to 1920x1080
    frame = cv.resize(frame, (target_width, target_height))
        
    # Run YOLO on the frame
    results = model(frame, conf=0.25)
    
    # Car count
    car_count = 0
    truck_count = 0
    motorcycle_count = 0

    cv.polylines(frame, [detection_area], isClosed=True, color=(0, 255, 255), thickness=2)

    # Draw detections on frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Get the center point of the detected vehicle
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if the vehicle is inside the detection area
            if cv.pointPolygonTest(detection_area, (center_x, center_y), False) >= 0:
                if class_id == 2:  # Car
                    car_count += 1
                    label = f'Car {confidence:.2f}'
                    color = (0, 255, 0)  # Green

                # elif class_id == 7:  # Truck
                #     truck_count += 1
                #     label = f'Truck {confidence:.2f}'
                #     color = (255, 0, 0)  # Blue

                # elif class_id == 3:  # Motorcycle
                #     motorcycle_count += 1
                #     label = f'Motorcycle {confidence:.2f}'
                #     color = (0, 165, 255)  # Orange

                else:
                    continue  # Skip other objects

                # Draw bounding box and label
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Display total count of each vehicle type inside the area
    cv.putText(frame, f'Cars: {car_count}  Trucks: {truck_count}  Motorcycles: {motorcycle_count}', 
               (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save processed frame
    out.write(frame)

    # Display (optional)
    cv.imshow("YOLO Car Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv.destroyAllWindows()