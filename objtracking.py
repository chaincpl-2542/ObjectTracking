import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model1 = YOLO("yolo11n.pt")
#model2 = YOLO("yolo11n-seg.pt")

video_path = "D:/GitHub/ObjectTracking/vdo1.mp4"
cap = cv.VideoCapture(video_path)

vdo_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)/3)
vdo_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)/3)

while True:
    ret, frame = cap.read()
    
   
    annotator = Annotator(frame, line_width = 5)
    
    if not ret:
        print("Video ended")
        break
    
    
    result = model1.track(frame, persist=True)
    
    if result[0].boxes.id is not None and result[0].masks is not None:
        masks = result[0].masks.xy
        ids = result[0].boxes.id.tolist()
        
        for mask, id in zip(masks, ids):
            annotator.seg_bbox(mask = mask,mask_color=colors(id, True), label=str(id))
    
    
    # for result in results:
    #     for box in result.boxes:
    #        coords = box.xyxy[0]
    #        x1, y1, x2, y2 = [int(coord) for coord in coords]
    #        cv.rectangle(resized_frame, (x1,y1),(x2,y2), (0, 255, 0), 2)
    
    resized_frame = cv.resize(frame,(vdo_width, vdo_height))
    
    cv.imshow('Video', resized_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()