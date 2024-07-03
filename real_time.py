import cv2
from ultralytics import YOLO

model_path: str = "best_model.pt"
model = YOLO(model_path)

color_map={"cardboard":(255, 0, 0), "work_paper": (0, 0, 255), "graph_paper": (0, 255, 0)}

def detect_and_show_real_time(model = model, class_names: dict[int, str] = model.names):
    cap = cv2.VideoCapture(0)  
        
    if not cap.isOpened():
        print("Error: no camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: end of stream")
            break

        results = model(frame)

        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            class_name = class_names[class_id]
            color = color_map[class_name]
            label = f'{class_name} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Real Time detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_and_show_real_time()