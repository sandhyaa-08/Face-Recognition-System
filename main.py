import cv2
import os
from deepface import DeepFace

# Folder path
db_path = "D:/Python AI/faces"

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    try:
        # DeepFace find (compare with database)
        result = DeepFace.find(
            img_path=frame,
            db_path=db_path,
            enforce_detection=False,
            model_name="Facenet"
        )

        if len(result) > 0 and len(result[0]) > 0:
            # Get best match
            best_match = result[0].iloc[0]

            identity_path = best_match["identity"]
            name = os.path.basename(identity_path).split(".")[0]

            # Face region
            x = int(best_match["source_x"])
            y = int(best_match["source_y"])
            w = int(best_match["source_w"])
            h = int(best_match["source_h"])

            # 🔴 Red box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Name display
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "Unknown", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("DeepFace Face Match", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
