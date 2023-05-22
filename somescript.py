import cv2
import json
from tempfile import NamedTemporaryFile
from deepface import DeepFace
import random

def analyze_from_video(source: str, save: bool = True):
    capture = cv2.VideoCapture(source)
    frames = []
    while 1:
        _, image = capture.read()
        try:
            if image is not None:
                frames.append(image)
            else:
                break
        except (AttributeError, TypeError):
            print("err")
            continue

    capture.release()

    cv2.destroyAllWindows()
    frames = random.sample(frames, 20)
    gender = ""
    age = 0
    age_when_woman = 0
    age_when_man = 0
    total_dominant_woman = 0
    total_dominant_man = 0
    for frame in frames:
        with NamedTemporaryFile("wb", suffix=".jpg") as file:
            cv2.imwrite(file.name, frame)
            file.seek(0)
            results = DeepFace.analyze(file.name, actions=("age", "gender"))
            for result in results:
                if result['gender']['Woman'] > result['gender']['Man']:
                    age_when_woman += result['age']
                    total_dominant_woman += 1
                else:
                    age_when_man += result['age']
                    total_dominant_man += 1

    if total_dominant_man > total_dominant_woman:
        age = age_when_man / total_dominant_man
        gender = "Man"
    else:
        age = age_when_man / total_dominant_man
        gender = "Woman"
    if save:
        with open(source + ".json", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "age": age,
                "gender": gender
            }))
