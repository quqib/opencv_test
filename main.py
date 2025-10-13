import cv2

model_path = 'face_detection_yunet_2023mar_int8.onnx'
img = cv2.imread('people.jpg')

h, w, _ = img.shape
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config='',
    input_size=(w, h),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# 检测人脸
_, faces = detector.detect(img)

if faces is not None:
    for i, face in enumerate(faces):
        # 人脸矩形框坐标
        x, y, w, h = map(int, face[:4])
        print(f"第 {i+1} 张人脸坐标：x={x}, y={y}, w={w}, h={h}")

        # 绘制人脸框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 关键点
        landmarks = face[4:14].reshape((5, 2)).astype(int)
        for (lx, ly) in landmarks:
            cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)

cv2.imshow('YuNet Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
