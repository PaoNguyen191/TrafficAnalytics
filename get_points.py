import cv2

# Đọc video
cap = cv2.VideoCapture('videos/traffic1.mp4') # Đổi lại đường dẫn nếu cần
ret, frame = cap.read()

# Tính tỷ lệ thu nhỏ để vừa màn hình (ví dụ thu nhỏ 3 lần)
scale = 0.3 
small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

print("Hãy click 6 điểm: 4 điểm đo tốc độ (đa giác) và 2 điểm đếm xe (đường thẳng)")
print("Nhấn phím 'q' để thoát.")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Nhân ngược lại với tỷ lệ để ra tọa độ trên video gốc
        real_x = int(x / scale)
        real_y = int(y / scale)
        print(f"[{real_x}, {real_y}],")
        
        # Vẽ chấm đỏ để đánh dấu
        cv2.circle(small_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Calibration', small_frame)

cv2.imshow('Calibration', small_frame)
cv2.setMouseCallback('Calibration', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()