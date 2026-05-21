from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import cv2
import os
import json

# Import trực tiếp máy phát từ main.py
from main import generate_frames 

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")

class PointsData(BaseModel):
    source_points: list[list[int]]
    counting_region: list[list[int]]
    video_path: str

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    filename = getattr(video, "filename", None) or "uploaded_video"
    safe_filename = os.path.basename(filename).replace(" ", "_")
    video_path = f"videos/{safe_filename}"
    
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_filename = "first_frame.jpg"
        cv2.imwrite(f"static/{frame_filename}", frame)
        return {
            "status": "success", 
            "frame_url": f"/static/{frame_filename}",
            "original_width": frame.shape[1],
            "original_height": frame.shape[0],
            "video_path": video_path
        }
    return {"status": "error", "message": "Giải mã tệp tin video tải lên không thành công!"}

@app.post("/save-points")
async def save_points(data: PointsData):
    with open("points_config.json", "w") as f:
        json.dump({
            "SOURCE_POINTS": data.source_points,
            "COUNTING_REGION": data.counting_region,
            "INPUT_VIDEO": data.video_path
        }, f, indent=4)
    return {"status": "success", "message": "Hệ thống lưu cấu hình thành công!"}

# --- API STREAM VIDEO REAL-TIME ---
@app.get("/video_feed")
def video_feed():
    # Gọi trực tiếp generator và trả về chuẩn multipart x-mixed-replace (Hiệu ứng video live)
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )