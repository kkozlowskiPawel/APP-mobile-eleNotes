import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from processing import detect_board_corners, process_whiteboard

app = FastAPI()


@app.post("/detect")
async def detect_corners(file: UploadFile = File(...)):
    """Auto-detect whiteboard corners. Returns 4 corner points as percentages (0-1)."""
    image_bytes = await file.read()
    corners = detect_board_corners(image_bytes)
    return JSONResponse(content={"corners": corners})


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    corners: str = Form(""),
    sensitivity: int = Form(80),
    steepness: float = Form(0.30),
    blur_sigma: int = Form(51),
    denoise: bool = Form(True),
    preserve_color: bool = Form(False),
):
    image_bytes = await file.read()
    corner_points = json.loads(corners) if corners else None
    png_bytes = process_whiteboard(
        image_bytes,
        corners=corner_points,
        sensitivity=sensitivity,
        steepness=steepness,
        blur_sigma=blur_sigma,
        denoise=denoise,
        preserve_color=preserve_color,
    )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="elenotes_result.png"'},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
