# Hand Gesture Chatbot Setup and Run Guide

## 1. Activate your virtual environment
Open PowerShell and run:

```powershell
cd "C:\Users\swaya\Desktop\python trainning (HAND GESTURE AI CHATBOT)"
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run once as administrator:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

## 2. Install dependencies
Inside the activated venv, run:

```powershell
pip install -r requirements.txt
```

## 3. Check camera backend and index
If you see `videoio(MSMF): can't grab frame`, the webcam backend or index is wrong.

### Change camera index
Open `config/settings.py` and set:

```python
CAMERA_INDEX = 0
```

If your webcam is on another index, try:

```python
CAMERA_INDEX = 1
```

If your camera still does not show a live feed, also try:

```python
CAMERA_INDEX = 2
```

### Use Windows-compatible backend
The project now tries several backends automatically: `CAP_DSHOW`, `CAP_MSMF`, and `CAP_ANY`.

## 4. Run the project
### Check application status
```powershell
python app.py --mode status
```

### Run the GUI app
```powershell
python app.py --mode gui
```

### Collect gesture samples
```powershell
python app.py --mode collect
```

When collection starts:
- A camera window opens with live video.
- Click the camera window once to give it focus.
- Press `SPACE` while your hand is visible in the frame.
- If you do not see `HAND DETECTED`, move your hand closer, use good lighting, and place it against a plain background.
- Press `q` in the camera window to quit, or `r` to reset the current session.

### Train the model
```powershell
python app.py --mode train
```

### Test components
```powershell
python app.py --mode test
```

## 5. What to do if camera still fails
1. Close other apps using the webcam (Zoom, Teams, browser).
2. Try `CAMERA_INDEX = 1` or `CAMERA_INDEX = 2` in `config/settings.py`.
3. Restart your PC and run again.
4. If your camera is external, connect it to a different USB port.

## 6. Notes for your environment
- You are using Python 3.14.
- `mediapipe` is not supported on Python 3.14, so the app uses OpenCV fallback mode.
- If you want full MediaPipe support, use Python 3.12 instead.

## 7. Optional: Set OpenAI API key
If you want GPT mode and have a key, set environment variable:

```powershell
setx OPENAI_API_KEY "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"
```

Then restart PowerShell.

In `config/settings.py`, set:

```python
CHATBOT_MODE = "gpt"
```

## 8. Example flow
1. Activate venv
2. `pip install -r requirements.txt`
3. `python app.py --mode status`
4. `python app.py --mode collect`
5. Press `SPACE` to capture samples, `q` to quit
6. `python app.py --mode train`
7. `python app.py --mode gui`

## 9. If you still see errors
Copy the exact error text and send it here so I can help fix it step-by-step.
