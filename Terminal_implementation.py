import cv2
import numpy as np
import time
import os
import datetime
from pathlib import Path

# --- 1. UNIVERSAL AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
    ENGINE_NAME = "LiteRT (Native 3.13)"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        ENGINE_NAME = "TFLite-Runtime"
    except ImportError:
        print("?? AI Engine Missing! Run: pip install ai-edge-litert")
        exit()

import services
from nlp_engine import NLPEngine
nlp = NLPEngine()

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92
STABILITY_REQ = 8 

# --- 3. PRE-PROCESSING ---
def process_frame(frame):
    # Downsample for speed
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    
    _, mask = cv2.threshold(cr, THRESHOLD_T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    canvas = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1500:
            cv2.drawContours(canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = canvas[y:y+h, x:x+w]
            # Returns the 96x96 image and the mask for display
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0, mask
            
    return np.zeros((96, 96), dtype='float32'), mask

# --- 4. INITIALIZE AI ---
ACTIVE_MODEL = "mobilenet" 
model_path = BASE_DIR / MODEL_FOLDER / f"sign_{ACTIVE_MODEL}.tflite"

interpreter = tflite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 5. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\n?? SYSTEM ONLINE | Engine: {ENGINE_NAME}")
print(f"?? Model: {ACTIVE_MODEL.upper()} | ?? Voice: ACTIVE")
print("Press 'q' in the video window to exit.")
print("--------------------------------------------------------------")

stable_label = ""
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start_t = time.time()
        
        # Perception
        ai_input, debug_mask = process_frame(frame)
        
        # Inference
        input_tensor = ai_input.reshape(1, 96, 96, 1)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Logic
        idx = np.argmax(output)
        conf = output[idx]
        label = CLASSES[idx]
        
        latency = (time.time() - start_t) * 1000
        
        # --- TERMINAL LOGGING ---
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {label:10} ({conf*100:3.0f}%) | Latency: {latency:4.1f}ms", end='\r')

        # --- SPEECH TRIGGER ---
        if conf > 0.85:
            if label == stable_label:
                frame_counter += 1
            else:
                stable_label, frame_counter = label, 0
                
            if frame_counter == STABILITY_REQ:
                if label != "Background":
                    print(f"\n[!] SPEECH: {label}")
                    nlp.process_and_speak(label)
                    frame_counter = -20 

        # --- VISUAL DISPLAY (The Flashy part) ---
        # 1. Overlay the prediction on the raw frame
        cv2.putText(frame, f"AI: {label} ({conf*100:.0f}%)", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 2. Show Windows
        cv2.imshow("Vision Hub (Raw)", frame)
        cv2.imshow("AI Perception (Binary Mask)", debug_mask)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()

