import cv2
import numpy as np
import time
import psutil
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
        print("🚨 AI Engine Missing!")
        exit()

# --- 2. IMPORT CUSTOM COGNITION MODULES ---
import services
from nlp_engine import NLPEngine
nlp = NLPEngine()

# --- 3. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

# CALCULATED PARAMETERS
THRESHOLD_T = 138.92
STABILITY_REQ = 10 
CONFIDENCE_GATE = 0.90
MIN_AREA = 3000

# --- 4. PRE-PROCESSING PIPELINE ---
def process_frame(frame):
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > MIN_AREA:
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            # Return both the AI input and the mask for display
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0, mask_temp
            
    return np.zeros((96, 96), dtype='float32'), final_canvas

# --- 5. INITIALIZE BRAIN ---
ACTIVE_MODEL = "mobilenet" 
model_path = BASE_DIR / MODEL_FOLDER / f"sign_{ACTIVE_MODEL}.tflite"

try:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"❌ Load Error: {e}"); exit()

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\n🚀 EDGE SYSTEM ONLINE | Press 'q' in video window to exit.")
print("--------------------------------------------------------------")

stable_label, frame_counter = "", 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        start_t = time.time()
        
        # A. PERCEPTION
        ai_input, debug_mask = process_frame(frame)
        
        # B. INFERENCE (With Null-Signal Interlock)
        if np.max(ai_input) == 0:
            label, conf = "Background", 1.0
        else:
            input_tensor = ai_input.reshape(1, 96, 96, 1)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            idx = np.argmax(output)
            label, conf = CLASSES[idx], output[idx]
        
        latency = (time.time() - start_t) * 1000
        
        # C. UI OVERLAY (The 'Flashy' part for the screen)
        # Draw a box for the text background
        cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"AI: {label}", (10, 30), 2, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"CONF: {conf*100:.1f}% | {latency:.1f}ms", (10, 60), 2, 0.6, (255, 255, 255), 1)

        # D. SHOW WINDOWS
        cv2.imshow("Vision Hub (640x480 Raw)", frame)
        cv2.imshow("AI Perception (96x96 Binary)", debug_mask)

        # E. COGNITION & SPEECH
        if label != "Background" and conf > CONFIDENCE_GATE:
            if label == stable_label: frame_counter += 1
            else: stable_label, frame_counter = label, 0
            
            if frame_counter == STABILITY_REQ:
                print(f"\n[!] EVENT: {label}")
                nlp.process_and_speak(label)
                frame_counter = -40 
        elif label == "Background":
            nlp.previous_sign = None
            stable_label, frame_counter = "", 0

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap.release()
    cv2.destroyAllWindows()
