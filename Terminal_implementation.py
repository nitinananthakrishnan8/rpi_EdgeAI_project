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
        print("🚨 AI Engine Missing! Run: pip install ai-edge-litert")
        exit()

# --- 2. IMPORT CUSTOM COGNITION MODULES ---
# Ensure nlp_engine.py and services.py are in the same folder
try:
    import services
    from nlp_engine import NLPEngine
    nlp = NLPEngine()
except ImportError as e:
    print(f"🚨 Dependency Error: {e}")
    exit()

# --- 3. CONFIGURATION (The 60-Mark Project Blueprint) ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 

# 🔥 THE TRUTH MAP: MUST MATCH YOUR COLAB TRAINING ORDER EXACTLY 🔥
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

# CALCULATED PARAMETERS
THRESHOLD_T = 138.92   # Derived from 3-Sigma Analysis
STABILITY_REQ = 10     # Consecutive frames to trigger speech (~0.3s)
CONFIDENCE_GATE = 0.90 # High precision for demo stability
MIN_AREA = 3000        # Minimum pixel area to be considered a hand

# --- 4. PRE-PROCESSING PIPELINE (The "Cr-Mean Morphological Filter") ---
def process_frame(frame):
    """ Transforms raw color into 96x96 binary morphology """
    # Downsample immediately for 400% processing speed increase
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    
    # Otsu's Algorithm for calculated binarization
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        
        # Validation: Is it skin and is it hand-sized?
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        # --- THE NULL-SIGNAL GUARD ---
        # Only create a white shape if it passes the calculated filters
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > MIN_AREA:
            # Reconstruct hand morphology
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            # Tight Crop
            crop = final_canvas[y:y+h, x:x+w]
            # Resize for AI input and Normalize
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    # Return ZERO-TENSOR if no valid signal detected (Forces Background class)
    return np.zeros((96, 96), dtype='float32')

# --- 5. INITIALIZE THE NEURAL MODELS ---
# Choose model: "mobilenet", "lstm", or "gru"
ACTIVE_MODEL = "mobilenet" 
model_path = BASE_DIR / MODEL_FOLDER / f"sign_{ACTIVE_MODEL}.tflite"

print(f"\n🔄 Loading Algorithm: {ACTIVE_MODEL.upper()}...")
try:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ System Perception Layer Ready.")
except Exception as e:
    print(f"❌ Critical Load Error: {e}")
    exit()

# --- 6. MAIN INFERENCE LOOP ---
cap = cv2.VideoCapture(0)
# Set low hardware resolution to save memory bus bandwidth
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"\n🚀 EDGE SYSTEM ONLINE | Engine: {ENGINE_NAME}")
print(f"📡 Status: 100% Autonomous | 🔊 Voice: Connected")
print("--------------------------------------------------------------")

stable_label = ""
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start_t = time.time()
        
        # A. PERCEPTION PHASE
        ai_input = process_frame(frame)
        
        # B. NULL-SIGNAL INTERLOCK
        # If the pre-processor output is empty, bypass AI to prevent ghosting
        if np.max(ai_input) == 0:
            label = "Background"
            conf = 1.0
        else:
            # C. NEURAL INFERENCE PHASE
            input_tensor = ai_input.reshape(1, 96, 96, 1)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            
            idx = np.argmax(output)
            conf = output[idx]
            label = CLASSES[idx]
        
        latency = (time.time() - start_t) * 1000
        
        # D. TELEMETRY OUTPUT (CLI)
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        # Overwrites line for clean dashboard feel
        print(f"[{ts}] AI: {label:12} ({conf*100:5.1f}%) | Latency: {latency:4.1f}ms | CPU: {psutil.cpu_percent()}%", end='\r')

        # E. COGNITION & SPEECH TRIGGER
        if label != "Background" and conf > CONFIDENCE_GATE:
            if label == stable_label:
                frame_counter += 1
            else:
                stable_label, frame_counter = label, 0
                
            if frame_counter == STABILITY_REQ:
                print(f"\n[!] EVENT: Confirmed '{label}'. Speaking...")
                # Triggers services.py and asynchronous audio
                sentence = nlp.process_and_speak(label)
                # Brief cooldown to prevent repetition
                frame_counter = -50 
        elif label == "Background":
            # Reset contextual memory when hand is removed
            nlp.previous_sign = None
            stable_label = ""
            frame_counter = 0

except KeyboardInterrupt:
    print("\n\n👋 System shutting down...")
finally:
    cap.release()
