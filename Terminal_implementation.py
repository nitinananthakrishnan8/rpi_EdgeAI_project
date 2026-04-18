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
        print("🚨 AI Engine Missing!"); exit()

# --- 2. MODULES & CONFIG ---
import services
from nlp_engine import NLPEngine
nlp = NLPEngine()

BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

# --- DYNAMIC PARAMETERS (These will be updated by pressing 's') ---
THRESHOLD_T = 138.92 # Default
MIN_AREA = 2500      # Default
STABILITY_REQ = 8 
CONFIDENCE_GATE = 0.90

# --- 3. THE CALIBRATION ALGORITHM (The 60-Mark Logic) ---
def perform_calibration(cap):
    print("\n" + "="*50)
    print("🛠️  STARTING SYSTEM SELF-CALIBRATION...")
    print("="*50)
    print("STEP 1: Move hand OUT of frame. Calculating Noise Floor...")
    time.sleep(2)
    
    cr_samples = []
    for _ in range(30): # Sample 30 frames of the empty wall
        ret, frame = cap.read()
        if not ret: continue
        ycrcb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2YCrCb)
        cr_samples.append(ycrcb[:,:,1])
    
    avg_cr = np.mean(cr_samples)
    std_cr = np.std(cr_samples)
    
    # CALCULATE 3-SIGMA THRESHOLD
    new_threshold = avg_cr + (3 * std_cr)
    
    print(f"STEP 2: Show an OPEN PALM to the camera. Calculating Area Gate...")
    time.sleep(2)
    
    areas = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: continue
        ycrcb = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:,:,1]
        _, mask = cv2.threshold(cr, new_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            areas.append(cv2.contourArea(max(contours, key=cv2.contourArea)))
    
    # CALCULATE MIN AREA (Set at 50% of your hand size to allow for fists)
    new_min_area = np.mean(areas) * 0.5 if areas else 2500
    
    print("\n✅ CALIBRATION COMPLETE!")
    print(f"📈 New Threshold (3-Sigma): {new_threshold:.2f}")
    print(f"📐 New Min Area Gate: {int(new_min_area)} pixels")
    print("="*50 + "\n")
    
    return new_threshold, new_min_area

# --- 4. PRE-PROCESSING ---
def process_frame(frame, t_val, a_val):
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
        
        if mean_cr > t_val and cv2.contourArea(c) > a_val:
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0, mask_temp
            
    return np.zeros((96, 96), dtype='float32'), final_canvas

# --- 5. INITIALIZE BRAIN ---
ACTIVE_MODEL = "mobilenet" 
model_path = BASE_DIR / MODEL_FOLDER / f"sign_{ACTIVE_MODEL}.tflite"
interpreter = tflite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)
print(f"\n🚀 SYSTEM ONLINE | Press 's' to Calibrate | Press 'q' to Exit")

stable_label, frame_counter = "", 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        start_t = time.time()
        
        # PERCEPTION
        ai_input, debug_mask = process_frame(frame, THRESHOLD_T, MIN_AREA)
        
        # INFERENCE & LOGIC
        label, conf = "Background", 1.0
        if np.max(ai_input) > 0:
            input_tensor = ai_input.reshape(1, 96, 96, 1)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            idx = np.argmax(output)
            label, conf = CLASSES[idx], output[idx]
        
        latency = (time.time() - start_t) * 1000
        
        # UI DISPLAY
        cv2.putText(frame, f"AI: {label} ({conf*100:.0f}%)", (10, 30), 2, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"T:{int(THRESHOLD_T)} A:{int(MIN_AREA)}", (10, 60), 2, 0.5, (255, 255, 255), 1)
        cv2.imshow("Live Feed", frame)
        cv2.imshow("AI Mask", debug_mask)

        # COGNITION & SPEECH
        if label != "Background" and conf > CONFIDENCE_GATE:
            if label == stable_label: frame_counter += 1
            else: stable_label, frame_counter = label, 0
            if frame_counter == STABILITY_REQ:
                nlp.process_and_speak(label)
                frame_counter = -40 
        elif label == "Background":
            nlp.previous_sign = None
            stable_label, frame_counter = "", 0

        # --- INTERACTIVE CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('s'):
            # TRIGGER LIVE CALIBRATION
            THRESHOLD_T, MIN_AREA = perform_calibration(cap)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap.release()
    cv2.destroyAllWindows()
