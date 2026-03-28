import cv2
import numpy as np
import time
import os
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

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
MODEL_PATH = BASE_DIR / MODEL_FOLDER / "sign_mobilenet.tflite"

# Ensure these match your training labels exactly
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92
SAMPLES_PER_CLASS = 50 

# --- 3. INITIALIZE AI ---
print(f"?? Loading Engine: {ENGINE_NAME}")
try:
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"? Model Loaded: {MODEL_PATH.name}")
except Exception as e:
    print(f"? Error loading model: {e}")
    exit()

def process_frame(frame):
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    _, mask = cv2.threshold(cr, THRESHOLD_T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1500:
            canvas = np.zeros_like(mask)
            cv2.drawContours(canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
    return np.zeros((96, 96), dtype='float32')

# --- 4. VALIDATION PROTOCOL ---
cap = cv2.VideoCapture(0)
results = {}

print("\n" + "="*50)
print("   EDGE AI ON-DEVICE VALIDATION PROTOCOL")
print("="*50)

for target_label in CLASSES:
    if target_label == "Background": continue
    
    print(f"\n?? NEXT TEST: [{target_label.upper()}]")
    input(f"   Prepare sign and press ENTER to start...")
    
    print("   Starting in 3...", end=" ", flush=True)
    time.sleep(1)
    print("2...", end=" ", flush=True)
    time.sleep(1)
    print("1...", end=" ", flush=True)
    time.sleep(1)
    print("?? CAPTURING")
    
    correct_count = 0
    for i in range(SAMPLES_PER_CLASS):
        ret, frame = cap.read()
        if not ret: break
        
        ai_input = process_frame(frame)
        input_tensor = ai_input.reshape(1, 96, 96, 1)
        
        # --- FIXED LINE: NO TRUNCATION ---
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Get Predicted Class
        idx = np.argmax(output)
        if CLASSES[idx] == target_label:
            correct_count += 1
            
        # Visual display
        cv2.putText(frame, f"TARGET: {target_label} ({i}/{SAMPLES_PER_CLASS})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Validation Pipeline", frame)
        cv2.waitKey(1)

    acc = (correct_count / SAMPLES_PER_CLASS) * 100
    results[target_label] = acc
    print(f"?? Accuracy for {target_label}: {acc:.1f}%")
    time.sleep(1)

# --- 5. FINAL REPORT ---
print("\n" + "="*50)
print("       HARDWARE VALIDATION FINAL REPORT")
print("="*50)
print(f"{'GESTURE':<15} | {'ACCURACY':<10}")
print("-" * 50)
for label, acc in results.items():
    print(f"{label:<15} | {acc:>8.1f}%")
print("="*50)

cap.release()
cv2.destroyAllWindows()