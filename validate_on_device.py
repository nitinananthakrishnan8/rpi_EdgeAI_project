import cv2
import numpy as np
import time
import os
from pathlib import Path
from tabulate import tabulate

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

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "Model_file" / "sign_mobilenet.tflite"
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92 

# --- 3. PRE-PROCESSING (Cr-Mean Filter) ---
def process_frame(frame):
    small = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask); cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2500:
            cv2.drawContours(final_canvas, [c], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

# --- 4. INITIALIZE ---
interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
test_results = []

print("\n" + "="*55)
print("     EDGE AI SYSTEM: LIVE HARDWARE VALIDATION")
print("="*55)
print(f"Engine: {ENGINE_NAME} | Target: 10 Gestures")

# --- 5. TEST LOOP ---
try:
    for target in CLASSES:
        if target == "Background": continue
        
        # UI PHASE: Waiting for User
        while True:
            ret, frame = cap.read()
            # Draw Instructions on Screen
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (630, 80), (0,0,0), -1)
            cv2.putText(display_frame, f"NEXT TEST: [{target.upper()}]", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, "PRESS [ENTER] IN TERMINAL TO START", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Validation Hub", display_frame)
            
            # Non-blocking check for Terminal Input
            print(f"\r👉 Prepare gesture for {target.upper()} and press [ENTER]...", end="")
            
            # This bit is tricky in pure Python; we will use a small wait
            if cv2.waitKey(1) == 13: # 13 is the Enter key in the CV2 window
                break
            # Alternative: Use standard input but the window won't update
            # To fix this, we use a timeout-based input or just Enter in CV2 window
            break # Let's use the standard Enter key in terminal for simplicity

        input(f"\n--- Starting {target.upper()} Test. Press ENTER to begin countdown ---")

        # COUNTDOWN PHASE
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, str(i), (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
            cv2.imshow("Validation Hub", frame)
            cv2.waitKey(1000) # Wait 1 second

        # CAPTURE PHASE
        ret, frame = cap.read()
        print(f"📸 Capturing {target}...")
        
        # Inference
        ai_input = process_frame(frame)
        input_tensor = ai_input.reshape(1, 96, 96, 1)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Results
        idx = np.argmax(output)
        pred_label = CLASSES[idx]
        conf = output[idx] * 100
        
        status = "✅ PASS" if pred_label.lower() == target.lower() else "❌ FAIL"
        test_results.append([target, pred_label, f"{conf:.1f}%", status])
        
        # Visual Confirmation
        color = (0, 255, 0) if status == "✅ PASS" else (0, 0, 255)
        cv2.putText(frame, f"PREDICTED: {pred_label}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("Validation Hub", frame)
        cv2.waitKey(2000) # Show result for 2 seconds

    # --- 6. FINAL NUMERICAL REPORT ---
    print("\n\n" + "="*60)
    print("           FINAL HARDWARE ACCURACY REPORT")
    print("="*60)
    headers = ["TARGET SIGN", "AI PREDICTION", "CONFIDENCE", "STATUS"]
    print(tabulate(test_results, headers=headers, tablefmt="grid"))
    
    total_pass = sum(1 for r in test_results if r[3] == "✅ PASS")
    accuracy = (total_pass / len(test_results)) * 100
    print(f"\n🎯 ON-DEVICE RELIABILITY SCORE: {accuracy:.1f}%")
    print("="*60)

except KeyboardInterrupt:
    print("\nValidation manually stopped.")
finally:
    cap.release()
    cv2.destroyAllWindows()
