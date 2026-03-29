import cv2
import numpy as np
import time
import psutil
import os
from pathlib import Path
from tabulate import tabulate

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
TEST_IMG_DIR = CURRENT_SCRIPT_DIR / "pi_test_set"
MODEL_DIR = CURRENT_SCRIPT_DIR.parent / "Model_file"

# The Alphabetical Order used during Training
CLASSES = ['Background', 'Call', 'Hello', 'L', 'Peace', 'Pinch', 'Pointing', 'Raised', 'Thumbsup', 'Yes']
MODELS = ["mobilenet", "lstm", "gru"]
THRESHOLD_T = 138.92

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- THE MISSING LINK: THE PRE-PROCESSOR ---
def process_frame(frame):
    """ Converts raw images into the 96x96 binary masks the AI was trained on """
    frame_resized = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    
    blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_canvas = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        # 3-Sigma Validation
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2500:
            hull = cv2.convexHull(c)
            cv2.drawContours(final_canvas, [hull], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[max(0,y-10):y+h+10, max(0,x-10):x+w+10]
            # Returns a 96x96 float32 array normalized to 0.0 - 1.0
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

def run_benchmark(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists():
        return [model_name.upper(), "FILE ERROR", "-", "-", "-", "-"]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []
    
    print(f"\n--- Benchmarking {model_name.upper()} ---")
    files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    for img_name in files:
        # e.g., "Call_3.jpg" -> "call"
        actual_label = img_name.split('_')[0].lower()
        
        # Load Raw Image
        raw_img = cv2.imread(str(TEST_IMG_DIR / img_name))
        if raw_img is None: continue
        
        # --- THE FIX: APPLY THE PRE-PROCESSOR ---
        # Transforms the raw color image into the 96x96 binary mask
        ai_input = process_frame(raw_img)

        # Prepare Tensor
        if model_name == "mobilenet":
            input_tensor = ai_input.reshape(1, 96, 96, 1)
        else:
            input_tensor = np.repeat(ai_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        # Inference
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start) * 1000)

        # Analysis
        idx = np.argmax(output)
        pred_label = CLASSES[idx].lower()
        
        if pred_label == actual_label:
            correct += 1
            
        total += 1

    acc = (correct / total) * 100 if total > 0 else 0
    avg_lat = np.mean(latencies) if latencies else 0
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").strip()
    
    return [model_name.upper(), f"{acc:.1f}%", f"{avg_lat:.1f}ms", f"{cpu}%", f"{ram}%", temp]

# --- EXECUTION ---
print("🚀 Launching Benchmark with Pre-processing Pipeline...")
results = [run_benchmark(m) for m in MODELS]

headers = ["ALGORITHM", "ACCURACY", "AVG LATENCY", "CPU LOAD", "RAM USAGE", "TEMP"]
print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
