import cv2
import numpy as np
import os
import time
import platform
from pathlib import Path
from tabulate import tabulate
import psutil

# --- 1. AI ENGINE ---
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
TEST_IMG_DIR = BASE_DIR / "blind_test" 
MODEL_DIR = BASE_DIR / "Model_file"

# 🔥 THE TRUTH: This MUST match your GESTURE_LABELS from Colab exactly 🔥
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

def run_evaluation(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists(): return [model_name.upper(), "MISSING", "-", "-", "-"]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []
    
    # Filter for files that contain "UNSEEN"
    files = [f for f in os.listdir(TEST_IMG_DIR) if "UNSEEN" in f]
    if not files: return [model_name.upper(), "NO_DATA", "-", "-", "-"]

    for img_name in files:
        # Get ground truth from filename: "Hello_UNSEEN_1.jpg" -> "hello"
        actual_label = img_name.split('_')[0].lower()
        
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Cleanup JPEG noise and normalize
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_input = binary.astype('float32') / 255.0

        if model_name == "mobilenet":
            tensor = img_input.reshape(1, 96, 96, 1)
        else:
            tensor = np.repeat(img_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        start_t = time.time()
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start_t) * 1000)

        # PREDICTION LOGIC
        pred_idx = np.argmax(output_data)
        predicted_label = CLASSES[pred_idx].lower()
        
        # Scoring
        if predicted_label == actual_label:
            correct += 1
        total += 1
    
    acc = (correct / total) * 100 if total > 0 else 0
    return [model_name.upper(), f"{acc:.1f}%", f"{np.mean(latencies):.2f}ms"]

# --- 3. EXECUTION ---
print("\n" + "="*70)
print(f"🔬 ARCHITECTURAL GENERALIZATION STUDY")
print(f"🏷️  CLASSES: {CLASSES}")
print("="*70)

report_data = []
for m in ["mobilenet", "lstm", "gru"]:
    print(f"⏳ Benchmarking {m.upper()}...")
    stats = run_evaluation(m)
    stats.extend([f"{psutil.cpu_percent()}%", f"{os.popen('vcgencmd measure_temp').readline().replace('temp=','').strip()}"])
    report_data.append(stats)

headers = ["Algorithm", "Accuracy (Blind)", "Latency", "CPU Load", "Temp"]
print("\n" + tabulate(report_data, headers=headers, tablefmt="grid"))
