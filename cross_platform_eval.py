import cv2
import numpy as np
import os
import time
import platform
from pathlib import Path
from tabulate import tabulate
import psutil

# --- 1. AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
TEST_IMG_DIR = BASE_DIR / "blind_test" 
MODEL_DIR = BASE_DIR / "Model_file"

# THE OFFICIAL MAP (Must match your Training order)
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

def run_evaluation(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    correct, total = 0, 0
    latencies = []
    
    files = [f for f in os.listdir(TEST_IMG_DIR) if "UNSEEN" in f]
    
    for img_name in files:
        # Extract ground truth: "Hello_UNSEEN_1.jpg" -> "hello"
        actual_label = img_name.split('_')[0].lower()
        
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # JPEG cleanup and normalization
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_input = binary.astype('float32') / 255.0

        if model_name == "mobilenet":
            tensor = img_input.reshape(1, 96, 96, 1)
        else:
            tensor = np.repeat(img_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        start_t = time.time()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start_t) * 1000)

        # THE FIX: Case-insensitive comparison
        pred_label = CLASSES[np.argmax(output)].lower()
        
        if pred_label == actual_label:
            correct += 1
        total += 1
    
    acc = (correct / total) * 100 if total > 0 else 0
    return [model_name.upper(), f"{acc:.1f}%", f"{np.mean(latencies):.2f}ms"]

# --- 3. EXECUTION ---
print("\n" + "="*60)
print("🚀 FINAL CROSS-PLATFORM ARCHITECTURAL STUDY")
print("="*60)

report_data = []
for m in ["mobilenet", "lstm", "gru"]:
    stats = run_evaluation(m)
    # Add hardware metrics
    stats.extend([f"{psutil.cpu_percent()}%", f"{psutil.virtual_memory().percent}%"])
    report_data.append(stats)

headers = ["Algorithm", "Accuracy (Unseen)", "Latency", "CPU", "RAM"]
print("\n" + tabulate(report_data, headers=headers, tablefmt="grid"))
