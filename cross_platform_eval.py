import sys
import cv2
import numpy as np
import os
import time
import platform
from pathlib import Path
from tabulate import tabulate
import psutil

# --- 1. UNIVERSAL AI ENGINE IMPORT (For Raspberry Pi) ---
try:
    import ai_edge_litert.interpreter as tflite
    ENGINE_NAME = "LiteRT (Native ARM)"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        ENGINE_NAME = "TFLite-Runtime (ARM)"
    except ImportError:
        print("🚨 CRITICAL ERROR: AI Engine Missing! Run: pip install ai-edge-litert")
        sys.exit(1)

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
# We use ONLY the unseen blind test data for the paper
TEST_IMG_DIR = BASE_DIR / "blind_test" 
MODEL_DIR = BASE_DIR / "Model_file"
# Alphabetical order from Colab training
CLASSES = ['Background', 'Call', 'Hello', 'L', 'Peace', 'Pinch', 'Pointing', 'Raised', 'Thumbsup', 'Yes']

# --- 3. HARDWARE TELEMETRY ---
def get_cpu_temp():
    """ Raspberry Pi specific temperature sensing """
    try:
        return os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
    except:
        return "N/A"

# --- 4. THE EVALUATION ENGINE ---
def run_evaluation(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists():
        return [model_name.upper(), "MODEL MISSING", "-", "-", "-", "-"]
        
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []
    
    files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        return [model_name.upper(), "NO IMAGES", "-", "-", "-", "-"]

    for img_name in files:
        # Ground truth from filename (e.g., "Hello_UNSEEN_123.jpg" -> "hello")
        actual_label = img_name.split('_')[0].lower()
        
        # Load the ALREADY PROCESSED 96x96 binary mask
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Clean JPEG artifacts to ensure pure binary input
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_input = binary.astype('float32') / 255.0

        # Prepare Tensor based on architecture
        if model_name == "mobilenet":
            tensor = img_input.reshape(1, 96, 96, 1)
        else:
            tensor = np.repeat(img_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        # Inference Timing
        start_t = time.time()
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start_t) * 1000)

        # Accuracy Check
        if CLASSES[np.argmax(output)].lower() == actual_label:
            correct += 1
        total += 1
    
    # Calculate final stats
    acc = (correct / total) * 100 if total > 0 else 0
    avg_lat = np.mean(latencies) if latencies else 0
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    
    return [model_name.upper(), f"{acc:.1f}%", f"{avg_lat:.2f}ms", f"{cpu}%", f"{ram}%", f"{get_cpu_temp()}C"]

# --- 5. EXECUTION ---
print("================================================================")
print(f"🔬 RUNNING ARCHITECTURAL EVALUATION | PLATFORM: {platform.processor()}")
print("================================================================")

report_data = []
for m in ["mobilenet", "lstm", "gru"]:
    print(f"Benchmarking {m}...")
    report_data.append(run_evaluation(m))

# Print the final table
headers = ["Algorithm", "Accuracy (Unseen)", "Avg Latency", "CPU Load", "RAM Used", "Temp"]
final_table = tabulate(report_data, headers=headers, tablefmt="grid")
print("\n" + final_table)

# Save to a text file for your report
with open("pi_evaluation_report.txt", "w") as f:
    f.write(f"=== RASPBERRY PI 5 PERFORMANCE COMPARISON ===\n")
    f.write(f"Engine: {ENGINE_NAME}\n")
    f.write(final_table)
print("\n💾 Report saved to pi_evaluation_report.txt")
