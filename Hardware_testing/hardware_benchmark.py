import cv2
import numpy as np
import time
import psutil
import os
from pathlib import Path
from tabulate import tabulate # For a beautiful table output

# --- 1. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = BASE_DIR / "Model_file"
TEST_IMG_DIR = BASE_DIR / "pi_test_set"
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
MODELS = ["mobilenet", "lstm", "gru"]

# Import AI Engine
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- 2. HARDWARE MONITOR FUNCTION ---
def get_sys_stats():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
    return cpu, ram, temp

# --- 3. BENCHMARKING ENGINE ---
def run_benchmark(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0
    latencies = []
    
    # Load and process all images in test set
    for img_name in os.listdir(TEST_IMG_DIR):
        if not img_name.endswith(('.jpg', '.png')): continue
        
        # Ground Truth Label
        actual_label = img_name.split('_')[0]
        
        # Load and Normalize
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        img_input = img.astype('float32') / 255.0
        
        # Prep Tensor
        if model_name == "mobilenet":
            input_tensor = img_input.reshape(1, 96, 96, 1)
        else:
            input_tensor = np.repeat(img_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        # Inference with Timing
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start) * 1000)

        # Accuracy Check
        pred_label = CLASSES[np.argmax(output)]
        if pred_label == actual_label:
            correct += 1
        total += 1

    avg_lat = np.mean(latencies)
    acc = (correct / total) * 100
    cpu, ram, temp = get_sys_stats()
    
    return [model_name.upper(), f"{acc:.1f}%", f"{avg_lat:.1f}ms", f"{cpu}%", f"{ram}%", f"{temp}C"]

# --- 4. EXECUTION ---
print("\n?? STARTING MULTI-ALGORITHM HARDWARE STRESS TEST...")
report_data = []

for m in MODELS:
    print(f"Testing {m}...")
    report_data.append(run_benchmark(m))

# --- 5. FINAL REPORT ---
headers = ["ALGORITHM", "ACCURACY", "AVG LATENCY", "CPU LOAD", "RAM USAGE", "TEMP"]
final_table = tabulate(report_data, headers=headers, tablefmt="grid")

print("\n" + final_table)

# Save to Notepad file
with open("hardware_comparison_report.txt", "w") as f:
    f.write("=== SYSTEM PERFORMANCE COMPARISON ===\n")
    f.write(f"Timestamp: {time.ctime()}\n")
    f.write(final_table)
print("\n?? Report saved to hardware_comparison_report.txt")