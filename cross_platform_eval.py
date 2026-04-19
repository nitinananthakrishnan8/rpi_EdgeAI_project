import cv2
import numpy as np
import os
import time
import platform
from pathlib import Path
from tabulate import tabulate
import psutil

# --- 1. UNIVERSAL AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
    ENGINE_NAME = "LiteRT (Native ARM)"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        ENGINE_NAME = "TFLite-Runtime (ARM)"
    except ImportError:
        print("🚨 AI Engine Missing! Please ensure LiteRT is installed.")
        exit()

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
TEST_IMG_DIR = BASE_DIR / "blind_test" 
MODEL_DIR = BASE_DIR / "Model_file"

# THE OFFICIAL MAP (Alphabetical order from Keras/Colab)
CLASSES = ['Background', 'Call', 'Hello', 'L', 'Peace', 'Pinch', 'Pointing', 'Raised', 'Thumbsup', 'Yes']

# --- 3. HARDWARE TELEMETRY ---
def get_cpu_temp():
    try:
        return os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
    except:
        return "N/A"

# --- 4. THE EVALUATION ENGINE ---
def run_evaluation(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists():
        return [model_name.upper(), "MISSING", "-", "-", "-", "-"]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # --- FIXED: Correctly assign input and output details ---
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []
    
    # Only test the "UNSEEN" images for the generalization report
    files = [f for f in os.listdir(TEST_IMG_DIR) if "UNSEEN" in f]
    if not files:
        return [model_name.upper(), "NO_UNSEEN_DATA", "-", "-", "-", "-"]

    for img_name in files:
        # Extract ground truth from filename: "Hello_UNSEEN_1.jpg" -> "hello"
        actual_label = img_name.split('_')[0].lower()
        
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # 1. Image Normalization (Mirroring the training conditions)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_input = binary.astype('float32') / 255.0

        # 2. Tensor Formatting
        if model_name == "mobilenet":
            tensor = img_input.reshape(1, 96, 96, 1)
        else:
            # RNN models expect (Batch, Time, H, W, C)
            tensor = np.repeat(img_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        # 3. Inference with Timing
        start_t = time.time()
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start_t) * 1000)

        # 4. Accuracy Logic (Case-Insensitive)
        pred_idx = np.argmax(output_data)
        predicted_label = CLASSES[pred_idx].lower()
        
        if predicted_label == actual_label:
            correct += 1
        total += 1
    
    # 5. Result Synthesis
    acc = (correct / total) * 100 if total > 0 else 0
    avg_lat = np.mean(latencies)
    
    return [model_name.upper(), f"{acc:.1f}%", f"{avg_lat:.2f}ms"]

# --- 5. EXECUTION ---
print("\n" + "="*70)
print(f"🔬 ARCHITECTURAL GENERALIZATION STUDY | PLATFORM: {platform.machine()}")
print(f"📡 AI ENGINE: {ENGINE_NAME}")
print("="*70)

report_data = []
for m in ["mobilenet", "lstm", "gru"]:
    print(f"⏳ Benchmarking {m.upper()} on Unseen Dataset...")
    stats = run_evaluation(m)
    
    # Add real-time hardware telemetry
    stats.extend([f"{psutil.cpu_percent()}%", f"{psutil.virtual_memory().percent}%", f"{get_cpu_temp()}C"])
    report_data.append(stats)

# Display the professional results table
headers = ["Algorithm", "Accuracy (Blind)", "Latency", "CPU Load", "RAM Used", "Temp"]
print("\n" + tabulate(report_data, headers=headers, tablefmt="grid"))

# Write to report file for your publication
with open("final_architectural_report.txt", "w") as f:
    f.write(f"=== SYSTEM PERFORMANCE REPORT | {time.ctime()} ===\n")
    f.write(tabulate(report_data, headers=headers, tablefmt="grid"))

print(f"\n✅ REPORT GENERATED: final_architectural_report.txt")
