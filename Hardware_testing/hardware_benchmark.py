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
# Match order exactly
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

def run_benchmark(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    try:
        import ai_edge_litert.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
    except: return [model_name.upper(), "ERROR", "-", "-", "-", "-"]

    correct, total = 0, 0
    latencies = []
    
    files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    for img_name in files:
        actual = img_name.split('_')[0].lower()
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        # Fix JPEG compression noise before inference
        _, binary = cv2.threshold(cv2.resize(img,(96,96)), 127, 255, cv2.THRESH_BINARY)
        img_in = binary.astype('float32') / 255.0

        input_tensor = img_in.reshape(1,96,96,1) if model_name=="mobilenet" else np.repeat(img_in[np.newaxis,:,:],5,axis=0).reshape(1,5,96,96,1)
        
        start = time.time()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
        latencies.append((time.time() - start) * 1000)

        if CLASSES[np.argmax(output)].lower() == actual: correct += 1
        total += 1

    return [model_name.upper(), f"{(correct/total)*100:.1f}%", f"{np.mean(latencies):.1f}ms", f"{psutil.cpu_percent()}%", f"{os.popen('vcgencmd measure_temp').readline().strip()}"]

# Execution
results = [run_benchmark(m) for m in ["mobilenet", "lstm", "gru"]]
print("\n" + tabulate(results, headers=["ALGORITHM", "ACCURACY", "LATENCY", "CPU", "TEMP"], tablefmt="grid"))
