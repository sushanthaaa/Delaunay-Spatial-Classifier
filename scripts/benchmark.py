import argparse
import pandas as pd
import time
import subprocess
import os
import shutil
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None
    df = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    return df[['x', 'y']].values, df['label'].values

# --- STATIC BENCHMARK ---
def run_static_benchmark(args, root):
    print(f"\n[{args.dataset.upper()}] STATIC BENCHMARK (Inference Speed)")
    
    train_csv = f"{root}/data/train/{args.dataset}_train.csv"
    test_X_csv = f"{root}/data/test/{args.dataset}_test_X.csv"
    test_y_csv = f"{root}/data/test/{args.dataset}_test_y.csv"
    cpp_exe = f"{root}/build/main"
    
    cpp_clean_out = f"{root}/results/clean_points.csv"
    cpp_preds_out = f"{root}/results/predictions.csv"
    final_clean_out = f"{root}/results/{args.dataset}_clean_points.csv"

    X_train, y_train = load_data(train_csv)
    X_test, y_true = load_data(test_y_csv)
    X_test_no_label, _ = load_data(test_X_csv)

    if X_train is None: return

    classifiers = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, algorithm='brute'),
        "SVM (RBF)": SVC(kernel='rbf'),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = []
    baseline_time = 0

    print("--- Python Baselines ---")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        start = time.perf_counter()
        y_pred = clf.predict(X_test_no_label)
        end = time.perf_counter()
        
        avg_time = ((end - start) / len(X_test)) * 1_000_000
        acc = accuracy_score(y_true, y_pred)
        if "KNN" in name: baseline_time = avg_time
        
        results.append({"Algorithm": name, "Accuracy": acc, "Time_us": avg_time})
        print(f"{name}: {avg_time:.2f} us")

    print("--- C++ Delaunay (Ours) ---")
    if os.path.exists(cpp_preds_out): os.remove(cpp_preds_out)
    if os.path.exists(cpp_clean_out): os.remove(cpp_clean_out)
    
    cmd = [cpp_exe, "static", train_csv, test_X_csv, f"{root}/results"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    cpp_time = 0.0
    for line in proc.stdout.split('\n'):
        if "Avg Time Per Point" in line:
            try: cpp_time = float(line.split()[-2])
            except: pass
            
    print(f"C++ Delaunay: {cpp_time:.4f} us")

    if os.path.exists(cpp_clean_out):
        shutil.move(cpp_clean_out, final_clean_out)

    if os.path.exists(cpp_preds_out):
        cpp_preds = pd.read_csv(cpp_preds_out, header=None).values.ravel()
        if len(cpp_preds) == len(y_true):
            acc = accuracy_score(y_true, cpp_preds)
            results.append({"Algorithm": "**Delaunay (Ours)**", "Accuracy": acc, "Time_us": cpp_time})

    print("\n" + "="*70)
    print(f"{'Algorithm':<25} | {'Accuracy':<10} | {'Inference (us)':<15} | {'Speedup'}")
    print("-" * 70)
    for r in results:
        speedup = baseline_time / r['Time_us'] if r['Time_us'] > 0 else 0
        print(f"{r['Algorithm']:<25} | {r['Accuracy']*100:.1f}%     | {r['Time_us']:<15.4f} | {speedup:.1f}x")
    print("="*70 + "\n")

# --- DYNAMIC BENCHMARK ---
def run_dynamic_benchmark(args, root):
    print(f"[{args.dataset.upper()}] DYNAMIC BENCHMARK (Update Cost: Insert/Delete)")
    
    base_csv = f"{root}/data/train/{args.dataset}_dynamic_base.csv"
    stream_csv = f"{root}/data/train/{args.dataset}_dynamic_stream.csv"
    
    cpp_log_filename = f"{args.dataset}_dynamic_logs.csv"
    cpp_log_dir = f"{root}/results/logs"
    cpp_log_path = f"{cpp_log_dir}/{cpp_log_filename}"
    
    os.makedirs(cpp_log_dir, exist_ok=True)

    X_base, y_base = load_data(base_csv)
    X_stream, y_stream = load_data(stream_csv)
    
    if X_base is None: return

    print("Running C++ Algorithm 1/2/3")
    cpp_exe = f"{root}/build/main"
    
    if os.path.exists(cpp_log_path): os.remove(cpp_log_path)
    
    cmd = [cpp_exe, "dynamic", base_csv, stream_csv, cpp_log_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print("C++ Error:", proc.stderr)
        return

    print("Measuring Competitor Retraining Cost")
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(kernel='rbf')
    
    knn_times = []
    svm_times = []
    
    current_X = list(X_base)
    current_y = list(y_base)
    test_limit = min(50, len(X_stream))
    
    for i in range(test_limit):
        current_X.append(X_stream[i])
        current_y.append(y_stream[i])
        
        start = time.perf_counter()
        knn.fit(current_X, current_y)
        knn_times.append((time.perf_counter() - start) * 1e9)
        
        start = time.perf_counter()
        svm.fit(current_X, current_y)
        svm_times.append((time.perf_counter() - start) * 1e9)

    avg_knn_ns = np.mean(knn_times)
    avg_svm_ns = np.mean(svm_times)
    
    print(f"⚠️  KNN Avg Retrain: {avg_knn_ns:.0f} ns")
    print(f"⚠️  SVM Avg Retrain: {avg_svm_ns:.0f} ns")

    if not os.path.exists(cpp_log_path):
        print(f"Error: Log file {cpp_log_path} not found.")
        return

    df = pd.read_csv(cpp_log_path)
    avg_insert = df[df['operation'] == 'insert']['time_ns'].mean()
    avg_move = df[df['operation'] == 'move']['time_ns'].mean()
    avg_delete = df[df['operation'] == 'delete']['time_ns'].mean()

    print(f"Algo 1 (Insert): {avg_insert:.0f} ns")
    print(f"Algo 3 (Move):   {avg_move:.0f} ns")
    print(f"Algo 2 (Delete): {avg_delete:.0f} ns")

    print("\n" + "="*70)
    print(f"{'Operation':<20} | {'Method':<20} | {'Time (ns)':<15} | {'Speedup'}")
    print("-" * 70)
    print(f"{'Add New Point':<20} | {'KNN (Retrain)':<20} | {avg_knn_ns:<15.0f} | 1.0x")
    print(f"{'Add New Point':<20} | {'SVM (Retrain)':<20} | {avg_svm_ns:<15.0f} | {avg_knn_ns/avg_svm_ns:.1f}x")
    print(f"{'Add New Point':<20} | {'**Algo 1 (Ours)**':<20} | {avg_insert:<15.0f} | **{avg_knn_ns/avg_insert:.1f}x**")
    print("-" * 70)
    print(f"{'Move Point':<20} | {'KNN (Retrain)':<20} | {avg_knn_ns:<15.0f} | 1.0x")
    print(f"{'Move Point':<20} | {'**Algo 3 (Ours)**':<20} | {avg_move:<15.0f} | **{avg_knn_ns/avg_move:.1f}x**")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--mode', required=True, choices=['static', 'dynamic'])
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.mode == 'static':
        run_static_benchmark(args, root)
    elif args.mode == 'dynamic':
        run_dynamic_benchmark(args, root)