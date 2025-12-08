import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

# IEEE Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}

def load_data(args):
    data = {}
    try:
        # FIX: Try to load CLEAN points first, fall back to TRAIN if missing
        if os.path.exists(f"{args.results_dir}/clean_points.csv"):
            data['train'] = pd.read_csv(f"{args.results_dir}/clean_points.csv", header=None, names=['x', 'y', 'label'])
            print("ℹ️ Plotting CLEANED data (Phase 1 Output)")
        else:
            data['train'] = pd.read_csv(f"{args.data_dir}/{args.dataset}_train.csv", header=None, names=['x', 'y', 'label'])
            print("⚠️ Clean points not found. Plotting RAW data.")

        data['triangles'] = pd.read_csv(f"{args.results_dir}/triangles.csv", header=None, names=['x1', 'y1', 'x2', 'y2'])
        data['boundaries'] = pd.read_csv(f"{args.results_dir}/boundaries.csv", header=None, names=['x1', 'y1', 'x2', 'y2'])
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        exit(1)
    return data

def plot_static(data, dataset_name, out_path):
    if 'train' not in data: return
    plt.figure(figsize=(6, 5))
    
    # Mesh
    if 'triangles' in data:
        segs = [[(r.x1, r.y1), (r.x2, r.y2)] for i, r in data['triangles'].iterrows()]
        plt.gca().add_collection(LineCollection(segs, colors='gray', linewidths=0.2, alpha=0.3))
    
    # Boundary
    if 'boundaries' in data:
        bsegs = [[(r.x1, r.y1), (r.x2, r.y2)] for i, r in data['boundaries'].iterrows()]
        plt.gca().add_collection(LineCollection(bsegs, colors='black', linewidths=1.5))

    # Points
    df = data['train']
    plt.scatter(df['x'], df['y'], c=df['label'].map(COLORS), s=15, alpha=0.7)
    
    plt.title(f"Delaunay Classification: {dataset_name.capitalize()}")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")

def plot_dynamic(log_file, out_path):
    try:
        df = pd.read_csv(log_file)
        insert = df[df['operation'] == 'insert']['time_ns'].reset_index(drop=True)
        plt.figure(figsize=(8, 4))
        plt.plot(insert, color='green', alpha=0.6, linewidth=1, label='Insert Time')
        plt.plot(insert.rolling(20).mean(), color='darkgreen', linewidth=2, label='Moving Avg')
        plt.yscale('log')
        plt.title("Dynamic Insertion Performance")
        plt.ylabel("Time (ns)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"✅ Saved: {out_path}")
    except Exception as e:
        print(f"⚠️ Dynamic plot skipped: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', required=False) # Made optional for dynamic mode
    parser.add_argument('--data_dir', default=f"{root}/data/train")
    parser.add_argument('--results_dir', default=f"{root}/results")
    args = parser.parse_args()

    os.makedirs(f"{args.results_dir}/figures", exist_ok=True)

    if args.mode == 'static' and args.dataset:
        data = load_data(args)
        plot_static(data, args.dataset, f"{args.results_dir}/figures/Static_{args.dataset}.png")
    
    elif args.mode == 'dynamic':
        plot_dynamic(f"{args.results_dir}/dynamic_logs.csv", f"{args.results_dir}/figures/Dynamic_Perf.png")