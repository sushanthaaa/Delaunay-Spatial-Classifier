import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}

def load_data(args):
    data = {}
    
    train_file = f"{args.data_dir}/{args.dataset}_train.csv"
    if not os.path.exists(train_file):
        print(f"Error: Dataset '{args.dataset}' not found at {train_file}")
        print("Please check the spelling or run data_generator.py first.")
        exit(1)

    try:
        clean_file = f"{args.results_dir}/{args.dataset}_clean_points.csv"
        
        if os.path.exists(clean_file):
            data['train'] = pd.read_csv(clean_file, header=None, names=['x', 'y', 'label'])
            print(f"Plotting CLEANED data ({clean_file})")
        else:
            data['train'] = pd.read_csv(train_file, header=None, names=['x', 'y', 'label'])
            print(f"Clean points not found. Plotting RAW data ({train_file})")

        data['triangles'] = pd.read_csv(f"{args.results_dir}/triangles.csv", header=None, names=['x1', 'y1', 'x2', 'y2'])
        data['boundaries'] = pd.read_csv(f"{args.results_dir}/boundaries.csv", header=None, names=['x1', 'y1', 'x2', 'y2'])
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit(1)
    return data

def plot_static(data, dataset_name, out_path):
    if 'train' not in data: return
    plt.figure(figsize=(6, 5))
    
    if 'triangles' in data:
        segs = [[(r.x1, r.y1), (r.x2, r.y2)] for i, r in data['triangles'].iterrows()]
        plt.gca().add_collection(LineCollection(segs, colors='gray', linewidths=0.2, alpha=0.3))
    
    if 'boundaries' in data:
        bsegs = [[(r.x1, r.y1), (r.x2, r.y2)] for i, r in data['boundaries'].iterrows()]
        plt.gca().add_collection(LineCollection(bsegs, colors='black', linewidths=1.5))

    df = data['train']
    plt.scatter(df['x'], df['y'], c=df['label'].map(COLORS), s=15, alpha=0.7)
    
    plt.title(f"Delaunay Classification: {dataset_name.capitalize()}")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

def plot_dynamic(log_file, out_path, dataset_name):
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found. Run benchmark --mode dynamic first.")
        return

    try:
        df = pd.read_csv(log_file)
        insert = df[df['operation'] == 'insert']['time_ns'].reset_index(drop=True)
        plt.figure(figsize=(8, 4))
        plt.plot(insert, color='green', alpha=0.6, linewidth=1, label='Insert Time')
        plt.plot(insert.rolling(20).mean(), color='darkgreen', linewidth=2, label='Moving Avg')
        plt.yscale('log')
        plt.title(f"Dynamic Insertion: {dataset_name.capitalize()}")
        plt.ylabel("Time (ns)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Dynamic plot skipped: {e}")

def plot_dynamic_snapshots(args):
    stages = ["dynamic_1_inserted", "dynamic_2_moved", "dynamic_3_deleted"]
    titles = ["Step 1: After Insertion", "Step 2: After Movement", "Step 3: After Deletion"]
    
    for stage, title in zip(stages, titles):
        tri_file = f"{args.results_dir}/{stage}_triangles.csv"
        bound_file = f"{args.results_dir}/{stage}_boundaries.csv"
        point_file = f"{args.results_dir}/{stage}_points.csv"
        
        if not os.path.exists(tri_file): continue
        
        print(f"Plotting {title}...")
        plt.figure(figsize=(6, 5))
        
        triangles = pd.read_csv(tri_file, header=None, names=['x1', 'y1', 'x2', 'y2'])
        mesh_segs = [[(r.x1, r.y1), (r.x2, r.y2)] for i, r in triangles.iterrows()]
        plt.gca().add_collection(LineCollection(mesh_segs, colors='gray', linewidths=0.2, alpha=0.3))
        
        boundaries = pd.read_csv(bound_file, header=None, names=['x1', 'y1', 'x2', 'y2'])
        bound_segs = [[(r.x1, r.y1), (r.x2, r.y2)] for i, r in boundaries.iterrows()]
        plt.gca().add_collection(LineCollection(bound_segs, colors='black', linewidths=1.5))
        
        if os.path.exists(point_file):
            points = pd.read_csv(point_file, header=None, names=['x', 'y', 'label'])
            plt.scatter(points['x'], points['y'], c=points['label'].map(COLORS), s=15, alpha=0.7, zorder=3)
        
        plt.title(f"{args.dataset.capitalize()} - {title}")
        plt.xlabel("Feature 1 (Scaled)")
        plt.ylabel("Feature 2 (Scaled)")
        plt.tight_layout()
        
        out_name = f"{args.results_dir}/figures/{args.dataset}_{stage}.png"
        plt.savefig(out_name)
        print(f"Saved: {out_name}")    

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_dir', default=f"{root}/data/train")
    parser.add_argument('--results_dir', default=f"{root}/results")
    args = parser.parse_args()

    os.makedirs(f"{args.results_dir}/figures", exist_ok=True)

    if args.mode == 'static':
        data = load_data(args)
        plot_static(data, args.dataset, f"{args.results_dir}/figures/Static_{args.dataset}.png")
        pass

    elif args.mode == 'dynamic':
        log_name = f"{args.dataset}_dynamic_logs.csv"
        out_name = f"Dynamic_{args.dataset}.png"
        
        plot_dynamic(
            f"{args.results_dir}/logs/{log_name}", 
            f"{args.results_dir}/figures/{out_name}",
            args.dataset
        )
        pass

    elif args.mode == 'dynamic_visual' and args.dataset:
        plot_dynamic_snapshots(args)