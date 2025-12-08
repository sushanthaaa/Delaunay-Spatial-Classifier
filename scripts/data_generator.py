import argparse
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <--- NEW IMPORT
import os

# --- GENERATION FUNCTIONS ---
def generate_moons(n_samples=1000, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

def generate_blobs(n_samples=1500, centers=3):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1.5, random_state=42)
    return X, y

def generate_iris():
    iris = datasets.load_iris()
    return iris.data, iris.target

def generate_wine():
    wine = datasets.load_wine()
    return wine.data, wine.target

def generate_cancer():
    cancer = datasets.load_breast_cancer()
    return cancer.data, cancer.target

def generate_digits():
    digits = datasets.load_digits()
    return digits.data, digits.target

def process_data(X, y):
    # 1. Normalize Features (Crucial for Delaunay distance checks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    return X_2d, y

def add_noise(X, y, n_noise=20):
    np.random.seed(99)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    noise_X = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(n_noise, 2))
    noise_y = np.random.randint(0, len(np.unique(y)), size=n_noise)
    return np.vstack([X, noise_X]), np.hstack([y, noise_y])

def save_csv(X, y, filename):
    df = pd.DataFrame(X, columns=['x', 'y'])
    if y is not None:
        df['label'] = y
    df.to_csv(filename, index=False, header=False)
    print(f"✅ Saved: {filename} ({len(X)} points)")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_out = os.path.join(project_root, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, 
                        choices=['moons', 'blobs', 'iris', 'wine', 'cancer', 'digits'])
    parser.add_argument('--out_dir', type=str, default=default_out)
    parser.add_argument('--noise', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(f"{args.out_dir}/train", exist_ok=True)
    os.makedirs(f"{args.out_dir}/test", exist_ok=True)

    # 1. Load Raw Data
    if args.type == 'moons': X, y = generate_moons()
    elif args.type == 'blobs': X, y = generate_blobs()
    elif args.type == 'iris': X, y = generate_iris()
    elif args.type == 'wine': X, y = generate_wine()
    elif args.type == 'cancer': X, y = generate_cancer()
    elif args.type == 'digits': X, y = generate_digits()

    # 2. Process (Scale + PCA)
    # Note: Synthetic data is already 2D/scaled, but processing assumes raw high-dim input.
    # For moons/blobs we skip scaling to preserve their shape, for others we apply it.
    if args.type in ['iris', 'wine', 'cancer', 'digits']:
        X, y = process_data(X, y)

    # 3. Add Noise
    if args.noise > 0:
        X, y = add_noise(X, y, args.noise)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    save_csv(X_train, y_train, f"{args.out_dir}/train/{args.type}_train.csv")
    save_csv(X_test, None, f"{args.out_dir}/test/{args.type}_test_X.csv")
    save_csv(X_test, y_test, f"{args.out_dir}/test/{args.type}_test_y.csv")

    # 5. Dynamic Split
    if len(X_train) < 200:
        base_size = min(100, int(len(X_train) * 0.8))
    else:
        base_size = int(len(X_train) * 0.5)

    if base_size < len(X_train):
        save_csv(X_train[:base_size], y_train[:base_size], f"{args.out_dir}/train/{args.type}_dynamic_base.csv")
        save_csv(X_train[base_size:], y_train[base_size:], f"{args.out_dir}/train/{args.type}_dynamic_stream.csv")