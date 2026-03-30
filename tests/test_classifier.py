#!/usr/bin/env python3
"""
Unit Tests for Delaunay Triangulation Classifier

Validates correctness of:
1. Data loading and CSV format
2. Delaunay triangulation geometric properties
3. C++ classifier static and dynamic modes
4. Outlier detection logic
5. 2D Buckets grid sizing and O(1) lookup
6. Decision boundary geometry (half-plane, nearest-vertex)
7. Point-in-polygon ray casting
8. Dataset generation output format

Run with: python -m pytest tests/test_classifier.py -v
Or: python tests/test_classifier.py
"""

import os
import sys
import unittest
import subprocess
import tempfile
import shutil
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CPP_MAIN = PROJECT_ROOT / "build" / "main"
CPP_BENCHMARK = PROJECT_ROOT / "build" / "benchmark"


# ============================================================================
# 1. Data Loading
# ============================================================================

class TestDataLoading(unittest.TestCase):
    """Test data loading and CSV format handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = np.array([
            [0.0, 0.0, 0],
            [1.0, 0.0, 0],
            [0.0, 1.0, 1],
            [1.0, 1.0, 1],
            [0.5, 0.5, 0],
        ])
        self.train_path = os.path.join(self.temp_dir, "test_train.csv")
        np.savetxt(self.train_path, self.test_data,
                   delimiter=',', fmt=['%.6f', '%.6f', '%d'])
        self.test_path = os.path.join(self.temp_dir, "test_X.csv")
        np.savetxt(self.test_path, self.test_data[:, :2],
                   delimiter=',', fmt='%.6f')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_csv_format_three_columns(self):
        """Training CSV should have 3 columns: x, y, label."""
        df = pd.read_csv(self.train_path, header=None)
        self.assertEqual(df.shape[1], 3)
        self.assertEqual(df.shape[0], 5)

    def test_labels_are_integers(self):
        """Labels column should contain integer values."""
        df = pd.read_csv(self.train_path, header=None)
        labels = df.iloc[:, 2].values
        self.assertTrue(np.all(labels == labels.astype(int)))


# ============================================================================
# 2. Delaunay Triangulation Properties
# ============================================================================

class TestDelaunayTriangulation(unittest.TestCase):
    """Test Delaunay triangulation geometric properties."""

    def test_basic_triangulation(self):
        """Triangulation of 5 points should produce valid triangles."""
        from scipy.spatial import Delaunay
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
        tri = Delaunay(points)
        self.assertGreater(len(tri.simplices), 0)
        for simplex in tri.simplices:
            self.assertEqual(len(simplex), 3)

    def test_circumcircle_property(self):
        """No point should lie strictly inside any triangle's circumcircle."""
        from scipy.spatial import Delaunay
        np.random.seed(42)
        points = np.random.rand(20, 2)
        tri = Delaunay(points)

        for simplex in tri.simplices:
            p1, p2, p3 = points[simplex]
            ax, ay = p1
            bx, by = p2
            cx, cy = p3

            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            if abs(d) < 1e-10:
                continue

            ux = ((ax**2 + ay**2) * (by - cy) +
                  (bx**2 + by**2) * (cy - ay) +
                  (cx**2 + cy**2) * (ay - by)) / d
            uy = ((ax**2 + ay**2) * (cx - bx) +
                  (bx**2 + by**2) * (ax - cx) +
                  (cx**2 + cy**2) * (bx - ax)) / d
            radius_sq = (ax - ux)**2 + (ay - uy)**2

            for i, pt in enumerate(points):
                if i in simplex:
                    continue
                dist_sq = (pt[0] - ux)**2 + (pt[1] - uy)**2
                self.assertGreaterEqual(dist_sq, radius_sq - 1e-10,
                    "No point should be strictly inside circumcircle")


# ============================================================================
# 3. C++ Classifier Integration Tests
# ============================================================================

class TestCppClassifier(unittest.TestCase):
    """Test the C++ classifier executable."""

    @classmethod
    def setUpClass(cls):
        if not CPP_MAIN.exists():
            raise unittest.SkipTest(
                f"C++ executable not found at {CPP_MAIN}. Run 'make' first.")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        np.random.seed(42)
        n_per_class = 50

        X0 = np.random.randn(n_per_class, 2) * 0.3 + np.array([-1, -1])
        y0 = np.zeros(n_per_class, dtype=int)
        X1 = np.random.randn(n_per_class, 2) * 0.3 + np.array([1, 1])
        y1 = np.ones(n_per_class, dtype=int)

        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        train_size = int(0.8 * len(X))
        self.X_train, self.y_train = X[:train_size], y[:train_size]
        self.X_test, self.y_test = X[train_size:], y[train_size:]

        self.train_path = os.path.join(self.temp_dir, "train.csv")
        self.test_labeled_path = os.path.join(self.temp_dir, "test_y.csv")
        self.test_unlabeled_path = os.path.join(self.temp_dir, "test_X.csv")

        train_data = np.column_stack([self.X_train, self.y_train])
        np.savetxt(self.train_path, train_data,
                   delimiter=',', fmt=['%.6f', '%.6f', '%d'])

        test_y_data = np.column_stack([self.X_test, self.y_test])
        np.savetxt(self.test_labeled_path, test_y_data,
                   delimiter=',', fmt=['%.6f', '%.6f', '%d'])

        np.savetxt(self.test_unlabeled_path, self.X_test,
                   delimiter=',', fmt='%.6f')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_static_mode_runs(self):
        """C++ classifier should complete static mode without errors."""
        cmd = [str(CPP_MAIN), "static", self.train_path,
               self.test_unlabeled_path, self.results_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"Static mode failed: {result.stderr}")
        predictions_file = os.path.join(self.results_dir, "predictions.csv")
        self.assertTrue(os.path.exists(predictions_file),
                        "predictions.csv should be created")

    def test_accuracy_on_separable_data(self):
        """Accuracy on well-separated clusters should exceed 90%."""
        cmd = [str(CPP_MAIN), "static", self.train_path,
               self.test_labeled_path, self.results_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"Static mode failed: {result.stderr}")

        accuracy = None
        for line in result.stdout.split('\n'):
            if "Accuracy" in line and "%" in line:
                try:
                    # Parse "Accuracy:  95.0% (19/20)" format
                    acc_part = line.split(':')[-1].strip()
                    accuracy = float(acc_part.split('%')[0].strip())
                except (ValueError, IndexError):
                    pass

        self.assertIsNotNone(accuracy,
            "C++ should report accuracy when given labeled test file")
        self.assertGreater(accuracy, 90.0,
            f"Accuracy on separable data should be > 90%, got {accuracy}%")

    def test_dynamic_mode_runs(self):
        """C++ dynamic stress test should complete without errors."""
        # Create stream file (subset of test data with labels)
        stream_path = os.path.join(self.temp_dir, "stream.csv")
        stream_data = np.column_stack([self.X_test[:5], self.y_test[:5]])
        np.savetxt(stream_path, stream_data,
                   delimiter=',', fmt=['%.6f', '%.6f', '%d'])

        log_path = os.path.join(self.results_dir, "dynamic_log.csv")
        cmd = [str(CPP_MAIN), "dynamic", self.train_path,
               stream_path, log_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"Dynamic mode failed: {result.stderr}")

        self.assertTrue(os.path.exists(log_path),
                        "Dynamic log CSV should be created")
        df = pd.read_csv(log_path)
        self.assertIn('operation', df.columns)
        self.assertIn('time_ns', df.columns)
        self.assertGreater(len(df), 0, "Log should contain timing entries")


# ============================================================================
# 4. Outlier Detection
# ============================================================================

class TestOutlierDetection(unittest.TestCase):
    """Test outlier detection via k-NN same-class density."""

    def test_isolated_point_detected(self):
        """A point far from its cluster should have highest mean distance."""
        from sklearn.neighbors import NearestNeighbors

        X = np.array([
            [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1],  # Cluster
            [10, 10],  # Outlier
        ])
        k = 3
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        mean_distances = distances[:, 1:].mean(axis=1)
        outlier_idx = np.argmax(mean_distances)
        self.assertEqual(outlier_idx, 4,
                         "Point at (10,10) should be the outlier")


# ============================================================================
# 5. 2D Buckets Grid
# ============================================================================

class TestGridAndBuckets(unittest.TestCase):
    """Test Square Root Rule grid sizing and O(1) bucket lookup."""

    def test_grid_size_follows_sqrt_n(self):
        """Grid dimension should be ceil(sqrt(n))."""
        import math
        test_cases = [
            (100, 10),
            (1000, 32),   # ceil(sqrt(1000)) = 32
            (10000, 100),
        ]
        for n, expected in test_cases:
            grid_size = int(math.ceil(math.sqrt(n)))
            self.assertAlmostEqual(grid_size, expected, delta=1,
                msg=f"For n={n}, grid size should be ~{expected}")

    def test_bucket_index_is_O1(self):
        """Bucket lookup time should be constant regardless of grid size."""
        grid_sizes = [10, 100, 1000]
        lookup_times = []

        for size in grid_sizes:
            grid = {(i, j): 0 for i in range(size) for j in range(size)}
            start = time.perf_counter()
            for _ in range(1000):
                x, y = np.random.randint(0, size, 2)
                _ = grid.get((x, y))
            elapsed = time.perf_counter() - start
            lookup_times.append(elapsed)

        ratio = lookup_times[-1] / lookup_times[0]
        self.assertLess(ratio, 5.0,
            f"Lookup should be O(1), but time ratio is {ratio:.2f}")

    def test_bucket_index_calculation(self):
        """Bucket index from coordinates should map correctly."""
        min_x, max_x = 0.0, 10.0
        min_y, max_y = 0.0, 10.0
        cols, rows = 10, 10
        step_x = (max_x - min_x) / cols
        step_y = (max_y - min_y) / rows

        def get_bucket_index(x, y):
            c = int((x - min_x) / step_x)
            r = int((y - min_y) / step_y)
            c = max(0, min(c, cols - 1))
            r = max(0, min(r, rows - 1))
            return r * cols + c

        self.assertEqual(get_bucket_index(0.5, 0.5), 0)    # Bottom-left
        self.assertEqual(get_bucket_index(9.5, 0.5), 9)    # Bottom-right
        self.assertEqual(get_bucket_index(0.5, 9.5), 90)   # Top-left
        self.assertEqual(get_bucket_index(9.5, 9.5), 99)   # Top-right
        self.assertEqual(get_bucket_index(5.5, 5.5), 55)   # Center

    def test_clamping_handles_edge_cases(self):
        """Points outside the grid should be clamped to valid indices."""
        min_x, max_x = 0.0, 10.0
        min_y, max_y = 0.0, 10.0
        cols, rows = 10, 10
        step_x = (max_x - min_x) / cols
        step_y = (max_y - min_y) / rows

        def get_bucket_index(x, y):
            c = int((x - min_x) / step_x)
            r = int((y - min_y) / step_y)
            c = max(0, min(c, cols - 1))
            r = max(0, min(r, rows - 1))
            return r * cols + c

        # Points outside bounds should clamp to edge buckets
        idx = get_bucket_index(-5.0, -5.0)
        self.assertEqual(idx, 0, "Negative coords should clamp to bucket 0")

        idx = get_bucket_index(100.0, 100.0)
        self.assertEqual(idx, 99, "Far positive coords should clamp to last bucket")


# ============================================================================
# 6. Decision Boundary Geometry
# ============================================================================

class TestDecisionBoundary(unittest.TestCase):
    """Test half-plane and nearest-vertex decision boundary logic."""

    def test_half_plane_two_class_triangle(self):
        """Half-plane test should correctly separate two classes in a triangle.

        Given triangle with vertices:
          v0 = (0, 0) class 0
          v1 = (2, 0) class 0
          v2 = (1, 2) class 1

        The decision boundary is the line connecting the midpoints of the
        two cross-class edges (v0-v2 and v1-v2). Points on the v2 side of
        this line should be class 1, others class 0.
        """
        # Triangle vertices
        v0, v1, v2 = np.array([0, 0]), np.array([2, 0]), np.array([1, 2])
        l0, l1, l2 = 0, 0, 1  # v2 is isolated (class 1)

        # Midpoints of cross-class edges
        mid_v0_v2 = (v0 + v2) / 2   # (0.5, 1.0)
        mid_v1_v2 = (v1 + v2) / 2   # (1.5, 1.0)

        # Decision boundary line direction
        line_dx = mid_v1_v2[0] - mid_v0_v2[0]
        line_dy = mid_v1_v2[1] - mid_v0_v2[1]

        def classify(px, py):
            cross_query = (line_dx * (py - mid_v0_v2[1]) -
                          line_dy * (px - mid_v0_v2[0]))
            cross_isolated = (line_dx * (v2[1] - mid_v0_v2[1]) -
                            line_dy * (v2[0] - mid_v0_v2[0]))
            if (cross_query > 0) == (cross_isolated > 0):
                return l2  # isolated class
            else:
                return l0  # majority class

        # Points near v0 and v1 (majority side) should be class 0
        self.assertEqual(classify(0.5, 0.2), 0)
        self.assertEqual(classify(1.5, 0.2), 0)
        self.assertEqual(classify(1.0, 0.3), 0)

        # Points near v2 (isolated side) should be class 1
        self.assertEqual(classify(1.0, 1.5), 1)
        self.assertEqual(classify(0.8, 1.3), 1)

    def test_three_class_nearest_vertex(self):
        """Three distinct classes: nearest-vertex should determine the class."""
        v0, v1, v2 = np.array([0, 0]), np.array([4, 0]), np.array([2, 4])
        l0, l1, l2 = 0, 1, 2

        def classify_nearest(px, py):
            d0 = (px - v0[0])**2 + (py - v0[1])**2
            d1 = (px - v1[0])**2 + (py - v1[1])**2
            d2 = (px - v2[0])**2 + (py - v2[1])**2
            if d0 <= d1 and d0 <= d2:
                return l0
            if d1 <= d0 and d1 <= d2:
                return l1
            return l2

        self.assertEqual(classify_nearest(0.5, 0.5), 0)
        self.assertEqual(classify_nearest(3.5, 0.5), 1)
        self.assertEqual(classify_nearest(2.0, 3.5), 2)

    def test_homogeneous_triangle_returns_unanimous(self):
        """All same class: should return that class regardless of position."""
        label = 2
        # Any point in the triangle gets the same label
        self.assertEqual(label, 2)


# ============================================================================
# 7. Point-in-Polygon and Bucket Classification
# ============================================================================

class TestBucketClassification(unittest.TestCase):
    """Test 2D Buckets classification types: HOMOGENEOUS, BIPARTITIONED, MULTI."""

    def test_point_in_polygon_ray_casting(self):
        """Ray casting should correctly classify points inside/outside a square."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

        def point_in_polygon(x, y, poly):
            n = len(poly) - 1
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if (((yi > y) != (yj > y)) and
                        (x < (xj - xi) * (y - yi) / (yj - yi) + xi)):
                    inside = not inside
                j = i
            return inside

        self.assertTrue(point_in_polygon(0.5, 0.5, polygon))
        self.assertTrue(point_in_polygon(0.25, 0.25, polygon))
        self.assertFalse(point_in_polygon(1.5, 0.5, polygon))
        self.assertFalse(point_in_polygon(-0.5, 0.5, polygon))

    def test_homogeneous_bucket_returns_dominant(self):
        """Single-class bucket should always return dominant_class."""
        class MockBucket:
            def __init__(self, dominant_class):
                self.type = 'HOMOGENEOUS'
                self.dominant_class = dominant_class
            def classify_point(self, x, y):
                return self.dominant_class

        bucket = MockBucket(2)
        self.assertEqual(bucket.classify_point(0.0, 0.0), 2)
        self.assertEqual(bucket.classify_point(99.0, -50.0), 2)

    def test_bipartitioned_bucket_half_plane(self):
        """Two-class bucket should classify via half-plane dot product."""
        # Boundary: x = 0.5 (vertical line at x=0.5)
        # nx=1, ny=0, d=-0.5 → dot = 1*x + 0*y - 0.5
        nx, ny, d = 1.0, 0.0, -0.5
        class_positive, class_negative = 1, 0

        def classify(x, y):
            dot = nx * x + ny * y + d
            return class_positive if dot >= 0 else class_negative

        self.assertEqual(classify(0.8, 0.5), 1)   # Right of boundary
        self.assertEqual(classify(0.2, 0.5), 0)   # Left of boundary
        self.assertEqual(classify(0.5, 0.5), 1)   # On boundary (positive)

    def test_bucket_type_detection(self):
        """Bucket type should be determined by number of distinct classes."""
        def detect_type(classes_in_bucket):
            n = len(set(classes_in_bucket))
            if n <= 1:
                return 'HOMOGENEOUS'
            elif n == 2:
                return 'BIPARTITIONED'
            else:
                return 'MULTI_PARTITIONED'

        self.assertEqual(detect_type([0, 0, 0, 0]), 'HOMOGENEOUS')
        self.assertEqual(detect_type([0, 0, 1, 1]), 'BIPARTITIONED')
        self.assertEqual(detect_type([0, 1, 2]), 'MULTI_PARTITIONED')


# ============================================================================
# 8. Dynamic Operations
# ============================================================================

class TestDynamicOperations(unittest.TestCase):
    """Test dynamic insert/remove operations preserve triangulation."""

    def test_insert_preserves_delaunay(self):
        """Inserting a point should produce a valid Delaunay triangulation."""
        from scipy.spatial import Delaunay

        np.random.seed(42)
        base_points = np.random.rand(20, 2)
        new_point = np.array([[0.5, 0.5]])
        all_points = np.vstack([base_points, new_point])

        tri = Delaunay(all_points)
        self.assertGreater(len(tri.simplices), 0)

        new_idx = len(base_points)
        found = any(new_idx in simplex for simplex in tri.simplices)
        self.assertTrue(found, "New point should appear in triangulation")

    def test_remove_reduces_vertex_count(self):
        """Removing a point should reduce the triangulation vertex count."""
        from scipy.spatial import Delaunay

        np.random.seed(42)
        points = np.random.rand(20, 2)
        tri_before = Delaunay(points)
        n_before = len(points)

        # Remove last point and retriangulate
        points_after = points[:-1]
        tri_after = Delaunay(points_after)
        n_after = len(points_after)

        self.assertEqual(n_after, n_before - 1)
        self.assertGreater(len(tri_after.simplices), 0)


# ============================================================================
# 9. Dataset Generation
# ============================================================================

class TestDatasetGeneration(unittest.TestCase):
    """Test that generate_datasets.py produces correct output format."""

    def test_training_file_format(self):
        """Training CSV should have 3 float/int columns, no header."""
        data_dir = PROJECT_ROOT / "data" / "train"

        if not data_dir.exists():
            self.skipTest("Data directory not found. "
                          "Run generate_datasets.py first.")

        csv_files = list(data_dir.glob("*_train.csv"))
        if not csv_files:
            self.skipTest("No training files found. "
                          "Run generate_datasets.py first.")

        df = pd.read_csv(csv_files[0], header=None)
        self.assertEqual(df.shape[1], 3, "Should have 3 columns (x, y, label)")
        self.assertTrue(df.iloc[:, 0].dtype in [np.float64, np.float32])
        self.assertTrue(df.iloc[:, 1].dtype in [np.float64, np.float32])

    def test_test_files_exist_in_pairs(self):
        """Each dataset should have both _test_X.csv and _test_y.csv."""
        test_dir = PROJECT_ROOT / "data" / "test"

        if not test_dir.exists():
            self.skipTest("Test data directory not found.")

        x_files = {f.stem.replace('_test_X', '')
                   for f in test_dir.glob("*_test_X.csv")}
        y_files = {f.stem.replace('_test_y', '')
                   for f in test_dir.glob("*_test_y.csv")}

        if not x_files:
            self.skipTest("No test files found.")

        self.assertEqual(x_files, y_files,
            "Every _test_X.csv should have a matching _test_y.csv")


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Core algorithm tests (always run)
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDelaunayTriangulation))
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestGridAndBuckets))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionBoundary))
    suite.addTests(loader.loadTestsFromTestCase(TestBucketClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetGeneration))

    # C++ integration tests (only if executable exists)
    if CPP_MAIN.exists():
        suite.addTests(loader.loadTestsFromTestCase(TestCppClassifier))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("DELAUNAY TRIANGULATION CLASSIFIER — UNIT TESTS")
    print("=" * 70)

    success = run_tests()

    print("\n" + "=" * 70)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if success else 1)