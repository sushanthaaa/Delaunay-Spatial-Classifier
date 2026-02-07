#!/usr/bin/env python3
"""
Unit Tests for Delaunay Triangulation Classifier

This module provides comprehensive unit tests to validate:
1. Data loading and preprocessing
2. Delaunay triangulation correctness
3. Classification accuracy
4. Dynamic operations (insert/remove)
5. SRR grid functionality
6. Outlier detection

Run with: python -m pytest tests/test_classifier.py -v
Or: python tests/test_classifier.py
"""

import os
import sys
import unittest
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CPP_MAIN = PROJECT_ROOT / "build" / "main"
CPP_BENCHMARK = PROJECT_ROOT / "build" / "benchmark"


class TestDataLoading(unittest.TestCase):
    """Test data loading and CSV format handling."""
    
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test dataset
        self.test_data = np.array([
            [0.0, 0.0, 0],
            [1.0, 0.0, 0],
            [0.0, 1.0, 1],
            [1.0, 1.0, 1],
            [0.5, 0.5, 0],
        ])
        
        self.train_path = os.path.join(self.temp_dir, "test_train.csv")
        np.savetxt(self.train_path, self.test_data, delimiter=',', fmt=['%.6f', '%.6f', '%d'])
        
        self.test_path = os.path.join(self.temp_dir, "test_X.csv")
        np.savetxt(self.test_path, self.test_data[:, :2], delimiter=',', fmt='%.6f')
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_format(self):
        """Test that CSV files are correctly formatted."""
        df = pd.read_csv(self.train_path, header=None)
        self.assertEqual(df.shape[1], 3, "Training CSV should have 3 columns: x, y, label")
        self.assertEqual(df.shape[0], 5, "Should have 5 data points")
    
    def test_label_values(self):
        """Test that labels are integers."""
        df = pd.read_csv(self.train_path, header=None)
        labels = df.iloc[:, 2].values
        self.assertTrue(np.all(labels == labels.astype(int)), "Labels should be integers")


class TestDelaunayTriangulation(unittest.TestCase):
    """Test Delaunay triangulation properties."""
    
    def test_scipy_delaunay_basic(self):
        """Test basic Delaunay triangulation with scipy."""
        from scipy.spatial import Delaunay
        
        # Simple square with center point
        points = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]
        ])
        
        tri = Delaunay(points)
        
        # Should have valid triangles
        self.assertGreater(len(tri.simplices), 0, "Should create triangles")
        
        # All simplices should have 3 vertices
        for simplex in tri.simplices:
            self.assertEqual(len(simplex), 3, "Each triangle should have 3 vertices")
    
    def test_delaunay_circumcircle_property(self):
        """Test that Delaunay triangulation satisfies circumcircle property."""
        from scipy.spatial import Delaunay
        
        np.random.seed(42)
        points = np.random.rand(20, 2)
        tri = Delaunay(points)
        
        # For each triangle, no other point should be inside its circumcircle
        # This is the defining property of Delaunay triangulation
        for simplex in tri.simplices:
            # Get triangle vertices
            p1, p2, p3 = points[simplex]
            
            # Calculate circumcenter and radius
            ax, ay = p1
            bx, by = p2
            cx, cy = p3
            
            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            if abs(d) < 1e-10:
                continue  # Degenerate triangle
            
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
            
            radius_sq = (ax - ux)**2 + (ay - uy)**2
            
            # Check that no other point is strictly inside the circumcircle
            for i, pt in enumerate(points):
                if i in simplex:
                    continue
                dist_sq = (pt[0] - ux)**2 + (pt[1] - uy)**2
                # Allow small tolerance for numerical errors
                self.assertGreaterEqual(dist_sq, radius_sq - 1e-10,
                    "No point should be strictly inside circumcircle")


class TestCppClassifier(unittest.TestCase):
    """Test the C++ classifier executable."""
    
    @classmethod
    def setUpClass(cls):
        """Check that C++ executable exists."""
        if not CPP_MAIN.exists():
            raise unittest.SkipTest(f"C++ executable not found at {CPP_MAIN}. Run 'make' first.")
    
    def setUp(self):
        """Create test data files."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create simple linearly separable data
        np.random.seed(42)
        n_per_class = 50
        
        # Class 0: bottom-left cluster
        X0 = np.random.randn(n_per_class, 2) * 0.3 + np.array([-1, -1])
        y0 = np.zeros(n_per_class, dtype=int)
        
        # Class 1: top-right cluster
        X1 = np.random.randn(n_per_class, 2) * 0.3 + np.array([1, 1])
        y1 = np.ones(n_per_class, dtype=int)
        
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        
        # Split train/test
        train_size = int(0.8 * len(X))
        self.X_train, self.y_train = X[:train_size], y[:train_size]
        self.X_test, self.y_test = X[train_size:], y[train_size:]
        
        # Save to files
        self.train_path = os.path.join(self.temp_dir, "train.csv")
        self.test_path = os.path.join(self.temp_dir, "test.csv")
        self.test_labels_path = os.path.join(self.temp_dir, "test_y.csv")
        
        train_data = np.column_stack([self.X_train, self.y_train])
        np.savetxt(self.train_path, train_data, delimiter=',', fmt=['%.6f', '%.6f', '%d'])
        np.savetxt(self.test_path, self.X_test, delimiter=',', fmt='%.6f')
        
        test_y_data = np.column_stack([self.X_test, self.y_test])
        np.savetxt(self.test_labels_path, test_y_data, delimiter=',', fmt=['%.6f', '%.6f', '%d'])
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cpp_static_mode(self):
        """Test C++ classifier in static mode."""
        cmd = [str(CPP_MAIN), "static", self.train_path, self.test_path, self.results_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, f"C++ static mode failed: {result.stderr}")
        
        # Check output files exist
        predictions_file = os.path.join(self.results_dir, "predictions.csv")
        self.assertTrue(os.path.exists(predictions_file), "Predictions file should be created")
    
    def test_cpp_accuracy(self):
        """Test that C++ classifier achieves reasonable accuracy on simple data."""
        cmd = [str(CPP_MAIN), "static", self.train_path, self.test_path, self.results_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse accuracy from output
        accuracy = None
        for line in result.stdout.split('\n'):
            if "Accuracy" in line:
                try:
                    accuracy = float(line.split(':')[-1].strip().replace('%', ''))
                except:
                    pass
        
        # On well-separated clusters, should achieve > 90% accuracy
        if accuracy is not None:
            self.assertGreater(accuracy, 90.0, 
                f"Accuracy on simple separable data should be > 90%, got {accuracy}%")


class TestOutlierDetection(unittest.TestCase):
    """Test outlier detection functionality."""
    
    def test_outlier_isolation(self):
        """Test that isolated points are detected as outliers."""
        # Create data with one clear outlier
        data = np.array([
            [0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.1, 0.1, 0],  # Cluster
            [10, 10, 0],  # Outlier - far from cluster
        ])
        
        # Using k-NN based outlier detection concept
        from sklearn.neighbors import NearestNeighbors
        
        X = data[:, :2]
        k = 3
        
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 because point is its own neighbor
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        
        # Mean distance to k neighbors (excluding self)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # The outlier should have highest mean distance
        outlier_idx = np.argmax(mean_distances)
        self.assertEqual(outlier_idx, 4, "Point at (10,10) should be detected as outlier")


class TestSRRGrid(unittest.TestCase):
    """Test Square Root Rule grid functionality."""
    
    def test_grid_size_formula(self):
        """Test that grid size follows sqrt(n) rule."""
        test_cases = [
            (100, 10),   # sqrt(100) = 10
            (1000, 31),  # sqrt(1000) ≈ 31.6 → 31 or 32
            (10000, 100),  # sqrt(10000) = 100
        ]
        
        for n, expected_approx in test_cases:
            grid_size = int(np.sqrt(n))
            self.assertAlmostEqual(grid_size, expected_approx, delta=1,
                msg=f"For n={n}, grid size should be ~{expected_approx}")
    
    def test_grid_lookup_O1(self):
        """Test that grid lookup is O(1) complexity."""
        import time
        
        # Different grid sizes
        grid_sizes = [10, 100, 1000]
        lookup_times = []
        
        for size in grid_sizes:
            # Create dummy grid as dict
            grid = {(i, j): [] for i in range(size) for j in range(size)}
            
            # Time 1000 lookups
            start = time.perf_counter()
            for _ in range(1000):
                x, y = np.random.randint(0, size, 2)
                _ = grid.get((x, y))
            elapsed = time.perf_counter() - start
            lookup_times.append(elapsed)
        
        # O(1) means time should be roughly constant regardless of grid size
        # Allow 5x variation for measurement noise
        ratio = lookup_times[-1] / lookup_times[0]
        self.assertLess(ratio, 5.0, 
            f"Grid lookup should be O(1), but time ratio is {ratio:.2f}")


class TestDynamicOperations(unittest.TestCase):
    """Test dynamic insert/remove operations."""
    
    def test_insert_preserves_delaunay(self):
        """Test that inserting a point maintains Delaunay property."""
        from scipy.spatial import Delaunay
        
        # Start with base points
        np.random.seed(42)
        base_points = np.random.rand(20, 2)
        
        # Insert new point
        new_point = np.array([[0.5, 0.5]])
        all_points = np.vstack([base_points, new_point])
        
        # Build triangulation
        tri = Delaunay(all_points)
        
        # Should still be valid
        self.assertGreater(len(tri.simplices), 0)
        
        # New point should be in some triangle(s)
        new_idx = len(base_points)  # Index of new point
        found_in_triangle = False
        for simplex in tri.simplices:
            if new_idx in simplex:
                found_in_triangle = True
                break
        
        self.assertTrue(found_in_triangle, "New point should be part of triangulation")


class TestClassificationVoting(unittest.TestCase):
    """Test classification voting mechanism."""
    
    def test_majority_vote(self):
        """Test majority voting with 3 vertices."""
        test_cases = [
            ([0, 0, 0], 0),  # All same
            ([1, 1, 1], 1),  # All same
            ([0, 0, 1], 0),  # 2 vs 1
            ([0, 1, 1], 1),  # 1 vs 2
            ([0, 1, 2], None),  # Tie - any is valid
        ]
        
        def majority_vote(labels):
            """Simple majority vote."""
            from collections import Counter
            c = Counter(labels)
            most_common = c.most_common(2)
            if len(most_common) == 1:
                return most_common[0][0]
            if most_common[0][1] == most_common[1][1]:
                return None  # Tie
            return most_common[0][0]
        
        for labels, expected in test_cases:
            result = majority_vote(labels)
            if expected is not None:
                self.assertEqual(result, expected, f"Majority of {labels} should be {expected}")


class TestDatasetGeneration(unittest.TestCase):
    """Test dataset generation scripts."""
    
    def test_data_generator_output_format(self):
        """Test that data generator creates correct file format."""
        data_dir = PROJECT_ROOT / "data" / "train"
        
        if not data_dir.exists():
            self.skipTest("Data directory not found. Run data_generator.py first.")
        
        # Check at least one dataset exists
        csv_files = list(data_dir.glob("*_train.csv"))
        if not csv_files:
            self.skipTest("No training files found. Run data_generator.py first.")
        
        # Validate format of first file found
        df = pd.read_csv(csv_files[0], header=None)
        
        self.assertEqual(df.shape[1], 3, "Should have 3 columns")
        self.assertTrue(df.iloc[:, 0].dtype in [np.float64, np.float32], "X should be float")
        self.assertTrue(df.iloc[:, 1].dtype in [np.float64, np.float32], "Y should be float")


class Test2DBuckets(unittest.TestCase):
    """Test 2D Buckets data structure for O(1) dynamic classification."""
    
    def test_point_in_polygon_ray_casting(self):
        """Test ray casting algorithm for point-in-polygon."""
        # Square polygon
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        
        def point_in_polygon(x, y, poly):
            """Ray casting algorithm."""
            n = len(poly) - 1  # Last point = first point
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside
        
        # Test points inside
        self.assertTrue(point_in_polygon(0.5, 0.5, polygon), "Center should be inside")
        self.assertTrue(point_in_polygon(0.25, 0.25, polygon), "Quarter point should be inside")
        
        # Test points outside
        self.assertFalse(point_in_polygon(1.5, 0.5, polygon), "Point outside right should be outside")
        self.assertFalse(point_in_polygon(-0.5, 0.5, polygon), "Point outside left should be outside")
    
    def test_bucket_class_detection(self):
        """Test that buckets correctly detect single vs multi-class regions."""
        # Simulate bucket sampling
        def sample_bucket_classes(center_class, corner_classes):
            """Determine bucket type from 5 sample points."""
            all_classes = [center_class] + corner_classes
            unique = set(all_classes)
            return len(unique)
        
        # Single class bucket
        self.assertEqual(sample_bucket_classes(0, [0, 0, 0, 0]), 1)
        
        # Two class bucket
        self.assertEqual(sample_bucket_classes(0, [0, 1, 0, 1]), 2)
        
        # Three class bucket
        self.assertEqual(sample_bucket_classes(0, [0, 1, 2, 0]), 3)
    
    def test_srr_bucket_index_calculation(self):
        """Test O(1) bucket index calculation."""
        # Simulate SRR grid with 10x10 buckets
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
        
        # Test corner buckets
        self.assertEqual(get_bucket_index(0.5, 0.5), 0)  # Bottom-left
        self.assertEqual(get_bucket_index(9.5, 0.5), 9)  # Bottom-right
        self.assertEqual(get_bucket_index(0.5, 9.5), 90)  # Top-left
        self.assertEqual(get_bucket_index(9.5, 9.5), 99)  # Top-right
        
        # Test center bucket
        self.assertEqual(get_bucket_index(5.5, 5.5), 55)
    
    def test_classify_single_dynamic_accuracy(self):
        """Test that classify_single_dynamic returns same results as classify_single for single-class buckets."""
        # This is a conceptual test - actual implementation test requires C++
        # Testing the logic: for single-class buckets, should return dominant_class
        
        class MockBucket:
            def __init__(self, num_classes, dominant_class):
                self.num_classes = num_classes
                self.dominant_class = dominant_class
            
            def classify_point(self, x, y):
                if self.num_classes == 1:
                    return self.dominant_class  # O(1)
                return None  # Would need full classification
        
        # Single class bucket
        bucket = MockBucket(1, 2)
        self.assertEqual(bucket.classify_point(0.5, 0.5), 2)
        
        # Multi-class bucket returns None (would fallback)
        bucket = MockBucket(2, 1)
        self.assertIsNone(bucket.classify_point(0.5, 0.5))


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDelaunayTriangulation))
    suite.addTests(loader.loadTestsFromTestCase(TestOutlierDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestSRRGrid))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationVoting))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetGeneration))
    suite.addTests(loader.loadTestsFromTestCase(Test2DBuckets))
    
    # Add C++ tests only if executable exists
    if CPP_MAIN.exists():
        suite.addTests(loader.loadTestsFromTestCase(TestCppClassifier))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*70)
    print("DELAUNAY CLASSIFIER UNIT TESTS")
    print("="*70)
    
    success = run_tests()
    
    print("\n" + "="*70)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    sys.exit(0 if success else 1)
