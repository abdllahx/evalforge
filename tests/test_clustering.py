import numpy as np

from evalforge.classifier.clustering import representative_indices


def test_representative_returns_first_k_when_undersized():
    emb = np.eye(3, dtype=np.float32)
    out = representative_indices(emb, member_indices=[0, 1], k=5)
    assert out == [0, 1]


def test_representative_picks_closest_to_centroid():
    # Three vectors, two cluster around (1,0), one outlier at (0,1).
    emb = np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.92, 0.1]],
        dtype=np.float32,
    )
    out = representative_indices(emb, member_indices=[0, 1, 2, 3], k=2)
    # The two closest to centroid should be from the {0, 1, 3} cluster
    assert 2 not in out


def test_representative_safe_with_zero_centroid():
    # Vectors that sum to ~zero centroid (norm near 0)
    emb = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    out = representative_indices(emb, member_indices=[0, 1], k=5)
    # Just returns both — no crash, no NaN
    assert set(out) == {0, 1}
