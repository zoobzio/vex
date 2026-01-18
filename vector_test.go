package vex

import (
	"math"
	"testing"
)

func TestVector_Normalize(t *testing.T) {
	t.Run("normalizes to unit length", func(t *testing.T) {
		vec := Vector{3, 4}
		normalized := vec.Normalize()

		norm := normalized.Norm()
		if math.Abs(norm-1.0) > 0.0001 {
			t.Errorf("expected norm 1.0, got %f", norm)
		}
	})

	t.Run("handles zero vector", func(t *testing.T) {
		vec := Vector{0, 0, 0}
		normalized := vec.Normalize()

		// Zero vector should remain zero
		for i, v := range normalized {
			if v != 0 {
				t.Errorf("expected 0 at index %d, got %f", i, v)
			}
		}
	})

	t.Run("preserves direction", func(t *testing.T) {
		vec := Vector{1, 2, 3}
		normalized := vec.Normalize()

		// Ratios should be preserved
		ratio1 := vec[1] / vec[0]
		ratio2 := normalized[1] / normalized[0]

		if math.Abs(ratio1-ratio2) > 0.0001 {
			t.Errorf("direction not preserved: original ratio %f, normalized ratio %f", ratio1, ratio2)
		}
	})
}

func TestVector_Norm(t *testing.T) {
	t.Run("calculates correct L2 norm", func(t *testing.T) {
		vec := Vector{3, 4}
		expected := 5.0

		if vec.Norm() != expected {
			t.Errorf("expected norm %f, got %f", expected, vec.Norm())
		}
	})

	t.Run("handles empty vector", func(t *testing.T) {
		vec := Vector{}
		if vec.Norm() != 0 {
			t.Errorf("expected norm 0 for empty vector, got %f", vec.Norm())
		}
	})
}

func TestVector_Dot(t *testing.T) {
	t.Run("calculates correct dot product", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{4, 5, 6}
		expected := 32.0 // 1*4 + 2*5 + 3*6

		if v1.Dot(v2) != expected {
			t.Errorf("expected dot product %f, got %f", expected, v1.Dot(v2))
		}
	})

	t.Run("returns 0 for mismatched dimensions", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{4, 5}

		if v1.Dot(v2) != 0 {
			t.Errorf("expected 0 for mismatched dimensions, got %f", v1.Dot(v2))
		}
	})

	t.Run("is commutative", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{4, 5, 6}

		if v1.Dot(v2) != v2.Dot(v1) {
			t.Errorf("dot product should be commutative")
		}
	})
}

func TestVector_CosineSimilarity(t *testing.T) {
	t.Run("identical vectors have similarity 1", func(t *testing.T) {
		vec := Vector{1, 2, 3}.Normalize()

		sim := vec.CosineSimilarity(vec)
		if math.Abs(sim-1.0) > 0.0001 {
			t.Errorf("expected similarity 1.0 for identical vectors, got %f", sim)
		}
	})

	t.Run("orthogonal vectors have similarity 0", func(t *testing.T) {
		v1 := Vector{1, 0}
		v2 := Vector{0, 1}

		sim := v1.CosineSimilarity(v2)
		if math.Abs(sim) > 0.0001 {
			t.Errorf("expected similarity 0 for orthogonal vectors, got %f", sim)
		}
	})

	t.Run("opposite vectors have similarity -1", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{-1, -2, -3}

		sim := v1.CosineSimilarity(v2)
		if math.Abs(sim-(-1.0)) > 0.0001 {
			t.Errorf("expected similarity -1.0 for opposite vectors, got %f", sim)
		}
	})

	t.Run("returns 0 for zero vectors", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{0, 0, 0}

		sim := v1.CosineSimilarity(v2)
		if sim != 0 {
			t.Errorf("expected 0 for zero vector, got %f", sim)
		}
	})

	t.Run("returns 0 for mismatched dimensions", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{4, 5}

		if v1.CosineSimilarity(v2) != 0 {
			t.Errorf("expected 0 for mismatched dimensions")
		}
	})
}

func TestVector_EuclideanDistance(t *testing.T) {
	t.Run("calculates correct distance", func(t *testing.T) {
		v1 := Vector{0, 0}
		v2 := Vector{3, 4}
		expected := 5.0

		dist := v1.EuclideanDistance(v2)
		if math.Abs(dist-expected) > 0.0001 {
			t.Errorf("expected distance %f, got %f", expected, dist)
		}
	})

	t.Run("distance to self is zero", func(t *testing.T) {
		vec := Vector{1, 2, 3}
		if vec.EuclideanDistance(vec) != 0 {
			t.Errorf("expected distance 0 to self")
		}
	})

	t.Run("is symmetric", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{4, 5, 6}

		if v1.EuclideanDistance(v2) != v2.EuclideanDistance(v1) {
			t.Errorf("euclidean distance should be symmetric")
		}
	})

	t.Run("returns MaxFloat64 for mismatched dimensions", func(t *testing.T) {
		v1 := Vector{1, 2, 3}
		v2 := Vector{4, 5}

		if v1.EuclideanDistance(v2) != math.MaxFloat64 {
			t.Errorf("expected MaxFloat64 for mismatched dimensions")
		}
	})
}

func TestVector_Similarity(t *testing.T) {
	v1 := Vector{1, 2, 3}.Normalize()
	v2 := Vector{4, 5, 6}.Normalize()

	t.Run("cosine metric", func(t *testing.T) {
		expected := v1.CosineSimilarity(v2)
		actual := v1.Similarity(v2, Cosine)
		if actual != expected {
			t.Errorf("expected %f, got %f", expected, actual)
		}
	})

	t.Run("dot product metric", func(t *testing.T) {
		expected := v1.Dot(v2)
		actual := v1.Similarity(v2, DotProduct)
		if actual != expected {
			t.Errorf("expected %f, got %f", expected, actual)
		}
	})

	t.Run("euclidean metric", func(t *testing.T) {
		dist := v1.EuclideanDistance(v2)
		expected := 1 / (1 + dist)
		actual := v1.Similarity(v2, Euclidean)
		if math.Abs(actual-expected) > 0.0001 {
			t.Errorf("expected %f, got %f", expected, actual)
		}
	})
}

func TestPool(t *testing.T) {
	t.Run("returns nil for empty input", func(t *testing.T) {
		result := Pool([]Vector{}, PoolMean)
		if result != nil {
			t.Errorf("expected nil for empty input")
		}
	})

	t.Run("returns single vector unchanged", func(t *testing.T) {
		vec := Vector{1, 2, 3}
		result := Pool([]Vector{vec}, PoolMean)

		for i := range vec {
			if result[i] != vec[i] {
				t.Errorf("single vector should be returned unchanged")
				break
			}
		}
	})

	t.Run("PoolMean averages vectors", func(t *testing.T) {
		vectors := []Vector{
			{0, 2, 4},
			{2, 4, 6},
		}
		expected := Vector{1, 3, 5}

		result := Pool(vectors, PoolMean)
		for i := range expected {
			if result[i] != expected[i] {
				t.Errorf("at index %d: expected %f, got %f", i, expected[i], result[i])
			}
		}
	})

	t.Run("PoolMax takes element-wise maximum", func(t *testing.T) {
		vectors := []Vector{
			{1, 5, 3},
			{4, 2, 6},
		}
		expected := Vector{4, 5, 6}

		result := Pool(vectors, PoolMax)
		for i := range expected {
			if result[i] != expected[i] {
				t.Errorf("at index %d: expected %f, got %f", i, expected[i], result[i])
			}
		}
	})

	t.Run("PoolFirst returns first vector", func(t *testing.T) {
		vectors := []Vector{
			{1, 2, 3},
			{4, 5, 6},
		}

		result := Pool(vectors, PoolFirst)
		for i := range vectors[0] {
			if result[i] != vectors[0][i] {
				t.Errorf("expected first vector")
				break
			}
		}
	})
}
