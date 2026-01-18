package vex

import "math"

// Normalize returns a unit vector (L2 normalized).
func (v Vector) Normalize() Vector {
	norm := v.Norm()
	if norm == 0 {
		return v
	}
	result := make(Vector, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result
}

// Norm returns the L2 norm (magnitude) of the vector.
func (v Vector) Norm() float64 {
	var sum float64
	for _, val := range v {
		sum += val * val
	}
	return math.Sqrt(sum)
}

// Dot computes the dot product with another vector.
func (v Vector) Dot(other Vector) float64 {
	if len(v) != len(other) {
		return 0
	}
	var sum float64
	for i := range v {
		sum += v[i] * other[i]
	}
	return sum
}

// CosineSimilarity computes cosine similarity with another vector.
// Returns value in range [-1, 1], where 1 means identical direction.
func (v Vector) CosineSimilarity(other Vector) float64 {
	if len(v) != len(other) {
		return 0
	}
	dot := v.Dot(other)
	normA := v.Norm()
	normB := other.Norm()
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (normA * normB)
}

// EuclideanDistance computes the Euclidean distance to another vector.
func (v Vector) EuclideanDistance(other Vector) float64 {
	if len(v) != len(other) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range v {
		diff := v[i] - other[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Similarity computes similarity using the specified metric.
func (v Vector) Similarity(other Vector, metric SimilarityMetric) float64 {
	switch metric {
	case Cosine:
		return v.CosineSimilarity(other)
	case DotProduct:
		return v.Dot(other)
	case Euclidean:
		// Convert distance to similarity (smaller distance = higher similarity)
		dist := v.EuclideanDistance(other)
		return 1 / (1 + dist)
	default:
		return v.CosineSimilarity(other)
	}
}

// Pool combines multiple vectors using the specified pooling mode.
func Pool(vectors []Vector, mode PoolingMode) Vector {
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors) == 1 {
		return vectors[0]
	}

	switch mode {
	case PoolFirst:
		return vectors[0]
	case PoolMax:
		return poolMax(vectors)
	case PoolMean:
		return poolMean(vectors)
	default:
		return poolMean(vectors)
	}
}

func poolMean(vectors []Vector) Vector {
	dims := len(vectors[0])
	result := make(Vector, dims)
	for _, vec := range vectors {
		for i, val := range vec {
			result[i] += val
		}
	}
	n := float64(len(vectors))
	for i := range result {
		result[i] /= n
	}
	return result
}

func poolMax(vectors []Vector) Vector {
	dims := len(vectors[0])
	result := make(Vector, dims)
	copy(result, vectors[0])
	for _, vec := range vectors[1:] {
		for i, val := range vec {
			if val > result[i] {
				result[i] = val
			}
		}
	}
	return result
}
