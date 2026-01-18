// Package testing provides test utilities for vex.
package testing

import (
	"context"
	"crypto/sha256"
	"math"
	"testing"

	"github.com/zoobzio/vex"
)

// MockProvider implements vex.Provider for testing.
type MockProvider struct {
	err           error
	name          string
	dimensions    int
	failAfter     int
	callCount     int
	deterministic bool
}

// MockConfig configures a MockProvider.
type MockConfig struct {
	Error         error
	Name          string
	Dimensions    int
	FailAfter     int
	Deterministic bool
}

// NewMockProvider creates a new mock provider.
func NewMockProvider(config MockConfig) *MockProvider {
	if config.Name == "" {
		config.Name = "mock"
	}
	if config.Dimensions == 0 {
		config.Dimensions = 1536
	}
	return &MockProvider{
		name:          config.Name,
		dimensions:    config.Dimensions,
		deterministic: config.Deterministic,
		failAfter:     config.FailAfter,
		err:           config.Error,
	}
}

// Name returns the provider name.
func (p *MockProvider) Name() string {
	return p.name
}

// Dimensions returns the output dimensionality.
func (p *MockProvider) Dimensions() int {
	return p.dimensions
}

// Embed generates mock embeddings.
func (p *MockProvider) Embed(_ context.Context, texts []string) (*vex.EmbeddingResponse, error) {
	p.callCount++

	if p.err != nil {
		return nil, p.err
	}

	if p.failAfter > 0 && p.callCount > p.failAfter {
		return nil, p.err
	}

	vectors := make([]vex.Vector, len(texts))
	for i, text := range texts {
		vectors[i] = p.generateVector(text)
	}

	return &vex.EmbeddingResponse{
		Vectors:    vectors,
		Model:      "mock-model",
		Dimensions: p.dimensions,
		Usage: vex.Usage{
			PromptTokens: len(texts) * 10,
			TotalTokens:  len(texts) * 10,
		},
	}, nil
}

// CallCount returns the number of Embed calls.
func (p *MockProvider) CallCount() int {
	return p.callCount
}

// Reset resets the call counter.
func (p *MockProvider) Reset() {
	p.callCount = 0
}

func (p *MockProvider) generateVector(text string) vex.Vector {
	vec := make(vex.Vector, p.dimensions)

	if p.deterministic {
		// Generate deterministic vector from text hash
		hash := sha256.Sum256([]byte(text))
		for i := 0; i < p.dimensions; i++ {
			// Use hash bytes to seed values
			idx := i % 32
			val := float64(hash[idx]) / 255.0
			vec[i] = val*2 - 1 // Range [-1, 1]
		}
	} else {
		// Generate pseudo-random vector
		for i := 0; i < p.dimensions; i++ {
			vec[i] = float64(i%100) / 100.0
		}
	}

	return vec.Normalize()
}

// AssertVectorDimensions checks vector has expected dimensions.
func AssertVectorDimensions(t *testing.T, vec vex.Vector, expected int) {
	t.Helper()
	if len(vec) != expected {
		t.Errorf("expected %d dimensions, got %d", expected, len(vec))
	}
}

// AssertVectorNormalized checks vector is unit length.
func AssertVectorNormalized(t *testing.T, vec vex.Vector, tolerance float64) {
	t.Helper()
	norm := vec.Norm()
	if math.Abs(norm-1.0) > tolerance {
		t.Errorf("expected normalized vector (norm=1.0), got norm=%f", norm)
	}
}

// AssertSimilarityInRange checks similarity is within expected range.
func AssertSimilarityInRange(t *testing.T, similarity, minVal, maxVal float64) {
	t.Helper()
	if similarity < minVal || similarity > maxVal {
		t.Errorf("expected similarity in [%f, %f], got %f", minVal, maxVal, similarity)
	}
}

// GenerateTestVector creates a test vector with specified properties.
func GenerateTestVector(dimensions int, seed int64) vex.Vector {
	vec := make(vex.Vector, dimensions)
	for i := 0; i < dimensions; i++ {
		// Simple deterministic generation
		val := float64((seed+int64(i))%1000) / 1000.0
		vec[i] = val*2 - 1
	}
	return vec.Normalize()
}

// GenerateSimilarVectors creates two vectors with known similarity.
func GenerateSimilarVectors(dimensions int, targetSimilarity float64) (baseVec vex.Vector, similarVec vex.Vector) {
	// Start with a base vector
	baseVec = GenerateTestVector(dimensions, 42)

	// Create a similar vector by interpolating with noise
	noise := GenerateTestVector(dimensions, 123)

	similarVec = make(vex.Vector, dimensions)
	weight := math.Sqrt(targetSimilarity)
	noiseWeight := math.Sqrt(1 - targetSimilarity)

	for i := 0; i < dimensions; i++ {
		similarVec[i] = weight*baseVec[i] + noiseWeight*noise[i]
	}

	return baseVec, similarVec.Normalize()
}

