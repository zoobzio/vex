package testing

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestMockProvider_Embed(t *testing.T) {
	t.Run("returns correct number of vectors", func(t *testing.T) {
		provider := NewMockProvider(MockConfig{Dimensions: 512})
		texts := []string{"hello", "world", "foo"}

		resp, err := provider.Embed(context.Background(), texts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(resp.Vectors) != len(texts) {
			t.Errorf("expected %d vectors, got %d", len(texts), len(resp.Vectors))
		}
	})

	t.Run("returns correct dimensions", func(t *testing.T) {
		dims := 768
		provider := NewMockProvider(MockConfig{Dimensions: dims})

		resp, err := provider.Embed(context.Background(), []string{"test"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(resp.Vectors[0]) != dims {
			t.Errorf("expected %d dimensions, got %d", dims, len(resp.Vectors[0]))
		}
	})

	t.Run("deterministic mode produces consistent output", func(t *testing.T) {
		provider := NewMockProvider(MockConfig{
			Dimensions:    256,
			Deterministic: true,
		})

		resp1, err1 := provider.Embed(context.Background(), []string{"test input"})
		if err1 != nil {
			t.Fatalf("unexpected error: %v", err1)
		}
		resp2, err2 := provider.Embed(context.Background(), []string{"test input"})
		if err2 != nil {
			t.Fatalf("unexpected error: %v", err2)
		}

		for i := range resp1.Vectors[0] {
			if resp1.Vectors[0][i] != resp2.Vectors[0][i] {
				t.Errorf("deterministic mode produced different vectors")
				break
			}
		}
	})

	t.Run("returns error when configured", func(t *testing.T) {
		expectedErr := errors.New("mock error")
		provider := NewMockProvider(MockConfig{Error: expectedErr})

		_, err := provider.Embed(context.Background(), []string{"test"})
		if !errors.Is(err, expectedErr) {
			t.Errorf("expected error %v, got %v", expectedErr, err)
		}
	})

	t.Run("tracks call count", func(t *testing.T) {
		provider := NewMockProvider(MockConfig{})

		if provider.CallCount() != 0 {
			t.Errorf("expected 0 calls, got %d", provider.CallCount())
		}

		//nolint:errcheck // test helper
		provider.Embed(context.Background(), []string{"a"})
		//nolint:errcheck // test helper
		provider.Embed(context.Background(), []string{"b"})

		if provider.CallCount() != 2 {
			t.Errorf("expected 2 calls, got %d", provider.CallCount())
		}

		provider.Reset()
		if provider.CallCount() != 0 {
			t.Errorf("expected 0 calls after reset, got %d", provider.CallCount())
		}
	})
}

func TestGenerateTestVector(t *testing.T) {
	t.Run("produces normalized vectors", func(t *testing.T) {
		vec := GenerateTestVector(512, 42)

		norm := vec.Norm()
		if math.Abs(norm-1.0) > 0.0001 {
			t.Errorf("expected normalized vector, got norm=%f", norm)
		}
	})

	t.Run("same seed produces same vector", func(t *testing.T) {
		v1 := GenerateTestVector(256, 123)
		v2 := GenerateTestVector(256, 123)

		for i := range v1 {
			if v1[i] != v2[i] {
				t.Errorf("same seed produced different vectors")
				break
			}
		}
	})

	t.Run("different seeds produce different vectors", func(t *testing.T) {
		v1 := GenerateTestVector(256, 1)
		v2 := GenerateTestVector(256, 2)

		same := true
		for i := range v1 {
			if v1[i] != v2[i] {
				same = false
				break
			}
		}
		if same {
			t.Error("different seeds should produce different vectors")
		}
	})
}

func TestGenerateSimilarVectors(t *testing.T) {
	t.Run("produces two distinct vectors", func(t *testing.T) {
		v1, v2 := GenerateSimilarVectors(512, 0.8)

		// Vectors should be different
		same := true
		for i := range v1 {
			if v1[i] != v2[i] {
				same = false
				break
			}
		}
		if same {
			t.Error("expected distinct vectors")
		}

		// Both should be normalized
		if math.Abs(v1.Norm()-1.0) > 0.0001 {
			t.Errorf("v1 not normalized: %f", v1.Norm())
		}
		if math.Abs(v2.Norm()-1.0) > 0.0001 {
			t.Errorf("v2 not normalized: %f", v2.Norm())
		}
	})
}

func TestAssertHelpers(t *testing.T) {
	t.Run("AssertVectorDimensions passes for correct dimensions", func(t *testing.T) {
		vec := GenerateTestVector(512, 1)
		// This should not fail
		AssertVectorDimensions(t, vec, 512)
	})

	t.Run("AssertVectorNormalized passes for unit vectors", func(t *testing.T) {
		vec := GenerateTestVector(256, 1)
		// This should not fail
		AssertVectorNormalized(t, vec, 0.0001)
	})

	t.Run("AssertSimilarityInRange passes for valid range", func(t *testing.T) {
		// This should not fail
		AssertSimilarityInRange(t, 0.5, 0.0, 1.0)
	})
}
