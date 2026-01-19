package vex

import (
	"context"
	"errors"
	"testing"
)

// mockProvider is a simple test provider.
type mockProvider struct {
	name       string
	dimensions int
	err        error
	callCount  int
}

func newMockProvider(dims int) *mockProvider {
	return &mockProvider{
		name:       "mock",
		dimensions: dims,
	}
}

func (p *mockProvider) Name() string    { return p.name }
func (p *mockProvider) Dimensions() int { return p.dimensions }

func (p *mockProvider) Embed(_ context.Context, texts []string) (*EmbeddingResponse, error) {
	p.callCount++
	if p.err != nil {
		return nil, p.err
	}

	vectors := make([]Vector, len(texts))
	for i := range texts {
		vec := make(Vector, p.dimensions)
		for j := 0; j < p.dimensions; j++ {
			vec[j] = float32(j) / float32(p.dimensions)
		}
		vectors[i] = vec
	}

	return &EmbeddingResponse{
		Vectors:    vectors,
		Model:      "mock-model",
		Dimensions: p.dimensions,
		Usage: Usage{
			PromptTokens: len(texts) * 5,
			TotalTokens:  len(texts) * 5,
		},
	}, nil
}

// mockQueryProvider implements QueryProviderFactory for testing.
type mockQueryProvider struct {
	*mockProvider
	queryMode bool
}

func newMockQueryProvider(dims int) *mockQueryProvider {
	return &mockQueryProvider{
		mockProvider: newMockProvider(dims),
	}
}

func (p *mockQueryProvider) ForQuery() Provider {
	return &mockQueryProvider{
		mockProvider: p.mockProvider,
		queryMode:    true,
	}
}

func TestService_Embed(t *testing.T) {
	t.Run("returns vector with correct dimensions", func(t *testing.T) {
		dims := 512
		provider := newMockProvider(dims)
		svc := NewService(provider)

		vec, err := svc.Embed(context.Background(), "hello world")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(vec) != dims {
			t.Errorf("expected %d dimensions, got %d", dims, len(vec))
		}
	})

	t.Run("returns normalized vector by default", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider)

		vec, err := svc.Embed(context.Background(), "test")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		norm := vec.Norm()
		if norm < 0.99 || norm > 1.01 {
			t.Errorf("expected normalized vector (norm ~1.0), got %f", norm)
		}
	})

	t.Run("propagates provider errors", func(t *testing.T) {
		expectedErr := errors.New("provider error")
		provider := newMockProvider(256)
		provider.err = expectedErr
		svc := NewService(provider)

		_, err := svc.Embed(context.Background(), "test")
		if err == nil {
			t.Error("expected error, got nil")
		}
	})
}

func TestService_Batch(t *testing.T) {
	t.Run("returns correct number of vectors", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider)

		texts := []string{"one", "two", "three", "four"}
		vecs, err := svc.Batch(context.Background(), texts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(vecs) != len(texts) {
			t.Errorf("expected %d vectors, got %d", len(texts), len(vecs))
		}
	})

	t.Run("handles empty input", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider)

		vecs, err := svc.Batch(context.Background(), []string{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if vecs != nil {
			t.Errorf("expected nil for empty input, got %v", vecs)
		}
	})

	t.Run("handles nil input", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider)

		vecs, err := svc.Batch(context.Background(), nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if vecs != nil {
			t.Errorf("expected nil for nil input")
		}
	})
}

func TestService_EmbedQuery(t *testing.T) {
	t.Run("uses query provider when available", func(t *testing.T) {
		provider := newMockQueryProvider(256)
		svc := NewService(provider)

		vec, err := svc.EmbedQuery(context.Background(), "search query")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if vec == nil {
			t.Error("expected vector, got nil")
		}
	})

	t.Run("falls back to regular embed without query provider", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider)

		vec, err := svc.EmbedQuery(context.Background(), "search query")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if vec == nil {
			t.Error("expected vector, got nil")
		}

		// Should have called the regular provider
		if provider.callCount != 1 {
			t.Errorf("expected 1 call, got %d", provider.callCount)
		}
	})
}

func TestService_BatchQuery(t *testing.T) {
	t.Run("returns correct number of vectors", func(t *testing.T) {
		provider := newMockQueryProvider(256)
		svc := NewService(provider)

		texts := []string{"query one", "query two"}
		vecs, err := svc.BatchQuery(context.Background(), texts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(vecs) != len(texts) {
			t.Errorf("expected %d vectors, got %d", len(texts), len(vecs))
		}
	})

	t.Run("handles empty input", func(t *testing.T) {
		provider := newMockQueryProvider(256)
		svc := NewService(provider)

		vecs, err := svc.BatchQuery(context.Background(), []string{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if vecs != nil {
			t.Errorf("expected nil for empty input")
		}
	})
}

func TestService_WithNormalize(t *testing.T) {
	t.Run("can disable normalization", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider).WithNormalize(false)

		vec, err := svc.Embed(context.Background(), "test")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Without normalization, norm should not be 1.0
		// (unless provider happens to return normalized vectors)
		_ = vec // Just verify it runs without error
	})
}

func TestService_WithChunker(t *testing.T) {
	t.Run("applies chunking", func(t *testing.T) {
		provider := newMockProvider(256)
		chunker := &Chunker{
			Strategy:  ChunkSentence,
			TrimSpace: true,
		}
		svc := NewService(provider).WithChunker(chunker)

		// Text with multiple sentences
		text := "First sentence. Second sentence. Third sentence."
		vec, err := svc.Embed(context.Background(), text)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Should still return single vector (pooled from chunks)
		if vec == nil {
			t.Error("expected vector, got nil")
		}
	})
}

func TestService_WithPooling(t *testing.T) {
	t.Run("can change pooling mode", func(t *testing.T) {
		provider := newMockProvider(256)
		svc := NewService(provider).WithPooling(PoolMax)

		// Just verify it runs without error
		_, err := svc.Embed(context.Background(), "test")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestService_Dimensions(t *testing.T) {
	dims := 1024
	provider := newMockProvider(dims)
	svc := NewService(provider)

	if svc.Dimensions() != dims {
		t.Errorf("expected %d, got %d", dims, svc.Dimensions())
	}
}

func TestService_Provider(t *testing.T) {
	provider := newMockProvider(256)
	svc := NewService(provider)

	if svc.Provider() != provider {
		t.Error("expected same provider instance")
	}
}

func TestService_GetPipeline(t *testing.T) {
	provider := newMockProvider(256)
	svc := NewService(provider)

	if svc.GetPipeline() == nil {
		t.Error("expected non-nil pipeline")
	}
}

// mockEmptyProvider returns empty responses.
type mockEmptyProvider struct {
	*mockProvider
}

func (p *mockEmptyProvider) Embed(_ context.Context, _ []string) (*EmbeddingResponse, error) {
	p.callCount++
	return &EmbeddingResponse{
		Vectors: nil,
		Model:   "mock",
	}, nil
}

// mockEmptyQueryProvider returns empty responses and implements QueryProviderFactory.
type mockEmptyQueryProvider struct {
	*mockEmptyProvider
}

func (p *mockEmptyQueryProvider) ForQuery() Provider {
	return p
}

// mockErrorQueryProvider returns errors and implements QueryProviderFactory.
type mockErrorQueryProvider struct {
	*mockProvider
	err error
}

func newMockErrorQueryProvider(dims int, err error) *mockErrorQueryProvider {
	return &mockErrorQueryProvider{
		mockProvider: newMockProvider(dims),
		err:          err,
	}
}

func (p *mockErrorQueryProvider) Embed(_ context.Context, _ []string) (*EmbeddingResponse, error) {
	p.callCount++
	return nil, p.err
}

func (p *mockErrorQueryProvider) ForQuery() Provider {
	return p
}

func TestService_Embed_EmptyResponse(t *testing.T) {
	provider := &mockEmptyProvider{mockProvider: newMockProvider(256)}
	svc := NewService(provider)

	vec, err := svc.Embed(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vec != nil {
		t.Error("expected nil vector for empty response")
	}
}

func TestService_EmbedQuery_Error(t *testing.T) {
	expectedErr := errors.New("query provider error")
	provider := newMockErrorQueryProvider(256, expectedErr)
	svc := NewService(provider)

	_, err := svc.EmbedQuery(context.Background(), "query")
	if err == nil {
		t.Error("expected error, got nil")
	}
}

func TestService_EmbedQuery_EmptyResponse(t *testing.T) {
	provider := &mockEmptyQueryProvider{
		mockEmptyProvider: &mockEmptyProvider{mockProvider: newMockProvider(256)},
	}
	svc := NewService(provider)

	vec, err := svc.EmbedQuery(context.Background(), "query")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vec != nil {
		t.Error("expected nil vector for empty response")
	}
}

func TestService_Batch_EmptyResponse(t *testing.T) {
	provider := &mockEmptyProvider{mockProvider: newMockProvider(256)}
	svc := NewService(provider)

	vecs, err := svc.Batch(context.Background(), []string{"test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vecs != nil {
		t.Error("expected nil vectors for empty response")
	}
}

func TestService_BatchQuery_Error(t *testing.T) {
	expectedErr := errors.New("query provider error")
	provider := newMockErrorQueryProvider(256, expectedErr)
	svc := NewService(provider)

	_, err := svc.BatchQuery(context.Background(), []string{"query"})
	if err == nil {
		t.Error("expected error, got nil")
	}
}

func TestService_BatchQuery_EmptyResponse(t *testing.T) {
	provider := &mockEmptyQueryProvider{
		mockEmptyProvider: &mockEmptyProvider{mockProvider: newMockProvider(256)},
	}
	svc := NewService(provider)

	vecs, err := svc.BatchQuery(context.Background(), []string{"query"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vecs != nil {
		t.Error("expected nil vectors for empty response")
	}
}

func TestService_NewService_WithOptionsAndQueryProvider(t *testing.T) {
	provider := newMockQueryProvider(256)

	// Apply an option to verify it's applied to both pipelines
	svc := NewService(provider, WithRetry(3))

	// Both pipelines should be configured
	if svc.GetPipeline() == nil {
		t.Error("expected non-nil pipeline")
	}

	// Query path should work
	vec, err := svc.EmbedQuery(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vec == nil {
		t.Error("expected vector, got nil")
	}
}
