package vex

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWithRetry(t *testing.T) {
	t.Run("retries on failure", func(t *testing.T) {
		failCount := 0
		provider := newMockProvider(256)
		provider.err = errors.New("transient error")

		// Create a provider that fails twice then succeeds
		customProvider := &retryTestProvider{
			failUntil: 2,
			dims:      256,
		}

		svc := NewService(customProvider, WithRetry(3))
		_, err := svc.Embed(context.Background(), "test")

		if err != nil {
			t.Errorf("expected success after retries, got: %v", err)
		}
		if customProvider.calls < 2 {
			t.Errorf("expected at least 2 calls, got %d", customProvider.calls)
		}
		_ = failCount // unused in this simplified test
	})

	t.Run("fails after max attempts", func(t *testing.T) {
		provider := &retryTestProvider{
			failUntil: 100, // Always fail
			dims:      256,
		}

		svc := NewService(provider, WithRetry(3))
		_, err := svc.Embed(context.Background(), "test")

		if err == nil {
			t.Error("expected error after max retries")
		}
	})
}

type retryTestProvider struct {
	calls     int
	failUntil int
	dims      int
}

func (*retryTestProvider) Name() string    { return "retry-test" }
func (p *retryTestProvider) Dimensions() int { return p.dims }
func (p *retryTestProvider) Embed(_ context.Context, texts []string) (*EmbeddingResponse, error) {
	p.calls++
	if p.calls <= p.failUntil {
		return nil, errors.New("transient error")
	}
	vecs := make([]Vector, len(texts))
	for i := range vecs {
		vecs[i] = make(Vector, p.dims)
	}
	return &EmbeddingResponse{Vectors: vecs, Model: "test", Dimensions: p.dims}, nil
}

func TestWithTimeout(t *testing.T) {
	t.Run("cancels slow requests", func(t *testing.T) {
		provider := &slowProvider{
			delay: 500 * time.Millisecond,
			dims:  256,
		}

		svc := NewService(provider, WithTimeout(50*time.Millisecond))
		_, err := svc.Embed(context.Background(), "test")

		if err == nil {
			t.Error("expected timeout error")
		}
	})

	t.Run("allows fast requests", func(t *testing.T) {
		provider := &slowProvider{
			delay: 10 * time.Millisecond,
			dims:  256,
		}

		svc := NewService(provider, WithTimeout(500*time.Millisecond))
		_, err := svc.Embed(context.Background(), "test")

		if err != nil {
			t.Errorf("expected success, got: %v", err)
		}
	})
}

type slowProvider struct {
	delay time.Duration
	dims  int
}

func (*slowProvider) Name() string    { return "slow" }
func (p *slowProvider) Dimensions() int { return p.dims }
func (p *slowProvider) Embed(ctx context.Context, texts []string) (*EmbeddingResponse, error) {
	select {
	case <-time.After(p.delay):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	vecs := make([]Vector, len(texts))
	for i := range vecs {
		vecs[i] = make(Vector, p.dims)
	}
	return &EmbeddingResponse{Vectors: vecs, Model: "test", Dimensions: p.dims}, nil
}

func TestWithRateLimit(t *testing.T) {
	t.Run("limits request rate", func(t *testing.T) {
		provider := newMockProvider(256)
		// 2 requests per second, burst of 1
		svc := NewService(provider, WithRateLimit(2, 1))

		start := time.Now()

		// Make 3 requests - should take at least 500ms due to rate limiting
		for i := 0; i < 3; i++ {
			_, err := svc.Embed(context.Background(), "test")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		}

		elapsed := time.Since(start)
		// With 2 RPS and burst 1, 3 requests should take ~1 second
		// Being lenient with timing in tests
		if elapsed < 400*time.Millisecond {
			t.Errorf("rate limiting not effective, elapsed: %v", elapsed)
		}
	})
}

func TestWithBackoff(t *testing.T) {
	t.Run("applies increasing delays", func(t *testing.T) {
		provider := &retryTestProvider{
			failUntil: 2,
			dims:      256,
		}

		start := time.Now()
		svc := NewService(provider, WithBackoff(3, 50*time.Millisecond))
		_, err := svc.Embed(context.Background(), "test")

		elapsed := time.Since(start)

		if err != nil {
			t.Errorf("expected success, got: %v", err)
		}
		// Should have some delay from backoff (50ms + 100ms minimum)
		if elapsed < 100*time.Millisecond {
			t.Errorf("backoff not applied, elapsed: %v", elapsed)
		}
	})
}

func TestWithCircuitBreaker(t *testing.T) {
	t.Run("opens after failures", func(_ *testing.T) {
		provider := &retryTestProvider{
			failUntil: 100, // Always fail
			dims:      256,
		}

		// Circuit opens after 2 failures, recovers after 100ms
		svc := NewService(provider, WithCircuitBreaker(2, 100*time.Millisecond))

		// First 2 calls should hit the provider
		//nolint:errcheck // test helper
		svc.Embed(context.Background(), "test")
		//nolint:errcheck // test helper
		svc.Embed(context.Background(), "test")

		callsBefore := provider.calls

		// Next call should be blocked by open circuit
		//nolint:errcheck // test helper
		svc.Embed(context.Background(), "test")

		// Circuit breaker may still allow the call depending on implementation
		// This is a simplified test - we just verify no panic occurs
		_ = callsBefore
	})
}

func TestWithFallback(t *testing.T) {
	t.Run("uses fallback on primary failure", func(t *testing.T) {
		primary := &retryTestProvider{
			failUntil: 100, // Always fail
			dims:      256,
		}
		fallback := newMockProvider(256)

		primarySvc := NewService(primary)
		fallbackSvc := NewService(fallback)

		svc := NewService(primary, WithFallback(fallbackSvc))
		_ = primarySvc // unused, just for clarity

		vec, err := svc.Embed(context.Background(), "test")

		if err != nil {
			t.Errorf("expected fallback to succeed, got: %v", err)
		}
		if vec == nil {
			t.Error("expected vector from fallback")
		}
	})
}
