//go:build integration

package integration

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/zoobzio/vex"
	"github.com/zoobzio/vex/cohere"
	"github.com/zoobzio/vex/openai"
	"github.com/zoobzio/vex/testing/integration/mocks"
)

// TestOpenAI_WithMockServer tests OpenAI provider against a local mock.
func TestOpenAI_WithMockServer(t *testing.T) {
	mock := mocks.NewOpenAIMock()
	server := httptest.NewServer(mock)
	defer server.Close()

	provider := openai.New(openai.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Model:   "text-embedding-3-small",
	})

	svc := vex.NewService(provider)
	ctx := context.Background()

	t.Run("embeds single text", func(t *testing.T) {
		vec, err := svc.Embed(ctx, "hello world")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(vec) != 1536 {
			t.Errorf("expected 1536 dimensions, got %d", len(vec))
		}
	})

	t.Run("embeds batch", func(t *testing.T) {
		texts := []string{"hello", "world", "foo", "bar"}
		vecs, err := svc.Batch(ctx, texts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(vecs) != len(texts) {
			t.Errorf("expected %d vectors, got %d", len(texts), len(vecs))
		}
	})

	t.Run("vectors are normalized", func(t *testing.T) {
		vec, _ := svc.Embed(ctx, "test")
		norm := vec.Norm()
		if norm < 0.99 || norm > 1.01 {
			t.Errorf("expected normalized vector, got norm=%f", norm)
		}
	})
}

// TestCohere_WithMockServer tests Cohere provider against a local mock.
func TestCohere_WithMockServer(t *testing.T) {
	mock := mocks.NewCohereMock()
	server := httptest.NewServer(mock)
	defer server.Close()

	provider := cohere.New(cohere.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	svc := vex.NewService(provider)
	ctx := context.Background()

	t.Run("embeds single text", func(t *testing.T) {
		vec, err := svc.Embed(ctx, "hello world")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(vec) != 1024 {
			t.Errorf("expected 1024 dimensions, got %d", len(vec))
		}
	})

	t.Run("embeds batch", func(t *testing.T) {
		texts := []string{"hello", "world"}
		vecs, err := svc.Batch(ctx, texts)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(vecs) != len(texts) {
			t.Errorf("expected %d vectors, got %d", len(texts), len(vecs))
		}
	})
}

// TestWithRetry_Integration tests retry logic with mock servers.
func TestWithRetry_Integration(t *testing.T) {
	callCount := 0
	mock := mocks.NewOpenAIMock()

	// Wrap mock to fail first 2 requests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		mock.ServeHTTP(w, r)
	}))
	defer server.Close()

	provider := openai.New(openai.Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	svc := vex.NewService(provider, vex.WithRetry(3))

	_, err := svc.Embed(context.Background(), "test")
	if err != nil {
		t.Errorf("expected success after retries, got: %v", err)
	}
	if callCount != 3 {
		t.Errorf("expected 3 calls, got %d", callCount)
	}
}

// TestWithTestcontainers demonstrates testcontainers usage.
// This test requires Docker to be running.
func TestWithTestcontainers(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testcontainers test in short mode")
	}

	ctx := context.Background()

	// Create a generic container running a simple HTTP server
	// In a real scenario, you'd run a more sophisticated mock
	req := testcontainers.ContainerRequest{
		Image:        "nginx:alpine",
		ExposedPorts: []string{"80/tcp"},
		WaitingFor:   wait.ForHTTP("/").WithStartupTimeout(30 * time.Second),
	}

	container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		t.Fatalf("failed to start container: %v", err)
	}
	defer container.Terminate(ctx)

	host, err := container.Host(ctx)
	if err != nil {
		t.Fatalf("failed to get host: %v", err)
	}

	port, err := container.MappedPort(ctx, "80")
	if err != nil {
		t.Fatalf("failed to get port: %v", err)
	}

	// Verify container is running
	endpoint := fmt.Sprintf("http://%s:%s", host, port.Port())
	t.Logf("Container running at %s", endpoint)

	// In a real test, you'd configure your mock server container
	// and test against it. This just demonstrates the pattern.
}
