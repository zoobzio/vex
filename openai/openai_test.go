package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProvider_Name(t *testing.T) {
	p := New(Config{APIKey: "test"})
	if p.Name() != "openai" {
		t.Errorf("expected 'openai', got %q", p.Name())
	}
}

func TestProvider_Dimensions(t *testing.T) {
	tests := []struct {
		model    string
		expected int
	}{
		{"text-embedding-ada-002", 1536},
		{"text-embedding-3-small", 1536},
		{"text-embedding-3-large", 3072},
		{"unknown-model", 1536}, // defaults to small
	}

	for _, tt := range tests {
		p := New(Config{APIKey: "test", Model: tt.model})
		if p.Dimensions() != tt.expected {
			t.Errorf("model %s: expected %d dimensions, got %d", tt.model, tt.expected, p.Dimensions())
		}
	}
}

func TestProvider_Embed(t *testing.T) {
	t.Run("successful embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Verify request
			if r.Method != "POST" {
				t.Errorf("expected POST, got %s", r.Method)
			}
			if r.URL.Path != "/embeddings" {
				t.Errorf("expected /embeddings, got %s", r.URL.Path)
			}
			if r.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("missing or incorrect authorization header")
			}

			// Return mock response
			resp := embeddingResponse{
				Object: "list",
				Data: []embeddingData{
					{Object: "embedding", Index: 0, Embedding: []float64{0.1, 0.2, 0.3}},
					{Object: "embedding", Index: 1, Embedding: []float64{0.4, 0.5, 0.6}},
				},
				Model: "text-embedding-3-small",
				Usage: usage{PromptTokens: 10, TotalTokens: 10},
			}
			if err := json.NewEncoder(w).Encode(resp); err != nil {
				t.Fatalf("failed to encode response: %v", err)
			}
		}))
		defer server.Close()

		p := New(Config{
			APIKey:  "test-key",
			BaseURL: server.URL,
		})

		resp, err := p.Embed(context.Background(), []string{"hello", "world"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(resp.Vectors) != 2 {
			t.Errorf("expected 2 vectors, got %d", len(resp.Vectors))
		}
		if resp.Model != "text-embedding-3-small" {
			t.Errorf("expected model 'text-embedding-3-small', got %q", resp.Model)
		}
		if resp.Usage.PromptTokens != 10 {
			t.Errorf("expected 10 prompt tokens, got %d", resp.Usage.PromptTokens)
		}
	})

	t.Run("handles empty input", func(t *testing.T) {
		p := New(Config{APIKey: "test"})

		resp, err := p.Embed(context.Background(), []string{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Vectors != nil {
			t.Errorf("expected nil vectors for empty input")
		}
	})

	t.Run("handles API error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			//nolint:errcheck // test helper
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]string{
					"message": "Invalid API key",
					"type":    "invalid_request_error",
				},
			})
		}))
		defer server.Close()

		p := New(Config{
			APIKey:  "bad-key",
			BaseURL: server.URL,
		})

		_, err := p.Embed(context.Background(), []string{"test"})
		if err == nil {
			t.Error("expected error for invalid API key")
		}
	})

	t.Run("handles rate limit error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			//nolint:errcheck // test helper
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]string{
					"message": "Rate limit exceeded",
					"type":    "rate_limit_error",
				},
			})
		}))
		defer server.Close()

		p := New(Config{
			APIKey:  "test-key",
			BaseURL: server.URL,
		})

		_, err := p.Embed(context.Background(), []string{"test"})
		if err == nil {
			t.Error("expected error for rate limit")
		}
	})

	t.Run("preserves vector order by index", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// Return vectors in reverse order
			resp := embeddingResponse{
				Data: []embeddingData{
					{Index: 1, Embedding: []float64{0.4, 0.5}},
					{Index: 0, Embedding: []float64{0.1, 0.2}},
				},
				Model: "test",
			}
			if err := json.NewEncoder(w).Encode(resp); err != nil {
				t.Fatalf("failed to encode response: %v", err)
			}
		}))
		defer server.Close()

		p := New(Config{APIKey: "test", BaseURL: server.URL})
		resp, err := p.Embed(context.Background(), []string{"a", "b"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// First vector should be [0.1, 0.2] based on index
		if resp.Vectors[0][0] != float32(0.1) {
			t.Errorf("vectors not ordered by index")
		}
	})

	t.Run("rejects invalid index from API", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			resp := embeddingResponse{
				Data: []embeddingData{
					{Index: 99, Embedding: []float64{0.1, 0.2}},
				},
				Model: "test",
			}
			//nolint:errcheck // test helper
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		p := New(Config{APIKey: "test", BaseURL: server.URL})
		_, err := p.Embed(context.Background(), []string{"test"})
		if err == nil {
			t.Error("expected error for invalid index")
		}
	})

	t.Run("rejects negative index from API", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			resp := embeddingResponse{
				Data: []embeddingData{
					{Index: -1, Embedding: []float64{0.1, 0.2}},
				},
				Model: "test",
			}
			//nolint:errcheck // test helper
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		p := New(Config{APIKey: "test", BaseURL: server.URL})
		_, err := p.Embed(context.Background(), []string{"test"})
		if err == nil {
			t.Error("expected error for negative index")
		}
	})
}

func TestConfig_Defaults(t *testing.T) {
	p := New(Config{APIKey: "test"})

	if p.model != "text-embedding-3-small" {
		t.Errorf("expected default model 'text-embedding-3-small', got %q", p.model)
	}
	if p.baseURL != "https://api.openai.com/v1" {
		t.Errorf("expected default base URL, got %q", p.baseURL)
	}
}
