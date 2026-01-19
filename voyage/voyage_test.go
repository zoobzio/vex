package voyage

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zoobzio/vex"
)

func TestProvider_Name(t *testing.T) {
	p := New(Config{APIKey: "test"})
	if p.Name() != "voyage" {
		t.Errorf("expected 'voyage', got %q", p.Name())
	}
}

func TestProvider_Dimensions(t *testing.T) {
	tests := []struct {
		model    string
		expected int
	}{
		{"voyage-3", DimensionsVoyage3},
		{"voyage-3-lite", DimensionsVoyage3Lite},
		{"voyage-large-2", DimensionsVoyageLarge2},
		{"unknown", DimensionsVoyage3}, // defaults
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
			if r.Method != "POST" {
				t.Errorf("expected POST, got %s", r.Method)
			}
			if r.URL.Path != "/embeddings" {
				t.Errorf("expected /embeddings, got %s", r.URL.Path)
			}

			resp := embeddingResponse{
				Object: "list",
				Data: []embeddingData{
					{Object: "embedding", Index: 0, Embedding: []float64{0.1, 0.2, 0.3}},
					{Object: "embedding", Index: 1, Embedding: []float64{0.4, 0.5, 0.6}},
				},
				Model: "voyage-3",
				Usage: usage{TotalTokens: 10},
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
		if resp.Model != "voyage-3" {
			t.Errorf("expected model 'voyage-3', got %q", resp.Model)
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
			json.NewEncoder(w).Encode(map[string]string{
				"detail": "Invalid API key",
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

	t.Run("preserves vector order by index", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// Return vectors in reverse order
			resp := embeddingResponse{
				Data: []embeddingData{
					{Index: 1, Embedding: []float64{0.4, 0.5}},
					{Index: 0, Embedding: []float64{0.1, 0.2}},
				},
				Model: "voyage-3",
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
				Model: "voyage-3",
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
				Model: "voyage-3",
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

func TestProvider_WithInputType(t *testing.T) {
	p := New(Config{APIKey: "test", InputType: InputTypeDocument})

	queryProvider := p.WithInputType(InputTypeQuery)

	if queryProvider.inputType != InputTypeQuery {
		t.Errorf("expected query input type")
	}
	if p.inputType != InputTypeDocument {
		t.Errorf("original provider should be unchanged")
	}
}

func TestProvider_ForQuery(t *testing.T) {
	p := New(Config{APIKey: "test", InputType: InputTypeDocument})

	queryProvider := p.ForQuery()

	// Should be a *Provider with query input type
	qp, ok := queryProvider.(*Provider)
	if !ok {
		t.Fatalf("expected *Provider, got %T", queryProvider)
	}
	if qp.inputType != InputTypeQuery {
		t.Errorf("expected query input type, got %s", qp.inputType)
	}

	// Original should be unchanged
	if p.inputType != InputTypeDocument {
		t.Errorf("original provider should be unchanged")
	}
}

func TestProvider_ImplementsQueryProviderFactory(_ *testing.T) {
	p := New(Config{APIKey: "test"})

	// Verify it implements QueryProviderFactory (compile-time check)
	var _ vex.QueryProviderFactory = p
}

func TestConfig_Defaults(t *testing.T) {
	p := New(Config{APIKey: "test"})

	if p.model != "voyage-3" {
		t.Errorf("expected default model 'voyage-3', got %q", p.model)
	}
	if p.baseURL != "https://api.voyageai.com/v1" {
		t.Errorf("expected default base URL, got %q", p.baseURL)
	}
	if p.inputType != InputTypeDocument {
		t.Errorf("expected default input type 'document'")
	}
}
