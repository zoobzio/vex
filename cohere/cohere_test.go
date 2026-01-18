package cohere

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
	if p.Name() != "cohere" {
		t.Errorf("expected 'cohere', got %q", p.Name())
	}
}

func TestProvider_Dimensions(t *testing.T) {
	p := New(Config{APIKey: "test"})
	if p.Dimensions() != DimensionsEmbedEnglishV3 {
		t.Errorf("expected %d, got %d", DimensionsEmbedEnglishV3, p.Dimensions())
	}
}

func TestProvider_Embed(t *testing.T) {
	t.Run("successful embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("expected POST, got %s", r.Method)
			}
			if r.URL.Path != "/embed" {
				t.Errorf("expected /embed, got %s", r.URL.Path)
			}

			// Verify request body contains input_type
			var req embeddingRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("failed to decode request: %v", err)
			}
			if req.InputType == "" {
				t.Error("expected input_type in request")
			}

			resp := embeddingResponse{
				ID: "test-id",
				Embeddings: []vex.Vector{
					{0.1, 0.2, 0.3},
					{0.4, 0.5, 0.6},
				},
				Meta: meta{
					BilledUnits: billedUnits{InputTokens: 10},
				},
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
				"message": "Invalid API key",
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
}

func TestProvider_WithInputType(t *testing.T) {
	p := New(Config{APIKey: "test", InputType: InputTypeSearchDocument})

	queryProvider := p.WithInputType(InputTypeSearchQuery)

	if queryProvider.inputType != InputTypeSearchQuery {
		t.Errorf("expected search_query input type")
	}
	// Original should be unchanged
	if p.inputType != InputTypeSearchDocument {
		t.Errorf("original provider should be unchanged")
	}
}

func TestConfig_Defaults(t *testing.T) {
	p := New(Config{APIKey: "test"})

	if p.model != "embed-english-v3.0" {
		t.Errorf("expected default model 'embed-english-v3.0', got %q", p.model)
	}
	if p.baseURL != "https://api.cohere.ai/v1" {
		t.Errorf("expected default base URL, got %q", p.baseURL)
	}
	if p.inputType != InputTypeSearchDocument {
		t.Errorf("expected default input type 'search_document'")
	}
}

func TestInputTypes(t *testing.T) {
	types := []InputType{
		InputTypeSearchDocument,
		InputTypeSearchQuery,
		InputTypeClassification,
		InputTypeClustering,
	}

	for _, it := range types {
		if it == "" {
			t.Error("input type should not be empty")
		}
	}
}
