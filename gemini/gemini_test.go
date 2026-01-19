package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zoobzio/vex"
)

func TestProvider_Name(t *testing.T) {
	p := New(Config{APIKey: "test"})
	if p.Name() != "gemini" {
		t.Errorf("expected 'gemini', got %q", p.Name())
	}
}

func TestProvider_Dimensions(t *testing.T) {
	p := New(Config{APIKey: "test"})
	if p.Dimensions() != DimensionsTextEmbedding004 {
		t.Errorf("expected %d, got %d", DimensionsTextEmbedding004, p.Dimensions())
	}
}

func TestProvider_Embed(t *testing.T) {
	t.Run("successful embedding", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				t.Errorf("expected POST, got %s", r.Method)
			}
			if !strings.Contains(r.URL.Path, "batchEmbedContents") {
				t.Errorf("expected batchEmbedContents in path, got %s", r.URL.Path)
			}
			if !strings.Contains(r.URL.RawQuery, "key=test-key") {
				t.Errorf("expected API key in query, got %s", r.URL.RawQuery)
			}

			resp := batchEmbedResponse{
				Embeddings: []embedding{
					{Values: []float64{0.1, 0.2, 0.3}},
					{Values: []float64{0.4, 0.5, 0.6}},
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
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"code":    401,
					"message": "Invalid API key",
					"status":  "UNAUTHENTICATED",
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

	t.Run("sends task type in request", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req batchEmbedRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("failed to decode request: %v", err)
			}

			if len(req.Requests) == 0 {
				t.Error("expected requests in body")
				return
			}
			if req.Requests[0].TaskType != string(TaskTypeRetrievalDocument) {
				t.Errorf("expected task type RETRIEVAL_DOCUMENT, got %s", req.Requests[0].TaskType)
			}

			resp := batchEmbedResponse{
				Embeddings: []embedding{{Values: []float64{0.1}}},
			}
			if err := json.NewEncoder(w).Encode(resp); err != nil {
				t.Fatalf("failed to encode response: %v", err)
			}
		}))
		defer server.Close()

		p := New(Config{
			APIKey:   "test",
			BaseURL:  server.URL,
			TaskType: TaskTypeRetrievalDocument,
		})
		//nolint:errcheck // test helper
		p.Embed(context.Background(), []string{"test"})
	})
}

func TestProvider_WithTaskType(t *testing.T) {
	p := New(Config{APIKey: "test", TaskType: TaskTypeRetrievalDocument})

	queryProvider := p.WithTaskType(TaskTypeRetrievalQuery)

	if queryProvider.taskType != TaskTypeRetrievalQuery {
		t.Errorf("expected RETRIEVAL_QUERY task type")
	}
	if p.taskType != TaskTypeRetrievalDocument {
		t.Errorf("original provider should be unchanged")
	}
}

func TestProvider_ForQuery(t *testing.T) {
	p := New(Config{APIKey: "test", TaskType: TaskTypeRetrievalDocument})

	queryProvider := p.ForQuery()

	// Should be a *Provider with query task type
	qp, ok := queryProvider.(*Provider)
	if !ok {
		t.Fatalf("expected *Provider, got %T", queryProvider)
	}
	if qp.taskType != TaskTypeRetrievalQuery {
		t.Errorf("expected RETRIEVAL_QUERY task type, got %s", qp.taskType)
	}

	// Original should be unchanged
	if p.taskType != TaskTypeRetrievalDocument {
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

	if p.model != "text-embedding-004" {
		t.Errorf("expected default model 'text-embedding-004', got %q", p.model)
	}
	if p.baseURL != "https://generativelanguage.googleapis.com/v1beta" {
		t.Errorf("expected default base URL, got %q", p.baseURL)
	}
	if p.taskType != TaskTypeRetrievalDocument {
		t.Errorf("expected default task type RETRIEVAL_DOCUMENT")
	}
}

func TestTaskTypes(t *testing.T) {
	types := []TaskType{
		TaskTypeRetrievalQuery,
		TaskTypeRetrievalDocument,
		TaskTypeSemantic,
		TaskTypeClassification,
		TaskTypeClustering,
	}

	for _, tt := range types {
		if tt == "" {
			t.Error("task type should not be empty")
		}
	}
}
