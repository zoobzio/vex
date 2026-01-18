// Package mocks provides mock server implementations for embedding providers.
package mocks

import (
	"encoding/json"
	"net/http"
	"strings"
)

// OpenAIMock handles OpenAI embedding API requests.
type OpenAIMock struct {
	Dimensions int
	Model      string
}

// NewOpenAIMock creates a new OpenAI mock with default settings.
func NewOpenAIMock() *OpenAIMock {
	return &OpenAIMock{
		Dimensions: 1536,
		Model:      "text-embedding-3-small",
	}
}

// ServeHTTP implements http.Handler for the OpenAI mock.
func (m *OpenAIMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Verify auth
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		m.writeError(w, http.StatusUnauthorized, "Missing or invalid Authorization header")
		return
	}

	if r.Method != "POST" || r.URL.Path != "/embeddings" {
		m.writeError(w, http.StatusNotFound, "Not found")
		return
	}

	var req openAIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		m.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	// Generate mock embeddings
	data := make([]openAIEmbedding, len(req.Input))
	for i := range req.Input {
		data[i] = openAIEmbedding{
			Object:    "embedding",
			Index:     i,
			Embedding: m.generateVector(i),
		}
	}

	resp := openAIResponse{
		Object: "list",
		Data:   data,
		Model:  m.Model,
		Usage: openAIUsage{
			PromptTokens: len(req.Input) * 5,
			TotalTokens:  len(req.Input) * 5,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (m *OpenAIMock) generateVector(seed int) []float64 {
	vec := make([]float64, m.Dimensions)
	for i := range vec {
		vec[i] = float64((seed*m.Dimensions+i)%1000) / 1000.0
	}
	return vec
}

func (m *OpenAIMock) writeError(w http.ResponseWriter, status int, message string) {
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]string{
			"message": message,
			"type":    "invalid_request_error",
		},
	})
}

type openAIRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIResponse struct {
	Object string            `json:"object"`
	Data   []openAIEmbedding `json:"data"`
	Model  string            `json:"model"`
	Usage  openAIUsage       `json:"usage"`
}

type openAIEmbedding struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type openAIUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
