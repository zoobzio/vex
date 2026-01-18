package mocks

import (
	"encoding/json"
	"net/http"
	"strings"
)

// CohereMock handles Cohere embedding API requests.
type CohereMock struct {
	Dimensions int
	Model      string
}

// NewCohereMock creates a new Cohere mock with default settings.
func NewCohereMock() *CohereMock {
	return &CohereMock{
		Dimensions: 1024,
		Model:      "embed-english-v3.0",
	}
}

// ServeHTTP implements http.Handler for the Cohere mock.
func (m *CohereMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		m.writeError(w, http.StatusUnauthorized, "Missing or invalid Authorization header")
		return
	}

	if r.Method != "POST" || r.URL.Path != "/embed" {
		m.writeError(w, http.StatusNotFound, "Not found")
		return
	}

	var req cohereRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		m.writeError(w, http.StatusBadRequest, "Invalid request body")
		return
	}

	embeddings := make([][]float64, len(req.Texts))
	for i := range req.Texts {
		embeddings[i] = m.generateVector(i)
	}

	resp := cohereResponse{
		ID:         "mock-id",
		Embeddings: embeddings,
		Meta: cohereMeta{
			BilledUnits: cohereBilledUnits{
				InputTokens: len(req.Texts) * 5,
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (m *CohereMock) generateVector(seed int) []float64 {
	vec := make([]float64, m.Dimensions)
	for i := range vec {
		vec[i] = float64((seed*m.Dimensions+i)%1000) / 1000.0
	}
	return vec
}

func (m *CohereMock) writeError(w http.ResponseWriter, status int, message string) {
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{
		"message": message,
	})
}

type cohereRequest struct {
	Model     string   `json:"model"`
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

type cohereResponse struct {
	ID         string      `json:"id"`
	Embeddings [][]float64 `json:"embeddings"`
	Meta       cohereMeta  `json:"meta"`
}

type cohereMeta struct {
	BilledUnits cohereBilledUnits `json:"billed_units"`
}

type cohereBilledUnits struct {
	InputTokens int `json:"input_tokens"`
}
