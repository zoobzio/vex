// Package cohere provides an embedding provider for the Cohere API.
package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/zoobzio/vex"
)

// Default dimensions for Cohere models.
const (
	DimensionsEmbedEnglishV3 = 1024
	DimensionsEmbedMultiV3   = 1024
)

// InputType specifies the type of text being embedded.
type InputType string

// Input type constants.
const (
	InputTypeSearchDocument InputType = "search_document"
	InputTypeSearchQuery    InputType = "search_query"
	InputTypeClassification InputType = "classification"
	InputTypeClustering     InputType = "clustering"
)

// Provider implements vex.Provider for Cohere embeddings API.
type Provider struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	inputType  InputType
	dimensions int
}

// Config holds configuration for the Cohere embedding provider.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	InputType  InputType
	Dimensions int
	Timeout    time.Duration
}

// New creates a new Cohere embedding provider.
func New(config Config) *Provider {
	if config.Model == "" {
		config.Model = "embed-english-v3.0"
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://api.cohere.ai/v1"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.Dimensions == 0 {
		config.Dimensions = DimensionsEmbedEnglishV3
	}
	if config.InputType == "" {
		config.InputType = InputTypeSearchDocument
	}

	return &Provider{
		apiKey:     config.APIKey,
		model:      config.Model,
		baseURL:    config.BaseURL,
		dimensions: config.Dimensions,
		inputType:  config.InputType,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "cohere"
}

// Dimensions returns the output vector dimensionality.
func (p *Provider) Dimensions() int {
	return p.dimensions
}

// WithInputType returns a new provider with the specified input type.
func (p *Provider) WithInputType(inputType InputType) *Provider {
	newP := *p
	newP.inputType = inputType
	return &newP
}

// ForQuery returns a provider configured for query embedding mode.
// Implements vex.QueryProviderFactory.
func (p *Provider) ForQuery() vex.Provider {
	return p.WithInputType(InputTypeSearchQuery)
}

// Embed generates embeddings for the given texts.
func (p *Provider) Embed(ctx context.Context, texts []string) (*vex.EmbeddingResponse, error) {
	if len(texts) == 0 {
		return &vex.EmbeddingResponse{
			Vectors:    nil,
			Model:      p.model,
			Dimensions: p.dimensions,
		}, nil
	}

	reqBody := embeddingRequest{
		Model:     p.model,
		Texts:     texts,
		InputType: string(p.inputType),
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/embed", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp errorResponse
		if err := json.Unmarshal(body, &errResp); err == nil && errResp.Message != "" {
			return nil, fmt.Errorf("cohere error (%d): %s", resp.StatusCode, errResp.Message)
		}
		return nil, fmt.Errorf("cohere error: status %d", resp.StatusCode)
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	vectors := make([]vex.Vector, len(embResp.Embeddings))
	for i, emb := range embResp.Embeddings {
		vectors[i] = toFloat32(emb)
	}

	dims := p.dimensions
	if len(vectors) > 0 && len(vectors[0]) > 0 {
		dims = len(vectors[0])
	}

	return &vex.EmbeddingResponse{
		Vectors:    vectors,
		Model:      p.model,
		Dimensions: dims,
		Usage: vex.Usage{
			PromptTokens: embResp.Meta.BilledUnits.InputTokens,
			TotalTokens:  embResp.Meta.BilledUnits.InputTokens,
		},
	}, nil
}

// toFloat32 converts a float64 slice to a vex.Vector (float32).
func toFloat32(f64 []float64) vex.Vector {
	result := make(vex.Vector, len(f64))
	for i, v := range f64 {
		result[i] = float32(v)
	}
	return result
}

// API types

type embeddingRequest struct {
	Model     string   `json:"model"`
	InputType string   `json:"input_type"`
	Texts     []string `json:"texts"`
}

type embeddingResponse struct {
	ID         string      `json:"id"`
	Embeddings [][]float64 `json:"embeddings"`
	Meta       meta        `json:"meta"`
}

type meta struct {
	BilledUnits billedUnits `json:"billed_units"`
}

type billedUnits struct {
	InputTokens int `json:"input_tokens"`
}

type errorResponse struct {
	Message string `json:"message"`
}
