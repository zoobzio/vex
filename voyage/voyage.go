// Package voyage provides an embedding provider for the Voyage AI API.
package voyage

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

// Default dimensions for Voyage models.
const (
	DimensionsVoyage3      = 1024
	DimensionsVoyage3Lite  = 512
	DimensionsVoyageLarge2 = 1536
)

// InputType specifies the type of text being embedded.
type InputType string

// Input type constants.
const (
	InputTypeDocument InputType = "document"
	InputTypeQuery    InputType = "query"
)

// Provider implements vex.Provider for Voyage AI embeddings API.
type Provider struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	inputType  InputType
	dimensions int
}

// Config holds configuration for the Voyage AI embedding provider.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	InputType  InputType
	Dimensions int
	Timeout    time.Duration
}

// New creates a new Voyage AI embedding provider.
func New(config Config) *Provider {
	if config.Model == "" {
		config.Model = "voyage-3"
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://api.voyageai.com/v1"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.Dimensions == 0 {
		config.Dimensions = dimensionsForModel(config.Model)
	}
	if config.InputType == "" {
		config.InputType = InputTypeDocument
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
	return "voyage"
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
	return p.WithInputType(InputTypeQuery)
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
		Input:     texts,
		InputType: string(p.inputType),
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/embeddings", bytes.NewReader(jsonBody))
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
		if err := json.Unmarshal(body, &errResp); err == nil && errResp.Detail != "" {
			return nil, fmt.Errorf("voyage error (%d): %s", resp.StatusCode, errResp.Detail)
		}
		return nil, fmt.Errorf("voyage error: status %d", resp.StatusCode)
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	vectors := make([]vex.Vector, len(embResp.Data))
	for _, d := range embResp.Data {
		if d.Index < 0 || d.Index >= len(vectors) {
			return nil, fmt.Errorf("invalid index %d from API", d.Index)
		}
		vectors[d.Index] = toFloat32(d.Embedding)
	}

	dims := p.dimensions
	if len(vectors) > 0 && len(vectors[0]) > 0 {
		dims = len(vectors[0])
	}

	return &vex.EmbeddingResponse{
		Vectors:    vectors,
		Model:      embResp.Model,
		Dimensions: dims,
		Usage: vex.Usage{
			PromptTokens: embResp.Usage.TotalTokens,
			TotalTokens:  embResp.Usage.TotalTokens,
		},
	}, nil
}

func dimensionsForModel(model string) int {
	switch model {
	case "voyage-3":
		return DimensionsVoyage3
	case "voyage-3-lite":
		return DimensionsVoyage3Lite
	case "voyage-large-2":
		return DimensionsVoyageLarge2
	default:
		return DimensionsVoyage3
	}
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
	InputType string   `json:"input_type,omitempty"`
	Input     []string `json:"input"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Model  string          `json:"model"`
	Data   []embeddingData `json:"data"`
	Usage  usage           `json:"usage"`
}

type embeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

type usage struct {
	TotalTokens int `json:"total_tokens"`
}

type errorResponse struct {
	Detail string `json:"detail"`
}
