// Package openai provides an embedding provider for the OpenAI API.
package openai

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

// Default model dimensions.
const (
	DimensionsAda002              = 1536
	DimensionsTextEmbedding3Small = 1536
	DimensionsTextEmbedding3Large = 3072
)

// Provider implements vex.Provider for OpenAI embeddings API.
type Provider struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	dimensions int
}

// Config holds configuration for the OpenAI embedding provider.
type Config struct {
	APIKey     string
	Model      string        // e.g. "text-embedding-3-small", "text-embedding-ada-002"
	BaseURL    string        // Optional, defaults to "https://api.openai.com/v1"
	Dimensions int           // Optional, model-specific default
	Timeout    time.Duration // Optional, defaults to 30s
}

// New creates a new OpenAI embedding provider.
func New(config Config) *Provider {
	if config.Model == "" {
		config.Model = "text-embedding-3-small"
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://api.openai.com/v1"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.Dimensions == 0 {
		config.Dimensions = dimensionsForModel(config.Model)
	}

	return &Provider{
		apiKey:     config.APIKey,
		model:      config.Model,
		baseURL:    config.BaseURL,
		dimensions: config.Dimensions,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "openai"
}

// Dimensions returns the output vector dimensionality.
func (p *Provider) Dimensions() int {
	return p.dimensions
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
		Model: p.model,
		Input: texts,
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
		if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("openai error (%d): %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("openai error: status %d", resp.StatusCode)
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

	return &vex.EmbeddingResponse{
		Vectors:    vectors,
		Model:      embResp.Model,
		Dimensions: len(vectors[0]),
		Usage: vex.Usage{
			PromptTokens: embResp.Usage.PromptTokens,
			TotalTokens:  embResp.Usage.TotalTokens,
		},
	}, nil
}

func dimensionsForModel(model string) int {
	switch model {
	case "text-embedding-ada-002":
		return DimensionsAda002
	case "text-embedding-3-small":
		return DimensionsTextEmbedding3Small
	case "text-embedding-3-large":
		return DimensionsTextEmbedding3Large
	default:
		return DimensionsTextEmbedding3Small
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
	Model string   `json:"model"`
	Input []string `json:"input"`
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
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}
