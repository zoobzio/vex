// Package gemini provides an embedding provider for the Google Gemini API.
package gemini

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

// Default dimensions for Gemini models.
const (
	DimensionsTextEmbedding004 = 768
)

// TaskType specifies the downstream task for the embedding.
type TaskType string

// Task type constants.
const (
	TaskTypeRetrievalQuery    TaskType = "RETRIEVAL_QUERY"
	TaskTypeRetrievalDocument TaskType = "RETRIEVAL_DOCUMENT"
	TaskTypeSemantic          TaskType = "SEMANTIC_SIMILARITY"
	TaskTypeClassification    TaskType = "CLASSIFICATION"
	TaskTypeClustering        TaskType = "CLUSTERING"
)

// Provider implements vex.Provider for Google Gemini embeddings API.
type Provider struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	taskType   TaskType
	dimensions int
}

// Config holds configuration for the Gemini embedding provider.
type Config struct {
	APIKey     string
	Model      string
	BaseURL    string
	TaskType   TaskType
	Dimensions int
	Timeout    time.Duration
}

// New creates a new Gemini embedding provider.
func New(config Config) *Provider {
	if config.Model == "" {
		config.Model = "text-embedding-004"
	}
	if config.BaseURL == "" {
		config.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.Dimensions == 0 {
		config.Dimensions = DimensionsTextEmbedding004
	}
	if config.TaskType == "" {
		config.TaskType = TaskTypeRetrievalDocument
	}

	return &Provider{
		apiKey:     config.APIKey,
		model:      config.Model,
		baseURL:    config.BaseURL,
		dimensions: config.Dimensions,
		taskType:   config.TaskType,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "gemini"
}

// Dimensions returns the output vector dimensionality.
func (p *Provider) Dimensions() int {
	return p.dimensions
}

// WithTaskType returns a new provider with the specified task type.
func (p *Provider) WithTaskType(taskType TaskType) *Provider {
	newP := *p
	newP.taskType = taskType
	return &newP
}

// ForQuery returns a provider configured for query embedding mode.
// Implements vex.QueryProviderFactory.
func (p *Provider) ForQuery() vex.Provider {
	return p.WithTaskType(TaskTypeRetrievalQuery)
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

	// Gemini uses batch embedding endpoint
	requests := make([]embedContentRequest, len(texts))
	for i, text := range texts {
		requests[i] = embedContentRequest{
			Model: "models/" + p.model,
			Content: content{
				Parts: []part{{Text: text}},
			},
			TaskType: string(p.taskType),
		}
	}

	reqBody := batchEmbedRequest{
		Requests: requests,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/models/%s:batchEmbedContents?key=%s", p.baseURL, p.model, p.apiKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

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
			return nil, fmt.Errorf("gemini error (%d): %s", resp.StatusCode, errResp.Error.Message)
		}
		return nil, fmt.Errorf("gemini error: status %d", resp.StatusCode)
	}

	var embResp batchEmbedResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	vectors := make([]vex.Vector, len(embResp.Embeddings))
	for i, emb := range embResp.Embeddings {
		vectors[i] = toFloat32(emb.Values)
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
			PromptTokens: len(texts), // Gemini doesn't return token counts
			TotalTokens:  len(texts),
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

type batchEmbedRequest struct {
	Requests []embedContentRequest `json:"requests"`
}

type embedContentRequest struct {
	Model    string  `json:"model"`
	TaskType string  `json:"taskType,omitempty"`
	Content  content `json:"content"`
}

type content struct {
	Parts []part `json:"parts"`
}

type part struct {
	Text string `json:"text"`
}

type batchEmbedResponse struct {
	Embeddings []embedding `json:"embeddings"`
}

type embedding struct {
	Values []float64 `json:"values"`
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Status  string `json:"status"`
		Code    int    `json:"code"`
	} `json:"error"`
}
