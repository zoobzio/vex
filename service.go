package vex

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/zoobzio/pipz"
)

// Identity for the embedding terminal processor.
var terminalID = pipz.NewIdentity("vex:terminal", "Embedding provider terminal")

// EmbedRequest represents a request flowing through the pipeline.
type EmbedRequest struct {
	Error     error
	Response  *EmbeddingResponse
	RequestID string
	Provider  string
	Texts     []string
}

// Service wraps an embedding provider with pipeline-based reliability.
type Service struct {
	pipeline    pipz.Chainable[*EmbedRequest]
	provider    Provider
	chunker     *Chunker
	poolingMode PoolingMode
	normalize   bool
}

// ServiceConfig configures a Service.
type ServiceConfig struct {
	Chunker     *Chunker
	PoolingMode PoolingMode
	Normalize   bool
}

// NewService creates a new embedding Service with the given provider and options.
func NewService(provider Provider, opts ...Option) *Service {
	terminal := NewTerminal(provider)

	// Apply options in reverse order (outermost first)
	pipeline := terminal
	for i := len(opts) - 1; i >= 0; i-- {
		pipeline = opts[i](pipeline)
	}

	return &Service{
		pipeline:    pipeline,
		provider:    provider,
		chunker:     DefaultChunker(),
		poolingMode: PoolMean,
		normalize:   true,
	}
}

// NewTerminal creates a terminal processor that calls the embedding provider.
func NewTerminal(provider Provider) pipz.Chainable[*EmbedRequest] {
	return pipz.Apply(terminalID, func(ctx context.Context, req *EmbedRequest) (*EmbedRequest, error) {
		start := time.Now()
		emitProviderCallStarted(ctx, provider.Name(), len(req.Texts))

		resp, err := provider.Embed(ctx, req.Texts)
		duration := time.Since(start)

		if err != nil {
			emitProviderCallFailed(ctx, provider.Name(), err, duration)
			req.Error = err
			return req, err
		}

		emitProviderCallCompleted(ctx, provider.Name(), resp, duration)
		req.Response = resp
		return req, nil
	})
}

// GetPipeline returns the internal pipeline for composition.
func (s *Service) GetPipeline() pipz.Chainable[*EmbedRequest] {
	return s.pipeline
}

// WithChunker sets the chunking strategy.
func (s *Service) WithChunker(c *Chunker) *Service {
	s.chunker = c
	return s
}

// WithPooling sets the pooling mode for chunked embeddings.
func (s *Service) WithPooling(mode PoolingMode) *Service {
	s.poolingMode = mode
	return s
}

// WithNormalize sets whether to L2-normalize output vectors.
func (s *Service) WithNormalize(normalize bool) *Service {
	s.normalize = normalize
	return s
}

// Embed generates an embedding for a single text.
func (s *Service) Embed(ctx context.Context, text string) (Vector, error) {
	vectors, err := s.Batch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, nil
	}
	return vectors[0], nil
}

// Batch generates embeddings for multiple texts.
func (s *Service) Batch(ctx context.Context, texts []string) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	requestID := uuid.New().String()
	start := time.Now()

	emitEmbedStarted(ctx, requestID, s.provider.Name(), len(texts))

	// Chunk texts if needed
	var allChunks []string
	var chunkMapping []int // maps chunk index to original text index
	for i, text := range texts {
		chunks := s.chunker.Chunk(text)
		for range chunks {
			chunkMapping = append(chunkMapping, i)
		}
		allChunks = append(allChunks, chunks...)
	}

	// Create and process request
	req := &EmbedRequest{
		Texts:     allChunks,
		RequestID: requestID,
		Provider:  s.provider.Name(),
	}

	processed, err := s.pipeline.Process(ctx, req)
	duration := time.Since(start)

	if err != nil {
		emitEmbedFailed(ctx, requestID, s.provider.Name(), err, duration)
		return nil, err
	}

	if processed.Response == nil || len(processed.Response.Vectors) == 0 {
		return nil, nil
	}

	// Pool chunks back to original texts
	vectors := s.poolChunks(texts, processed.Response.Vectors, chunkMapping)

	// Normalize if configured
	if s.normalize {
		for i, v := range vectors {
			vectors[i] = v.Normalize()
		}
	}

	emitEmbedCompleted(ctx, requestID, s.provider.Name(), processed.Response, duration)

	return vectors, nil
}

// poolChunks combines chunk vectors back into per-text vectors.
func (s *Service) poolChunks(texts []string, chunkVectors []Vector, mapping []int) []Vector {
	result := make([]Vector, len(texts))

	// Group vectors by original text index
	grouped := make([][]Vector, len(texts))
	for i, vec := range chunkVectors {
		if i < len(mapping) {
			textIdx := mapping[i]
			grouped[textIdx] = append(grouped[textIdx], vec)
		}
	}

	// Pool each group
	for i, vecs := range grouped {
		if len(vecs) > 0 {
			result[i] = Pool(vecs, s.poolingMode)
		}
	}

	return result
}

// Dimensions returns the output vector dimensionality from the provider.
func (s *Service) Dimensions() int {
	return s.provider.Dimensions()
}

// Provider returns the underlying embedding provider.
func (s *Service) Provider() Provider {
	return s.provider
}
