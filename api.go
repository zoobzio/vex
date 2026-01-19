// Package vex provides type-safe embedding vector generation for Go.
package vex

import "context"

// Vector represents an embedding vector.
// Uses float32 for compatibility with vector databases (pgvector, Pinecone, Qdrant, etc.).
type Vector []float32

// Usage tracks token consumption for an embedding request.
type Usage struct {
	PromptTokens int
	TotalTokens  int
}

// EmbeddingResponse contains the result of an embedding request.
type EmbeddingResponse struct {
	Model      string
	Vectors    []Vector
	Usage      Usage
	Dimensions int
}

// Provider defines the interface for embedding backends.
type Provider interface {
	// Embed generates embedding vectors for the given texts.
	Embed(ctx context.Context, texts []string) (*EmbeddingResponse, error)

	// Name returns the provider identifier.
	Name() string

	// Dimensions returns the output vector dimensionality.
	Dimensions() int
}

// QueryProviderFactory is optionally implemented by providers that distinguish
// query vs document embeddings. Providers implementing this interface can
// generate query-optimized embeddings for improved retrieval quality.
type QueryProviderFactory interface {
	Provider
	// ForQuery returns a provider configured for query embedding mode.
	ForQuery() Provider
}

// SimilarityMetric defines how vectors are compared.
type SimilarityMetric int

const (
	// Cosine measures the cosine of the angle between vectors.
	Cosine SimilarityMetric = iota
	// DotProduct computes the dot product of vectors.
	DotProduct
	// Euclidean computes the Euclidean distance between vectors.
	Euclidean
)

// ChunkStrategy defines how long texts are split before embedding.
type ChunkStrategy int

const (
	// ChunkNone performs no chunking.
	ChunkNone ChunkStrategy = iota
	// ChunkSentence splits on sentence boundaries.
	ChunkSentence
	// ChunkParagraph splits on paragraph boundaries.
	ChunkParagraph
	// ChunkFixed splits into fixed-size chunks.
	ChunkFixed
)

// PoolingMode defines how multiple chunk vectors are combined.
type PoolingMode int

const (
	// PoolMean averages all vectors.
	PoolMean PoolingMode = iota
	// PoolFirst uses only the first vector.
	PoolFirst
	// PoolMax takes element-wise maximum.
	PoolMax
)
