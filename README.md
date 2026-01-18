# vex

[![CI](https://github.com/zoobzio/vex/actions/workflows/ci.yml/badge.svg)](https://github.com/zoobzio/vex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zoobzio/vex/branch/main/graph/badge.svg)](https://codecov.io/gh/zoobzio/vex)
[![Go Report Card](https://goreportcard.com/badge/github.com/zoobzio/vex)](https://goreportcard.com/report/github.com/zoobzio/vex)
[![CodeQL](https://github.com/zoobzio/vex/actions/workflows/codeql.yml/badge.svg)](https://github.com/zoobzio/vex/actions/workflows/codeql.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/zoobzio/vex.svg)](https://pkg.go.dev/github.com/zoobzio/vex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Version](https://img.shields.io/github/go-mod/go-version/zoobzio/vex)](https://go.dev/)
[![Release](https://img.shields.io/github/v/release/zoobzio/vex)](https://github.com/zoobzio/vex/releases)

Type-safe embedding vector generation for Go. Provider-agnostic, composable reliability, observable.

## Text In, Vectors Out

```go
provider := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
svc := vex.NewService(provider)

vec, _ := svc.Embed(ctx, "hello world")
// vec is a []float64 of length 1536
```

## Install

```bash
go get github.com/zoobzio/vex
```

Requires Go 1.24 or higher.

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/zoobzio/vex"
    "github.com/zoobzio/vex/openai"
)

func main() {
    ctx := context.Background()

    // Create provider
    provider := openai.New(openai.Config{
        APIKey: os.Getenv("OPENAI_API_KEY"),
        Model:  "text-embedding-3-small",
    })

    // Create service with reliability options
    svc := vex.NewService(provider,
        vex.WithRetry(3),
        vex.WithTimeout(30*time.Second),
    )

    // Embed single text
    vec, err := svc.Embed(ctx, "The quick brown fox")
    if err != nil {
        panic(err)
    }
    fmt.Printf("Vector dimensions: %d\n", len(vec))

    // Embed batch
    texts := []string{"hello", "world", "foo", "bar"}
    vecs, err := svc.Batch(ctx, texts)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Embedded %d texts\n", len(vecs))

    // Compare vectors
    similarity := vecs[0].CosineSimilarity(vecs[1])
    fmt.Printf("Similarity: %.4f\n", similarity)
}
```

## Providers

| Provider | Models | Import |
|----------|--------|--------|
| OpenAI | text-embedding-3-small, text-embedding-3-large, ada-002 | `vex/openai` |
| Cohere | embed-english-v3.0, embed-multilingual-v3.0 | `vex/cohere` |
| Voyage | voyage-3, voyage-3-lite, voyage-large-2 | `vex/voyage` |
| Gemini | text-embedding-004 | `vex/gemini` |

## Reliability

Built on [pipz](https://github.com/zoobzio/pipz) for composable reliability:

```go
svc := vex.NewService(provider,
    vex.WithRetry(3),                           // Retry failed requests
    vex.WithBackoff(3, 100*time.Millisecond),   // Exponential backoff
    vex.WithTimeout(30*time.Second),            // Request timeout
    vex.WithCircuitBreaker(5, time.Minute),     // Circuit breaker
    vex.WithRateLimit(10, 20),                  // Rate limiting
    vex.WithFallback(backupService),            // Fallback provider
)
```

## Chunking

Handle long texts by splitting and pooling:

```go
chunker := &vex.Chunker{
    Strategy:  vex.ChunkSentence,  // or ChunkParagraph, ChunkFixed
    MaxSize:   512,
    Overlap:   50,
}

svc := vex.NewService(provider).
    WithChunker(chunker).
    WithPooling(vex.PoolMean)  // or PoolMax, PoolFirst
```

## Vector Operations

```go
// Normalise to unit vector
normalised := vec.Normalize()

// Similarity metrics
cosine := vec1.CosineSimilarity(vec2)
dot := vec1.Dot(vec2)
euclidean := vec1.EuclideanDistance(vec2)

// Generic similarity
sim := vec1.Similarity(vec2, vex.Cosine)
```

## Why Vex?

- **Provider-agnostic**: Swap providers without changing application code
- **Type-safe**: Go generics and strong typing throughout
- **Composable reliability**: Mix and match retry, timeout, circuit breaker, rate limiting
- **Observable**: Hook signals via [capitan](https://github.com/zoobzio/capitan) for monitoring
- **Chunking built-in**: Handle long texts with configurable splitting and pooling

## Documentation

- [API Reference](https://pkg.go.dev/github.com/zoobzio/vex)
- [Examples](./docs/examples/)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development workflow.

## License

MIT - see [LICENSE](./LICENSE)
