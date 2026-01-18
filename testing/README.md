# Testing

This directory contains test utilities and organised test suites for vex.

## Structure

```
testing/
├── helpers.go          # Test utilities and mock provider
├── helpers_test.go     # Tests for helpers themselves
├── benchmarks/         # Performance benchmarks
│   └── README.md
└── integration/        # Integration tests (require API keys)
    └── README.md
```

## Running Tests

```bash
# Run all tests
make test

# Run unit tests only (short mode)
make test-unit

# Run integration tests (requires API keys)
make test-integration

# Run benchmarks
make test-bench

# Generate coverage report
make coverage
```

## Mock Provider

The `testing` package provides a `MockProvider` for unit testing without hitting real APIs:

```go
import vextesting "github.com/zoobzio/vex/testing"

func TestMyFeature(t *testing.T) {
    provider := vextesting.NewMockProvider(vextesting.MockConfig{
        Dimensions:    1536,
        Deterministic: true,  // Same input = same output
    })

    svc := vex.NewService(provider)
    vec, err := svc.Embed(ctx, "test")
    // ...
}
```

## Test Helpers

```go
// Assert vector dimensions
vextesting.AssertVectorDimensions(t, vec, 1536)

// Assert vector is normalized
vextesting.AssertVectorNormalized(t, vec, 0.0001)

// Assert similarity in range
vextesting.AssertSimilarityInRange(t, sim, 0.8, 1.0)

// Generate test vectors
vec := vextesting.GenerateTestVector(1536, seed)

// Generate vectors with target similarity
v1, v2 := vextesting.GenerateSimilarVectors(1536, 0.9)
```

## Writing Tests

- Use table-driven tests where appropriate
- Test both success and failure cases
- Use the mock provider for unit tests
- Reserve integration tests for real API validation
- Keep tests deterministic and fast
