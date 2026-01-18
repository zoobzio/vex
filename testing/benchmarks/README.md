# Benchmarks

Performance benchmarks for vex.

## Running Benchmarks

```bash
# Run all benchmarks
make test-bench

# Run specific benchmark
go test -bench=BenchmarkVector ./testing/benchmarks/...

# Run with memory allocation stats
go test -bench=. -benchmem ./testing/benchmarks/...

# Run with more iterations
go test -bench=. -benchtime=5s ./testing/benchmarks/...
```

## Benchmark Categories

### Vector Operations
- `BenchmarkVector_Normalize` - L2 normalization
- `BenchmarkVector_CosineSimilarity` - Cosine similarity calculation
- `BenchmarkVector_Dot` - Dot product calculation
- `BenchmarkVector_EuclideanDistance` - Euclidean distance

### Pooling
- `BenchmarkPool_Mean` - Mean pooling
- `BenchmarkPool_Max` - Max pooling

### Chunking
- `BenchmarkChunker_Sentence` - Sentence chunking
- `BenchmarkChunker_Fixed` - Fixed-size chunking

## Interpreting Results

```
BenchmarkVector_CosineSimilarity-8    5000000    234 ns/op    0 B/op    0 allocs/op
```

- `5000000` - Number of iterations
- `234 ns/op` - Nanoseconds per operation
- `0 B/op` - Bytes allocated per operation
- `0 allocs/op` - Allocations per operation

## Performance Guidelines

- Vector operations should be allocation-free where possible
- Cosine similarity on 1536-dim vectors should be < 1µs
- Pooling should scale linearly with vector count
