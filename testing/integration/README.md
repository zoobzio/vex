# Integration Tests

Integration tests for vex using testcontainers and mock servers.

## Structure

```
integration/
├── go.mod              # Separate module (isolates testcontainers dep)
├── integration_test.go # Integration test suite
├── mocks/
│   ├── openai.go       # OpenAI API mock server
│   └── cohere.go       # Cohere API mock server
└── README.md
```

## Running Tests

```bash
# From repository root
cd testing/integration
go test -tags integration -v ./...

# Or use make target from root
make test-integration
```

## Mock Servers

The `mocks` package provides HTTP handlers that simulate provider APIs:

```go
mock := mocks.NewOpenAIMock()
server := httptest.NewServer(mock)
defer server.Close()

provider := openai.New(openai.Config{
    APIKey:  "test-key",
    BaseURL: server.URL,
})
```

Mock servers:
- Validate request format (auth headers, paths, methods)
- Return realistic response structures
- Generate deterministic embeddings for reproducible tests

## Testcontainers

For more realistic integration testing, use testcontainers to run mock servers in Docker:

```go
req := testcontainers.ContainerRequest{
    Image:        "your-mock-server:latest",
    ExposedPorts: []string{"8080/tcp"},
    WaitingFor:   wait.ForHTTP("/health"),
}

container, _ := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
    ContainerRequest: req,
    Started:          true,
})
defer container.Terminate(ctx)
```

## Workspace Isolation

This module is isolated via `go.work` to keep testcontainers and other heavy dependencies out of the main `go.mod`. The workspace file at the repository root includes both modules:

```
go.work
├── . (main vex module)
└── ./testing/integration
```

## Adding New Provider Mocks

1. Create `mocks/provider.go` implementing `http.Handler`
2. Add test cases to `integration_test.go`
3. Run `go mod tidy` to update dependencies
