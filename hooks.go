package vex

import (
	"context"
	"time"

	"github.com/zoobzio/capitan"
)

// Signals for hook events.
var (
	EmbedStarted          = capitan.NewSignal("vex.embed.started", "Embedding request initiated")
	EmbedCompleted        = capitan.NewSignal("vex.embed.completed", "Embedding request succeeded")
	EmbedFailed           = capitan.NewSignal("vex.embed.failed", "Embedding request failed")
	ProviderCallStarted   = capitan.NewSignal("vex.provider.call.started", "Provider HTTP call initiated")
	ProviderCallCompleted = capitan.NewSignal("vex.provider.call.completed", "Provider HTTP call succeeded")
	ProviderCallFailed    = capitan.NewSignal("vex.provider.call.failed", "Provider HTTP call failed")
)

// Keys for hook event fields.
var (
	RequestIDKey    = capitan.NewStringKey("vex.request.id")
	ProviderKey     = capitan.NewStringKey("vex.provider")
	ModelKey        = capitan.NewStringKey("vex.model")
	InputCountKey   = capitan.NewIntKey("vex.input.count")
	DimensionsKey   = capitan.NewIntKey("vex.dimensions")
	DurationMsKey   = capitan.NewIntKey("vex.duration.ms")
	PromptTokensKey = capitan.NewIntKey("vex.tokens.prompt")
	TotalTokensKey  = capitan.NewIntKey("vex.tokens.total")
	ErrorKey        = capitan.NewStringKey("vex.error")
)

// emitEmbedStarted emits a signal when embedding begins.
func emitEmbedStarted(ctx context.Context, requestID string, provider string, inputCount int) {
	capitan.Info(ctx, EmbedStarted,
		RequestIDKey.Field(requestID),
		ProviderKey.Field(provider),
		InputCountKey.Field(inputCount),
	)
}

// emitEmbedCompleted emits a signal when embedding succeeds.
func emitEmbedCompleted(ctx context.Context, requestID string, provider string, resp *EmbeddingResponse, duration time.Duration) {
	capitan.Info(ctx, EmbedCompleted,
		RequestIDKey.Field(requestID),
		ProviderKey.Field(provider),
		ModelKey.Field(resp.Model),
		DimensionsKey.Field(resp.Dimensions),
		DurationMsKey.Field(int(duration.Milliseconds())),
		PromptTokensKey.Field(resp.Usage.PromptTokens),
		TotalTokensKey.Field(resp.Usage.TotalTokens),
	)
}

// emitEmbedFailed emits a signal when embedding fails.
func emitEmbedFailed(ctx context.Context, requestID string, provider string, err error, duration time.Duration) {
	capitan.Error(ctx, EmbedFailed,
		RequestIDKey.Field(requestID),
		ProviderKey.Field(provider),
		DurationMsKey.Field(int(duration.Milliseconds())),
		ErrorKey.Field(err.Error()),
	)
}

// emitProviderCallStarted emits a signal when a provider HTTP call begins.
func emitProviderCallStarted(ctx context.Context, provider string, inputCount int) {
	capitan.Info(ctx, ProviderCallStarted,
		ProviderKey.Field(provider),
		InputCountKey.Field(inputCount),
	)
}

// emitProviderCallCompleted emits a signal when a provider HTTP call succeeds.
func emitProviderCallCompleted(ctx context.Context, provider string, resp *EmbeddingResponse, duration time.Duration) {
	capitan.Info(ctx, ProviderCallCompleted,
		ProviderKey.Field(provider),
		ModelKey.Field(resp.Model),
		DimensionsKey.Field(resp.Dimensions),
		DurationMsKey.Field(int(duration.Milliseconds())),
		PromptTokensKey.Field(resp.Usage.PromptTokens),
		TotalTokensKey.Field(resp.Usage.TotalTokens),
	)
}

// emitProviderCallFailed emits a signal when a provider HTTP call fails.
func emitProviderCallFailed(ctx context.Context, provider string, err error, duration time.Duration) {
	capitan.Error(ctx, ProviderCallFailed,
		ProviderKey.Field(provider),
		DurationMsKey.Field(int(duration.Milliseconds())),
		ErrorKey.Field(err.Error()),
	)
}
