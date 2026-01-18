package vex

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/zoobzio/capitan"
)

func TestHookSignals(t *testing.T) {
	// Verify signal constants are defined correctly
	signals := []capitan.Signal{
		EmbedStarted,
		EmbedCompleted,
		EmbedFailed,
		ProviderCallStarted,
		ProviderCallCompleted,
		ProviderCallFailed,
	}

	for _, sig := range signals {
		if sig.Name() == "" {
			t.Error("signal has empty name")
		}
	}
}

func TestHookKeys(t *testing.T) {
	// Verify key constants are defined
	keys := []string{
		RequestIDKey.Name(),
		ProviderKey.Name(),
		ModelKey.Name(),
		InputCountKey.Name(),
		DimensionsKey.Name(),
		DurationMsKey.Name(),
		PromptTokensKey.Name(),
		TotalTokensKey.Name(),
		ErrorKey.Name(),
	}

	for _, key := range keys {
		if key == "" {
			t.Error("key has empty name")
		}
	}
}

func TestEmitEmbedStarted(_ *testing.T) {
	// Test that emit functions don't panic
	ctx := context.Background()
	emitEmbedStarted(ctx, "req-123", "openai", 5)
	// No panic = success
}

func TestEmitEmbedCompleted(_ *testing.T) {
	ctx := context.Background()
	resp := &EmbeddingResponse{
		Model:      "text-embedding-3-small",
		Dimensions: 1536,
		Usage: Usage{
			PromptTokens: 10,
			TotalTokens:  10,
		},
	}
	emitEmbedCompleted(ctx, "req-123", "openai", resp, 100*time.Millisecond)
	// No panic = success
}

func TestEmitEmbedFailed(_ *testing.T) {
	ctx := context.Background()
	err := errors.New("test error")
	emitEmbedFailed(ctx, "req-123", "openai", err, 50*time.Millisecond)
	// No panic = success
}

func TestEmitProviderCallStarted(_ *testing.T) {
	ctx := context.Background()
	emitProviderCallStarted(ctx, "openai", 3)
	// No panic = success
}

func TestEmitProviderCallCompleted(_ *testing.T) {
	ctx := context.Background()
	resp := &EmbeddingResponse{
		Model:      "text-embedding-3-small",
		Dimensions: 1536,
		Usage: Usage{
			PromptTokens: 10,
			TotalTokens:  10,
		},
	}
	emitProviderCallCompleted(ctx, "openai", resp, 100*time.Millisecond)
	// No panic = success
}

func TestEmitProviderCallFailed(_ *testing.T) {
	ctx := context.Background()
	err := errors.New("provider error")
	emitProviderCallFailed(ctx, "openai", err, 50*time.Millisecond)
	// No panic = success
}

func TestSignalNames(t *testing.T) {
	tests := []struct {
		signal   capitan.Signal
		expected string
	}{
		{EmbedStarted, "vex.embed.started"},
		{EmbedCompleted, "vex.embed.completed"},
		{EmbedFailed, "vex.embed.failed"},
		{ProviderCallStarted, "vex.provider.call.started"},
		{ProviderCallCompleted, "vex.provider.call.completed"},
		{ProviderCallFailed, "vex.provider.call.failed"},
	}

	for _, tt := range tests {
		if tt.signal.Name() != tt.expected {
			t.Errorf("expected signal name %q, got %q", tt.expected, tt.signal.Name())
		}
	}
}

func TestKeyNames(t *testing.T) {
	tests := []struct {
		name     string
		expected string
	}{
		{RequestIDKey.Name(), "vex.request.id"},
		{ProviderKey.Name(), "vex.provider"},
		{ModelKey.Name(), "vex.model"},
		{InputCountKey.Name(), "vex.input.count"},
		{DimensionsKey.Name(), "vex.dimensions"},
		{DurationMsKey.Name(), "vex.duration.ms"},
		{PromptTokensKey.Name(), "vex.tokens.prompt"},
		{TotalTokensKey.Name(), "vex.tokens.total"},
		{ErrorKey.Name(), "vex.error"},
	}

	for _, tt := range tests {
		if tt.name != tt.expected {
			t.Errorf("expected key name %q, got %q", tt.expected, tt.name)
		}
	}
}
