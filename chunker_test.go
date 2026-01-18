package vex

import (
	"strings"
	"testing"
)

func TestChunker_ChunkNone(t *testing.T) {
	chunker := &Chunker{Strategy: ChunkNone}
	text := "This is a test. With multiple sentences."

	chunks := chunker.Chunk(text)

	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0] != text {
		t.Errorf("expected original text, got %s", chunks[0])
	}
}

func TestChunker_ChunkSentence(t *testing.T) {
	chunker := &Chunker{
		Strategy:  ChunkSentence,
		TrimSpace: true,
	}

	t.Run("splits on periods", func(t *testing.T) {
		text := "First sentence. Second sentence. Third sentence."
		chunks := chunker.Chunk(text)

		if len(chunks) != 3 {
			t.Errorf("expected 3 chunks, got %d", len(chunks))
		}
	})

	t.Run("splits on exclamation marks", func(t *testing.T) {
		text := "Hello! How are you? I'm fine."
		chunks := chunker.Chunk(text)

		if len(chunks) != 3 {
			t.Errorf("expected 3 chunks, got %d", len(chunks))
		}
	})

	t.Run("handles text without sentence endings", func(t *testing.T) {
		text := "No sentence ending here"
		chunks := chunker.Chunk(text)

		if len(chunks) != 1 {
			t.Errorf("expected 1 chunk, got %d", len(chunks))
		}
	})

	t.Run("filters empty chunks", func(t *testing.T) {
		text := "First.  Second."
		chunks := chunker.Chunk(text)

		for _, chunk := range chunks {
			if chunk == "" {
				t.Error("empty chunk should be filtered")
			}
		}
	})
}

func TestChunker_ChunkParagraph(t *testing.T) {
	chunker := &Chunker{
		Strategy:  ChunkParagraph,
		TrimSpace: true,
	}

	t.Run("splits on double newlines", func(t *testing.T) {
		text := "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
		chunks := chunker.Chunk(text)

		if len(chunks) != 3 {
			t.Errorf("expected 3 chunks, got %d", len(chunks))
		}
	})

	t.Run("handles single paragraph", func(t *testing.T) {
		text := "Just one paragraph with no breaks."
		chunks := chunker.Chunk(text)

		if len(chunks) != 1 {
			t.Errorf("expected 1 chunk, got %d", len(chunks))
		}
	})

	t.Run("filters empty paragraphs", func(t *testing.T) {
		text := "First.\n\n\n\nSecond."
		chunks := chunker.Chunk(text)

		for _, chunk := range chunks {
			if strings.TrimSpace(chunk) == "" {
				t.Error("empty chunk should be filtered")
			}
		}
	})
}

func TestChunker_ChunkFixed(t *testing.T) {
	t.Run("splits at max size", func(t *testing.T) {
		chunker := &Chunker{
			Strategy:  ChunkFixed,
			MaxSize:   10,
			Overlap:   0,
			TrimSpace: false,
		}

		text := "12345678901234567890" // 20 chars
		chunks := chunker.Chunk(text)

		if len(chunks) != 2 {
			t.Errorf("expected 2 chunks, got %d", len(chunks))
		}
		if len(chunks[0]) != 10 {
			t.Errorf("expected chunk size 10, got %d", len(chunks[0]))
		}
	})

	t.Run("handles overlap", func(t *testing.T) {
		chunker := &Chunker{
			Strategy:  ChunkFixed,
			MaxSize:   10,
			Overlap:   3,
			TrimSpace: false,
		}

		text := "1234567890123456" // 16 chars
		chunks := chunker.Chunk(text)

		// With overlap of 3, step is 7
		// Chunk 1: 0-10, Chunk 2: 7-16
		if len(chunks) < 2 {
			t.Errorf("expected at least 2 chunks with overlap, got %d", len(chunks))
		}
	})

	t.Run("returns single chunk for short text", func(t *testing.T) {
		chunker := &Chunker{
			Strategy: ChunkFixed,
			MaxSize:  100,
		}

		text := "Short text"
		chunks := chunker.Chunk(text)

		if len(chunks) != 1 {
			t.Errorf("expected 1 chunk for short text, got %d", len(chunks))
		}
	})

	t.Run("handles zero or negative max size", func(t *testing.T) {
		chunker := &Chunker{
			Strategy: ChunkFixed,
			MaxSize:  0,
		}

		text := "Some text"
		chunks := chunker.Chunk(text)

		if len(chunks) != 1 {
			t.Errorf("expected original text for zero max size")
		}
	})
}

func TestChunker_TrimSpace(t *testing.T) {
	t.Run("trims whitespace when enabled", func(t *testing.T) {
		chunker := &Chunker{
			Strategy:  ChunkParagraph,
			TrimSpace: true,
		}

		text := "  First paragraph.  \n\n  Second paragraph.  "
		chunks := chunker.Chunk(text)

		for _, chunk := range chunks {
			if chunk != strings.TrimSpace(chunk) {
				t.Errorf("chunk should be trimmed: %q", chunk)
			}
		}
	})

	t.Run("preserves whitespace when disabled", func(t *testing.T) {
		chunker := &Chunker{
			Strategy:  ChunkFixed,
			MaxSize:   20,
			TrimSpace: false,
		}

		text := "  hello world  "
		chunks := chunker.Chunk(text)

		// With TrimSpace false, whitespace should be preserved
		if len(chunks) == 0 {
			t.Fatal("expected at least one chunk")
		}
		if chunks[0] != text {
			t.Errorf("expected whitespace preserved, got %q", chunks[0])
		}
	})
}

func TestDefaultChunker(t *testing.T) {
	chunker := DefaultChunker()

	if chunker.Strategy != ChunkNone {
		t.Errorf("expected ChunkNone strategy")
	}
	if chunker.MaxSize != 512 {
		t.Errorf("expected MaxSize 512, got %d", chunker.MaxSize)
	}
	if chunker.Overlap != 50 {
		t.Errorf("expected Overlap 50, got %d", chunker.Overlap)
	}
	if !chunker.TrimSpace {
		t.Error("expected TrimSpace to be true")
	}
}
