package vex

import (
	"strings"
	"unicode"
)

// Chunker splits text into smaller pieces for embedding.
type Chunker struct {
	Strategy    ChunkStrategy
	MaxSize     int  // Maximum chunk size in characters (for ChunkFixed)
	Overlap     int  // Overlap between chunks (for ChunkFixed)
	TrimSpace   bool // Trim whitespace from chunks
}

// DefaultChunker returns a chunker with sensible defaults.
func DefaultChunker() *Chunker {
	return &Chunker{
		Strategy:  ChunkNone,
		MaxSize:   512,
		Overlap:   50,
		TrimSpace: true,
	}
}

// Chunk splits text according to the configured strategy.
func (c *Chunker) Chunk(text string) []string {
	if c.Strategy == ChunkNone {
		return []string{text}
	}

	var chunks []string
	switch c.Strategy {
	case ChunkSentence:
		chunks = c.chunkBySentence(text)
	case ChunkParagraph:
		chunks = c.chunkByParagraph(text)
	case ChunkFixed:
		chunks = c.chunkByFixed(text)
	default:
		chunks = []string{text}
	}

	if c.TrimSpace {
		for i, chunk := range chunks {
			chunks[i] = strings.TrimSpace(chunk)
		}
	}

	// Filter empty chunks
	result := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if chunk != "" {
			result = append(result, chunk)
		}
	}
	return result
}

func (*Chunker) chunkBySentence(text string) []string {
	var chunks []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])

		// Check for sentence-ending punctuation followed by space or end
		if isSentenceEnd(runes[i]) {
			if i+1 >= len(runes) || unicode.IsSpace(runes[i+1]) {
				chunks = append(chunks, current.String())
				current.Reset()
			}
		}
	}

	// Don't forget remaining text
	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	return chunks
}

func (*Chunker) chunkByParagraph(text string) []string {
	// Split on double newlines
	paragraphs := strings.Split(text, "\n\n")
	chunks := make([]string, 0, len(paragraphs))
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p != "" {
			chunks = append(chunks, p)
		}
	}
	return chunks
}

func (c *Chunker) chunkByFixed(text string) []string {
	if c.MaxSize <= 0 {
		return []string{text}
	}

	runes := []rune(text)
	if len(runes) <= c.MaxSize {
		return []string{text}
	}

	var chunks []string
	step := c.MaxSize - c.Overlap
	if step <= 0 {
		step = c.MaxSize
	}

	for i := 0; i < len(runes); i += step {
		end := i + c.MaxSize
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}

	return chunks
}

func isSentenceEnd(r rune) bool {
	return r == '.' || r == '!' || r == '?'
}
