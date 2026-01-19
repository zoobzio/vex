package benchmarks

import (
	"testing"

	"github.com/zoobzio/vex"
)

func BenchmarkVector_Normalize(b *testing.B) {
	vec := make(vex.Vector, 1536)
	for i := range vec {
		vec[i] = float32(i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vec.Normalize()
	}
}

func BenchmarkVector_Norm(b *testing.B) {
	vec := make(vex.Vector, 1536)
	for i := range vec {
		vec[i] = float32(i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vec.Norm()
	}
}

func BenchmarkVector_Dot(b *testing.B) {
	v1 := make(vex.Vector, 1536)
	v2 := make(vex.Vector, 1536)
	for i := range v1 {
		v1[i] = float32(i) / 1536.0
		v2[i] = float32(1536-i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v1.Dot(v2)
	}
}

func BenchmarkVector_CosineSimilarity(b *testing.B) {
	v1 := make(vex.Vector, 1536)
	v2 := make(vex.Vector, 1536)
	for i := range v1 {
		v1[i] = float32(i) / 1536.0
		v2[i] = float32(1536-i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v1.CosineSimilarity(v2)
	}
}

func BenchmarkVector_EuclideanDistance(b *testing.B) {
	v1 := make(vex.Vector, 1536)
	v2 := make(vex.Vector, 1536)
	for i := range v1 {
		v1[i] = float32(i) / 1536.0
		v2[i] = float32(1536-i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v1.EuclideanDistance(v2)
	}
}

func BenchmarkPool_Mean(b *testing.B) {
	vectors := make([]vex.Vector, 10)
	for i := range vectors {
		vec := make(vex.Vector, 1536)
		for j := range vec {
			vec[j] = float32(i*1536+j) / 15360.0
		}
		vectors[i] = vec
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vex.Pool(vectors, vex.PoolMean)
	}
}

func BenchmarkPool_Max(b *testing.B) {
	vectors := make([]vex.Vector, 10)
	for i := range vectors {
		vec := make(vex.Vector, 1536)
		for j := range vec {
			vec[j] = float32(i*1536+j) / 15360.0
		}
		vectors[i] = vec
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = vex.Pool(vectors, vex.PoolMax)
	}
}
