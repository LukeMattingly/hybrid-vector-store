package hybrid_vector_store

import (
	"errors"
	"math"
)

// ErrUnKnownDistanceKind is returned when an unknown distance kind is provided to NewDistance
var ErrUknownDistanceKind = errors.New("unknown distance kind")

// DistanceKind represents the type of distance metric to use for vector comparisons
// Different distance metrics are suitable for different use cases:
// Eucliedia (L2): Measures absolute spatial distance between points
// Cosine: MEasure angular similarity, independent of magnitude
// DotProduct: Measure engative inner product, useful for Maximum Inner Product Search ( MIPS)

type DistanceKind string

const (

	// Euclidean (L2) distance measure the straight- line distance between two points.
	// Use this when the magnitude of vectors matters.
	// Formular: sqrt(sum(a[i] - b[i])^2))
	Euclidean DistanceKind = "l2"

	// Cosine distance measure the angular different between vectors (1- cosine similarity)
	// Use this when you care about direction but no magnitude (e.g. text embeddings)
	// Formula: 1- dot(a,b) / (||a|| * ||b||)
	// Range: [0, 2] where 0 = identical direction, 1 = orthogonal, 2 = opposite
	Cosine DistanceKind = "cosine"

	// DotProduct ocmputes negative inner product, useful for Maximum Inner Product Search
	// Use this when vectors are already normalized or when you want to find a maximum similarity
	// Formula: -dot(a,b)
	DotProduct DistanceKind = "dot"
)

// Singleton instances of distance strategies.
// These are stateless and can be safely reused across goroutines.
var (
	euclideanDistance  = euclidean{}
	cosineDist         = cosine{}
	dotProductDistance = dot{}
)

// Distance is the interface for computing distances between vectors.
// Implementations provide different distance metrics for vector similarity search

type Distance interface {
	//Calculate computes the distance between two vectors a and b.
	// The vectors must have the same dimensionality.
	// Returns a float 32 representing the distance (lower values = more similar).
	Calculate(a, b []float32) float32

	// CalculateBatch computes distances from multiple query vectors to a single target vector
	// This is more efficient than calling Calculate multiples times as it can optimize computations (e.g. precomputing norms for cosine distance)
	//
	// Parameters:
	// - queries: slice of query vectors(each vector is []float32)
	// - target: single target vecotr to compare against
	//
	// Returns:
	// - slice of distances where result[i] is the distance from queries[i] to target
	//
	// All vectors (queries and target) must have the same dimensionality.
	CalculateBatch(queries [][]float32, target []float32) []float32
}

// NewDistance retursn a singlton Distance implementation for the specified metric type.
// The returned instances are stateless and safe for concurrent use across goroutines
// Returns ErrUnknownDistanceKind if the distance kind is not recognized.
// Example:
// dist, err := NewDistance(Euclidean)
//
//	if err != nil{
//			log.Fatal(err)
//	}
//
// distance := dist.Calculate([]float32{1,2,3}, []float32{4,5,6})
func NewDistance(t DistanceKind) (Distance, error) {
	switch t {
	case Euclidean:
		return euclideanDistance, nil
	case Cosine:
		return cosineDist, nil
	case DotProduct:
		return dotProductDistance, nil
	default:
		return nil, ErrUknownDistanceKind
	}
}

type euclidean struct{}

func (e euclidean) Calculate(a, b []float32) float32 {
	return l2Distance(a, b)
}

func (e euclidean) CalculateBatch(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		results[i] = l2Distance(query, target)
	}
	return results
}

// l2Distance computes tthe Euclidean (L2) distance between two vectors.
// This is the most common distance metric, measure straight-line distance
//
// Time complexity O(n) where n is the vector dimension
func l2Distance(a, b []float32) float32 {
	validateLengths(a, b)
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt((float64(sum))))
}

// cosine implements the Distance interface using cosine distance.
// This measure angular similiarty between vectors, independent of their magnitude
type cosine struct{}

func (c cosine) Calculate(a, b []float32) float32 {
	return cosineDistance(a, b)
}

func (c cosine) CalculateBatch(queries [][]float32, target []float32) []float32 {
	//Optimize by precomputing the target's norm once
	normTarget := norm(target)

	results := make([]float32, len(queries))
	for i, query := range queries {
		normQuery := norm(query)
		results[i] = cosineDistanceWithNorms(query, target, normQuery, normTarget)
	}
	return results
}

// cosineDistanceWithNorms computes cosine distance using precomputed norms.
// This is more efficient when you need to compute distances from one vector
// to many others, as you can precompute the target vector's norm once.
//
// Parameters:
//   - a, b: the two vectors to compare
//   - normA, normB: precomputed norms (magnitudes) of vectors a and b
//
// This function is ~6x faster than cosineDistance when norms are precomputed.
//
// Time complexity: O(n) for dot product only (norms already computed)
func cosineDistanceWithNorms(a, b []float32, normA, normB float32) float32 {
	dot := dotProduct(a, b)

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (normA * normB)
	// Clamp to [-1, 1] to handle floating point precision errors
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

// cosineDistance computes the cosine distance between two vectors
// Cosine distance = 1- cosine similarity, measuring angular difference.
//
//	Formula: 1 - (dot(a, b) / (||a|| * ||b||))
//
// Range: [0:2]
// - 0: vectors point in the same direction (identical)
// - 1: vectors or orthogonal (perpendicular)
// - 2: vectors point in opposite directions
//
// Special cases:
// - If either vector has zero magnitude, returns 1.0
// - Clamps similarity to [-1, 1] to handle floating point errors
//
// Time complexity O(n) where n is the vector dimension
func cosineDistance(a, b []float32) float32 {
	const epsilon = 1e-6

	dot := dotProduct(a, b)
	normA := norm(a)
	if normA < epsilon {
		return 1.0
	}
	normB := norm(b)
	if normB < epsilon {
		return 1.0
	}

	similarity := dot / (normA * normB)
	// Clamp to [-1, 1] to handle floating point precision errors
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}
	return 1 - similarity
}

// dotProduct computes the dot product (inner product) of two vectors.
// This measure how much two vectors align with each other.
//
// Formula: sum(a[i] * b[i])
//
// Returns:
// - Positive value: vector point in similar directions
// - Zero: vectors are orthogonal (perpendicular)
// - Negative value: vectors point in opposite directions
//
// Time complexity O(n) where n is the vector dimension
func dotProduct(a, b []float32) float32 {
	validateLengths(a, b)
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// dot implements the Distance interface using negative inner product.
// This is useful for Maximum Inner Product Search (MIPS).
type dot struct{}

func (d dot) Calculate(a, b []float32) float32 {
	return innerProduct(a, b)
}

func (d dot) CalculateBatch(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		results[i] = innerProduct(query, target)
	}
	return results
}

// innerProduct computes the negative inner product of two vectors.
// This is used in Maximum Inner Product Search (MIPS), where we want to
// find vectors with the highest inner product (maximum similarity).
// By negating, we convert a maximization problem into a minimization problem.
//
// Formula: -sum(a[i] * b[i])
//
// Use cases:
//   - Recommendation systems
//   - Information retrieval
//   - When working with already normalized vectors
//
// Time complexity: O(n) where n is the vector dimension
func innerProduct(a, b []float32) float32 {
	return -dotProduct(a, b)
}

// norm computes the l2 norm (euclidiean length/magnitude) of a vector.
// This represents the legnth of the vector from origin
//
// Formula: sqrt(sum(v[i]^2))
//
// Time Complexity: O(n) where n is the vector dimension
func norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

func validateLengths(a, b []float32) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
}
