package hybrid_vector_store

import (
	"math"
	"testing"
)

const epislon = 1e-6

func almostEqual(a, b float32) bool {
	return math.Abs(float64(a-b)) < epislon
}

func TestNewDistance(t *testing.T) {
	tests := []struct {
		name          string
		distanceKind  DistanceKind
		expectError   bool
		expectedError error
	}{
		{
			name:         "euclidean distance",
			distanceKind: Euclidean,
			expectError:  false,
		},
		{
			name:         "cosine distance",
			distanceKind: Cosine,
			expectError:  false,
		},
		{
			name:         "dot product distance",
			distanceKind: DotProduct,
			expectError:  false,
		},
		{
			name:          "unknown distance",
			distanceKind:  "unknown",
			expectError:   true,
			expectedError: ErrUknownDistanceKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist, err := NewDistance(tt.distanceKind)
			if tt.expectError {
				if err == nil {
					t.Errorf("NewDistance(%s) expected error, got nil", tt.distanceKind)
				}
				if tt.expectedError != nil && err != tt.expectedError {
					t.Errorf("NewDistance(%s) expected error %v, got %v", tt.distanceKind, tt.expectedError, err)
				}
			} else {
				if err != nil {
					t.Errorf("NewDistance(%s) unexpected error: %v", tt.distanceKind, err)
				}
				if dist == nil {
					t.Errorf("NewDistance(%s) returned nil distance", tt.distanceKind)
				}
			}
		})
	}
}
