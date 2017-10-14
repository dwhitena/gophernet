package gophernet

import (
	"context"
	"testing"
)

var inputNeurons = 4
var outputNeurons = 3
var hiddenNeurons = 3
var numEpochs = 5000
var learningRate = 0.3

var config = GopherNetConfig{
	InputNeurons:  inputNeurons,
	OutputNeurons: outputNeurons,
	HiddenNeurons: hiddenNeurons,
	NumEpochs:     numEpochs,
	LearningRate:  learningRate,
}

func initTest(ctx context.Context, t *testing.T) *GopherNet {
	return NewNetwork(config)
}
func TestGopherNetConfig(t *testing.T) {

	if config.InputNeurons != inputNeurons {
		t.Errorf("config.InputNeurons != %v", inputNeurons)
	}

	if config.OutputNeurons != outputNeurons {
		t.Errorf("config.OutputNeurons != %v", outputNeurons)
	}

	if config.HiddenNeurons != hiddenNeurons {
		t.Errorf("config.HiddenNeurons != %v", hiddenNeurons)
	}

	if config.NumEpochs != numEpochs {
		t.Errorf("config.NumEpochs != %v", numEpochs)
	}

	if config.LearningRate != learningRate {
		t.Errorf("config.LearningRate != %v", learningRate)
	}
}
