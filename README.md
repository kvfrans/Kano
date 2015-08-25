###Installation
```npm install kano```

###Usage

```
var kano = require("kano")

var network = kano.neuralnet(numInputs,numOutputs,neuronsPerLayer,numLayers);

var result = network.forward(arrayOfInnputs)
```
Highly reccomend sigmoiding the result of the neural network.

Training:

trainUntilConverge(network,[
			{input: [1,1], result:[1,1]},
			{input: [1,0], result:[0,1]},
			{input: [0,1], result:[1,0]},
			{input: [0,0], result:[0,0]},
		],0.5);
