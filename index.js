var mnist = require('mnist');
var fs = require('fs');

var credit = "kevin";
module.exports = {
	credit: function() {
		return credit;
	},
	neuron: function()
	{
		var neuron = new HiddenNeuron(2,[1,1,1],false);
		return neuron.forward([1,1]);
	},
	layer: function()
	{
		var layer = new HiddenLayer(2,2,false,false);
		return layer.forward([1,1]);
	},
	network: function()
	{
		var network = new NeuralNetwork(2,2,2,2,false,false);
		return network.forward([1,1]);
	},
	xorStructure: function()
	{
		var network = new NeuralNetwork(2,1,3,1,true,false);
		return network.outputLayer.neurons[0].weights;
	},
	adjustment: function()
	{
		var network = new NeuralNetwork(2,1,3,1,true,false);
		network.adjust([1,1],[0]);
		return network.outputLayer.neurons[0].weights;
	},
	trainXOR: function()
	{
		var network = new NeuralNetwork(2,1,3,1,true,true);
		trainUntilConverge(network,[
			{input: [1,1], result:[0]},
			{input: [0,1], result:[1]},
			{input: [1,0], result:[1]},
			{input: [0,0], result:[0]},
		],0.5);
		return network;
	},
	trainMultiXOR: function()
	{
		var network = new NeuralNetwork(2,1,10,2,true,true);
		trainUntilConverge(network,[
			{input: [1,1], result:[0]},
			{input: [0,1], result:[1]},
			{input: [1,0], result:[1]},
			{input: [0,0], result:[0]},
		],0.5);
		return network;
	},
	trainMultipleOutput: function()
	{
		var network = new NeuralNetwork(2,2,10,2,true,true);
		trainUntilConverge(network,[
			{input: [1,1], result:[1,1]},
			{input: [0,1], result:[0,1]},
			{input: [1,0], result:[1,0]},
			{input: [0,0], result:[0,0]},
		],0.5);
		return network;
	},
	trainOpposites: function()
	{
		var network = new NeuralNetwork(2,2,10,2,true,true);
		trainUntilConverge(network,[
			{input: [1,1], result:[1,1]},
			{input: [1,0], result:[0,1]},
			{input: [0,1], result:[1,0]},
			{input: [0,0], result:[0,0]},
		],0.5);
		return network;
	},
	trainColors: function()
	{
		var network = new NeuralNetwork(6,5,10,2,true,true);
		trainUntilConverge(network,[
			{input: [13,2,4,13,5,7], result:[1,0,0,0,0]}, //red
			{input: [15,2,2,6,1,3], result:[1,0,0,0,0]},
			{input: [13,9,1,14,1,8], result:[1,0,0,0,0]},
			{input: [9,6,2,8,1,11], result:[1,0,0,0,0]},
			{input: [14,15,4,8,3,6], result:[1,0,0,0,0]},
			{input: [10,6,4,5,4,1], result:[1,0,0,0,0]},

			{input: [4,4,6,12,11,3], result:[0,1,0,0,0]},
			{input: [4,1,8,3,10,7], result:[0,1,0,0,0]}, //blue
			{input: [5,9,10,11,14,3], result:[0,1,0,0,0]},
			{input: [8,1,12,15,14,0], result:[0,1,0,0,0]},
			{input: [5,2,11,3,13,9], result:[0,1,0,0,0]},
			{input: [2,12,3,14,5,9], result:[0,1,0,0,0]},

			{input: [4,14,12,13,12,4], result:[0,0,1,0,0]}, //green
			{input: [10,2,13,14,13,0], result:[0,0,1,0,0]},
			{input: [8,7,13,3,7,12], result:[0,0,1,0,0]},
			{input: [9,0,12,6,9,5], result:[0,0,1,0,0]},
			{input: [2,6,10,6,5,11], result:[0,0,1,0,0]},
			{input: [0,3,12,9,10,9], result:[0,0,1,0,0]},

			{input: [15,4,11,3,5,0], result:[0,0,0,1,0]}, //yellow/orange
			{input: [15,2,7,8,4,11], result:[0,0,0,1,0]},
			{input: [14,11,9,7,4,14], result:[0,0,0,1,0]},
			{input: [14,5,10,11,3,5], result:[0,0,0,1,0]},
			{input: [13,3,5,4,0,0], result:[0,0,0,1,0]},
			{input: [15,3,9,12,1,2], result:[0,0,0,1,0]},

			{input: [6,7,4,1,7,2], result:[0,0,0,0,1]}, //purple
			{input: [10,14,10,8,13,3], result:[0,0,0,0,1]},
			{input: [9,1,3,13,8,8], result:[0,0,0,0,1]},
			{input: [9,10,1,2,11,3], result:[0,0,0,0,1]},
			{input: [11,15,5,5,14,13], result:[0,0,0,0,1]},
			{input: [11,14,9,0,13,4], result:[0,0,0,0,1]},
		],0.1);
		saveNetwork(network,"networks/flatui.json");
		return network;
	},
	trainDigits: function()
	{
		var set = mnist.set(8000,2000);
		console.log(set.training[0]);
		console.log(set.training[0]["output"]);


		var network = new NeuralNetwork(784,10,25,1,true,true);
		trainMNIST(network,0.5);
		for(var i = 0; i < 10; i++)
		{
			displayDigit(set.test[i]["input"]);
			console.log(network.forward(set.test[i]["input"]));
			console.log(chooseOne(network.forward(set.test[i]["input"])));
		}
	},
}

var WeightAlpha = 0.01;

function sigmoid(t,sigmoidActive) {
	if(!sigmoidActive)
	{
		return t;
	}
    return 1/(1+Math.pow(Math.E, -t));
}

var HiddenNeuron = function(numInputs,weights,sigmoidActive)
{
	this.numInputs = numInputs;
	this.weights = weights;
	this.error = 0;
	this.result = 0;
	this.forward = function(inputs)
	{
		var total = 0;
		for(var i = 0; i < inputs.length; i++)
		{
			total += inputs[i] * this.weights[i];
		}
		total += weights[inputs.length];
		this.result = sigmoid(total,sigmoidActive);
		return this.result;
	}
};

var HiddenLayer = function(numInputs,neuronsPerLayer,sigmoidActive,randomInit)
{
	this.neurons = [];
	this.numInputs = numInputs;
	for(var i = 0; i < neuronsPerLayer; i++)
	{
		var weights = [];
		for(var x = 0; x < numInputs+1; x++)
		{
			if(randomInit)
			{
				weights.push(Math.random()-0.5);
			}
			else
			{
				weights.push(1);
			}
		}
		var neuron = new HiddenNeuron(this.numInputs,weights,sigmoidActive);
		this.neurons.push(neuron);
	}
	this.forward = function(inputs)
	{
		var outputs = [];
		for(var i = 0; i < this.neurons.length; i++)
		{
			outputs.push(this.neurons[i].forward(inputs));
		}
		return outputs;
	}
}


var NeuralNetwork = function(numInputs, numOutputs, neuronsPerLayer, numLayers, sigmoidActive, randomInit)
{
	this.numInputs = numInputs;
	this.numOutputs = numOutputs;
	this.neuronsPerLayer = neuronsPerLayer;
	this.numLayers = numLayers;

	this.hiddenlayers = [];
	this.outputLayer = new HiddenLayer(this.neuronsPerLayer,numOutputs,sigmoidActive);

	this.hiddenlayers.push(new HiddenLayer(this.numInputs,this.neuronsPerLayer,sigmoidActive,randomInit));
	for(var i = 0; i < this.numLayers-1; i++)
	{
		this.hiddenlayers.push(new HiddenLayer(this.neuronsPerLayer,this.neuronsPerLayer,sigmoidActive, randomInit));
	}
	this.forward = function(inputs)
	{
		var currentInput = inputs;
		for(var i = 0; i < this.numLayers; i++)
		{
			currentInput = this.hiddenlayers[i].forward(currentInput);
		}
		return this.outputLayer.forward(currentInput);
	}
	this.adjust = function(inputs,desired)
	{
		//desired is an array of the results that you want.
		var results = this.forward(inputs);
		for(var i = 0; i < this.outputLayer.neurons.length; i++)
		{
			this.outputLayer.neurons[i].error = (desired[i] - results[i]) * results[i]*(1-results[i]);
			for(var x = 0; x < this.hiddenlayers[numLayers-1].neurons.length; x++)
			{
				this.outputLayer.neurons[i].weights[x] += WeightAlpha * this.outputLayer.neurons[i].error * this.hiddenlayers[numLayers-1].neurons[x].result;
			}
			this.outputLayer.neurons[i].weights[this.hiddenlayers[numLayers-1].neurons.length] += WeightAlpha * this.outputLayer.neurons[i].error * 1;
		}
		//k = layers
		//i = neurons in the layer
		//x = neurons in the top layer
		for(var k = this.numLayers-1; k >= 0; k--)
		{

			if(k == this.numLayers-1) //if its the top layer
			{
				for(var i = 0; i < this.hiddenlayers[k].neurons.length; i++) //loop through neurons
				{
					this.hiddenlayers[k].neurons[i].error = 0;
					for(var x = 0; x < this.outputLayer.neurons.length; x++) //loop through top neurons
					{
						this.hiddenlayers[k].neurons[i].error += this.outputLayer.neurons[x].weights[i] * this.outputLayer.neurons[x].error;
					}
					// console.log(this.hiddenlayers[k].neurons[i].error);
					this.hiddenlayers[k].neurons[i].error = this.hiddenlayers[k].neurons[i].result * (1-this.hiddenlayers[k].neurons[i].result) * this.hiddenlayers[k].neurons[i].error;
				}
			}
			else
			{
				for(var i = 0; i < this.hiddenlayers[k].neurons.length; i++) //loop through neurons
				{
					this.hiddenlayers[k].neurons[i].error = 0;
					for(var x = 0; x < this.hiddenlayers[k+1].neurons.length; x++) //loop through
					{
						this.hiddenlayers[k].neurons[i].error += this.hiddenlayers[k+1].neurons[x].weights[i] * this.hiddenlayers[k+1].neurons[x].error;
					}
					this.hiddenlayers[k].neurons[i].error = this.hiddenlayers[k].neurons[i].result * (1-this.hiddenlayers[k].neurons[i].result) * this.hiddenlayers[k].neurons[i].error;
				}
			}

			if(k > 0)
			{
				for(var i = 0; i < this.hiddenlayers[k].neurons.length; i++) //loop through neurons
				{
					for(var x = 0; x < this.hiddenlayers[k].neurons[i].weights.length - 1; x++) //loop through weights coming from neurons
					{
						this.hiddenlayers[k].neurons[i].weights[x] += WeightAlpha * this.hiddenlayers[k-1].neurons[i].result * this.hiddenlayers[k].neurons[i].error;
					}
					this.hiddenlayers[k].neurons[i].weights[this.hiddenlayers[k].neurons[i].weights.length-1] += WeightAlpha * 1 * this.hiddenlayers[k].neurons[i].error;
				}
			}
			else
			{
				for(var i = 0; i < this.hiddenlayers[k].neurons.length; i++) //loop through neurons
				{
					for(var x = 0; x < this.hiddenlayers[k].neurons[i].weights.length - 1; x++) //loop through weights coming from neurons
					{
						this.hiddenlayers[k].neurons[i].weights[x] += WeightAlpha * inputs[x] * this.hiddenlayers[k].neurons[i].error;
					}
					this.hiddenlayers[k].neurons[i].weights[this.hiddenlayers[k].neurons[i].weights.length-1] += WeightAlpha * 1 * this.hiddenlayers[k].neurons[i].error;
				}
			}
		}
	}
}


function drawNetwork(network)
{
	// var results = [[]];
	for(var y = 0; y <= 20; y += 1)
	{
		var layer = "";
		for(var x = 0; x <= 20; x += 1)
		{
			var res = network.forward([x/20.0,y/20.0]);
			if(res > 0.5)
			{
				layer = layer + "X";
			}
			else
			{
				layer = layer + "-";
			}

		}
		console.log(layer);
	}
}




function testNetwork(network,tests,error)
{
	var allgood = true;
	for(var i = 0; i < tests.length; i++)
	{
		var currenttest = tests[i];
		// console.log(currenttest);
		var result = network.forward(currenttest.input);
		// console.log(Math.round(result[0]) + " " + currenttest.result[0]);
		for(var x = 0; x < result.length; x++)
		{
			// if(Math.round(result[x]) != currenttest.result[x])
			if(Math.abs(result[x] - currenttest.result[x]) > error)
			{
				allgood = false;
			}
		}
	}
	// console.log(allgood);
	return allgood;
}

function chooseOne(results)
{
  var highest = 0;
  var highestindex = 0;
  for(var i = 0; i < results.length; i++)
  {
    if(results[i] > highest)
    {
      highest = results[i];
      highestindex = i;
    }
  }
  return highestindex;
}

function trainUntilConverge(network,data,error)
{
	var x = 0;
	while(testNetwork(network,data,error) == false)
	{
		x++;
		for(var i = 0; i < data.length; i++)
		{
			network.adjust(data[i].input,data[i].result);
		}
		// console.log(x);
	}
	console.log(x);
	// drawNetwork(network);
	// console.log(testNetwork(network,[
	// 	{input: [1,1], result:[0]},
	// 	{input: [0,1], result:[1]},
	// 	{input: [1,0], result:[1]},
	// 	{input: [0,0], result:[0]},
	// ]));
	return network;
}


function testMNIST(network,tests,error)
{
	var allgood = true;
	for(var i = 0; i < tests.length; i++)
	{
		var currenttest = tests[i];
		// console.log(currenttest);
		var result = network.forward(currenttest.input);
		// console.log(Math.round(result[0]) + " " + currenttest.result[0]);
		for(var x = 0; x < result.length; x++)
		{
			// if(Math.round(result[x]) != currenttest.result[x])
			if(Math.abs(result[x] - currenttest.output[x]) > error)
			{
				allgood = false;
			}
		}
	}
	// console.log(allgood);
	return allgood;
}

function trainMNIST(network,error)
{
	var set = mnist.set(8000,2000);
	var training = set.training;
	var x = 0;
	console.log("started training");
	// while(testMNIST(network,set.training,error) == false)
	for(var z = 0; z < 1500; z++)
	{
		x++;
		for(var i = 0; i < training.length; i++)
		{
			network.adjust(training[i].input,training[i].output);
		}
		console.log(x);
	}
	console.log(x);
	// drawNetwork(network);
	// console.log(testNetwork(network,[
	// 	{input: [1,1], result:[0]},
	// 	{input: [0,1], result:[1]},
	// 	{input: [1,0], result:[1]},
	// 	{input: [0,0], result:[0]},
	// ]));
	return network;
}

function displayDigit(digit)
{
	var i = 0;
	for(var y = 0; y < 28; y++)
	{
		var row = "";
		for(var x = 0; x < 28; x++)
		{
			if(digit[i] > 0)
			{
				row += "O"
			}
			else
			{
				row += "-";
			}
			i++;
		}
		console.log(row);
	}
}

function saveNetwork(network,outputFilename)
{
	fs.writeFile(outputFilename, JSON.stringify(network, null, 4), function(err)
	{
	    if(err) {
	      console.log(err);
	    } else {
	      console.log("JSON saved to " + outputFilename);
	    }
	});
}
