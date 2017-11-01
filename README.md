# NeuralNetwork
A C++ library for creating and training artificial neural networks

## Features
* Uses standard C++11 library.
* Simple design and usage.

## Installation
None required; simply dowload the library, then include it in your main.cpp
* Required graph.h library as a dependency

## Usage Examples
### Creating a Neural Network
To create a basic neural network with 2 inputs, 1 output and one hidden layer with 2 neurons
```c++
NeuralNetwork<string> testNeuralNetwork;

//first step is to define the neurons with their (initial) biases
unordered_map<string,double> neuronBiases = {
	{"i1",0},{"i2",0},{"h1",0.75},{"h2",0.25},{"o1",0.5}
};
testNeuralNetwork.defineBiases(neuronBiases);

//next denote which layers each neuron belongs to
vector<unordered_set<string>> neuronLayers = {
	{"i1","i2"},{"h1","h2"},{"o1"}
};
testNeuralNetwork.defineLayers(neuronLayers);

//add connections between nodes in sequential layers (with initial weights)
testNeuralNetwork.addConnection("i1","h1",0.75);
testNeuralNetwork.addConnection("i1","h2",0.25);
testNeuralNetwork.addConnection("i2","h1",0.25);
testNeuralNetwork.addConnection("i2","h2",0.75);
testNeuralNetwork.addConnection("h1","o1",0.75);
testNeuralNetwork.addConnection("h2","o1",0.25);

testNeuralNetwork.print();
```

Output will be:
```console
o1(0.5) -> 
h2(0.25) -> o1(0.25), 
h1(0.75) -> o1(0.75), 
i2(0) -> h2(0.75), h1(0.25), 
i1(0) -> h2(0.25), h1(0.75), 
```

### Training a Neural Network
To train the neural network created in the example above to replicate an XOR gate
```c++
//perform training on given data by backpropagation
//data is given is that of an XOR gate
vector<pair<unordered_map<string,double>,unordered_map<string,double>>> trainingData = {
	{{{"i1",0},{"i2",0}}	,	{{"o1",0}}},
	{{{"i1",0},{"i2",1}}	,	{{"o1",1}}},
	{{{"i1",1},{"i2",0}}	,	{{"o1",1}}},
	{{{"i1",1},{"i2",1}}	,	{{"o1",0}}}
};
test.train(trainingData, 10000, 0.5);

//test out the results of training
cout << "Results:\n";
for(int i = 0; i <=1; i++) for(int j = 0; j <=1; j++){
	unordered_map<string,double> results = test.compute(unordered_map<string,double>{{"i1",i},{"i2",j}});
	cout << "given input i1=" << i << " and i2=" << j << "\n";
	for(auto o = results.begin(); o != results.end(); ++o){
		cout << o->first << "=" << o->second << "\n";
	}
}
```

Output will be:
```console
Results:
given input i1=0 and i=0
o1=0.027044
given input i1=0 and i2=1
o1=0.97626
given input i1=1 and i2=0
o1=0.976217
given input i1=1 and i2=1
o1=0.0284718
```
