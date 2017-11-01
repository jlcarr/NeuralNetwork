#include <iostream>
#include "graph.h"


#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <queue>
#include <algorithm>
#include <vector>
#include <math.h>
#include <utility>
#include <cstdlib>

using namespace std;


template <class Tkey>
class NeuralNetwork :public data_graph<Tkey,double,double>{
	public:
		//list the neuron in each layer
		vector<unordered_set<Tkey>> layers;
		//the values last computed at each neuron. Could combind with layers?
		unordered_map<Tkey,double> values;
	
		void defineLayers(vector<unordered_set<Tkey>> layersIn){
			layers = layersIn;
		}
	
		void defineBiases(unordered_map<Tkey,double> biasesIn){
			for(auto v = biasesIn.begin(); v != biasesIn.end(); ++v){
				this->addVertex(v->first, v->second);
			}
		}
	
		void addConnection(Tkey tail, Tkey head, double weight){
			this->addEdge(tail, head, weight);
		}
	
		//compute the function
		unordered_map<Tkey,double> compute(unordered_map<Tkey,double> in){
			//reset values to hold only the current input
			values = in;
			
			//go layer by layer
			for (int i = 0; i < layers.size(); i++){
				//for each neuron in the layer
				for (auto neuron = layers[i].begin(); neuron != layers[i].end(); ++neuron){
					//send its output to each connection
					for(auto target = this->E[*neuron].begin(); target != this->E[*neuron].end(); ++target){
						//cout << *neuron << "->" << *target << " weight:" << this->Edata(*neuron,*target) << " value:" << values[*neuron] << "\n";
						if(values.find(*target) != values.end()) values[*target] += this->Edata(*neuron,*target)*values[*neuron];
						else values[*target] = this->Edata(*neuron,*target)*values[*neuron];
					}
				}
				//apply biases and activation functions
				if(i+1<layers.size()) for (auto neuron = layers[i+1].begin(); neuron != layers[i+1].end(); ++neuron) values[*neuron] = 1.0/(1.0+exp(-values[*neuron]-this->Vdata(*neuron)));
			}
			
			//build return value
			unordered_map<Tkey,double> returnValue;
			for(auto neuron = (--layers.end())->begin(); neuron!=(--layers.end())->end(); ++neuron) returnValue[*neuron]=values[*neuron];
			
			return returnValue;
		}
	
	
		void train(vector<pair<unordered_map<Tkey,double>,unordered_map<Tkey,double>>> trainingData, int iterations, double learningRate){
			//temp weight updates
			unordered_map<Tkey, unordered_map<Tkey,double>> weightsUpdates;
			unordered_map<Tkey, double> biasUpdates;
			
			
			//cout << "Begin Training\n";
			for(int i=0; i<iterations; i++) for(auto t = trainingData.begin(); t!= trainingData.end(); ++t){
				//cout << "Iteration:" << i << "\n";
				//perform a trial run to get the neuron values
				compute(t->first);
				//compute error
				
				//compute the weight updates
				for(auto tail = this->E.begin(); tail != this->E.end(); ++tail){
					for(auto head = tail->second.begin(); head != tail->second.end(); ++head){
						//cout << "Update: " << tail->first << "->" << *head << "\n";
						//cout << "headvalue: " << values[*head] << " tailvalue: " << values[tail->first] << "\n";
						weightsUpdates[tail->first][*head] = -learningRate*values[tail->first]*values[*head]*(1.0-values[*head])*backProp(*head,t->second);
						//cout << weightsUpdates[tail->first][*head] << "\n";
					}
				}
				//compute the bias updates
				for(auto neuron = this->V.begin(); neuron != this->V.end(); ++neuron){
					biasUpdates[*neuron] = -learningRate*values[*neuron]*(1.0-values[*neuron])*backProp(*neuron,t->second);
				}
				
				//perform the weight updates
				for(auto tail = this->E.begin(); tail != this->E.end(); ++tail){
					for(auto head = tail->second.begin(); head != tail->second.end(); ++head){
						this->addEdge(tail->first,*head, this->Edata(tail->first,*head)+weightsUpdates[tail->first][*head]);
					}
				}
				//perform the bias updates
				for(auto neuron = this->V.begin(); neuron != this->V.end(); ++neuron){
					this->addVertex(*neuron,this->Vdata(*neuron)+biasUpdates[*neuron]);
				}
			}
		}
	
	
	private:
		double backProp(Tkey neuron, unordered_map<Tkey,double>& trainingOutputs){
			//if at output node
			if((--layers.end())->find(neuron) != (--layers.end())->end()) return (values[neuron]-trainingOutputs[neuron]);
			
			double result = 0;
			for(auto w = this->E[neuron].begin(); w != this->E[neuron].end(); ++w){
				result += (this->Edata(neuron,*w))*values[*w]*(1.0-values[*w])*backProp(*w, trainingOutputs);
			}
			return result;
			
		}
};
