//
// Created by timian on 12/19/18.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::string& path, std::string& filename, float inputMaxValue, int inputMaxIndex) {
    maxValue = inputMaxValue;
    maxIndex = inputMaxIndex;
    type = NN;
    model.LoadModel(path + filename);
}

float NeuralNetwork::predict(float key) {
    key = key/maxValue;
//Create a 1D tensor, on length 1 for input data
    Tensor in(1);
    in.data_ = {key};
    Tensor out;
    model.Apply(&in, &out);
    double dataPoint = out.data_[0] * (maxIndex - 1);
    return dataPoint;
}