//
// Created by timian on 12/19/18.
//

#ifndef LIF_NEURALNETWORK_H
#define LIF_NEURALNETWORK_H

#include <string>
#include "Model.h"
#include "keras_model.h"

class NeuralNetwork : public Model {
private:

public:
    NeuralNetwork(std::string &path, std::string &filename, float inputMaxValue, int inputMaxIndex);
    KerasModel model;
    float maxValue;

    float predict(float value) override;
    int maxIndex;
};


#endif //LIF_NEURALNETWORK_H
