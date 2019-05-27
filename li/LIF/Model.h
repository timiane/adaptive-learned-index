//
// Created by timian on 12/19/18.
//


#ifndef LIF_MODEL_H
#define LIF_MODEL_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "modelsEnum.cpp"

class Model {
public:
    ModelEnum type;

    ModelEnum getType();
    virtual float predict(float value);
};


#endif //LIF_MODEL_H
