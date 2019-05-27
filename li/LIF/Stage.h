//
// Created by smadderfar on 12/19/18.
//

#ifndef LIF_LAYER_H
#define LIF_LAYER_H

#include "Model.h"
#include <memory>

class Stage {
public:
    Stage() = default;;
    std::vector<Model*> models;
    void addModel(Model* model);
    Model* getModel(long position);
    ~Stage() = default;;
};


#endif //LIF_LAYER_H
