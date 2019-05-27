//
// Created by smadderfar on 12/19/18.
//

#include "Stage.h"

void Stage::addModel(Model* newModel) {
    models.push_back(newModel);
}

Model* Stage::getModel(long position) {
    return models[position];
}
