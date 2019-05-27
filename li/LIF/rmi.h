//
// Created by timian on 12/19/18.
//
#include <vector>
#include "Model.h"
#include "NeuralNetwork.h"
#include "linearReg.h"
#include "multivariateLinearReg.h"
#include "Stage.h"

#ifndef LIF_RMI_H
#define LIF_RMI_H


class rmi {
private:
    std::vector<std::pair<double,int>> data_;
    std::vector<std::pair<int, ModelEnum>> stages_;
    std::vector<Stage> layers_;

public:
    rmi(std::string filepath, std::string name, std::vector<std::pair<int, ModelEnum>> stages);
    void make_rmi(std::string path, std::string dataset);
    double predict(float key);
    static std::vector<std::pair<double,int>> loadData(std::string path, std::string filename);
    std::vector<std::pair<double,int>> getData();
    ~rmi() = default;;
};

#endif //LIF_RMI_H
