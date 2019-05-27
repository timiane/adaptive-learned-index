//
// Created by timian on 12/19/18.
//

#ifndef LIF_LINEARREG_H
#define LIF_LINEARREG_H


#include <string>
#include "Model.h"

class linearReg : public Model {
public:
    explicit linearReg(std::vector<std::pair<double, int>> inputData);

    float predict(float key) override;

    void train();

private:
    std::vector<std::pair<double, int>> data;
    double a = 0;
    double b = 0;
};


#endif //LIF_LINEARREG_H
