//
// Created by timian on 12/19/18.
//

#ifndef LIF_MULTIVARIATELINEARREG_H
#define LIF_MULTIVARIATELINEARREG_H


#include <string>
#include <functional>
#include "Model.h"

class multivariateLinearReg: public Model {
public:
    explicit multivariateLinearReg(std::string dataset);
    float predict(float key) override;
    float logx(float base, float x);
private:
    float logNorm(float x);
    float osm(float x);
    float sas(float x);
    float norm(float x);
    float beta1(float x);
    float beta2(float x);
    float beta3(float x);
    float beta4(float x);
    float beta5(float x);
    float beta6(float x);
    float beta7(float x);
    float beta8(float x);
    std::function<float(float)> pFunc;
};


#endif //LIF_MULTIVARIATELINEARREG_H
