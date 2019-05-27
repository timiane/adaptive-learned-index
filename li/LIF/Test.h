//
// Created by timian on 3/25/19.
//

#ifndef LIF_TEST_H
#define LIF_TEST_H

#endif //LIF_TEST_H
#include "Model.h"
#include <math.h>
class Test {
public:
    void createTests(std::vector<std::string> testData, int threshold);
    std::vector<std::pair<std::string, float>> getTestResult();
    void createTestResults(int threshhold, const std::string &datasource, Model* model, const std::vector<std::pair<double,int>> &data);
private:
    std::vector<std::pair<std::string, float>> result;
    std::vector<std::pair<Model*, std::string>> models;
};