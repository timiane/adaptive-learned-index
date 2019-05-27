//
// Created by timian on 3/25/19.
//

#include "Test.h"
#include "rmi.h"

void Test::createTests(std::vector <std::string> testData, int threshold) {
    for (auto &testDataLocation: testData) {
        std::string modelPath = "../../stages/" + testDataLocation + "/export/";
        std::string modelName = "0.0.model";
        std::vector<std::pair<double,int>> data = rmi::loadData("../../data/", testDataLocation + ".csv");
        auto max_value = max_element(data.begin(), data.end())->first;
        NeuralNetwork* neuralNetwork = new NeuralNetwork(modelPath, modelName, max_value, data.size());
        std::pair<Model*, std::string> modelNamePair = std::pair<Model*, std::string>(neuralNetwork, testDataLocation);
        models.emplace_back(modelNamePair);

        createTestResults(threshold, testDataLocation, neuralNetwork, data);

        multivariateLinearReg *mvr = new multivariateLinearReg(testDataLocation);
        models.emplace_back(std::pair<Model*, std::string>(mvr, testDataLocation));
        createTestResults(threshold, testDataLocation, mvr, data);


        linearReg *lr = new linearReg(data);
        lr->train();
        models.emplace_back(std::pair<Model*, std::string>(lr, testDataLocation));
        createTestResults(threshold, testDataLocation, lr, data);
        std::cout<< "\n\n";
    }
}

void Test::createTestResults(int threshhold, const std::string &datasource, Model* model, const std::vector<std::pair<double,int>> &data) {
    double error = 0;
    double prediction = 0;
    double inf = 0;
    for(auto &dataElement : data) {
        prediction = model->predict(dataElement.first);
        if(prediction == -INFINITY || prediction == INFINITY) { inf += 1;}
        if((std::abs(prediction - dataElement.second)) > threshhold) error += 1;
    }
    error = error/data.size() * 100;
    std::cout << "infnity numbers --> " << inf << "\n";
    std::string print = ToString(model->getType()) + datasource + " error: " + std::to_string(error);
    std::cout << print + "\n";
    result.emplace_back(std::pair<std::string,float>(ToString(model->getType()) + datasource, error));
}

std::vector<std::pair<std::string, float>> Test::getTestResult() {
    return result;
}


