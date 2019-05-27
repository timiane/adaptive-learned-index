#include <utility>

//
// Created by timian on 12/19/18.
//

#include "rmi.h"
#include <stdexcept>
#include <list>
#include <typeinfo>

using namespace std;

rmi::rmi(string filepath, string name, vector<pair<int, ModelEnum>> stages) {
    stages_ = move(stages);
    data_ = loadData(move(filepath), move(name));
}

void rmi::make_rmi(string models_path, string dataset) {
    int stages_height = stages_.size();
    int biggest_models_size_in_stage;

    for (auto &stage : stages_) {
        if (stage.first > biggest_models_size_in_stage) biggest_models_size_in_stage = stage.first;
    }

    vector<vector<vector<pair<double, int>>>> temp_records;
    vector<vector<pair<double, int>>> _;
    _.push_back(data_);
    temp_records.push_back(_);
    vector<Stage> layers;
    for (int i = 0; i < stages_height; i++) {
        Stage tempLayer;
        vector<pair<double, int>> vectors[stages_[i + 1].first];
        for (int j = 0; j < stages_[i].first; j++) {
            ModelEnum modelEnum = stages_[i].second;
            vector<pair<double, int>> tempData = temp_records[i][j];

            if (modelEnum == NN) {
                std::string filename = to_string(i) + "." + to_string(j) + ".model";
                auto max_value = max_element(tempData.begin(), tempData.end())->first;
                NeuralNetwork *nn = new NeuralNetwork(models_path, filename, max_value, _[0].size());
                tempLayer.addModel(nn);
            } else if (modelEnum == MVR) {
                multivariateLinearReg *mvr = new multivariateLinearReg(dataset);
                tempLayer.addModel(mvr);
            } else if (modelEnum == LR) {
                vector<pair<double, int>> tempData;
                if (temp_records[i][j].size() > 0) {
                    tempData = temp_records[i][j];
                }
                auto lr = new linearReg(tempData);
                lr->train();
                tempLayer.addModel(lr);
            } else {
                throw runtime_error("Choose a supported model type");
            }

            if (i < stages_height - 1) {
                for (auto r : tempData) {
                    double prediction = tempLayer.getModel(j)->predict(r.first);
                    double scale = floor((tempData.size() / stages_[i + 1].first));
                    int nextIndex = (int) (prediction / scale);
                    if (nextIndex > (sizeof(vectors) / sizeof(vectors[0])) - 1)
                        nextIndex = sizeof(vectors) / sizeof(vectors[0]) - 1;
                    else if (nextIndex < 0) nextIndex = 0;
                    vectors[nextIndex].push_back(r);
                }
            }

        }
        vector<vector<pair<double, int>>> tempVector;
        for (int b = 0; b < sizeof(vectors) / sizeof(vectors[0]); b++) {
            tempVector.push_back(vectors[b]);
        }
        temp_records.push_back(tempVector);

        layers.push_back(tempLayer);
    }
    layers_ = layers;
}

double rmi::predict(float key) {
    int modelIndex = 0;
    for (int i = 0; i < layers_.size(); i++) {
        if (modelIndex < 0) {
            modelIndex = 0;
        } else if (modelIndex > layers_[i].models.size() - 1) {
            modelIndex = (int) layers_[i].models.size() - 1;
        }

        auto tempModel = layers_[i].getModel(modelIndex);

        if (i == layers_.size() - 1) {
            return tempModel->predict(key);
        }

        double dataSize = data_.size();
        double stageSize = stages_[i + 1].first;
        double scale = floor(dataSize / stageSize);
        double prediction = tempModel->predict(key);
        modelIndex = (int) floor(prediction / scale);
    }
    return 0;
}

std::vector<std::pair<double,int>> rmi::loadData(string path, string filename) {
    std::vector<std::pair<double,int>> result;
    ifstream myfile(path + filename);
    string line;
    int i = 0;
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            result.emplace_back(stod(line), i);
            i++;
        }
        myfile.close();
        sort(result.begin(), result.end());
    } else {
        cout << "unable to open file" << '\n';
    }
    return result;
}

std::vector<pair<double, int>> rmi::getData() {
    return data_;
}

