#include "rmi.h"
#include "Stage.h"
#include "Test.h"
#include <chrono>
#include <math.h>
#include <iostream>
#include <fstream>

// model biased binary search as used in the paper, but we need to find l, and r such that
// we dont search  from start to finish of the array, but within the error margin instead.
// for that we need to know the minimum and maximum error but i cannot find a solution for this????
double
modelBiasedBinarySearch(const std::vector<std::pair<double, int>> &arr, double l, double r, double x, double middle) {

    int i = 0;
    int steps = 0;
    while (l <= r) {
        if (l < 0) {
            l = 0;
        }
        if (r > arr.size()) {
            r = arr.size();
        }
        steps++;
        int m;
        if (i == 0) {
            if (middle < 0) {
                m = 0;
            } else if (middle > arr.size() - 1) {
                m = arr.size() - 1;
            } else {
                m = int(middle);
            }
            l = (int) l;
            r = (int) r;
            i++;
        } else {
            m = l + (r - l) / 2;
        }
        // Check if x is present at mid
        if (arr[m].first == x) {
            std::cout << "Key: " << x << "; Predicted index: " << middle << "; Actual index "
                      << m << "; Found in " << steps << " steps \n";

            return m;
        }
        // If x greater, ignore left half
        if (arr[m].first < x)
            l = m + 1;

            // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

int main(int argc, char **argv) {

std::vector<std::string> testData;
testData.emplace_back("beta2");
testData.emplace_back("beta3");
testData.emplace_back("beta4");
testData.emplace_back("beta5");
testData.emplace_back("beta6");
testData.emplace_back("beta7");


Test* test = new Test();
test->createTests(testData, 12800);
//    std::vector<std::pair<int, ModelEnum>> stages;
//    stages.emplace_back(1, NN);
//    stages.emplace_back(1000, LR);
//
//    rmi recusive_model_index("../../data/", "osm.csv", stages);
//    recusive_model_index.make_rmi("../../stages/osm/export/", "osm"); // <-- here to tell which dataset
//    std::chrono::steady_clock::time_point begin;
//    std::chrono::steady_clock::time_point end;
//    double time = 0;
//
//    auto data = recusive_model_index.getData();
//    int i = 0;
//    double predictedKey = 0;
//    double maxError = 0;
//
//    double errorAmmout = 0;
//    double minError = 0;
//    for (auto key: data) {
//        begin = std::chrono::steady_clock::now();
//        predictedKey = recusive_model_index.predict(key.first);
//        end = std::chrono::steady_clock::now();
//        double currentError = predictedKey - key.second;
//        if (abs(currentError) > 128) {
//            errorAmmout += 1;
//        }
//        if (currentError > maxError) {
//            maxError = currentError;
//        } else if (currentError < minError) {
//            minError = currentError;
//        }
//        if (i == 0) {
//            i++;
//            continue;
//        }
//        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
//        i++;
//    }
//    minError = abs(minError);
//    std::cout << "Average lookup time " << time / data.size() << " ns \n";
//    std::cout << "Max Error = " << maxError << "\n";
//    std::cout << "Min Error = " << minError << "\n";
//    double keyToPredict = data[data.size() - 1].first;
//    double predictedIndex = recusive_model_index.predict(keyToPredict);
//    modelBiasedBinarySearch(data, predictedIndex - minError, predictedIndex + maxError, keyToPredict, predictedIndex);
//    double key = 0;
//
//    while (key != -1) {
//        std::cin >> key;
//        keyToPredict = data[key].first;
//        predictedIndex = recusive_model_index.predict(keyToPredict);
//        modelBiasedBinarySearch(data, predictedIndex - minError, predictedIndex + maxError, keyToPredict,
//                                predictedIndex);
//    }

    return 0;
}
