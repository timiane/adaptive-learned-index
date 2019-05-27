#include <iostream>
#include "keras_model.h"
#include <chrono>

auto file_name = "../0.model";
using namespace std;

int main() {
    //Initialize model
    KerasModel model;
    model.LoadModel(file_name);

    chrono::high_resolution_clock::time_point begin_insert;
    chrono::high_resolution_clock::time_point end_insert;
    long elapsed_time_insert;
    cout << "starting to predict \n";

    //Create a 1D tensor, on length 1 for input data
    double mean = 0;
    int amount = 0;
    for(int i = 0; i < 100; i++) {
        float key = i * .1;
        Tensor in(1);
        in.data_ = {key};
        Tensor out;
        begin_insert = chrono::high_resolution_clock::now();
        model.Apply(&in, &out);
        end_insert = chrono::high_resolution_clock::now();
        if(i == 0) continue;
        elapsed_time_insert = chrono::duration_cast<chrono::nanoseconds>(end_insert - begin_insert).count();
        mean += elapsed_time_insert;
        amount += 1;
        cout << "Predicted value: " << out.data_[0] << " elapsing: " << elapsed_time_insert << "ns \n";
    }
    cout << "mean: " << mean/amount << "ns \n";

    return 0;
}