#include <iostream>
#include "stx-btree-0.9/include/stx/btree.h"
#include "stx-btree-0.9/include/stx/btree_multimap.h"
#include "stx-btree-0.9/include/stx/btree_multiset.h"
#include "stx-btree-0.9/speedtest/speedtest.cc"
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
#include <chrono>
#include <climits>
#include <unistd.h>
#include <random>
#include "malloc_count-0.7.1/malloc_count.h"

using namespace std;

auto osm_path = "../../data/osm.csv";
auto norm_dist_path = "../../data/norm_dist/log.csv";
auto saskathewan_path = "../../data/saskatachewan_data.csv";
auto current_path = osm_path;

int number_of_lookups = 2000000;

template<typename KeyType>
struct traits_nodebug : stx::btree_default_set_traits<KeyType> {
    static const bool selfverify = true;
    static const bool debug = false;

    static const int leafslots = 128;
    static const int innerslots = 128;
};

#define BTREE_DEBUG

vector<double> load_file() {
    ifstream myfile(current_path);
    vector<double> data;

    string line;
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            data.push_back(stod(line));
        }
        myfile.close();
        sort(data.begin(), data.end());
    } else {
        cout << "unable to open file" << '\n';
    }
    return data;

}
typedef stx::btree_multiset<double,
        less<>, traits_nodebug<double>> btree_type;
//
//long calculate_memory_usage(btree_type btree1, vector<double> dataArray) {
//    malloc_count_reset_peak();
//    long before = malloc_count_current();
//    //insert load of data into tree here, this calculation ins in bytes
//    btree1.bulk_load(dataArray.begin(), dataArray.end());
//    long after = malloc_count_current();
//    cout << "size of btree in bytes: " << after - before << "\n";
//}

int main() {




    // We store all data in a vector as they are easy to handle and it is iterable.
    // This is great because then we can bulk insert them into the btree
    vector<double> dataArray = load_file();
    cout << "data size: " << dataArray.size() << '\n';

    btree_type btree1;
//    calculate_memory_usage(btree1, dataArray);
    //@begin_insert and @end_insert will measure the time taken in milliseconds
    //for how long it takes to insert all data points
    auto begin_insert = chrono::steady_clock::now();

    btree1.bulk_load(dataArray.begin(), dataArray.end());


    auto end_insert = chrono::steady_clock::now();
    long elapsed_time_insert = chrono::duration_cast<chrono::nanoseconds>(end_insert - begin_insert).count();
    btree1.verify();

    //Initialization of the values such that they don't get allocated every run in the loop
    long elapsed_time_lookup = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, (int)dataArray.size());
    chrono::steady_clock::time_point begin_lookup;
    chrono::steady_clock::time_point end_lookup;
    long lookup_time;
    double key;

    //Then we do the same for @begin_lookup and @end_lookup to measure how long a lookup takes.
    //How it is done now is we do @number_of_lookups amount of lookups and take the average time
    for(int k = 0; k < 5; k++) {

        for (int i = 0; i < number_of_lookups + 1; i++) {
            begin_lookup = chrono::steady_clock::now();
            key = btree1.find(dataArray[dis(gen)]).key();
            end_lookup = chrono::steady_clock::now();
            if (i == 0) {
                continue;
            }
            lookup_time = chrono::duration_cast<chrono::nanoseconds>(end_lookup - begin_lookup).count();
//        cout << "Lookup found key:" << key << " lookup time: " << lookup_time << "ns\n";
            elapsed_time_lookup += lookup_time;
        }
    }

    cout << "elapsed insert time: " << elapsed_time_insert << "ns" << '\n';
    cout << "elapsed lookup time: " << (elapsed_time_lookup / 5) /number_of_lookups << "ns" << '\n';

    return 0;
}

