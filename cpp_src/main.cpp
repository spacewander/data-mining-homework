#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
//#include "similarities.hpp"
#include "global.h"

#define TRAIN_FILE "../data-rs/80train.txt"
#define TEST_FILE "../data-rs/test.txt"
#define RESULT_FILE "../results.txt"

int my_round(float input)
{
    if (input > 4) {
        if (input > 4.395) {
            return 5;
        }
        else {
            return 4;
        }
    }
    else if (input < 2) {
        if (input < 1.65) {
            return 1;
        }
        else {
            return 2;
        }
    }
    return round(input);
}


void fillWithRand(vector<double> &p)
{
    if (p.size() == 0) {
        float base = 1 / DIM;
        p.resize(DIM);
        for (int i = 0; i < DIM; ++i) {
            p[i] = base + (rand() % 100 +1) / 1000;
        }
    }
}


double dot(vector<double> &p, vector<double> &q)
{
    double result = 0.0;
    int min = p.size() > q.size() ? q.size() : p.size();
    for (int i = 0; i < min; ++i) {
        result += p[i] * q[i];
    }
    return result;
}

double evaluate(int userId, int movieId)
{
    double result = mean + bu[userId - 1] + bi[movieId - 1] + 
        dot(p[userId - 1], q[movieId - 1]);
    if (result < 1.0) {
        result = 1.0;
    }
    if (result > 5.0) {
        result = 5.0;
    }
    return result;
}

double learnPQ(int i, float *data, double rmae)
{
    int userId = data[i * 3];
    int movieId = data[i * 3 + 1];
    float value = data[i * 3 + 2];
    double evaluation = evaluate(userId, movieId);
    double err = value - evaluation;
    rmae += err * err;
    bu[userId - 1] += alpha * (err - BETA * bu[userId - 1]);
    bi[movieId - 1] += alpha * (err - BETA * bi[movieId - 1]);
    for (int k = 0; k < DIM; ++k) {
    //for (int j = 0; j < DIM; ++j) {
        //int k = rand() % DIM;
        p[userId - 1][k] += alpha * (err * q[movieId - 1][k] - 
                BETA * p[userId - 1][k]);
        q[movieId - 1][k] += alpha * (err * p[userId - 1][k] - 
                BETA * q[movieId - 1][k]);
    }

    return rmae;
}

int main()
{
    using namespace std;
    float *data = new float[80000 * 3];
    ifstream trainData(TRAIN_FILE);
    float value;
    int userId;
    int movieId;

    int timestamp;
    int num = 0;
    float sum = 0;
    while (trainData >> userId) {
        trainData >> movieId;
        trainData >> value;
        trainData >> timestamp; // ignore timestamp
        data[num * 3] = userId;
        data[num * 3 + 1] = movieId;
        data[num * 3 + 2] = value;

        ++num;
        sum += value;
        ++un[userId - 1];
        ++in[movieId - 1];
        bu[userId - 1] += value;
        bi[movieId - 1] += value;

        fillWithRand(p[userId - 1]);
        fillWithRand(q[movieId - 1]);
    }
    mean = sum / num;
    for (int i = 0; i < USER_NUM; ++i) {
        if (un[i] == 0) {
            continue;
        }
        bu[i] = (bu[i] / un[i]) - mean;
    }
    for (int i = 0; i < ITEM_NUN; ++i) {
        if (in[i] == 0) {
            continue;
        }
        bi[i] = (bi[i] / in[i]) - mean;
    }

    trainData.close();


    double preRmae = 1.0;
    for (int step = 0; step < 25; ++step) {
        double rmae = 0.0;
        for (int i = 0; i < 80000; ++i) {
            rmae = learnPQ(i, data, rmae);
        }

        rmae /= 80000;
        cout << "step: " << step << " rmae: " << rmae << endl;
        if (rmae <= preRmae) {
            preRmae = rmae;
        }
        else {
            break;
        }
        alpha *= 0.885;
    }

    ifstream testData(TEST_FILE);
    ofstream resultsData(RESULT_FILE);
    float MAE = 0;
    float RMAE = 0;
    float evaluation = 0;
    int length = 0;
    while (testData >> userId) {
        testData >> movieId;
        testData >> value;
        testData >> timestamp; // ignore timestamp

        evaluation = my_round(evaluate(userId, movieId));
        //cout << userId << "\t" << movieId << "\t" << evaluation
            //<< "\t" << value << endl;
        resultsData << userId << "\t" << movieId << "\t" << evaluation
            << "\t" << value << endl;
        MAE += abs(evaluation - value);
        RMAE += (evaluation - value) * (evaluation - value);
        ++length;
    }
    testData.close();
    resultsData.close();

    cout << "MAE : " << MAE / length << endl;
    cout << "RMAE : " << sqrt(RMAE / length) << endl;
    delete[] data;
    return 0;
}
