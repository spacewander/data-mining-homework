#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include "similarities.hpp"

#define TRAIN_FILE "../data-rs/80train.txt"
#define TEST_FILE "../data-rs/test.txt"
#define RESULT_FILE "../results.txt"

int main()
{
    using namespace std;
    ifstream trainData(TRAIN_FILE);
    float value;
    int userId;
    int movieId;
    unordered_map<int, unordered_map<int, float> > usersWithMovies;
    unordered_map<int, unordered_map<int, float> > moviesWithUsers;

    int timestamp;
    auto usersIt = usersWithMovies.begin();
    auto moviesIt = moviesWithUsers.begin();
    while (trainData >> userId) {
        trainData >> movieId;
        trainData >> value;
        trainData >> timestamp; // ignore timestamp
        if ((usersIt = usersWithMovies.find(userId)) != usersWithMovies.end()) {
            usersIt->second[movieId] = value;
        }
        else {
            usersWithMovies[userId] = unordered_map<int, float>();
            usersWithMovies[userId][movieId] = value;
        }
        if ((moviesIt = moviesWithUsers.find(movieId)) != moviesWithUsers.end()) {
            moviesIt->second[userId] = value;
        }
        else {
            moviesWithUsers[movieId] = unordered_map<int, float>();
            moviesWithUsers[movieId][userId] = value;
        }
    }

    trainData.close();

    ifstream testData(TEST_FILE);
    ofstream resultsData(RESULT_FILE);
    float MAE = 0;
    float RMAE = 0;
    int evaluation = 0;
    int length = 0;
    while (testData >> userId) {
        testData >> movieId;
        testData >> value;
        testData >> timestamp; // ignore timestamp

        evaluation = evaluate(userId, movieId, usersWithMovies, moviesWithUsers);
        cout << userId << "\t" << movieId << "\t" << evaluation
            << "\t" << value << endl;
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
    return 0;
}
