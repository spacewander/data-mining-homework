#ifndef SIMILARITIES_H
#define SIMILARITIES_H

#include <cmath>
#include <map>
#include <unordered_map>
#include <iostream>

using namespace std;

typedef const unordered_map<int, unordered_map<int, float> > moviesData;

float manhattan_distances(int movieId, int otherMovie, moviesData &moviesWithUsers)
{
    auto usersFromMovieA = moviesWithUsers.find(movieId);
    auto usersFromMovieB = moviesWithUsers.find(otherMovie);
    if (usersFromMovieA == moviesWithUsers.end() || usersFromMovieB == moviesWithUsers.end()) {
        return 0;
    }
    auto first = (*usersFromMovieA).second;
    auto common = (*usersFromMovieB).second;

    float distance = 0;
    int length = 0;
    for (auto i : first) {
        auto commonIter = common.begin();
        if ((commonIter = common.find(i.first)) != common.end()) {
            distance += abs(i.second - (*commonIter).second);
            ++length;
        }
    }
    if (length == 0) {
        return 0;
    }
    return 1.0 - distance / float(length);
}

float averageMovieValue(moviesData &moviesWithUsers, int movieId)
{
    auto pair = moviesWithUsers.find(movieId);
    if (pair == moviesWithUsers.end()) {
        return 4.0;
    }
    float sum = 0;
    int length = 0;
    for (auto i : (*pair).second) {
        sum += i.second;
        ++length;
    }
    return sum / length;
}

int evaluate(int userId, int movieId, moviesData &usersWithMovies, 
        moviesData &moviesWithUsers)
{
    auto moviesFromUserPair = usersWithMovies.find(userId);
    if (moviesFromUserPair == usersWithMovies.end()) {
        return averageMovieValue(moviesWithUsers, movieId);
    }
    float value = 0;
    float sim = 0;
    float sumSim = 0;
    auto moviesFromUser = (*moviesFromUserPair).second;
    // for movies user watched
    for (auto i : moviesFromUser) {
        if (i.first == movieId) {
            return i.second;
        }
         //get sim from two movies
        sim = manhattan_distances(movieId, i.first, moviesWithUsers);
        if (sim <= 0) {
            continue;
        }
        //sim /= pow(sim, 2.5);
        value += sim * i.second;
        sumSim += sim;
    }
    if (sumSim <= 0.0) {
        return averageMovieValue(moviesWithUsers, movieId);
    }
    int evaluation = round(value / sumSim);
    if (evaluation > 5) {
        evaluation = 5;
    }
    else if (evaluation < 1) {
        evaluation = 1;
    }
    return evaluation;
}


#endif /* SIMILARITIES_H */
