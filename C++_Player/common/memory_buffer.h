#pragma once
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <stack>
#include <vector>
#include <array>
#include <set>
#include <iterator>
#include <queue>
#include <cassert>
#include <cmath>
#include <map>
#include <unordered_map>
#include <random>
#include <unordered_set>
#include <bitset>
#include <string>

template <class T> class memory_buffer {
public:
    memory_buffer(size_t sz)  {
        this->tol_size = sz;
        this->rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    void add(const T &item) {
        q.push_back(item);
        while (q.size() > this->tol_size) q.pop_front();
    }

    std::vector<T> sample(int tol) {
        if (tol > (int) q.size()) {
            printf("Error: Try to sample %d many elements from a buffer of size %d\n", tol, (int) q.size());
            exit(1);
        }

        std::vector<T> output;
        std::vector<int> seed(q.size());
        for (int i = 0 ; i < tol; ++i) {
            seed[i] = 1;
        }

        std::shuffle(seed.begin(), seed.end(), rng);
        int i = 0;
        for (auto elm : q) {
            if (seed[i]) output.push_back(elm);
            ++i;
        }

        return output;
    }

    size_t size() {
        return q.size();
    }

    


private:
    std::deque<T> q;
    size_t tol_size;
    std::default_random_engine rng = std::default_random_engine {};
};