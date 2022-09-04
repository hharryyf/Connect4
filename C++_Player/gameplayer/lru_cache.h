/*
    implementation of the LRU cache for the transposition table of the alpha-beta agent
    copied from https://stackoverflow.com/questions/2504178/lru-cache-design
*/
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

template <class KEY_T, class VAL_T> class LRUCache{

public:
        LRUCache(int cache_size_=1000000):cache_size(cache_size_){
                ;
        };

        void put(const KEY_T &key, const VAL_T &val){
                auto it = item_map.find(key);
                if(it != item_map.end()){
                        item_list.erase(it->second);
                        item_map.erase(it);
                }
                item_list.push_front(make_pair(key,val));
                item_map.insert(make_pair(key, item_list.begin()));
                clean();
        };
        
        int get_size() {
            return (int) item_list.size();
        }

        inline int max_size() {
            return cache_size;
        }

        bool exist(const KEY_T &key){
            return (item_map.count(key)>0);
        };

        VAL_T get(const KEY_T &key){
            assert(exist(key));
            auto it = item_map.find(key);
            item_list.splice(item_list.begin(), item_list, it->second);
            return it->second->second;
        };

        std::list< std::pair<KEY_T,VAL_T> > item_list;
        std::unordered_map<KEY_T, decltype(item_list.begin()) > item_map;
        int cache_size;

private:
        void clean(void){
            while((int) item_map.size()>cache_size){
                auto last_it = item_list.end(); last_it --;
                item_map.erase(last_it->first);
                item_list.pop_back();
            }
        };
};
