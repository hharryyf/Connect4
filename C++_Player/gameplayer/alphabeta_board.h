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
#include "config.h"
#include <fstream>
#include "lru_cache.h"

class alphabeta_board {
public:

    // initialize the data structure for alpha-beta player
    void init() {
        board.init();
        row.init(ROW, board);
        col.init(COL, board);
        diag.init(DIAG, board);
        antidiag.init(ANTIDIAG, board);
        this->store_cache(true);
        std::cout << "transposition table maximum size " << table.max_size() << std::endl;
        std::cout << "transposition table start size " << table.get_size() << std::endl;
    }

    // if we can take the current move
    bool can_move(int c) {
        return this->board.canplay(c);
    }

    /* "piece" take a move in column c
        if piece is 0, we remove the top move in column c 
        otherwise, we place "piece" in clumn c
        precondition: if piece is 0, height[c] cannot be -1
                      &&
                        0 <= c < 7
                      && if piece is 1/-1, height[c] cannot be more than 5
    */
    void update(int c, int piece) {
        // top-most row in column c that has a piece
        int r = board.height[c];    
        board.play(c, piece);
        if (piece != 0) {
            r++;
        }

        // we set (r, c) to be piece
        set_bit(r, c, piece);

        for (int i = r; i < 6; ++i) {
            row.recalculate(i, board);
            diag.recalculate(board.diag[i][c], board);
            antidiag.recalculate(board.antidiag[i][c], board);
        }

        col.recalculate(c, board);
        // debug();
    }

    /*
        check if player "piece" can take a move in column c and win the game
        precondition: the board status is still undetermine && piece \in {1, -1}
    */
    bool killmove(int c, int piece) {
        assert(piece != 0);
        check_move(c, piece);
        bool ok = (status() == piece);
        check_move(c, 0);
        return ok;
    }
    
    /*
        check the status of the board
        @return: 1 -- X win, -1 -- O win, 0 -- full board, 2 -- undetermine
    */
    int status() {
        if (diag.get_status() != 0) return diag.get_status();
        if (antidiag.get_status() != 0) return antidiag.get_status();
        if (row.get_status() != 0) return row.get_status();
        if (col.get_status() != 0) return col.get_status();
        if (board.move >= 42) return 0;
        return 2;
    }

    /*
        heuristic score of a board
    */
    double heuristic() {
        int s = status();
        if (s == 1) return four;
        if (s == -1) return -four;
        if (s == 0) return 0;
        return diag.get_score() + antidiag.get_score() + row.get_score() + col.get_score();
    }

    /*
        count the number of moves on the board
    */
    int get_move() {
        return this->board.move;
    }

    /*
        print the board
    */
    std::string print_board() {
        std::string ret;
        for (int i = 5; i >= 0; --i) {
            for (int j = 0 ; j < 7; ++j) {
                if (board.a[i][j] == 0) {
                    ret.append(".");
                } else if (board.a[i][j] == 1) {
                    ret.append("X");
                } else {
                    ret.append("O");
                }
                
                if (j != 6) {
                    ret.append("|");
                } else {
                    ret.append("\n");
                }
            }

            if (i != 0) {
                for (int j = 0 ; j < 7; ++j) {
                    ret.append("_");
                    if (j != 6) {
                        ret.append(" ");
                    } else {
                        ret.append("\n");
                    }
                }
            }
        }
        
        return ret;
    }

    /*
        print out the board information for debug
    */
    void debug() {
        std::cout << "current board" << std::endl;
        std::cout << print_board() << std::endl; 
        std::cout << "board heuristic: " << heuristic() << " status: " << status() << std::endl;
        board.debug();
        std::cout << "information of the diagnal" << std::endl;
        diag.debug();
        std::cout << "information of the anti-diagnal" << std::endl;
        antidiag.debug();
        std::cout << "information of the row" << std::endl;
        row.debug();
        std::cout << "information of the column" << std::endl;
        col.debug();        
    }

    /*
        Note that this simply means that given the current bitboard,
        we can go to a terminal winning state, the return value is <heuristic score, move>
    */
    std::pair<double, int> get_cached_move() {
        if (table.exist(this->bitboard)) return table.get(this->bitboard);
        return std::make_pair(0, -1);
    }

    /*
        store the bitboard status and move
    */
    void cache_state(std::pair<double, int> &score) {
        table.put(this->bitboard, score);
    }

    void print_bit_board() {
        std::cout << "reach bitboard terminal state" << std::endl;
        std::cout << bitboard << std::endl;
    }

    int get_cache_size() {
        return this->table.get_size();
    }

    void store_cache(bool isread=false) {
        if (!isread) {
            std::ofstream file;
            file.open(cache_file);
            auto iter = this->table.item_list.rbegin();
            file << this->table.get_size() << std::endl;
            while (iter != this->table.item_list.rend()) {
                file << iter->first.to_string() << " " << iter->second.first << " " << iter->second.second << std::endl;
                ++iter;
            }

        } else {
            std::ifstream infile;
            infile.open(cache_file);
            if (!infile.is_open()) return;
            int n;
            infile >> n;
            while (n-- > 0) {
                std::string ch;
                double score;
                int move;
                infile >> ch >> score >> move;
                std::string state(ch);
                std::bitset<84> bt(state);
                this->table.put(bt, std::make_pair(score, move));
            }   
        }
    }

private:

    void set_bit(int r, int c, int bt) {
        int id1 = (r * 7 + c) << 1, id2 = ((r * 7 + c) << 1) | 1;
        if (bt == 1) {
            bitboard.set(id1, false);
            bitboard.set(id2, true);
        } else if (bt == 0) {
            bitboard.set(id1, false);
            bitboard.set(id2, false);
        } else {
            bitboard.set(id1, true);
            bitboard.set(id2, false);
        }
    }

    void check_move(int c, int piece) {
        int r = board.height[c];    
        board.play(c, piece);
        if (piece != 0) {
            r++;
        }

        col.recalculate(c, board);
        row.recalculate(r, board);
        diag.recalculate(board.diag[r][c], board);
        antidiag.recalculate(board.antidiag[r][c], board);
    }

    struct gameboard {
        int a[6][7], height[7], move = 0;
        int diag[6][7], antidiag[6][7];
        double pw[29];
        void init() {
            move = 0;
            for (int i = 0 ; i < 6; ++i) {
                for (int j = 0 ; j < 7; ++j) {
                    a[i][j] = 0;
                    height[j] = -1;
                }
            }

            pw[0] = 1.0;
            for (int i = 1; i < 29; ++i) pw[i] = pw[i-1] * discount;
            
            for (int i = 0; i < 6; ++i) {
                for (int j = 0 ; j < 7; ++j) {
                    diag[i][j] = antidiag[i][j] = -1;
                }
            }

            int id = 0;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0 ; j < 7; ++j) {
                    if (diag[i][j] == -1 && std::min(6 - i, 7 - j) >= 4) {
                        int tx = i, ty = j;
                        while(tx < 6 && ty < 7) {
                            diag[tx][ty] = id;
                            tx++, ty++;
                        }
                        id++;
                    }
                }
            }

            id = 0;

            for (int i = 0; i <= 5; ++i) {
                for (int j = 6 ; j >= 0; --j) {
                    if (antidiag[i][j] == -1 && std::min(6 - i, j + 1) >= 4) {
                        int tx = i, ty = j;
                        while(tx < 6 && ty >= 0) {
                            antidiag[tx][ty] = id;
                            tx++, ty--;
                        }
                        id++;
                    }
                }
            }
        }

        bool canplay(int column) {
            return column >= 0 && column < 7 && height[column] < 5;
        }

        void play(int column, int piece) {
            assert(column >= 0 && column <= 6);
            if (piece != 0) {
                assert(canplay(column));
                a[++height[column]][column] = piece;
                ++move;
            } else {
                assert(height[column] >= 0);
                a[height[column]][column] = 0;
                --height[column];
                --move;
            }
        }

        void debug() {
            std::cout << "total number of moves: " << move << std::endl; 
            std::cout << "the height array [";
            for (int i = 0 ; i < 7; ++i) {
                std::cout << " " << height[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "diagnal: group" << std::endl;
            for (int i = 5 ; i >= 0; --i) {
                for (int j = 0 ; j < 7; ++j) {
                    std::cout << diag[i][j] << " ";
                }
                std::cout << std::endl;
            }

            std::cout << "anti-diagnal: group" << std::endl;
            for (int i = 5 ; i >= 0; --i) {
                for (int j = 0 ; j < 7; ++j) {
                    std::cout << antidiag[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
    };

    struct group {
        std::vector<std::pair<int, int>> window[8];
        int status[8];
        double score[8];
        int x_win = 0, o_win = 0;
        double tol_score = 0;
        void init(int type, gameboard &board) {
            x_win = o_win = 0;
            tol_score = 0.0;
            for (int i = 0 ; i < 8; ++i) status[i] = 0, score[i] = 0.0;
            if (type == ROW) {
                for (int i = 0 ; i < 6; ++i) {
                    for (int j = 0 ; j < 7; ++j) {
                        window[i].emplace_back(i, j);
                    }
                }
            } else if (type == COL) {
                for (int i = 0 ; i < 6; ++i) {
                    for (int j = 0 ; j < 7; ++j) {
                        window[j].emplace_back(i, j);
                    }
                }
            } else if (type == DIAG) {
                for (int i = 0 ; i < 6; ++i) {
                    for (int j = 0 ; j < 7; ++j) {
                        if (board.diag[i][j] != -1) window[board.diag[i][j]].emplace_back(i, j);
                    }
                }
            } else if (type == ANTIDIAG) {
                for (int i = 0 ; i < 6; ++i) {
                    for (int j = 0 ; j < 7; ++j) {
                        if (board.antidiag[i][j] != -1) window[board.antidiag[i][j]].emplace_back(i, j);
                    }
                }
            }
        }

        void recalculate(int index, gameboard &brd) {
            if (index < 0) return;
            if (status[index] > 0) {
                x_win -= status[index];
            } else {
                o_win += status[index];
            }            

            tol_score -= score[index];
            score[index] = 0;
            status[index] = 0;
            int x_count = 0, o_count = 0;
            int potential = 0;
            for (int i = 0; i < (int) window[index].size(); ++i) {
                if (brd.a[window[index][i].first][window[index][i].second] == 1) x_count++;
                if (brd.a[window[index][i].first][window[index][i].second] == -1) o_count++;
                if (brd.a[window[index][i].first][window[index][i].second] == 0) {
                    potential = potential + window[index][i].first - brd.height[window[index][i].second];
                }
                if (i >= 3) {
                    if (x_count == 0) {
                        if (o_count == 2) {
                            score[index] = score[index] - brd.pw[potential] * two;
                        } else if (o_count == 3) {
                            score[index] = score[index] - brd.pw[potential] * three;
                        } else if (o_count == 4) {
                             score[index] = score[index] - brd.pw[potential] * four;
                             status[index]--;
                        } else if (o_count == 1) {
                             score[index] = score[index] - brd.pw[potential] * one;
                        }
                    } else if (o_count == 0) {
                        if (x_count == 2) {
                            score[index] = score[index] + brd.pw[potential] * two;
                        } else if (x_count == 3) {
                            score[index] = score[index] + brd.pw[potential] * three;
                        } else if (x_count == 4) {
                            score[index] = score[index] + brd.pw[potential] * four;
                            status[index]++;
                        } else if (x_count == 1) {
                            score[index] = score[index] + brd.pw[potential] * one;
                        }
                    }

                    if (brd.a[window[index][i-3].first][window[index][i-3].second] == 1) x_count--;
                    if (brd.a[window[index][i-3].first][window[index][i-3].second] == -1) o_count--;
                    if (brd.a[window[index][i-3].first][window[index][i-3].second] == 0) {
                        potential = potential - window[index][i-3].first + brd.height[window[index][i-3].second];
                    }
                }
            }

            if (status[index] > 0) x_win += status[index];
            if (status[index] < 0) o_win -= status[index];
            tol_score += score[index];
        }

        int get_status() {
            if (x_win > 0) return 1;
            if (o_win > 0) return -1;
            return 0;
        }

        double get_score() {
            return tol_score;
        }

        void debug() {
            std::cout << "score = " << get_score() << " x_win: " << x_win << " o_win: " << o_win << std::endl;
            for (int i = 0 ; i < 8; ++i) {
                if (!window[i].empty()) {
                    std::cout << "window elements: " << std::endl;
                    for (auto elim : window[i]) {
                        std::cout << "(" << elim.first << ", " << elim.second << ") ";
                    }
                    std::cout << std::endl;
                    std::cout << "window score: " << score[i] << " window count: " << status[i] << std::endl;
                }
            }
        }
    };

    // a bit representation of the board, prepare for the transposition table 
    // for position (r, c), we have index (r * 7 + c) * 2 and (r * 7 + c) * 2 + 1 describes this cell
    // if (r, c) has value 0, these index have value (0, 0)
    // if (r, c) has value -1, these index have value (1, 0)
    // if (r, c) has value 1, these index have value (0, 1)
    std::bitset<84> bitboard;
    LRUCache<std::bitset<84>, std::pair<double, int>> table;
    // the watched entries
    group diag, antidiag, row, col;
    // the gameboard
    gameboard board;
};