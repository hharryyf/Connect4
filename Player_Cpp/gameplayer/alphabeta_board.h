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

class alphabeta_board {
public:
    void init() {
        board.init();
        row.init(ROW, board);
        col.init(COL, board);
        diag.init(DIAG, board);
        antidiag.init(DIAG, board);
    }

    bool can_move(int c) {
        return this->board.canplay(c);
    }

    void update(int c, int piece) {
        // top-most row in column c that has a piece
        int r = board.height[c];    
        board.play(c, piece);
        if (piece != 0) {
            r++;
        }

        for (int i = r; i < 6; ++i) {
            row.recalculate(i, board);
            diag.recalculate(board.diag[i][c], board);
            antidiag.recalculate(board.antidiag[i][c], board);
        }

        col.recalculate(c, board);
        // debug();
    }

    bool killmove(int c, int piece) {
        assert(piece != 0);
        check_move(c, piece);
        bool ok = (status() == piece);
        check_move(c, 0);
        return ok;
    }

    int status() {
        if (diag.get_status() != 0) return diag.get_status();
        if (antidiag.get_status() != 0) return antidiag.get_status();
        if (row.get_status() != 0) return row.get_status();
        if (col.get_status() != 0) return col.get_status();
        if (board.move >= 42) return 0;
        return 2;
    }

    double heuristic() {
        int s = status();
        if (s == 1) return four;
        if (s == -1) return -four;
        if (s == 0) return 0;
        return diag.get_score() + antidiag.get_score() + row.get_score() + col.get_score();
    }

    int get_move() {
        return this->board.move;
    }

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

private:

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

            for (int i = 5; i >= 0; --i) {
                for (int j = 6 ; j >= 0; --j) {
                    if (antidiag[i][j] == -1 && std::min(i + 1, 7 - j) >= 4) {
                        int tx = i, ty = j;
                        while(tx >= 0 && ty < 7) {
                            antidiag[tx][ty] = id;
                            tx--, ty++;
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

    group diag, antidiag, row, col;
    gameboard board;
};