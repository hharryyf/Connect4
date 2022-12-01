#pragma warning(push, 0)
#include <torch/torch.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#pragma warning(pop)
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <vector>
#include "bit_board.h"
#include "connect4_board.h"
#include "gameplayer.h"
#include "alpha_beta/alphabeta.h"
#include "human_play/humanplayer.h"
#include "deepq_mcts/mcts_pure.h"
#include "deepq_mcts/mcts_zero.h"
#include <winsock2.h>
#include <Windows.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#define ALPHA_BETA_X 0
#define MCTS_DEEP_X 1
#define MCTS_DEEP_O 2
#define ALPHA_BETA_O 3

void tensor_test();

void bit_board_unit_test(int T=1000000);

int play_game(gameplayer *player1, gameplayer *player2
    ,std::string player1_name, std::string player2_name, 
    ConfigObject config1, ConfigObject config2, bool detail);

void play_group_of_games(int T, gameplayer *player1, gameplayer *player2,
    std::string player1_name, std::string player2_name, 
    ConfigObject config1, ConfigObject config2);

int main(int argc, char *argv[]) {
    // run 10000 test-cases for bitboard
    bit_board_unit_test(10000);
    srand(time(NULL));
    alphabeta_player player1;
    alphabeta_player player2;
    mcts_pure player3;
    mcts_pure player4;
    human_player player5;
    human_player player6;
    ConfigObject config1;
    ConfigObject config2;
    int type1, type2, d;
    gameplayer *g1;
    gameplayer *g2;
    std::cout << "please input player 1\n1 for alpha-beta, 2 for pure-mcts, 3 for human: ";
    std::cin >> type1;
    if (type1 == 1) {
        g1 = &player1;
        std::cout << "please input the searching depth: ";
        std::cin >> d;
        config1.Set_alpha_beta_depth(d >= 3 ? d : 3);
    } else if (type1 == 2) {
        g1 = &player3;
        std::cout << "please input the number of MCTS iteration: ";
        std::cin >> d;
        config1.Set_mcts_play_iteration(d >= 10000 ? d : 10000).Set_c_puct(5);
    } else if (type1 == 3) {
        g1 = &player5;
    } else {
        std::cerr << "player 1 type must be within {1, 2, 3}" << std::endl;
        return 1;
    }

    std::cout << "please input player 2\n1 for alpha-beta, 2 for pure-mcts, 3 for human: ";
    std::cin >> type2;
    if (type2 == 1) {
        g2 = &player2;
        std::cout << "please input the searching depth: ";
        std::cin >> d;
        config2.Set_alpha_beta_depth(d >= 3 ? d : 3);
    } else if (type2 == 2) {
        g2 = &player4;
        std::cout << "please input the number of MCTS iteration: ";
        std::cin >> d;
        config2.Set_mcts_play_iteration(d >= 10000 ? d : 10000).Set_c_puct(5);
    } else if (type2 == 3) {
        g2 = &player6;
    } else {
        std::cerr << "player 2 type must be within {1, 2, 3}" << std::endl;
        return 1;
    }

    //config1.Set_alpha_beta_depth(11);
    //config2.Set_mcts_play_iteration(500000);
    play_group_of_games(10, g1, g2, "Alpha-Beta-d-11", "MCTS-500000", config1, config2);
    return 0;
}


void play_group_of_games(int T, gameplayer *player1, gameplayer *player2
        ,std::string player1_name, std::string player2_name, 
        ConfigObject config1, ConfigObject config2) {
    int win = 0, lose = 0, draw = 0;
    for (int i = 1; i <= T; ++i) {
        int res = 0;
        if (i % 2 == 1) {
            res = play_game(player1, player2, player1_name, player2_name, config1, config2, false);
            if (res == 1) {
                win++;
            } else if (res == 0) {
                draw++;
            } else {
                lose++;
            }
        } else {
            res = play_game(player2, player1, player2_name, player1_name, config2, config1, false);
            if (res == -1) {
                win++;
            } else if (res == 0) {
                draw++;
            } else {
                lose++;
            }
        }

        printf("(%.1lf - %.1lf)\n", (1.0 * win + 0.5 * draw), (1.0 * lose + 0.5 * draw));
    }

    printf("Final result of %d games <%s vs %s>\n %.1lf - %.1lf", 
        T, player1->display_name().c_str(), player2->display_name().c_str(),
        (1.0 * win + 0.5 * draw), (1.0 * lose + 0.5 * draw));
}

int play_game(gameplayer *player1, gameplayer *player2, 
        std::string player1_name, std::string player2_name,
        ConfigObject config1, ConfigObject config2, bool detail) {
    connect4_board brd;
    brd.init();
    
    player1->init(1, player1_name, config1);

    player2->init(-1, player2_name, config2);

    clock_t t1 = 0, t2 = 0;
    int previous = -1, current = 1;
    while (brd.get_status() == 2) {
        int move = 0;
        if (current == 1) {
            std::cout << player1->display_name() << " move" << std::endl;
            clock_t start = clock();
            move = player1->play(previous);
            clock_t end = clock();
            t1 = t1 + end - start; 
            printf("Move %d taken: %.2fs\n", brd.get_move() + 1, (double)(end - start)/CLOCKS_PER_SEC);
        } else {
            std::cout << player2->display_name() << " move" << std::endl;
            clock_t start = clock();
            move = player2->play(previous);
            clock_t end = clock();
            t2 = t2 + end - start;
            printf("Move %d taken: %.2fs\n", brd.get_move() + 1, (double)(end - start)/CLOCKS_PER_SEC);
        }

        previous = move;
        brd.update(move, current);
        brd.show_board();
        current *= -1;
    }

    std::cout << brd.print_board() << std::endl;
    if (brd.get_status() == 0) {
        std::cout << "Draw!" << std::endl;
    } else if (brd.get_status() == 1) {
        std::cout << player1->display_name() << " Win!" << std::endl;
    } else {
        std::cout << player2->display_name() << " Win!" << std::endl;
    }

    if (brd.get_status() == 0) {
        player1->game_over(0);
        player2->game_over(0);
    } else if (brd.get_status() == 1) {
        player1->game_over(1);
        player2->game_over(-1);
    } else {
        player1->game_over(-1);
        player2->game_over(1);
    }
    printf("Game Time %s: %.2lfs, %s: %.2lfs\n", player1->display_name().c_str(), (double) t1/CLOCKS_PER_SEC, 
                                                player2->display_name().c_str(), (double) t2/CLOCKS_PER_SEC);
    return brd.get_status();
}

void tensor_test() {
    // tensor = torch::rand({2, 3});
    int n = 5, m = 4;
    // Just creating some dummy data for example
    std::vector<std::vector<double>> vect(n, std::vector<double>(m, 0)); 
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            vect[i][j] = i+j;

    // Copying into a tensor
    auto options = torch::TensorOptions().dtype(at::kDouble);
    torch::Tensor tensor = torch::zeros({n,m}, options);
    for (int i = 0; i < n; i++)
        tensor.slice(0, i,i+1) = torch::from_blob(vect[i].data(), {m}, options);
    std::cout << tensor << std::endl;
    std::cout << tensor[0][2].item<double>() << std::endl;
    std::cout << "Pytorch start success" << std::endl;
}


void bit_board_unit_test(int T) {
    int npass = 0, nfail = 0;
    clock_t judge_time = 0, fast_time = 0;
    srand(time(NULL));
    for (int i = 1 ; i <= T; ++i) {
        connect4_board judge;
        bit_board board = bit_board();
        std::vector<int> rec;
        int curr = 1;
        bool ok = true;
        judge.init();
        while (judge.get_status() == 2) {
            int nxt = rand() % 7;
            if (judge.canplay(nxt) != board.can_move(nxt)) {
                ok = false;
                break;
            }

            if (judge.canplay(nxt)) {
                rec.push_back(nxt);
                clock_t start = clock();
                judge.update(nxt, curr);
                clock_t end = clock();
                judge_time += end - start;
                start = clock();
                board.do_move(nxt);
                end = clock();
                fast_time += end - start;
                curr *= -1;
            }
            
            bool answer, observe;
            clock_t start = clock();
            answer = judge.get_status();
            clock_t end = clock();
            judge_time += end - start;

            start = clock();
            observe = board.has_winner().second;
            end = clock();
            fast_time += end - start;
            if (answer != observe || judge.get_move() != board.get_move()) {
                ok = false;
                break;
            }
        }
        
        if (!ok) {
            printf("-------- Test %d --------\n", i);
            printf("FAIL\nMove details: [");
            for (auto v : rec) {
                printf("%d, ", v);
            }
            printf("]\n");

            printf("judge status = %d, board status = %d\n", judge.get_status(), board.has_winner().second);
            printf("total move by judges = %d, total move by bitboard = %d, judge can move: ", judge.get_move(), board.get_move());
            for (int j = 0 ; j < 7; ++j) {
                if (judge.canplay(j)) printf("%d ", j);
            }
            printf("\n");
            printf("bit board can move: ");
            for (int j = 0 ; j < 7; ++j) {
                if (board.can_move(j)) printf("%d ", j);
            }
            printf("\n");
            judge.show_board();
            nfail++;
            break;
        } 

        npass++;
    }

    printf("%d cases PASS, %d cases FAIL, brute force takes %.2lfs, bit-board takes %.2lfs\n", 
                        npass, nfail, (double) judge_time / CLOCKS_PER_SEC, (double) fast_time / CLOCKS_PER_SEC);
    if (!nfail) {
        printf("Unit test for bit_board\nOK!\n");
    } else {
        printf("Unit test for bit_board\nFAIL!\n");
    }
}
