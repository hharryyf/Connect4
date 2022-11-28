#pragma warning(push, 0)
#include <torch/torch.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#pragma warning(pop)
#include <iostream>
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

void play_game();

void tensor_test();

void bit_board_unit_test(int T=1000000);

int main(int argc, char *argv[]) {
    bit_board_unit_test();
    return 0;
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
        printf("OK!\n");
    } else {
        printf("Wrong Answer!\n");
    }
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

void play_game() {
    alphabeta_player b, b2;
    human_player human;
    connect4_board brd;
    brd.init();
    b.init(1, "Alpha-Beta AI X Player");
    b2.init(-1, "Alpha-Beta AI O Player");
    human.init(-1, "Human Player");
    gameplayer *player, *player2;
    /* Do your stuff here */
    
    clock_t t1 = 0, t2 = 0;
    player = &b;
    player2 = &human;

    // player->debug();
    int previous = -1, current = 1;
    while (brd.get_status() == 2) {
        int move = 0;
        if (current == 1) {
            std::cout << player->display_name() << " move" << std::endl;
            clock_t start = clock();
            move = player->play(previous);
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
        std::cout << player->display_name() << " Win!" << std::endl;
    } else {
        std::cout << player2->display_name() << " Win!" << std::endl;
    }

    if (brd.get_status() == 0) {
        player->game_over(0);
        player2->game_over(0);
    } else if (brd.get_status() == 1) {
        player->game_over(1);
        player2->game_over(-1);
    } else {
        player->game_over(-1);
        player2->game_over(1);
    }
    printf("Game Time %s: %.2lfs, %s: %.2lfs\n", player->display_name().c_str(), (double) t1/CLOCKS_PER_SEC, 
                                                player2->display_name().c_str(), (double) t2/CLOCKS_PER_SEC);
}