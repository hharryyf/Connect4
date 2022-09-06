#include <torch/torch.h>
#include <iostream>
#include "connect4_board.h"
#include "gameplayer.h"
#include "alpha_beta/alphabeta.h"
#include "human_play/humanplayer.h"
#define ALPHA_BETA_X 0
#define MCTS_DEEP_X 1
#define MCTS_DEEP_O 2
#define ALPHA_BETA_O 3

int main(int argc, char *argv[]) {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << "Pytorch start success" << std::endl;
    alphabeta_player b;
    human_player human, human2;
    connect4_board brd;
    brd.init();
    b.init(1);
    human.init(-1);
    human2.init(1);
    int type = 0;
    gameplayer *player;
    /* Do your stuff here */
    
    clock_t t1 = 0, t2 = 0;
    if (type == ALPHA_BETA_X) {
        player = &b;
    } else {
        player = &human2;
    }

    // player->debug();
    int previous = -1, current = 1;
    while (brd.get_status() == 2) {
        int move = 0;
        if (current == 1) {
            clock_t start = clock();
            move = player->play(previous);
            clock_t end = clock();
            t1 = t1 + end - start; 
            printf("Move %d taken: %.2fs\n", brd.get_move(), (double)(end - start)/CLOCKS_PER_SEC);
        } else {
            clock_t start = clock();
            move = human.play(previous);
            clock_t end = clock();
            t2 = t2 + end - start;
            printf("Move %d taken: %.2fs\n", brd.get_move(), (double)(end - start)/CLOCKS_PER_SEC);
        }

        previous = move;
        brd.update(move, current);
        current *= -1;
    }

    std::cout << brd.print_board() << std::endl;
    if (brd.get_status() == 0) {
        std::cout << "Draw!" << std::endl;
    } else if (brd.get_status() == 1) {
        std::cout << player->display_name() << " Win!" << std::endl;
    } else {
        std::cout << human.display_name() << " Win!" << std::endl;
    }

    human.game_over();
    player->game_over();
    printf("Game Time %s: %.2lfs, %s: %.2lfs\n", player->display_name().c_str(), (double) t1/CLOCKS_PER_SEC, 
                                                human.display_name().c_str(), (double) t2/CLOCKS_PER_SEC);
    return 0;
}