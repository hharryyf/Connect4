#include <torch/torch.h>
#include <iostream>
#include "connect4_board.h"
#include "gameplayer.h"
#include "alpha_beta/alphabeta.h"
#include "human_play/humanplayer.h"

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    alphabeta_player player;
    human_player human;
    connect4_board brd;
    brd.init();
    player.init(1);
    human.init(-1);
    int previous = -1, current = 1;
    while (brd.get_status() == 2) {
        int move = 0;
        if (current == 1) {
            move = player.play(previous); 
        } else {
            move = human.play(previous);
        }

        previous = move;
        brd.update(move, current);
        current *= -1;
    }

    std::cout << brd.print_board() << std::endl;
    if (brd.get_status() == 0) {
        std::cout << "Draw!" << std::endl;
    } else if (brd.get_status() == 1) {
        std::cout << player.display_name() << " Win!" << std::endl;
    } else {
        std::cout << human.display_name() << " Win!" << std::endl;
    }

    human.game_over();
    player.game_over();
    return 0;
}