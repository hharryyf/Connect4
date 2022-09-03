#include <torch/torch.h>
#include <iostream>
#include "gameplayer/connect4_board.h"
#include "gameplayer/gameplayer.h"
#include "gameplayer/alphabeta.h"
#include "gameplayer/humanplayer.h"
#include "gameplayer/alphabeta_board.h"

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    alphabeta_player player;
    board brd;
    brd.init();
    player.init(1);
    int move = player.play(-1);
    brd.update(move, 1);
    brd.update(0, -1);
    move = player.play(0);
    brd.update(move, 1);
    std::cout << brd.print_board() << std::endl;
    return 0;
}