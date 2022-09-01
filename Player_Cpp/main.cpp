#include <torch/torch.h>
#include <iostream>
#include "gameplayer/connect4_board.h"
#include "gameplayer/gameplayer.h"
#include "gameplayer/alphabeta.h"
#include "gameplayer/humanplayer.h"

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    alphabeta_player player;
    player.init(1);
    std::cout << player.play(-1) << std::endl;
    human_player human;
    human.init(1);
}