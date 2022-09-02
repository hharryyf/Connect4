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
    alphabeta_board board;
    board.init();
    board.debug();
    std::cout << board.print_board() << std::endl;
    board.update(3, 1);
    board.debug();
    board.update(3, -1);
    board.debug();
    board.update(2, 1);
    board.debug();
    board.update(3, -1);
    board.debug();
    board.update(4, 1);
    board.debug();
    return 0;
}