#include "mcts_pure.h"


void pure_mcts_tree::playout(bit_board board) {
    auto curr = this->root;
    while (curr != nullptr) {
        if (curr->is_leaf()) break;
        auto mpn = curr->selection(this->c_puct);
        board.do_move(mpn.first);
        curr = mpn.second;
    }

    auto action_prob = this->policy_value_function(board);
    if (!board.has_winner().first) {
        curr->expansion(action_prob);
    }

    int reward = evaluate_rollout(board);
    curr->update_recursive(-1.0 * reward);
}


int pure_mcts_tree::get_move(bit_board board) {
    for (int i = 0 ; i < this->num_playout; ++i) {
        this->playout(board);
    }

    this->root->debug();
    return this->root->select_move();  
}


void pure_mcts_tree::update_with_move(int move) {
    auto nxt = this->root->get_children(move);
    if (nxt == nullptr) {
        this->root = std::make_shared<mcts_node>(mcts_node(nullptr, 1.0));
    } else {
        this->root = nxt;
        this->root->set_parent(nullptr);
    }
}


int mcts_pure::play(int previous_move) {    
    if (this->board.game_end()) {
        printf("cannot play when the game ends!\n");
        exit(1);
    }

    if (previous_move != -1) { 
        this->board.do_move(previous_move);
    }

    
    if (this->board.game_end()) {
        printf("cannot play when the game ends!\n");
        exit(1);
    }

    if (previous_move == -1) {
        // we play in the center for the first move
        // this uses some expert knowledge
        this->mcts.update_with_move(-1);
        this->board.do_move(3);
        return 3;
    }

    // general case
    int move = this->mcts.get_move(this->board);
    this->board.do_move(move);
    this->mcts.update_with_move(-1);
    return move;
}

int mcts_pure::force_move(int previous_move, int move) {
    if (previous_move != -1) {
        this->board.do_move(previous_move);
    }

    this->board.do_move(move);
    return move;
}

void mcts_pure::game_over(int result) {

}

void mcts_pure::debug() {

}

