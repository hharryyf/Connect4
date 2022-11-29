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

int mcts_pure::force_play(int position) {
    this->board.do_move(position);
    this->mcts.update_with_move(position);
    return position;
}

int mcts_pure::play(int previous_move) {

    assert(!this->board.game_end());
    
    if (previous_move != -1) { 
        this->mcts.update_with_move(previous_move);
        this->board.do_move(previous_move);
    }

    assert(!this->board.game_end());
    
    int move = this->mcts.get_move(this->board);
    this->board.do_move(move);
    return move;
}

void mcts_pure::game_over(int result) {

}

void mcts_pure::debug() {

}

