#include "mcts_pure.h"


void pure_mcts_tree::playout(bit_board board) {
    auto curr = this->root;
    board.debug();
    while (curr != nullptr) {
        if (curr->is_leaf()) break;
        auto mpn = curr->selection(this->c_puct);
        board.do_move(mpn.first);
        curr = mpn.second;
    }

    //board.debug();
    auto action_prob = this->policy_value_function(board);
    if (!board.has_winner().first) {
        //printf("enter here!\n");
        curr->expansion(action_prob);
    }

    int reward = evaluate_rollout(board);
    printf("reward = %d\nstart backpropagation\n", reward);
    curr->update_recursive(-1.0 * reward);
    printf("finish backproagation");
    this->root->debug();
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
    this->mcts.update_with_move(-1);
    return position;
}

int mcts_pure::play(int previous_move) {

    assert(!this->board.game_end());
    printf("before\n");
    board.debug();
    if (previous_move != -1) { 
        this->board.do_move(previous_move);
    }

    assert(!this->board.game_end());
    
    int move = this->mcts.get_move(this->board);
    this->board.do_move(move);
    this->mcts.update_with_move(-1);
    printf("after\n");
    board.debug();
    return move;
}

void mcts_pure::game_over(int result) {

}

void mcts_pure::debug() {

}

