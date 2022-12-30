#include "combine_player.h"

void combine_player::init(int turn, std::string name, ConfigObject config) {
    this->g1->init(turn, name, config);
    this->g2->init(turn, name, config);
    this->num_move = 0;
    this->name = name;
} 


int combine_player::play(int previous_move) {
    if (previous_move != -1) this->num_move++;
    int win = this->g2->winning_move(previous_move);
    if (win != -1) {
        this->num_move++;
        this->g1->force_move(previous_move, win);
        this->g2->force_move(previous_move, win);
    } else {
        if (this->num_move < 18) {
            this->num_move++;
            win = this->g2->force_move(previous_move, this->g1->play(previous_move));
        } else {
            this->num_move++;
            win = this->g1->force_move(previous_move, this->g2->play(previous_move));
        }
    }

    return win;
    // if (this->num_move < 12) {
    //     this->num_move++;
    //     return this->g2->force_move(previous_move, this->g1->play(previous_move));
    // }

    // this->num_move++;
    // return this->g1->force_move(previous_move, this->g2->play(previous_move));
}

int combine_player::force_move(int previous_move, int move) {
    this->g1->force_move(previous_move, move);
    this->g2->force_move(previous_move, move);
    return move;
}

std::string combine_player::display_name() {
    return this->name;
}

void combine_player::set_players(gameplayer *g1, gameplayer *g2) {
    this->g1 = g1;
    this->g2 = g2;
}

void combine_player::game_over(int result) {

}

void combine_player::debug() {

}