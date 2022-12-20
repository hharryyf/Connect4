#pragma warning(push, 0)
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#pragma warning(pop)
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <vector>
#include "bit_board.h"
#include "connect4_board.h"
#include "gameplayer.h"
#include "memory_buffer.h"
#include "alpha_beta/alphabeta.h"
#include "alpha_beta/alphabeta_board.h"
#include "human_play/humanplayer.h"
#include "deepq_mcts/mcts_pure.h"
#include "deepq_mcts/mcts_zero.h"
#include "dirichlet.h"
#include <winsock2.h>
#include <Windows.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#define ALPHA_BETA_X 0
#define MCTS_DEEP_X 1
#define MCTS_DEEP_O 2
#define ALPHA_BETA_O 3

void tensor_test();

void test_load_model();

void test_dirichlet();

void alpha_beta_board_unit_test(int T=1000000);

void bit_board_unit_test(int T=1000000);

void start_interactive_game();

int play_game(gameplayer *player1, gameplayer *player2
    ,std::string player1_name, std::string player2_name, 
    ConfigObject config1, ConfigObject config2, bool detail);

double play_group_of_games(int T, gameplayer *player1, gameplayer *player2,
    std::string player1_name, std::string player2_name, 
    ConfigObject config1, ConfigObject config2);


void start_training_game(int T);

void test_memory_buffer();

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("please run ./main -h for help\n");
        return 0;
    }

    if (strcmp(argv[1], "-h") == 0) {
        printf("-a for running unit test for alpha_beta_board\n");
        printf("-b for running unit test for bit_board\n");
        printf("-d for running unit test for dirichlet distribution\n");
        printf("-g for running games\n");
        printf("-h for help\n");
        printf("-t for running tests\n");
        printf("-l [number of games] for training the deepQ network\n");
        return 0;
    }

    if (strcmp(argv[1], "-t") == 0) {
        tensor_test();
        test_load_model();
    } else if (strcmp(argv[1], "-a") == 0) {
        alpha_beta_board_unit_test(1000000);
    } else if (strcmp(argv[1], "-b") == 0) {
        bit_board_unit_test(1000000);
    } else if (strcmp(argv[1], "-d") == 0) {
        test_dirichlet();
    } else if (strcmp(argv[1], "-g") == 0) {
        start_interactive_game();
    } else if (strcmp(argv[1], "-l") == 0) {
        if (argc == 2) {
            start_training_game(10000);
        } else {
            start_training_game(atoi(argv[2]));
        }
    } else if (strcmp(argv[1], "-m") == 0) {
        test_memory_buffer();   
    } else {
        printf("command %s not supported\n", argv[1]);
    }
    return 0;
}

void test_memory_buffer() {
    printf("please input the size of the buffer: ");
    int sz;
    char s[3];
    scanf("%d", &sz);
    printf("+ [integer] to add an element\n? [number] to sample [number] many elements\n# to get the size of the buffer\n");
    memset(s, 0, sizeof(s));
    memory_buffer<int> buffer(sz);
    while (scanf("%s", s) != EOF) {
        if (s[0] == '+') {
            int v;
            scanf("%d", &v);
            buffer.add(v);
        } else if (s[0] == '?') {
            int v;
            scanf("%d", &v);
            auto output = buffer.sample(v);
            for (auto vv : output) {
                printf("%d ", vv);
            }
            printf("\n");
        } else if (s[0] == '#') {
            printf("size = %d\n", (int) buffer.size());
        } else {
            printf("command not supported!\n");
            break;
        }
    }
}

void test_load_model() {
    try {
        int sz = 2;
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // we load the model and do 1 iteration of backpropagation
        auto module = torch::jit::load("../../model/resblock_old.pt");
        std::vector<torch::jit::IValue> inputs;
        std::vector<at::Tensor> parameters;
        for (const auto& params : module.parameters()) {
	        parameters.push_back(params);
        }
        torch::optim::Adam optimizer(parameters, /*lr=*/0.1);
        std::vector<int> state = std::vector<int>(sz * 3 * 6 * 7, 1);
        std::cout << state.size() << std::endl;
        auto t = torch::from_blob(state.data(), {sz, 3, 6, 7}).clone();
        auto target_p = torch::zeros({sz, 7});
        auto target_v = torch::ones({sz, 1});
        auto target_v_e = torch::exp(target_v);
        std::cout << "target_v = " << target_v << " target_v_exp = " << target_v_e << std::endl;
        std::cout << "target v has type " << target_v.dtype() << std::endl;
        t[0][0][0][0] = 0.0;
        inputs.push_back(t);
        auto output = module.forward(inputs);
        std::cout << output.toTuple()->elements()[0].toTensor() << std::endl;
        std::cout << output.toTuple()->elements()[1].toTensor() << std::endl;
        for (int i = 0 ; i < sz; ++i) {
            std::cout << "item of score [" << i << "]" << output.toTuple()->elements()[1].toTensor()[i][0].item<float>() << std::endl;
            for (int j = 0 ; j < 7; ++j) {
                std::cout << "item of policy [" << j << "]" << output.toTuple()->elements()[0].toTensor()[i][j].item<float>() << std::endl;    
            }
        }
        optimizer.zero_grad();
        auto loss_v = torch::mse_loss(output.toTuple()->elements()[1].toTensor(), target_v);
        std::cout << "value loss " <<  loss_v << std::endl;
        auto loss_p = -torch::mean(torch::sum(target_p * output.toTuple()->elements()[0].toTensor(), 1));
        std::cout << "policy loss " <<  loss_p << " inner term " << torch::sum(target_p * output.toTuple()->elements()[0].toTensor(), 1) << std::endl;
        auto loss = loss_v + loss_p;
        loss.backward();
        optimizer.step();
        module.save("../../model/resblock_old.pt");
        std::cout << module.forward(inputs).toTuple()->elements()[0].toTensor() << std::endl;
        std::cout << module.forward(inputs).toTuple()->elements()[1].toTensor() << std::endl;
    }
    
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return;
    }

    std::cout << "Model Load successfully\n";
}

void start_interactive_game() {
    alphabeta_player player1;
    alphabeta_player player2;
    mcts_pure player3;
    mcts_pure player4;
    human_player player5;
    human_player player6;
    mcts_zero player7;
    mcts_zero player8;
    ConfigObject config1;
    ConfigObject config2;
    config1.Set_c_puct(3).Set_dqn_decay(0.0001).Set_dqn_lr(0.002).Set_dqn_noise_portion(0.25).Set_dqn_temp(1e-3).Set_dirichlet_alpha(0.3).Set_reload(true);
    config2.Set_c_puct(3).Set_dqn_decay(0.0001).Set_dqn_lr(0.002).Set_dqn_noise_portion(0.25).Set_dqn_temp(1e-3).Set_dirichlet_alpha(0.3).Set_reload(true);
    int type1, type2, d;
    gameplayer *g1;
    gameplayer *g2;
    std::string name1, name2;
    int tolgame = 10;
    std::cout << "please input player 1\n1 for alpha-beta, 2 for pure-mcts, 3 for human, 4 for mcts-zero: ";
    std::cin >> type1;
    if (type1 == 1) {
        g1 = &player1;
        std::cout << "please input the searching depth: ";
        std::cin >> d;
        config1.Set_alpha_beta_depth(d >= 3 ? d : 3);
        name1 = std::string("Alpha-Beta-d-").append(std::to_string(d));
    } else if (type1 == 2) {
        g1 = &player3;
        std::cout << "please input the number of MCTS iteration: ";
        std::cin >> d;
        config1.Set_mcts_play_iteration(d >= 1000 ? d : 1000).Set_c_puct(5);
        name1 = std::string("Pure-MCTS-").append(std::to_string(d));
    } else if (type1 == 3) {
        g1 = &player5;
        name1 = std::string("Human");
    } else if (type1 == 4) {
        config1.Set_mcts_play_iteration(5000).Set_c_puct(3);
        name1 = std::string("MCTS-Zero");
        player7.init(1, name1, config1);
        config1.Set_reload(false);
        player7.set_train(config1, false);
        g1 = &player7;
    } else {
        std::cerr << "player 1 type must be within {1, 2, 3, 4}" << std::endl;
        exit(1);
    }

    std::cout << "please input player 2\n1 for alpha-beta, 2 for pure-mcts, 3 for human, 4 for mcts-zero: ";
    std::cin >> type2;
    if (type2 == 1) {
        g2 = &player2;
        std::cout << "please input the searching depth: ";
        std::cin >> d;
        config2.Set_alpha_beta_depth(d >= 3 ? d : 3);
        name2 = std::string("Alpha-Beta-d-").append(std::to_string(d));
    } else if (type2 == 2) {
        g2 = &player4;
        std::cout << "please input the number of MCTS iteration: ";
        std::cin >> d;
        config2.Set_mcts_play_iteration(d >= 1000 ? d : 1000).Set_c_puct(5);
        name2 = std::string("Pure-MCTS-").append(std::to_string(d));
    } else if (type2 == 3) {
        g2 = &player6;
        name2 = std::string("Human");
    } else if (type2 == 4) {
        config2.Set_mcts_play_iteration(5000).Set_c_puct(3);
        name2 = std::string("MCTS-Zero");
        player8.init(1, name2, config2);
        config2.Set_reload(false);
        player8.set_train(config2, false);
        g2 = &player8;
    } else {
        std::cerr << "player 2 type must be within {1, 2, 3, 4}" << std::endl;
        exit(1);
    }

    std::cout << "please input the total number of games: ";
    std::cin >> tolgame;
    play_group_of_games(tolgame, g1, g2, name1, name2, config1, config2);
}

double play_group_of_games(int T, gameplayer *player1, gameplayer *player2
        ,std::string player1_name, std::string player2_name, 
        ConfigObject config1, ConfigObject config2) {
    int win = 0, lose = 0, draw = 0;
    for (int i = 1; i <= T; ++i) {
        int res = 0;
        if (i % 2 == 1) {
            res = play_game(player1, player2, player1_name, player2_name, config1, config2, false);
            if (res == 1) {
                win++;
            } else if (res == 0) {
                draw++;
            } else {
                lose++;
            }
        } else {
            res = play_game(player2, player1, player2_name, player1_name, config2, config1, false);
            if (res == -1) {
                win++;
            } else if (res == 0) {
                draw++;
            } else {
                lose++;
            }
        }

        printf("(%.1lf - %.1lf)\n", (1.0 * win + 0.5 * draw), (1.0 * lose + 0.5 * draw));
    }

    printf("Final result of %d games <%s vs %s>\n %.1lf - %.1lf\n", 
        T, player1->display_name().c_str(), player2->display_name().c_str(),
        (1.0 * win + 0.5 * draw), (1.0 * lose + 0.5 * draw));
    return (1.0 * win + 0.5 * draw) / T;
}

int play_game(gameplayer *player1, gameplayer *player2, 
        std::string player1_name, std::string player2_name,
        ConfigObject config1, ConfigObject config2, bool detail) {
    connect4_board brd;
    brd.init();
    
    player1->init(1, player1_name, config1);

    player2->init(-1, player2_name, config2);

    clock_t t1 = 0, t2 = 0;
    int previous = -1, current = 1;
    while (brd.get_status() == 2) {
        int move = 0;
        if (current == 1) {
            std::cout << player1->display_name() << " move" << std::endl;
            clock_t start = clock();
            move = player1->play(previous);
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
        std::cout << player1->display_name() << " Win!" << std::endl;
    } else {
        std::cout << player2->display_name() << " Win!" << std::endl;
    }

    if (brd.get_status() == 0) {
        player1->game_over(0);
        player2->game_over(0);
    } else if (brd.get_status() == 1) {
        player1->game_over(1);
        player2->game_over(-1);
    } else {
        player1->game_over(-1);
        player2->game_over(1);
    }
    printf("Game Time %s: %.2lfs, %s: %.2lfs\n", player1->display_name().c_str(), (double) t1/CLOCKS_PER_SEC, 
                                                player2->display_name().c_str(), (double) t2/CLOCKS_PER_SEC);
    return brd.get_status();
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
    torch::Tensor tensor = torch::zeros({n,m});
    for (int i = 0; i < n; i++) {
        for (int j = 0 ; j < m; ++j) {
            tensor[i][j] = vect[i][j];
        }
    }
        //tensor.slice(0, i,i+1) = torch::from_blob(vect[i].data(), {m}, options);
    std::cout << tensor << std::endl;
    std::cout << tensor[0][2].item<double>() << std::endl;
    std::vector<float> vc = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    auto tensor2 = torch::from_blob(vc.data(), {1, 2, 3, 3}).clone();
    std::cout << tensor2 << std::endl;
    vc.clear();
    std::cout << "after delete the vector " << std::endl;
    std::cout << tensor2 << std::endl;
    std::cout << "Pytorch start success!" << std::endl;
}

void test_dirichlet() {
    std::default_random_engine rng = std::default_random_engine {};
    rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<double> vc;
    int sz;
    printf("please input the size of the distribution vector [finish with new-line]: ");
    scanf("%d", &sz);
    printf("please input the input vector of the given size [separate with space & finish with new-line]: ");
    while (sz-- > 0) {
        double v;
        scanf("%lf", &v);
        vc.push_back(v);
    }

    dirichlet_distribution<std::default_random_engine> d(vc);
    for (int i = 1; i <= 10; ++i) {
        auto ret = d(rng);
        double sm = 0.0;
        for (auto &v : ret) sm = sm + v;
        // we use the property of the distribution that the sum of the elements must be 1
        printf("dirichlet vector number [%d] (sum=%.4lf) : ", i, sm);
        for (auto &v : ret) printf("%.3lf ", v);
        printf("\n");

        if (fabs(sm - 1.0) > 1e-7) {
            printf("sum of the distribution vector is far from 1, error!\n");
            exit(1);
        }
    }

    printf("dirichlet load success!\n");
}

void bit_board_unit_test(int T) {
    int npass = 0, nfail = 0;
    clock_t judge_time = 0, fast_time = 0;
    srand(time(NULL));
    std::vector<std::pair<double, double>> consume;
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
        if (i % 1000 == 0) consume.emplace_back((double) judge_time / CLOCKS_PER_SEC, (double) fast_time / CLOCKS_PER_SEC);
    }

    printf("%d cases PASS, %d cases FAIL, brute force takes %.2lfs, bit-board takes %.2lfs\n", 
                        npass, nfail, (double) judge_time / CLOCKS_PER_SEC, (double) fast_time / CLOCKS_PER_SEC);
    if (!nfail) {
        printf("Unit test for bit_board\nOK!\n");
    } else {
        printf("Unit test for bit_board\nFAIL!\n");
    }

    // print the cumlative time
    int i = 1000;
    for (auto p : consume) {
        printf("%d,%.2lf,%.2lf\n", i, p.first, p.second);
        i = i + 1000;
    }
}

void alpha_beta_board_unit_test(int T) {
    int npass = 0, nfail = 0;
    clock_t judge_time = 0, fast_time = 0;
    srand(time(NULL));
    std::vector<std::pair<double, double>> consume;
    for (int i = 1 ; i <= T; ++i) {
        connect4_board judge;
        alphabeta_board board = alphabeta_board(false);
        std::vector<int> rec;
        int curr = 1;
        bool ok = true;
        judge.init();
        board.init();
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
                board.update(nxt, curr);
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
            observe = board.status();
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

            printf("judge status = %d, board status = %d\n", judge.get_status(), board.status());
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
        if (i % 1000 == 0) consume.emplace_back((double) judge_time / CLOCKS_PER_SEC, (double) fast_time / CLOCKS_PER_SEC);
    }

    printf("%d cases PASS, %d cases FAIL, brute force takes %.2lfs, alpha-beta takes %.2lfs\n", 
                        npass, nfail, (double) judge_time / CLOCKS_PER_SEC, (double) fast_time / CLOCKS_PER_SEC);
    if (!nfail) {
        printf("Unit test for alpha-beta_board\nOK!\n");
    } else {
        printf("Unit test for alpha-beta_board\nFAIL!\n");
    }

    // print the cumlative time
    int i = 1000;
    for (auto p : consume) {
        printf("%d,%.2lf,%.2lf\n", i, p.first, p.second);
        i = i + 1000;
    }
}

void start_training_game(int tol_game) {
    std::cout << "start training total " << tol_game << " games" << std::endl;
    ConfigObject config;
    ConfigObject mcts_player_config;
    mcts_zero player;
    mcts_player_config = mcts_player_config.Set_c_puct(3).Set_mcts_play_iteration(1000);
    double winning_rate = 0.0;
    printf("please insert the start winning rate: ");
    scanf("%lf", &winning_rate);
    config = config.Set_c_puct(3).Set_dqn_decay(0.0001).Set_dqn_lr(0.002).Set_dqn_noise_portion(0.25).Set_dqn_temp(1e-3).Set_dirichlet_alpha(0.3).Set_mcts_play_iteration(1000).Set_mcts_train_iteration(500).Set_reload(true);
    player.init(1, "Mcts-Zero-Player", config);
    config = config.Set_reload(false);
    player.set_train(config, true);
    memory_buffer<std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<double>, int>> buffer(10000);
    int mini_batch_sz = 512, epoch = 5;
    for (int t = 1; t <= tol_game; ++t) {
        auto play_data = std::get<1>(player.self_play(config.get_temp()));
        auto board_states = std::get<0>(play_data);
        auto move_probabilities = std::get<1>(play_data);
        auto winners = std::get<2>(play_data);
        printf("batch %d has episode length %d\n", t, (int) board_states.size());
        // add the self-play data + the augmented data to the memory buffer
        for (int i = 0 ; i < (int) board_states.size(); ++i) {
            buffer.add(std::make_tuple(board_states[i], move_probabilities[i], winners[i]));
            for (int j = 0 ; j < 6; ++j) {
                reverse(board_states[i][0][j].begin(), board_states[i][0][j].end());
                reverse(board_states[i][1][j].begin(), board_states[i][1][j].end());
            }
            reverse(move_probabilities[i].begin(), move_probabilities[i].end());
            buffer.add(std::make_tuple(board_states[i], move_probabilities[i], winners[i]));
        }

        if ((int) buffer.size() > mini_batch_sz) {
            // train the neural network for the data buffer
            auto training_data = buffer.sample(mini_batch_sz);
            board_states.clear();
            move_probabilities.clear();
            winners.clear();
            for (int i = 0 ; i < (int) training_data.size(); ++i) {
                board_states.push_back(std::get<0>(training_data[i]));
                move_probabilities.push_back(std::get<1>(training_data[i]));
                winners.push_back(std::get<2>(training_data[i]));
            }
            for (int i = 0 ; i < epoch; ++i) {
                auto p = player.train_step(board_states, move_probabilities, winners);
                printf("loss: %lf, entropy: %lf\n", std::get<0>(p), std::get<1>(p));
            }
        }

        // evaluate the current policy
        if (t % 50 == 0) {
            player.save_model("../../model/current_model.pt");
            config = config.Set_mcts_play_iteration(mcts_player_config.get_mcts_play_iteration());
            player.set_train(config, false);
            mcts_pure player2;
            player2.init(1, std::string("MCTS-Pure") + std::string(std::to_string(mcts_player_config.get_mcts_play_iteration())), mcts_player_config);
            auto nxt_win = play_group_of_games(10, &player, &player2, player.display_name(), player2.display_name(), config, mcts_player_config);
            if (nxt_win > winning_rate) {
                printf("New best policy!\n");
                player.save_model("../../model/best_model.pt");
                winning_rate = nxt_win;
            }

            if (nxt_win == 1.0) {
                mcts_player_config = mcts_player_config.Set_mcts_play_iteration(mcts_player_config.get_mcts_play_iteration() + 1000);
            }
            player.set_train(config, true);
            player.reset_player();
        }
        
    }
}