#include <boost/algorithm/string.hpp>
#include "Hex.hpp"

#include "hex_algorithms.hpp"

using namespace shark;
using namespace Hex;

#define NUM_EPSIODES_CMA 50000
#define NUM_EPSIODES_TD 10000000

/******************\
 *  Base Trainer  *
\******************/
template <class AlgorithmType>
class ModelTrainer {
public:
    ModelTrainer() {}
    virtual void playExampleGame() = 0;
    virtual void playAgainstRandom() = 0;
    virtual void printTrainingStatus() = 0;
    virtual void step() = 0;
    virtual void saveModel(std::string modelName) = 0;
protected:
    AlgorithmType m_algorithm;
    size_t m_steps = 0;
    float m_wins_vs_random = 0;
    float m_games_vs_random_played = 0;
    std::deque<float> m_last_wins;
    std::size_t m_games_played;

    void updateRandomPlayStatus(bool blue_won) {
        m_games_vs_random_played++;
        if (blue_won) {
            std::cout << "Network strategy (blue) won!" << std::endl;
            m_wins_vs_random++;
            m_last_wins.push_back(1);
        } else {
            std::cout << "Random strategy (red) won!" << std::endl;
            m_last_wins.push_back(0);
        }
        std::cout << "blue winrate: " << m_wins_vs_random / m_games_vs_random_played << std::endl;
        if (m_last_wins.size() > 100) {
            m_last_wins.pop_front();
        }
        float sum = 0;
        for (int i=0; i < m_last_wins.size(); i++) {
            sum += m_last_wins[i];
        }
        std::cout << "blue winrate last " << m_last_wins.size() << " games: " << sum / m_last_wins.size() << std::endl;
    }
};


/******************\
 *  CMA  Trainer  *
\******************/
class ModelTrainerCMA : public ModelTrainer<CMAAlgorithm> {
    CMANetworkStrategy m_player2;
public:
    ModelTrainerCMA() {
        m_player2.setColor(Red);
    }

    void playExampleGame() override {
        Game game = m_algorithm.GetGame();
        CMANetworkStrategy player1 = m_algorithm.GetStrategy();
        SelfRLCMA cma = m_algorithm.GetCMA();
        game.reset();
        player1.setParameters(cma.mean());
        m_player2.setParameters(cma.mean());
        std::cout << game.asciiState() << std::endl;
        while (game.takeStrategyTurn({&player1, &m_player2})) {
            std::cout << game.asciiState() << std::endl;
        }
        std::cout << game.asciiState() << std::endl;
        std::cout << "End of example game." << std::endl;
    }

    void playAgainstRandom() override {
        RandomStrategy random_player;
        Game game = m_algorithm.GetGame();
        CMANetworkStrategy player1 = m_algorithm.GetStrategy();
        SelfRLCMA cma = m_algorithm.GetCMA();

        game.reset();
        player1.setParameters(cma.mean());
        // std::cout << game.asciiState() << std::endl;
        while (game.takeStrategyTurn({&player1, &random_player})) {
            //   std::cout << game.asciiState() << std::endl;
        }

        std::cout << game.asciiState() << std::endl;
        std::cout << "end of random game" << std::endl;
        updateRandomPlayStatus(game.getRank(Blue));
    }

    void printTrainingStatus() override {
        std::cout<<"Training games: " << m_steps << "\nSigma: " << m_algorithm.GetCMA().sigma() << std::endl;
        std::cout<<"Value " << m_algorithm.GetCMA().solution().value << std::endl;
        std::cout<< "Learn: " << m_algorithm.GetCMA().rate() << std::endl;
    }

    void step() override {
        m_algorithm.EpisodeStep(m_steps);
        m_steps++;
    }

    void saveModel(std::string modelName) override {
        m_algorithm.GetStrategy().setParameters(m_algorithm.GetCMA().mean());
        m_algorithm.GetStrategy().saveStrategy(modelName);
    }
};


/******************\
 *  TD   Trainer  *
\******************/
class ModelTrainerTD : public ModelTrainer<TDAlgorithm> {
private:
public:
    ModelTrainerTD() {}

    void playExampleGame() override {
        Game game = m_algorithm.GetGame();
        TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
        game.reset();
        std::cout << game.asciiState() << std::endl;
        bool won = false;
        while (!won) {
            RealVector feasibleMoves;
            if (game.ActivePlayer() == Hex::Red) {
                feasibleMoves = game.getFeasibleMoves( TDplayer1.rotateField(game.getGameBoard() ) );
            } else {
                feasibleMoves = game.getFeasibleMoves( game.getGameBoard() );
            }
            std::pair<double, int> chosen_move = TDplayer1.getChosenMove(feasibleMoves, game.getGameBoard(), game.ActivePlayer(), false);
            won = !game.takeTurn(chosen_move.second);
            std::cout << game.asciiState() << std::endl;
        }
        std::cout << game.asciiState() << std::endl;
    }
    void playAgainstRandom() override {
        RandomStrategy random_player;
        TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
        Game game = m_algorithm.GetGame();
        game.reset();

        bool won = false;
        while (!won) {
            if (game.ActivePlayer() == Blue) {
                std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game.getFeasibleMoves(game.getGameBoard()), game.getGameBoard(), Blue, false);
                won = !game.takeTurn(chosen_move.second);
            } else {
                won = !game.takeStrategyTurn({NULL, &random_player});
            }
        }
        std::cout << game.asciiState() << std::endl;
        std::cout << "end of random game" << std::endl;
        updateRandomPlayStatus(game.getRank(Blue));
    }

    void printTrainingStatus() override {
        //std::cout<<"Training games: " << m_steps << "\nSigma: " << m_algorithm->GetCMA().sigma() << std::endl;
        //std::cout<<"Value " << m_algorithm->GetCMA().solution().value << std::endl;
        //std::cout<< "Learn: " << m_algorithm->GetCMA().rate() << std::endl;
    }

    void step() override {
        m_algorithm.EpisodeStep(m_steps);
        m_steps++;
    }

    void saveModel(std::string modelName) override {
        m_algorithm.GetStrategy().saveStrategy(modelName);
    }
};


/*******************\
 *  Training Loop  *
\*******************/
template<class TrainerType>
void trainingLoop(std::string modelName) {
    TrainerType trainer;

    for (int i=0; i < NUM_EPSIODES_CMA; i++) {
        //if (i % 10 == 0) {
        //    trainer.playAgainstRandom();
        //}
        //if (i % 50 == 0) {
        //    std::cout << std::endl << "Training status: " << std::endl;
        //    trainer.printTrainingStatus();
        //}
        if (i % 100 == 0) {
            trainer.saveModel(modelName);
        }
        if (i % 100 == 0) {
            trainer.playExampleGame();
        }
        trainer.step();
    }
}


/********************\
 *  For python app  *
\********************/
void initializePythonSettings() {
    std::cout << "__MODEL_GOOD__" << std::endl;
    std::string response = "";
    std::getline(std::cin, response);
    std::cout << "__BOARD_SIZE__ " << BOARD_SIZE << std::endl;
}

void playHexTDVsHuman(std::string model) {
    HumanStrategy human_player(true);
    TDNetworkStrategy TDplayer1;
    if (model.length()) {
        TDplayer1.loadStrategy(model);
    }
    Game game;
    game.reset();
    initializePythonSettings();
    std::cout << game.asciiStatePython() << std::endl;
    bool won = false;
    while (!won) {
        if (game.ActivePlayer() == Blue) {
            std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game.getFeasibleMoves(game.getGameBoard()), game.getGameBoard(), Blue, false);
            won = !game.takeTurn(chosen_move.second);
        } else {
            won = !game.takeStrategyTurn({NULL, &human_player});
        }
        std::cout << game.asciiStatePython() << std::endl;
    }
    std::cout << game.asciiStatePython() << std::endl;
    std::cout << "__GAME_OVER__ " << (game.getRank(0) ? 0 : 1) << std::endl;
}

void playHexCMAVsHuman(std::string model) {
    HumanStrategy human_player(true);
    CMANetworkStrategy CMAplayer1;
    if (model.length()) {
        CMAplayer1.loadStrategy(model);
    }
    Game game;
    game.reset();
    initializePythonSettings();
    std::cout << game.asciiStatePython() << std::endl;
    while (game.takeStrategyTurn({&CMAplayer1, &human_player})) {
        std::cout << game.asciiStatePython() << std::endl;
    }
    std::cout << game.asciiStatePython() << std::endl;
    std::cout << "__GAME_OVER__ " << (game.getRank(0) ? 0 : 1) << std::endl;
}


/**********\
 *  Main  *
\**********/
int main (int argc, char* argv[]) {
    shark::random::globalRng().seed(time(NULL));

    if (argc > 3) {
        std::cout << "usage: (what: traincma/cma, traintd/td, cmaplay, tdplay) (model)" << std::endl;
        exit(1);
    }

    std::string what  = (argc >= 2) ? argv[1] : "";
    std::string model = (argc == 3) ? argv[2] : "";

    if (what.length() == 0) {
        std::cout << "what to run? Options are: traincma (or cma), traintd (or td), cmaplay, tdplay" << std::endl;
        getline(std::cin, what);
    }

    bool train_td;
    if (boost::iequals(what, "traincma") || boost::iequals(what, "cma")) {
        train_td = false;
    }
    else if (boost::iequals(what, "traintd") || boost::iequals(what, "td")) {
        train_td = true;
    }
    else if (boost::iequals(what, "cmaplay")) {
        train_td = false;
        playHexCMAVsHuman(model);
        return 0;
    }
    else if (boost::iequals(what, "tdplay")) {
        playHexTDVsHuman(model);
        return 0;
    } else {
        std::cout << "invalid input. Options are: traincma (or cma), traintd (or td), cmaplay, tdplay" << std::endl;
        return 1;
    }

    if (train_td) {
        trainingLoop<ModelTrainerTD>("TDmodel");
    } else {
        trainingLoop<ModelTrainerCMA>("CMAmodel");
    }

    return 0;
}
