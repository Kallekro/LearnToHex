#include <boost/algorithm/string.hpp>
#include "Hex.hpp"
#include "hex_algorithms.hpp"

using namespace shark;
using namespace Hex;

struct RandomGameStats {
    double wins_vs_random = 0;
    double games_vs_random_played = 0;
    std::deque<double> last_wins;
};

struct PreviousModelStats {

};

/******************\
 *  Base Trainer  *
\******************/
template <class AlgorithmType>
class ModelTrainer {
public:
    ModelTrainer() {}
    virtual void playExampleGame() = 0;
    virtual void playAgainstRandom() = 0;
    virtual void playAgainstModel(std::string model) = 0;
    virtual void printTrainingStatus() = 0;
    virtual void step() = 0;
    virtual void saveModel(std::string modelName) = 0;
    virtual void loadModel(std::string modelName) = 0;
    size_t NumberOfEpisodes() { return m_number_of_episodes; }
    AlgorithmType GetAlgorithm() { return m_algorithm; }

    std::string lastModel;
protected:
    AlgorithmType m_algorithm;
    size_t m_number_of_episodes;
    size_t m_steps = 0;

    struct RandomGameStats m_randomGameStats;
    //struct

    void updateRandomPlayStats(bool blue_won) {
        m_randomGameStats.games_vs_random_played++;
        if (blue_won) {
            std::cout << "Network strategy (blue) won!" << std::endl;
            m_randomGameStats.wins_vs_random++;
            m_randomGameStats.last_wins.push_back(1);
        } else {
            std::cout << "Random strategy (red) won!" << std::endl;
            m_randomGameStats.last_wins.push_back(0);
        }
        std::cout << "blue winrate: " << m_randomGameStats.wins_vs_random / m_randomGameStats.games_vs_random_played << std::endl;
        if (m_randomGameStats.last_wins.size() > 100) {
            m_randomGameStats.last_wins.pop_front();
        }
        double sum = 0;
        for (int i=0; i < m_randomGameStats.last_wins.size(); i++) {
            sum += m_randomGameStats.last_wins[i];
        }
        std::cout << "blue winrate last " << m_randomGameStats.last_wins.size() << " games: " << sum / m_randomGameStats.last_wins.size() << std::endl;
    }
};


/*********************\
 *  CSA-ES  Trainer  *
\*********************/
class ModelTrainerCSA : public ModelTrainer<CSAAlgorithm> {
private:
    CSANetworkStrategy m_player2;
public:
    ModelTrainerCSA() {
        m_number_of_episodes = 1000000;
        m_player2.setColor(Red);
    }

    void playExampleGame() override {
        Game game = m_algorithm.GetGame();
        CSANetworkStrategy player1 = m_algorithm.GetStrategy();
        SelfRLCMA csa = m_algorithm.GetCSA();
        game.reset();
        player1.setParameters(csa.mean());
        m_player2.setParameters(csa.mean());
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
        CSANetworkStrategy player1 = m_algorithm.GetStrategy();
        SelfRLCMA csa = m_algorithm.GetCSA();

        game.reset();
        player1.setParameters(csa.mean());
        // std::cout << game.asciiState() << std::endl;
        while (game.takeStrategyTurn({&player1, &random_player})) {
            //   std::cout << game.asciiState() << std::endl;
        }

        std::cout << game.asciiState() << std::endl;
        std::cout << "end of random game" << std::endl;
        updateRandomPlayStats(game.getRank(Blue));
    }

    void playAgainstModel(std::string model) override {}

    void printTrainingStatus() override {
        std::cout<<"Training games: " << m_steps << "\nSigma: " << m_algorithm.GetCSA().sigma() << std::endl;
        std::cout<<"Value " << m_algorithm.GetCSA().solution().value << std::endl;
        std::cout<< "Learn: " << m_algorithm.GetCSA().rate() << std::endl;
    }

    void step() override {
        m_algorithm.EpisodeStep(m_steps);
        m_steps++;
    }

    void saveModel(std::string modelName) override {
        m_algorithm.GetStrategy().setParameters(m_algorithm.GetCSA().mean());
        m_algorithm.GetStrategy().saveStrategy(modelName);
    }

    void loadModel(std::string modelName) override {
        m_algorithm.GetStrategy().loadStrategy(modelName);
    }
};


/******************\
 *  TD   Trainer  *
\******************/
class ModelTrainerTD : public ModelTrainer<TDAlgorithm> {
public:
    ModelTrainerTD() {
        m_number_of_episodes = 100000000;
    }

    void playExampleGame() override {
        Game game = m_algorithm.GetGame();
        TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
        game.reset();
        std::cout << game.asciiState() << std::endl;
        bool won = false;
        unsigned state = 0;
        while (!won) {
            std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game, false);
            won = !game.takeTurn(chosen_move.second);
            std::cout << game.asciiState() << std::endl;
            state++;
        }
        std::cout << "End of example game." << std::endl;
    }
    void playAgainstRandom() override {
        RandomStrategy random_player;
        TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
        Game game = m_algorithm.GetGame();
        game.reset();

        bool won = false;
        while (!won) {
            if (game.ActivePlayer() == Blue) {
                std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game, false);
                won = !game.takeTurn(chosen_move.second);
            } else {
                won = !game.takeStrategyTurn({NULL, &random_player});
            }
        }
        std::cout << game.asciiState() << std::endl;
        std::cout << "End of model vs random game." << std::endl;
        updateRandomPlayStats(game.getRank(Blue));
    }

    void playAgainstModel(std::string model) override {
        // initialize players with models
        TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
        TDNetworkStrategy TDplayer2;
        TDplayer2.loadStrategy(model);
        Game game;
        game.reset();

        double total_games = 1000;
        double new_model_wins = 0;
        for (int i=0; i < total_games; i++) {
            bool won = false;
            while (!won) {
                std::pair<double, int> chosen_move;
                if (game.ActivePlayer() == Blue) {
                    chosen_move = TDplayer1.getChosenMove(game, false);
                } else {
                    chosen_move = TDplayer2.getChosenMove(game, false);
                }
                won = !game.takeTurn(chosen_move.second);
            }
        }
        double winrate = total_games / new_model_wins;
        std::cout << winrate << " winrate in " << total_games << " games played against previous model.";
    }

    void printTrainingStatus() override {
        std::cout << "Step " << m_steps << std::endl;
    }

    void step() override {
        m_algorithm.EpisodeStep(m_steps);
        m_steps++;
    }

    void saveModel(std::string modelName) override {
        m_algorithm.GetStrategy().saveStrategy(modelName);
    }

    void loadModel(std::string modelName) override {
        m_algorithm.GetStrategy().loadStrategy(modelName);
    }
};


/*******************\
 *  Training Loop  *
\*******************/
template<class TrainerType>
void trainingLoop(std::string modelName) {
    TrainerType trainer;
    for (int i=0; i < trainer.NumberOfEpisodes(); i++) {
        if (i % 1000 == 0) { // Play example game and save model
            trainer.playExampleGame();
            trainer.saveModel(modelName);
        }
        if (i % 1500 == 0) { // Play against a random strategy and display stats
            trainer.playAgainstRandom();
        }
        //if (i % 1000 == 0) { // Play against the previous model
        //    if (i != 0) {
        //        trainer.playAgainstModel(trainer.lastModel);
        //    }
        //    std::string modelName = "TDModel_autosave";
        //    trainer.saveModel(modelName);
        //    trainer.lastModel = modelName + ".model";
        //}
        if (i % 100 == 0) {
            //std::cout << std::endl << "Training status: " << std::endl;
            trainer.printTrainingStatus();

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
            std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game, false);
            won = !game.takeTurn(chosen_move.second);
        } else {
            won = !game.takeStrategyTurn({NULL, &human_player});
        }
        std::cout << game.asciiStatePython() << std::endl;
    }
    std::cout << game.asciiStatePython() << std::endl;
    std::cout << "__GAME_OVER__ " << (game.getRank(0) ? 0 : 1) << std::endl;
}

void playHexCSAVsHuman(std::string model) {
    HumanStrategy human_player(true);
    CSANetworkStrategy CSAplayer1;
    if (model.length()) {
        CSAplayer1.loadStrategy(model);
    }
    Game game;
    game.reset();
    initializePythonSettings();
    std::cout << game.asciiStatePython() << std::endl;
    while (game.takeStrategyTurn({&CSAplayer1, &human_player})) {
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
        std::cout << "usage: (what: traines/es, traintd/td, esplay, tdplay) (model)" << std::endl;
        exit(1);
    }

    std::string what  = (argc >= 2) ? argv[1] : "";
    std::string model = (argc == 3) ? argv[2] : "";

    if (what.length() == 0) {
        std::cout << "what to run? Options are: traines (or es), traintd (or td), esplay, tdplay" << std::endl;
        getline(std::cin, what);
    }

    bool train_td;
    if (boost::iequals(what, "traines") || boost::iequals(what, "es")) {
        train_td = false;
    }
    else if (boost::iequals(what, "traintd") || boost::iequals(what, "td")) {
        train_td = true;
    }
    else if (boost::iequals(what, "esplay")) {
        train_td = false;
        playHexCSAVsHuman(model);
        return 0;
    }
    else if (boost::iequals(what, "tdplay")) {
        playHexTDVsHuman(model);
        return 0;
    } else {
        std::cout << "invalid input. Options are: traines (or es), traintd (or td), esplay, tdplay" << std::endl;
        return 1;
    }

    if (train_td) {
        std::cout << "Training model with TD algorithm." << std::endl;
        trainingLoop<ModelTrainerTD>("TDmodel");
    } else {
        std::cout << "Training model with CSA-ES algorithm." << std::endl;
        trainingLoop<ModelTrainerCSA>("CSAmodel");
    }

    return 0;
}
