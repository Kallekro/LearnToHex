#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "Hex.hpp"
#include "hex_algorithms.hpp"

using namespace shark;
using namespace Hex;

struct RandomGameStats {
    double wins_vs_random = 0;
    double games_vs_random_played = 0;
    std::deque<double> last_wins;

    double blue_winrate = 0;
    double blue_winrate_last_x_games = 0;
};

struct PreviousModelGameStats {
    double total_wins = 0;
    double games_played = 0;

    double new_model_winrate = 0;
};

/******************\
 *  Base Trainer  *
\******************/
template <class AlgorithmType, class StrategyType>
class ModelTrainer {
public:
    ModelTrainer(std::string randomStatsFilename, std::string previousModelStatsFilename) {
        boost::filesystem::path modelsdir("models/");
        boost::filesystem::create_directory(modelsdir);
        boost::filesystem::path logsdir("logs/");
        boost::filesystem::create_directory(logsdir);

        randomStatsTotalWinrateOutStream.open("logs/" + randomStatsFilename + "_totalWinrate.log");
        randomStatsCurrentWinrateOutStream.open("logs/" + randomStatsFilename + "_currentWinrate.log");
        previousModelStatsCurrentOutStream.open("logs/" + previousModelStatsFilename + "_currentWinrate.log");
        previousModelStatsTotalOutStream.open("logs/" + previousModelStatsFilename + "_totalWinrate.log");
    }
    ~ModelTrainer() {
        randomStatsTotalWinrateOutStream.close();
        randomStatsCurrentWinrateOutStream.close();
        previousModelStatsCurrentOutStream.close();
        previousModelStatsTotalOutStream.close();
    }

    std::string lastModel;

    virtual void playExampleGame() = 0;
    virtual void playAgainstRandom() = 0;
    virtual int playGameWithStrategies(std::vector<StrategyType*> const& strategies) = 0;
    virtual void printTrainingStatus() = 0;
    virtual void step() = 0;
    virtual void saveModel(std::string modelName) = 0;
    virtual void loadModel(std::string modelName) = 0;
    size_t NumberOfEpisodes() { return m_number_of_episodes; }
    AlgorithmType GetAlgorithm() { return m_algorithm; }

    struct RandomGameStats GetRandomPlayStats() { return m_randomGameStats; }

    void displayRandomPlayStats() {
        if (m_silent) { return; }
        std::cout << "Displaying stats from random games:" << std::endl;
        std::cout << "Blue winrate: " << m_randomGameStats.blue_winrate << std::endl;
        std::cout << "Blue winrate last " << m_randomGameStats.last_wins.size() << " games: " << m_randomGameStats.blue_winrate_last_x_games << std::endl;
    }

    void logRandomPlayStats() {
        randomStatsTotalWinrateOutStream << m_steps << " "
                                         << m_randomGameStats.blue_winrate << std::endl;
        randomStatsCurrentWinrateOutStream << m_steps << " "
                                           << m_randomGameStats.blue_winrate_last_x_games << std::endl;
    }

    void playAgainstModel(std::string model) {
        // initialize players with models
        StrategyType player1 = m_algorithm.GetStrategy();
        StrategyType player2;
        player2.loadStrategy("models/" + model);

        double total_games = 100;
        double new_model_wins = 0;
        for (int i=0; i < total_games; i++) {
            new_model_wins += playGameWithStrategies({&player1, &player2});
        }
        double winrate = new_model_wins / total_games;

        m_previousModelGameStats.games_played += total_games;
        m_previousModelGameStats.total_wins += new_model_wins;
        m_previousModelGameStats.new_model_winrate = m_previousModelGameStats.total_wins / m_previousModelGameStats.games_played;

        previousModelStatsCurrentOutStream << m_steps << " "
                                           << winrate << std::endl;
        previousModelStatsTotalOutStream << m_steps << " "
                                         << m_previousModelGameStats.new_model_winrate << std::endl;

        if (!m_silent) {
            std::cout << winrate << " newest model winrate in " << total_games << " games played against previous model." << std::endl;
        }

    }

    void RandomPlayersBaseline() {
        Game game = m_algorithm.GetGame();

        RandomStrategy rPlayer1;
        RandomStrategy rPlayer2;

        double total_games_played = 0;
        double player1_wins = 0;

        std::ofstream randomPlayersBaselineStatsOutstream("logs/randomPlayersBaseline.log");

        for (int i=0; i < 500; i++) {
            for (int i=0; i < 100; i ++) {
                game.reset();
                while (game.takeStrategyTurn({&rPlayer1, &rPlayer2})) {}
                if (game.getRank(0) == 0) {
                    player1_wins++;
                }
                total_games_played++;
            }
            double winrate = player1_wins / total_games_played;
            randomPlayersBaselineStatsOutstream << i << " " << winrate << std::endl;
        }
        randomPlayersBaselineStatsOutstream.close();
    }

protected:
    bool m_silent = false;

    AlgorithmType m_algorithm;
    size_t m_number_of_episodes;
    size_t m_steps = 0;

    struct RandomGameStats m_randomGameStats;
    struct PreviousModelGameStats m_previousModelGameStats;

	std::ofstream randomStatsTotalWinrateOutStream;
	std::ofstream randomStatsCurrentWinrateOutStream;
    std::ofstream previousModelStatsCurrentOutStream;
    std::ofstream previousModelStatsTotalOutStream;

    void updateRandomPlayStats(bool blue_lost) {
        m_randomGameStats.games_vs_random_played++;
        if (blue_lost) {
            m_randomGameStats.last_wins.push_back(0);
        } else {
            m_randomGameStats.wins_vs_random++;
            m_randomGameStats.last_wins.push_back(1);
        }
        if (m_randomGameStats.last_wins.size() > 100) {
            m_randomGameStats.last_wins.pop_front();
        }
        m_randomGameStats.blue_winrate = m_randomGameStats.wins_vs_random / m_randomGameStats.games_vs_random_played;

        double sum = 0;
        for (int i=0; i < m_randomGameStats.last_wins.size(); i++) {
            sum += m_randomGameStats.last_wins[i];
        }
        m_randomGameStats.blue_winrate_last_x_games = sum / m_randomGameStats.last_wins.size();
    }


};


/*********************\
 *  CSA-ES  Trainer  *
\*********************/
class ModelTrainerCSA : public ModelTrainer<CSAAlgorithm, CSANetworkStrategy> {
private:
    CSANetworkStrategy m_player2;
public:
    ModelTrainerCSA(std::string randomStatsFilename, std::string previousModelStatsFilename) : ModelTrainer(randomStatsFilename, previousModelStatsFilename) {
        m_number_of_episodes = 50000;
        m_player2.setColor(Red);
    }

    void playExampleGame() override {
        Game game = m_algorithm.GetGame();
        CSANetworkStrategy player1 = m_algorithm.GetStrategy();
        SelfRLCMA csa = m_algorithm.GetCSA();
        game.reset();
        player1.setParameters(csa.mean());
        m_player2.setParameters(csa.mean());
        if (!m_silent) { std::cout << game.asciiState() << std::endl; }
        while (game.takeStrategyTurn({&player1, &m_player2})) {
            if (!m_silent) { std::cout << game.asciiState() << std::endl; }
        }
        if (!m_silent) {
            std::cout << game.asciiState() << std::endl;
            std::cout << "End of example game." << std::endl;
        }
    }

    void playAgainstRandom() override {
        RandomStrategy random_player;
        Game game = m_algorithm.GetGame();
        CSANetworkStrategy player1 = m_algorithm.GetStrategy();
        SelfRLCMA csa = m_algorithm.GetCSA();

        game.reset();
        player1.setParameters(csa.mean());
        while (game.takeStrategyTurn({&player1, &random_player})) { }
        updateRandomPlayStats(game.getRank(Blue));
    }


    int playGameWithStrategies(std::vector<CSANetworkStrategy*> const& strategies) override {
        Game game;
        game.reset();
        CSANetworkStrategy* ESplayer1 = (CSANetworkStrategy*)strategies[0];
        CSANetworkStrategy* ESplayer2 = (CSANetworkStrategy*)strategies[1];
        while (game.takeStrategyTurn({ESplayer1, ESplayer2})) {}
        if (game.getRank(0) == 0) {
            return 1;
        } else {
            return 0;
        }
    }


    //void playAgainstModel(std::string model) override {
    //    // initialize players with models
    //    CSANetworkStrategy ESplayer1 = m_algorithm.GetStrategy();
    //    CSANetworkStrategy ESplayer2;
    //    ESplayer2.loadStrategy(model);
    //    Game game;
//
    //    double total_games = 100;
    //    double new_model_wins = 0;
    //    for (int i=0; i < total_games; i++) {
    //        game.reset();
    //        while (game.takeStrategyTurn({&ESplayer1, &ESplayer2})) { }
    //        if (game.getRank(0) == 0) { new_model_wins++; }
    //    }
    //    double winrate = total_games / new_model_wins;
    //    std::cout << winrate << " newest model winrate in " << total_games << " games played against previous model.";
    //}

    void printTrainingStatus() override {
        std::cout<<"Training games: " << m_steps << "\nSigma: " << m_algorithm.GetCSA().sigma() << std::endl;
        std::cout<< "Learn: " << m_algorithm.GetCSA().rate() << std::endl;
    }

    void step() override {
        m_algorithm.EpisodeStep(m_steps);
        m_steps++;
    }

    void saveModel(std::string modelName) override {
        m_algorithm.GetStrategy().setParameters(m_algorithm.GetCSA().mean());
        m_algorithm.GetStrategy().saveStrategy("models/" + modelName);
    }

    void loadModel(std::string modelName) override {
        m_algorithm.GetStrategy().loadStrategy("models/" + modelName);
    }
};


/******************\
 *  TD   Trainer  *
\******************/
class ModelTrainerTD : public ModelTrainer<TDAlgorithm, TDNetworkStrategy> {
public:
    ModelTrainerTD(std::string randomStatsFilename, std::string previousModelStatsFilename) : ModelTrainer(randomStatsFilename, previousModelStatsFilename) {
        m_number_of_episodes = 50000;
    }

    void playExampleGame() override {
        Game game = m_algorithm.GetGame();
        TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
        game.reset();
        if (!m_silent) { std::cout << game.asciiState() << std::endl; }
        bool won = false;
        unsigned state = 0;
        while (!won) {
            std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game, false);
            won = !game.takeTurn(chosen_move.second);
            if (!m_silent) { std::cout << game.asciiState() << std::endl; }
            state++;
        }
        if (!m_silent) {
           std::cout << "End of example game." << std::endl;
        }
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
        updateRandomPlayStats(game.getRank(Blue));
    }

    int playGameWithStrategies(std::vector<TDNetworkStrategy*> const& strategies) override {
        Game game;
        game.reset();
        bool won = false;
        TDNetworkStrategy* TDplayer1 = (TDNetworkStrategy*)strategies[0];
        TDNetworkStrategy* TDplayer2 = (TDNetworkStrategy*)strategies[1];
        while (!won) {
            std::pair<double, int> chosen_move;
            if (game.ActivePlayer() == Blue) {
                chosen_move = TDplayer1->getChosenMove(game, false);
            } else {
                chosen_move = TDplayer2->getChosenMove(game, false);
            }
            won = !game.takeTurn(chosen_move.second);
        }
        if (game.getRank(0) == 0) {
            return 1;
        } else {
            return 0;
        }
    }

    //void playAgainstModel(std::string model) override {
    //    // initialize players with models
    //    TDNetworkStrategy TDplayer1 = m_algorithm.GetStrategy();
    //    TDNetworkStrategy TDplayer2;
    //    TDplayer2.loadStrategy(model);
    //    Game game;
//
    //    double total_games = 100;
    //    double new_model_wins = 0;
    //    for (int i=0; i < total_games; i++) {
    //        bool won = false;
    //        game.reset();
    //        while (!won) {
    //            std::pair<double, int> chosen_move;
    //            if (game.ActivePlayer() == Blue) {
    //                chosen_move = TDplayer1.getChosenMove(game, false);
    //            } else {
    //                chosen_move = TDplayer2.getChosenMove(game, false);
    //            }
    //            won = !game.takeTurn(chosen_move.second);
    //        }
    //        if (game.getRank(0) == 0) { new_model_wins++; }
    //    }
    //    double winrate = new_model_wins / total_games;
    //    std::cout << winrate << " newest model winrate in " << total_games << " games played against previous model.";
    //}

    void printTrainingStatus() override {
        std::cout << "Step " << m_steps << std::endl;
    }

    void step() override {
        m_algorithm.EpisodeStep(m_steps);
        m_steps++;
    }

    void saveModel(std::string modelName) override {
        m_algorithm.GetStrategy().saveStrategy("models/" + modelName);
    }

    void loadModel(std::string modelName) override {
        m_algorithm.GetStrategy().loadStrategy("models/" + modelName);
    }
};


/*******************\
 *  Training Loop  *
\*******************/
template<class TrainerType>
void trainingLoop(std::string modelName) {
    std::string prefix = modelName + std::to_string(BOARD_SIZE) + "x" + std::to_string(BOARD_SIZE);
    TrainerType trainer(prefix + "randomStats", prefix + "previousModelStats");

    // Uncomment to create random players baseline
    //trainer.RandomPlayersBaseline();

    double highest_winrate = 0;
    for (int i=0; i < trainer.NumberOfEpisodes(); i++) {
        if (i % 1000 == 0 && i != 0) { // Play example game and save model
            trainer.playExampleGame();
            trainer.saveModel(modelName);
        }
        if (i % 100 == 0 && i != 0) { // Play against a random strategy and display stats
            for (int game_i = 0; game_i < 100; game_i++) {
                trainer.playAgainstRandom();
            }
            trainer.displayRandomPlayStats();
            trainer.logRandomPlayStats();

            double cur_model_winrate = trainer.GetRandomPlayStats().blue_winrate_last_x_games;
            if (cur_model_winrate > highest_winrate) {
                highest_winrate = cur_model_winrate;
                trainer.saveModel("highestWR");
            }
        }
        if (i % 100 == 0) { // Play against the previous model
            if (i != 0) {
                trainer.playAgainstModel(trainer.lastModel);
            }
            std::string modelName = prefix + "_autosave";
            trainer.saveModel(modelName);
            trainer.lastModel = modelName + ".model";
        }
        if (i % 100 == 0 ) {
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

void playHexTDVsHuman(std::string model, bool for_python) {
    HumanStrategy human_player(for_python);
    TDNetworkStrategy TDplayer1;
    if (model.length()) {
        TDplayer1.loadStrategy(model);
    }
    Game game;
    game.reset();
    if (for_python) {
        initializePythonSettings();
        std::cout << game.asciiStatePython() << std::endl;
    } else {
        std::cout << game.asciiState() << std::endl;
    }

    bool won = false;
    while (!won) {
        if (game.ActivePlayer() == Blue) {
            std::pair<double, int> chosen_move = TDplayer1.getChosenMove(game, false);
            won = !game.takeTurn(chosen_move.second);
        } else {
            won = !game.takeStrategyTurn({NULL, &human_player});
        }
        std::cout << (for_python ? game.asciiStatePython() : game.asciiState()) << std::endl;

    }
    if (for_python) {
        std::cout << game.asciiStatePython() << std::endl;
        std::cout << "__GAME_OVER__ " << (game.getRank(0) == 0 ? 0 : 1) << std::endl;
    } else {
        std::cout << game.asciiState() << std::endl;
        std::cout << "Game over. Player " << (game.getRank(0) == 0 ? 1 : 2) << " won!" << std::endl;
    }
}

void playHexCSAVsHuman(std::string model, bool for_python) {
    HumanStrategy human_player(for_python);
    CSANetworkStrategy CSAplayer1;
    if (model.length()) {
        CSAplayer1.loadStrategy(model);
    }
    Game game;
    game.reset();
    if (for_python) {
        initializePythonSettings();
        std::cout << game.asciiStatePython() << std::endl;
    } else {
        std::cout << game.asciiState() << std::endl;
    }

    while (game.takeStrategyTurn({&CSAplayer1, &human_player})) {
        std::cout << (for_python ? game.asciiStatePython() : game.asciiState()) << std::endl;
    }

    if (for_python) {
        std::cout << game.asciiStatePython() << std::endl;
        std::cout << "__GAME_OVER__ " << (game.getRank(0) ? 0 : 1) << std::endl;
    } else {
        std::cout << game.asciiState() << std::endl;
        std::cout << "Game over. Player " << (game.getRank(0) == 0 ? 1 : 2) << " won!" << std::endl;
    }

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
        playHexCSAVsHuman(model, false);
        return 0;
    }
    else if (boost::iequals(what, "espython")) {
        playHexCSAVsHuman(model, true);
        return 0;
    }
    else if (boost::iequals(what, "tdplay")) {
        playHexTDVsHuman(model, false);
        return 0;
    }
    else if (boost::iequals(what, "tdpython")) {
        playHexTDVsHuman(model, true);
        return 0;
    }
    else {
        std::cout << "invalid input. Options are: traines (or es), traintd (or td), esplay, tdplay" << std::endl;
        return 1;
    }

    if (model.length() > 0) {
        model += "_";
    }

    if (train_td) {
        std::cout << "Training model with TD algorithm." << std::endl;
        trainingLoop<ModelTrainerTD>(model + "TDmodel");
    } else {
        std::cout << "Training model with CSA-ES algorithm." << std::endl;
        trainingLoop<ModelTrainerCSA>(model + "CSAmodel");
    }

    return 0;
}
