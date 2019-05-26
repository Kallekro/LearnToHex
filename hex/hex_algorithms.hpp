#ifndef HEXMLALGORITHMS_H
#define HEXMLALGORITHMS_H

#include "SelfRLCMA2.h"
#include "Hex.hpp"
#include "hex_strategies.hpp"

using namespace shark;

namespace Hex {

/********************\
 *  Base Algorithm  *
\********************/
template <class StrategyType>
class HexMLAlgorithm {
protected:
    Game m_game;
    StrategyType m_strategy;
public:
    HexMLAlgorithm() {}

    Game GetGame() { return m_game; }
    StrategyType GetStrategy() { return m_strategy; }
    virtual void EpisodeStep(unsigned episode) = 0;
};


/*******************\
 *  TD Algorithm   *
\*******************/
class TDAlgorithm : public HexMLAlgorithm<TDNetworkStrategy> {
private:
    RealVector m_weights;
    double learning_rate = 0.1;
public:
    TDAlgorithm() : HexMLAlgorithm() {
        m_weights = blas::normal(random::globalRng(), m_strategy.numParameters(), 0.0, 1.0/m_strategy.numParameters(), blas::cpu_tag());
    }

    Game GetGame() { return m_game; }
    TDNetworkStrategy GetStrategy() { return m_strategy; }

    void EpisodeStep(unsigned episode) override {
        m_game.reset();
        bool won = false;
        m_strategy.setParameters(m_weights);
        // for computing derivatives
        std::vector<RealVector> states;
        RealVector rewards;
        RealVector values;
        RealVector nextValues;

        int step_i = 0;

        // save state
        RealVector input(2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE), 0.0);
        //m_strategy.createInput(game.getGameBoard(), m_game.ActivePlayer(), input);
        //states.push_back(input);
        //values.push_back(m_strategy.evaluateNetwork(input));
        //rewards.push_back(0);

        while (!won) {
            //std::cout << step_i << ", ";
            blas::matrix<Hex::Tile> field = m_game.getGameBoard();
            unsigned int activePlayer = m_game.ActivePlayer();

            // save state S
            m_strategy.createInput(m_game.getGameBoard(), m_game.ActivePlayer(), input);
            states.push_back(input);
            // save value of state S
            values.push_back(m_strategy.evaluateNetwork(input));

            std::pair<double, int> chosen_move = m_strategy.getChosenMove(m_game.getFeasibleMoves(), m_game.getGameBoard(), activePlayer);
            // take action
            try {
                if (chosen_move.second < 0 || chosen_move.second >= Hex::BOARD_SIZE*Hex::BOARD_SIZE) {
                    std::cout << "Chosen move for player 1 " << chosen_move.second << " out of range." << std::endl;
                    //for (int i=0; i < move_values.size(); i++) {
                    //    std::cout << move_values[i].first;
                    //}
                    std::cout << std::endl;
                    exit(1);
                } else {
                    // FLIP
                    //if (activePlayer == Hex::Red) {
                    //    int x = chosen_move.second % Hex::BOARD_SIZE;
                    //    int y = chosen_move.second / Hex::BOARD_SIZE;
                    //    chosen_move.second = y*Hex::BOARD_SIZE + x;
                    //}
                    won = !m_game.takeTurn(chosen_move.second);

                    if (episode % 1000 == 0) {
                        std::cout << m_game.asciiState() << std::endl;
                    }
                }
            } catch (std::invalid_argument& e) {
                std::cout << std::endl;
                throw(e);
            }

            // opponent turn
            //if(!won ) {
            //    rewards.push_back(0);
            //    activePlayer = m_game.ActivePlayer();
            //    RealVector feasible_moves_opponent = m_game.getFeasibleMoves();
            //    blas::matrix<Hex::Tile> fieldCopy_opponent = getFieldCopy(m_game.getGameBoard());
            //    std::vector<std::pair<double, int>> move_values_opponent = getMoveValues(fieldCopy_opponent, activePlayer, m_strategy, feasible_moves_opponent);
            //    std::pair<double, int> chosen_move_opponent = chooseMove(move_values_opponent, activePlayer, feasible_moves_opponent);
            //    // take action
            //    try {
            //        if (chosen_move_opponent.second < 0 || chosen_move_opponent.second >= Hex::BOARD_SIZE*Hex::BOARD_SIZE) {
            //            std::cout << "Chosen move for player 2 " << chosen_move_opponent.second << " out of range." << std::endl;
            //            for (int i=0; i < move_values_opponent.size(); i++) {
            //                std::cout << move_values_opponent[i].first;
            //            }
            //            std::cout << std::endl;
            //            exit(1);
            //        } else {
            //            won = !m_game.takeTurn(chosen_move_opponent.second);
            //        }
//
            //    } catch (std::invalid_argument& e) {
            //        std::cout << std::endl;
            //        throw(e);
            //    }
//
            //} else {
            //    //rewards.push_back(1);
            //    if (m_game.ActivePlayer() == Hex::Blue) {
            //        rewards.push_back(1);
            //    } else {
            //        rewards.push_back(0);
            //    }
            //}
            // save next value
            if(!won ) {
                rewards.push_back(0);
            } else {
                if (m_game.ActivePlayer() == Hex::Blue) {
                    rewards.push_back(1);
                } else {
                    rewards.push_back(0);
                }
            }
            m_strategy.createInput(m_game.getGameBoard(), m_game.ActivePlayer(), input);
            nextValues.push_back(m_strategy.evaluateNetwork(input));

            step_i++;
        }
        //nextValues.push_back(rewards[step_i]);

        RealVector statePoint(2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE));
        RealVector valuePoint(1);
        // batch of states
        Batch<RealVector>::type stateBatch = Batch<RealVector>::createBatch(statePoint, states.size());
        // batch of values/outputs/predictions
        Batch<RealVector>::type valueBatch = Batch<RealVector>::createBatch(valuePoint, states.size());

        //std::cout << "ASSERT " << (values.size() == states.size())  << std::endl;
        //std::cout << values.size() << ", " << states.size() << std::endl;
        for (int i=0; i < states.size(); i++) {
            if (states[i].size() != 2*(Hex::BOARD_SIZE*Hex::BOARD_SIZE)) {
                std::cout << "state " << i << std::endl;
            }
            getBatchElement(stateBatch, i) = states[i];
            getBatchElement(valueBatch, i)(0) = values(i);
        }


        //std::cout << "R: " << rewards.size() << " V: " << values.size() << " NV: " << nextValues.size() << std::endl;
//
        //std::cout << "SIZE statebatch: " <<  batchSize(stateBatch) << std::endl;
       // std::cout << "SIZE coeff size2: " <<  m_strategy.GetMoveModel().outputShape().numElements() << std::endl;
//
        //std::cout << "Rewards " << rewards << std::endl;
        //std::cout << "Values " << values << std::endl;
        //std::cout << "Next values " << nextValues << std::endl;

        // computes td-errors
        RealMatrix coeffs(states.size(), m_strategy.GetMoveModel().outputShape().numElements()); // b td_main.cpp:286
        column(coeffs, 0) = rewards + nextValues - values;
        //RealVector coeffVec = rewards + nextValues - values; // TODO: values[::-1]
        //column(coeffs, 0) = coeffVec;
        //for (int i=0; i<step_i; i++) {
        //    coeffs(i, 0) =  rewards(i) + nextValues(i) - values(i);
        //}



        //std::cout << "Coeffs " << coeffs << std::endl;

        // batch of
        boost::shared_ptr<State> state = m_strategy.createState();
        m_strategy.GetMoveModel().eval(stateBatch, valueBatch, *state); // compiles

        RealVector derivative;
        m_strategy.GetMoveModel().weightedParameterDerivative(stateBatch, valueBatch, coeffs, *state, derivative); // b td_main.cpp:271
        //std::cout << "States: " << stateMat << std::endl;
        //std::cout << "Rewards: " << rewards << std::endl;
        //std::cout << "Values: " << values << std::endl;
        //std::cout << "NValues: " << nextValues << std::endl;
        //std::cout << "Derivatives: " << derivative << std::endl;
        //std::cout << m_strategy.GetMoveModel().numberOfParameters() << std::endl;
        //std::cout << "WWW  " << m_weights << std::endl;
        //if (m_game.ActivePlayer() == Hex::Blue) {
            m_weights -= learning_rate*derivative;
        //}
    }
};

/***********************\
 *  SelfPlayTwoPlayer  *
\***********************/
template<class Game, class Strategy>
class SelfPlayTwoPlayer : public SingleObjectiveFunction {
private:
	Game m_game;
	Strategy m_baseStrategy;
public:
	SelfPlayTwoPlayer(Game const& game, Strategy const& strategy)
	: m_game(game), m_baseStrategy(strategy){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_NOISY;
	}

    Game getGame() const {
        return m_game;
    }

	std::size_t numberOfVariables()const{
		return m_baseStrategy.numParameters();
	}

	SearchPointType proposeStartingPoint() const {
		return blas::normal(random::globalRng(), numberOfVariables(), 0.0, 1.0/numberOfVariables(), shark::blas::cpu_tag());
	}

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == 2*numberOfVariables());
		m_evaluationCounter++;

		auto x0 = subrange(x,0, numberOfVariables());
		auto x1 = subrange(x, numberOfVariables(), x.size());

		thread_local Strategy strategy0;
		thread_local Strategy strategy1;
		thread_local Game game = m_game;

		strategy0.setParameters(x0);
		strategy1.setParameters(x1);

		//simulate
		game.reset();
		double logW = 0.0;
        double max = BOARD_SIZE*BOARD_SIZE;
		double count = 0;
        while(game.takeStrategyTurn({&strategy0, &strategy1})){
            count++;
			logW += game.logImportanceWeight({&strategy0, &strategy1});
        }

		double r = game.getRank(1);
		double y = 2*r - 1;
		double rr = 1/(1+std::exp(y*logW));
        return r;
	}
};

/*******************\
 *  CMA Algorithm  *
\*******************/
class CMAAlgorithm : public HexMLAlgorithm<CMANetworkStrategy> {
private:
    SelfPlayTwoPlayer<Game, CMANetworkStrategy> m_objective;
    SelfRLCMA m_cma;
public:
    CMAAlgorithm() : HexMLAlgorithm(), m_objective(m_game, m_strategy) {
        m_strategy.setColor(Blue);

        std::size_t d = m_objective.numberOfVariables();
        std::size_t lambda = SelfRLCMA::suggestLambda(d);

        m_cma.init(m_objective, m_objective.proposeStartingPoint(), lambda, 1.0);
    }

    void EpisodeStep(unsigned episode) {
		m_cma.step(m_objective);
    }

    SelfRLCMA GetCMA() {
        return m_cma;
    }
};
}
#endif