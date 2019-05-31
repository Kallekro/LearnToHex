#ifndef HEXMLALGORITHMS_H
#define HEXMLALGORITHMS_H

#include "SelfRLCMA.h"
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
    double m_learning_rate = 0.1;
    double m_lambda = 0.7;
public:
    TDAlgorithm() : HexMLAlgorithm() {
        m_weights = blas::normal(random::globalRng(), m_strategy.numParameters(), 0.0, 1.0/m_strategy.numParameters(), blas::cpu_tag());
    }

    Game GetGame() { return m_game; }
    TDNetworkStrategy GetStrategy() { return m_strategy; }

    // Take one step in the algorithm (run episode/game and calculate new weights)
    void EpisodeStep(unsigned episode) override {
        m_game.reset();
        bool won = false;
        m_strategy.setParameters(m_weights);

        // save states, values and rewards for computing derivatives
        std::vector<RealVector> states;
        RealVector rewards;
        RealVector values;
        RealVector nextValues;

        // game turns elapsed
        int step_i = 0;

        // variable for storing input to the neural network
        RealVector input((Hex::BOARD_SIZE*Hex::BOARD_SIZE), 0.0);

        // Play game and record states, values and rewards
        while (!won) {
            unsigned playerWithTurn = m_game.ActivePlayer();

            // choose an action
            std::pair<double, int> chosen_move = m_strategy.getChosenMove(m_game, true);

            // take action
            try {
                if (chosen_move.second < 0 || chosen_move.second >= Hex::BOARD_SIZE*Hex::BOARD_SIZE) {
                    std::cout << "Chosen move for player 1 " << chosen_move.second << " out of range." << std::endl;
                    std::cout << std::endl;
                    exit(1);
                } else {
                    // create input
                    blas::matrix<Tile> fieldCopy;
                    if (playerWithTurn == Red) {
                        fieldCopy = m_strategy.rotateField(m_game.getGameBoard(), false);
                    } else {
                        fieldCopy = m_game.getGameBoard();
                    }
                    m_strategy.createInput(fieldCopy, playerWithTurn, input);
                    // push state, encoded like the state used in the neural network
                    states.push_back(input);
                    // push value
                    values.push_back(chosen_move.first);

                    won = !m_game.takeTurn(chosen_move.second);

                    // push 1 as reward if game is over, else 0
                    if (!won) {
                        rewards.push_back(0.0);
                    } else {
                        rewards.push_back(1.0);
                    }
                    // push previous value as states' next value
                    if (step_i > 0) {
                        nextValues.push_back(1 - values[step_i]);
                    }
                }
            } catch (std::invalid_argument& e) {
                std::cout << std::endl;
                throw(e);
            }
            step_i++;
        }
        // push last "next" value (for t+1)
        nextValues.push_back(1.0);

        // Now we calculate the derivative of the model after evaluating the batched states
        RealVector statePoint((Hex::BOARD_SIZE*Hex::BOARD_SIZE));
        RealVector valuePoint(1);
        // batch of states
        Batch<RealVector>::type stateBatch = Batch<RealVector>::createBatch(statePoint, states.size());
        // batch of values/outputs/predictions
        Batch<RealVector>::type valueBatch = Batch<RealVector>::createBatch(valuePoint, states.size());

        for (int i=0; i < states.size(); i++) {
            getBatchElement(stateBatch, i) = states[i];
            getBatchElement(valueBatch, i)(0) = nextValues(i);
        }

        std::cout << "R: " << rewards << " V: " << values << " NV: " << nextValues << std::endl;

        // eligibility trace (not used right now)
        RealVector eTrace(step_i, 0.0);
        for (int k=1 ; k < step_i; k++) {
            eTrace(k) = pow(m_lambda, step_i - k);
        }

        // computes td-errors
        RealMatrix coeffs(states.size(), m_strategy.GetMoveModel().outputShape().numElements());
        column(coeffs, 0) = (rewards + nextValues - values); //* eTrace;

        boost::shared_ptr<State> state = m_strategy.createState();
        m_strategy.GetMoveModel().eval(stateBatch, valueBatch, *state);

        RealVector derivative;
        m_strategy.GetMoveModel().weightedParameterDerivative(stateBatch, valueBatch, coeffs, *state, derivative);

        // update weights
        m_weights -= m_learning_rate*derivative;
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
        double max = BOARD_SIZE*BOARD_SIZE;
		double count = 0;
        while(game.takeStrategyTurn({&strategy0, &strategy1})){
            count++;
        }

		double r = game.getRank(1);
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