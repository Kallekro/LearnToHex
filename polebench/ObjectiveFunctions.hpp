#ifndef OBJECTIVE_FUNS_HPP
#define OBJECTIVE_FUNS_HPP

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/SinglePole.h>
#include "Strategies.hpp"

using namespace shark;

//////////    Shark objective functions    //////////

/////   CMA Polebalancing ObjectiveFunction    /////
class SinglePoleObjectiveFunctionCMA : public SingleObjectiveFunction {
public:
	SinglePoleObjectiveFunctionCMA(std::size_t numberOfVariables)
    : m_numberOfVariables(numberOfVariables)  {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_NOISY;
	}
	std::string name() const
	{ return "SinglePoleTest"; }
	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	bool hasScalableDimensionality()const{
		return true;
	}
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}
	SearchPointType proposeStartingPoint() const {
		return blas::normal(random::globalRng(), numberOfVariables(), 0.0, 1.0/numberOfVariables(), shark::blas::cpu_tag());
	}
	double eval(SearchPointType const&  x) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;

		NetworkStrategy strat1;
		SinglePole singlePole1 = SinglePole(true);
		singlePole1.init();

		auto x0 = subrange(x, 0, numberOfVariables());
		strat1.setParameters(x0);
		double maxcount = 200.0;
		double count1 = 0;
		double count2 = 0;
		shark::RealVector v(numberOfVariables(), 0.0);
		std::vector<double> angles;
		while(!singlePole1.failure() && count1 < maxcount) {
			count1++;
			singlePole1.getState(v);
			double out = strat1.getMoveAction(v);
			singlePole1.move(abs(out));
		}
		return -count1; 
    }
private:
	std::size_t m_numberOfVariables;
};

//////////    Custom objective functions   //////////

/////   TDL Objective function    /////


template<class SinglePole, class Strategy>
class SinglePoleObjectiveFunctionTDL {
public:
    SinglePoleObjectiveFunctionTDL (SinglePole const& singlePole, Strategy const& strat) {
        m_baseStrategy = strat;
        m_singlePole = singlePole;
    }
	std::string name() const
	{ return "SinglePoleTest"; }
	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	bool hasScalableDimensionality()const{
		return true;
	}
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = m_baseStrategy.numParameters();
	}
    void init() {
    }

    double eval() {
        // What to eval here? 
    }

private:
    std::size_t m_numberOfVariables;
    NetworkStrategy m_baseStrategy;
    SinglePole m_singlePole;
};
#endif