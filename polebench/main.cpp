
#include "SelfRLCMA.h"
#include <shark/ObjectiveFunctions/Benchmarks/PoleSimulators/SinglePole.h>

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConvolutionalModel.h>//single convolutional layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>
#include <shark/Algorithms/DirectSearch/CMA.h>

#include <stdlib.h>
#include "unistd.h"

using namespace shark;


//typedef boost::shared_ptr<SingleObjectiveFunction> Function;


class Strategy{
    public:
        virtual double getMoveAction(RealVector &v) = 0;
        virtual std::size_t numParameters() const = 0;
        virtual void setParameters(shark::RealVector const& parameters) = 0;
};

class NetworkStrategy: public Strategy{
private:
	LinearModel<RealVector, TanhNeuron> m_inLayer;
	LinearModel<RealVector, TanhNeuron> m_outLayer;
	LinearModel<RealVector, TanhNeuron> m_moveNet;

    unsigned m_color;

public:
	NetworkStrategy(){
		m_inLayer.setStructure({4, 1, 1}, 20);
		m_outLayer.setStructure(m_inLayer.outputShape(), 1);

		m_moveNet.setStructure(m_inLayer.inputShape(), 1); 
	}

	double getMoveAction(RealVector &inputs) override{
		RealVector response = m_moveNet(inputs);
		double result = response[0];
		return result;
	}

	std::size_t numParameters() const override{
		return m_moveNet.numberOfParameters();
	}

	void setParameters(shark::RealVector const& parameters) override{
		auto p1 = subrange(parameters, 0, m_moveNet.numberOfParameters());
		m_moveNet.setParameterVector(p1);
	}

};

class SinglePoleTest : public SingleObjectiveFunction {
public:
	SinglePoleTest(std::size_t numberOfVariables)
    : m_numberOfVariables(numberOfVariables) {
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
		// uncomment below when RL CMA
		//SIZE_CHECK(x.size() == 2*numberOfVariables());
		SIZE_CHECK(x.size() == numberOfVariables());
		
		m_evaluationCounter++;

		thread_local NetworkStrategy strat1;
		thread_local SinglePole singlePole1 = SinglePole(true);
		singlePole1.init();

		auto x0 = subrange(x, 0, numberOfVariables());
		strat1.setParameters(x0);
		double maxcount = 100000.0;
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
		return 1 - (count1/maxcount) ;

		//thread_local NetworkStrategy strat2;
		//thread_local SinglePole singlePole2 = SinglePole(true);
		//singlePole2.init();
		//auto x1 = subrange(x, numberOfVariables(), x.size());
		//strat2.setParameters(x1);
		//while(!singlePole2.failure() && count2 < maxcount) {
			//count2++;
			//singlePole2.getState(v);
			//double out = strat2.getMoveAction(v);
			//singlePole2.move(out);
		//}
		//if (count1 < count2 ) {
			//return 1;
		//} else {
			//return 0;
		//}

		////std::cout << count1 << " --vs-- " << count2 << std::endl;

		//return 1 - count1/maxcount; 
    }
private:
	std::size_t m_numberOfVariables;
};

int main() {
    shark::random::globalRng().seed(time(NULL));
    //SinglePoleTest single_pole(numvars);

	NetworkStrategy strat;
#if 1
	std::size_t numvars = 4;
	////////////// TD-LAMBDA /////////////////	
	// algorithm parameters
	double rate = 0.01;
	double lambda = 0.5;
	double decayRate  = 0.5;

	// initialize algorithm variables
	RealVector initWeights(numvars);
	auto weights = subrange(initWeights, 0, numvars);
	auto solution = subrange(initWeights, 0, numvars);

	// stopping criteria
	int maxcount = 10000;
	int max = 0;

	for (int t=0; t<5000; t++) {
		// initialize simulator
		SinglePole singlePole = SinglePole(true);
		singlePole.init();
		// eligibility trace
		RealVector z(numvars);
		// stopping criteria
		int count = 0;

		// holding the current state
		remora::vector<double> state(singlePole.noVars(), 0.0);

		auto value=[&](double c, double m){
			return 1.0 - (c / m);
		};

		do { // for every timestep
			// set the policy 
			strat.setParameters(weights);
			count++;
			if (count > max) {max=count;}			
			if (count > maxcount) {break; }

			singlePole.getState(state);
			// choose action based on policy
			double move = strat.getMoveAction(state);	
			// take action (!)
			singlePole.move(abs(move));
			// save previous state
			RealVector oldState = state; 
			// observe new state 
			singlePole.getState(state);
			// get reward
			double reward = (singlePole.failure() ? 0 : 1) ;

			if (value(count, maxcount) < 0.00001) {
				std::cout << "Trail " << t << std::endl;
				std::cout <<"Count " << count << "\n" << "Value " << (value(count, maxcount)) << std::endl;
				std::cout << "State " << state << "\n\n" << std::endl;
				
			}
			// update eligibility trace
			z = decayRate*lambda + oldState;
			// compute td-error
			double tdError = reward + decayRate*value(count, maxcount) - value((count-1), maxcount); 
			// update the weights
			solution = weights;
			weights = weights + rate * tdError * z;
		
			state = oldState;
		} while (!singlePole.failure());

		auto playBest=[&](){
        	SinglePole singlePole = SinglePole(true);
			singlePole.init();

			strat.setParameters(solution);

			double count = 0;
			double maxcount = 10000.0;
			shark::RealVector v(singlePole.noVars(), 0.0);
			while (!singlePole.failure() && count <= maxcount) {
				singlePole.getState(v);
				double output = strat.getMoveAction(v);
				singlePole.move(output);
				count++;
			}
			std::cout << "Reached count " << count << std::endl;
		};
		if (count > maxcount) {
			std::cout << "Running experiment" << std::endl;
			playBest();
		}	
	}
}
#elif 0
    //SelfRLCMA cma;
	CMA cma;
    cma.init(single_pole, single_pole.proposeStartingPoint());


	auto runExperiment=[&](){
        SinglePole singlePole = SinglePole(true);
		singlePole.init();

		strat.setParameters(cma.solution().point);

		double count = 0;
		double maxcount = 100000.0;
		shark::RealVector v(singlePole.noVars(), 0.0);
		while (!singlePole.failure() && count <= maxcount) {
			singlePole.getState(v);
			std::cout << v << std::endl;
			double output = strat.getMoveAction(v);
			singlePole.move(output);
			count++;
		}
		std::cout << "Reached count " << count << std::endl;
	};

    for (std::size_t t = 0; t != 50000; ++t) {
        cma.step(single_pole);
		//runExperiment();
		if (t % 1 == 0) {
			runExperiment();
			std::cout << "t " << t << std::endl;
			std::cout << "sigma " << cma.sigma() << std::endl;
			std::cout << "value " << cma.solution().value << std::endl;
		}

    }
}
#endif








