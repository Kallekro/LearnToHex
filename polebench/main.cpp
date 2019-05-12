#include "SelfRLCMA.h"
#include "Strategies.hpp"
#include "ObjectiveFunctions.hpp"

#include <shark/Algorithms/DirectSearch/CMA.h>

#include <stdlib.h>
#include "unistd.h"

using namespace shark;


int main() {
    //shark::random::globalRng().seed(time(NULL));
	shark::random::globalRng().seed(123);//std::chrono::system_clock::now().time_since_epoch().count());
	// 4 features/variables
	std::size_t numvars = 4;

	NetworkStrategy strat;
    SinglePoleObjectiveFunctionCMA single_pole(numvars, strat);

#if 1
	////////////// TD-LAMBDA /////////////////	
	// algorithm parameters
	double rate = 0.01;
	double lambda = 0.6;
	double decayRate  = 0.5;

	// initialize algorithm variables
	RealVector initWeights(numvars);
	auto weights = subrange(initWeights, 0, numvars);
	auto solution = subrange(initWeights, 0, numvars);

	// stopping criteria
	int maxcount = 10000;

	// Value function ? 
	auto value=[&](double c, double m){
		return 1.0 - (c / m);
	};

	auto playBest=[&](){
		SinglePole singlePole = SinglePole(true);
		singlePole.init();

		strat.setParameters(solution);

		double c = 0;
		shark::RealVector v(singlePole.noVars(), 0.0);
		while (!singlePole.failure() && c < maxcount) {
			singlePole.getState(v);
			std::cout << v << std::endl;
			double output = strat.getMoveAction(v);
			singlePole.move(output);
			c++;
		}
		std::cout << "Reached count " << c << std::endl;
	};

	// loop for each training episode.
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

		if (t % 100 == 0) {
			std::cout << "t " << t << std::endl;
		}
		do { // for every timestep
			// set the policy 
			strat.setParameters(weights);

			// increment time-step counter
			count++;
			if (count > maxcount) {break; }

			singlePole.getState(state);
			// choose action based on policy
			double move = strat.getMoveAction(state);	
			// save current state
			RealVector oldState = state; 
			// take action (!)
			singlePole.move(abs(move));
			// observe new state 
			singlePole.getState(state);
			// get reward
			double reward = (singlePole.failure() ? 0 : 1) ;

			//if ( count > maxcount-10 ) {
			//	std::cout << "Trail " << t << std::endl;
			//	std::cout <<"Count " << count << "\n" << "Value " << (value(count, maxcount)) << std::endl;
			//	std::cout << "State " << state << "\n\n" << std::endl;
			//} 
			// update eligibility trace
			z = decayRate*lambda + oldState;
			// compute td-error
			double tdError = reward + decayRate*value(count, maxcount) - value((count-1), maxcount); 
			// update the weights
			solution = weights;
			weights = weights + rate * tdError * z;
		
			state = oldState;
		} while (!singlePole.failure());
		//std::cout << state << std::endl;
		//std::cout << count << std::endl;
		if (count > maxcount) {
			std::cout << "Running experiment" << std::endl;
			playBest();
			std::string playerin;
			getline(std::cin, playerin);
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








