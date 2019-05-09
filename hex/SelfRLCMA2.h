#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_SelfRLCMA_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_SelfRLCMA_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Core/Threading/Algorithms.h>
#include <boost/math/distributions/chi_squared.hpp>
#include <shark/Algorithms/DirectSearch/LMCMA.h>
namespace shark {
	
	
	

class SelfRLCMA : public AbstractSingleObjectiveOptimizer<RealVector >{
public:

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "sRLCMA"; }
	
	/// \brief Calculates lambda for the supplied dimensionality n.
	static std::size_t suggestLambda( std::size_t dimension ) {
		std::size_t lambda = std::size_t( 4. + ::floor( 3 *::log( static_cast<double>( dimension ) ) ) );
		return lambda + lambda % 2;
	}

	void read( InArchive & archive ){}
	void write( OutArchive & archive ) const{}

	using AbstractSingleObjectiveOptimizer<RealVector >::init;
	/// \brief Initializes the algorithm for the supplied objective function.
	void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		SIZE_CHECK(p.size() == function.numberOfVariables());
		checkFeatures(function);
		std::size_t lambda = SelfRLCMA::suggestLambda( p.size() );
		init(
			function,
			p,
			lambda,
			3.0/std::sqrt(double(p.size()))
		);
	}

	/// \brief Initializes the algorithm for the supplied objective function.
	void init( 
		ObjectiveFunctionType const& function, 
		SearchPointType const& initialSearchPoint,
		std::size_t lambda,
		double initialSigma
	){
		m_numberOfVariables = initialSearchPoint.size();
		m_lambda = lambda;
		
		m_firstIter = true;
		
		//variables for mean
		m_mean = blas::repeat(0.0, m_numberOfVariables);
		
		//variables for step size
		m_path = blas::repeat(0.0, m_numberOfVariables);
		m_gammaPath = 0.0;
		m_sigma = initialSigma;
		
		//variables for noise estimation
		m_ztest = 10;
		m_fvar = 10.0;
		m_sigmanoise = 1.0;
		m_rate = 1.0;
		m_muEff = 0.0;
		
		//pick starting point as best point in the set
		m_mean = initialSearchPoint;
		
		//initialize offspring array
		m_offspring.resize(m_lambda);
		for( std::size_t i = 0; i < m_offspring.size(); i++ ) {
			m_offspring[i].chromosome()  = blas::repeat(0.0, m_numberOfVariables);
			m_offspring[i].searchPoint()  = blas::repeat(0.0, m_numberOfVariables);
		}
	}

	/// \brief Executes one iteration of the algorithm.
	void step(ObjectiveFunctionType const& function){
		std::vector<IndividualType> offspring = generateOffspring();
		
		struct Eval{
			std::size_t first;
			std::size_t second;
			std::size_t third;
			double result1;
			double result2;
		};
		auto evalHelper=[&](Eval eval){
			auto& individual1 = offspring[eval.first];
			auto& individual2 = offspring[eval.second];
			auto& individual3 = offspring[eval.third];
			eval.result1 = function(individual1.searchPoint() | individual2.searchPoint());
			eval.result2 = function(individual1.searchPoint() | individual3.searchPoint());
			return eval;
		};
		
		std::vector<Eval> evaluations(offspring.size());

		//pick two partners for each element
		for(std::size_t i = 0; i != offspring.size(); ++i){
			
			std::size_t second = (i+m_lambda/2) % offspring.size();
			if(second < m_lambda/2)
				second = m_lambda/2 - 1 - second;
			
			std::size_t third = (i+m_lambda/2 + m_lambda/4) % offspring.size();
			if(third < m_lambda/2)
				third = m_lambda/2 - 1 - third;
			evaluations[i] = {i, second, third, 0.0, 0.0};
		}
		
		//evaluate and accumulate results
		RealVector f1(offspring.size(),0.0);
		RealVector f2(offspring.size(),0.0);
		threading::mapApply(
			evaluations,
			evalHelper, 
			[&](Eval eval){
				f1(eval.first) += eval.result1/2.0;
				f1(eval.second) += (1-eval.result1)/2.0;
				f2(eval.first) += eval.result2/2.0;
				f2(eval.third) += (1-eval.result2)/2.0;
			},
			threading::globalThreadPool()
		);


		//calculate average loss and variance under re-evaluation
		double var_noise = 0.0;
		double fmean = 0.0;
		double fvar = 0.0;
		for(std::size_t i = 0; i != offspring.size(); ++i){
			auto& individual= offspring[i];
			individual.unpenalizedFitness() = (f1(i)+f2(i))/2;
			individual.penalizedFitness() = individual.unpenalizedFitness();
			var_noise += sqr(f1(i)-f2(i)/2);
			fmean += individual.unpenalizedFitness();
		}
		var_noise /= offspring.size();
		fmean /= offspring.size();
		for(auto& individual: offspring){
			fvar += sqr(individual.unpenalizedFitness() - fmean);
		}
		fvar /= offspring.size() - 1;
		
		double cztest = std::pow(m_rate,1.5) / 100;
		
		//update noise statistics
		m_fvar = (1-cztest)*m_fvar + cztest * fvar;
		m_sigmanoise = (1-cztest)*m_sigmanoise + cztest * var_noise;
		m_ztest = 0.5*m_fvar/m_sigmanoise + 0.5;
		m_rate = 1.0/(1.0+1.0/(m_ztest-1.0));
		
		updatePopulation(offspring);
	}

	
	double sigma() const {
		return m_sigma;
	}
	
	double rate() const {
		return m_rate;
	}

	RealVector generatePolicy()const{
		return m_mean + remora::normal(random::globalRng(), m_numberOfVariables, 0.0, sqr(m_sigma), remora::cpu_tag());
	}
	
	RealVector const& mean()const{
		return m_mean;
	}
protected:
	/// \brief The type of individual used for the CMA
	typedef Individual<RealVector, double, RealVector> IndividualType;

	/// \brief Samples lambda individuals from the search distribution	
	std::vector<IndividualType> & generateOffspring( ) const{
		auto sampler = [&](std::size_t i){
			RealVector& z = m_offspring[i].chromosome();
			RealVector& x = m_offspring[i].searchPoint();
			noalias(z) = remora::normal(random::globalRng(), m_numberOfVariables, 0.0, 1.0, remora::cpu_tag());
			noalias(x) = m_mean + m_sigma * z;
		};
		
		threading::parallelND(m_offspring.size(), 0, sampler,threading::globalThreadPool());
		
		return m_offspring;
	}

	/// \brief Updates the strategy parameters based on the supplied offspring population.
	void updatePopulation( std::vector<IndividualType > const& offspring){
		//compute the weights
		RealVector weights(m_lambda, 0.0);
		for (std::size_t i = 0; i < m_lambda; i++){
			weights(i) = -offspring[i].penalizedFitness();
		}
		weights -=min(weights);
		weights /= norm_1(weights);

		double cPath = 2*(m_muEff + 2.)/(m_numberOfVariables + m_muEff + 5.) * m_rate;
		double dPath = 2 * m_rate * cPath/std::sqrt(cPath * (2-cPath));
		double cmuEff = 0.01;
		//first iteration: initialize all paths with true data from the function
		if(m_firstIter){
			m_firstIter = false;
			cmuEff = 1.0;
			m_muEff = 1.0 / sum(sqr(weights));
		}

		//gradient of mean
		RealVector dMean( m_numberOfVariables, 0. );
		RealVector stepZ( m_numberOfVariables, 0. );
		for (std::size_t i = 0; i < m_lambda; i++){
			noalias(dMean) += (weights(i) - 1.0/m_lambda) * offspring[i].searchPoint();
			noalias(stepZ) += weights(i) * offspring[i].chromosome();
		}

		noalias(m_path)= (1-cPath) * m_path + std::sqrt(cPath * (2-cPath) * m_muEff) * stepZ;
		m_gammaPath = sqr(1-cPath) * m_gammaPath+ cPath * (2-cPath);
		double deviationStepLen = norm_2(m_path)/std::sqrt(m_numberOfVariables) - std::sqrt(m_gammaPath);
		
		//performing steps in variables
		noalias(m_mean) +=  m_rate * dMean;
		m_sigma *= std::exp(deviationStepLen*dPath);
		m_muEff = (1-cmuEff) * m_muEff + cmuEff / sum(sqr(weights));
	}
	
private:
	mutable std::vector<IndividualType > m_offspring;
	std::size_t m_numberOfVariables; ///< Stores the dimensionality of the search space.
	std::size_t m_lambda; ///< The size of the offspring population, needs to be larger than mu.
	
	//mean of search distribution
	RealVector m_mean;
	
	//Variables governing step size update
	RealVector m_path;
	double m_gammaPath;

	double m_sigma;//global step-size
	double m_muEff;
	//variables for noise estimation and global learning rate
	double m_ztest;
	double m_fvar;
	double m_sigmanoise;
	double m_rate;

	bool m_firstIter;
};
//~ class SelfRLCMA : public AbstractSingleObjectiveOptimizer<RealVector >{
//~ public:

	//~ /// \brief From INameable: return the class name.
	//~ std::string name() const
	//~ { return "sRLCMA"; }
	
	//~ /// \brief Calculates lambda for the supplied dimensionality n.
	//~ static std::size_t suggestLambda( std::size_t dimension ) {
		//~ std::size_t lambda = std::size_t( 4. + ::floor( 3 *::log( static_cast<double>( dimension ) ) ) );
		//~ return lambda + lambda % 2;
	//~ }

	//~ void read( InArchive & archive ){}
	//~ void write( OutArchive & archive ) const{}

	//~ using AbstractSingleObjectiveOptimizer<RealVector >::init;
	//~ /// \brief Initializes the algorithm for the supplied objective function.
	//~ void init( ObjectiveFunctionType const& function, SearchPointType const& p) {
		//~ SIZE_CHECK(p.size() == function.numberOfVariables());
		//~ checkFeatures(function);
		//~ std::vector<RealVector> points(1,p);
		//~ std::vector<double> functionValues(10.0);

		//~ std::size_t lambda = SelfRLCMA::suggestLambda( p.size() );
		//~ doInit(
			//~ points,
			//~ functionValues,
			//~ lambda,
			//~ 3.0/std::sqrt(double(p.size()))
		//~ );
	//~ }

	//~ /// \brief Initializes the algorithm for the supplied objective function.
	//~ void init( 
		//~ ObjectiveFunctionType const& function, 
		//~ SearchPointType const& initialSearchPoint,
		//~ std::size_t lambda,
		//~ double initialSigma
	//~ ){
		//~ std::vector<RealVector> points(1,initialSearchPoint);
		//~ std::vector<double> functionValues(1,0.0);
		//~ doInit(
			//~ points,
			//~ functionValues,
			//~ lambda,
			//~ initialSigma
		//~ );
	//~ }

	//~ /// \brief Executes one iteration of the algorithm.
	//~ void step(ObjectiveFunctionType const& function){
		//~ std::vector<IndividualType> offspring = generateOffspring();
		//~ //evaluate pairs
		//~ for(auto& individual : offspring){
			//~ individual.penalizedFitness() = 0.0;
		//~ }
		
		//~ for(std::size_t i = 0; i != offspring.size(); ++i){
			//~ auto& individual1 = offspring[i];
			//~ std::size_t second = (i+m_lambda/2) % offspring.size();
			//~ if(second < m_lambda/2)
				//~ second = m_lambda/2 - 1 - second;
			//~ auto& individual2 = offspring[second];
			//~ double loss = function(individual1.searchPoint() | individual2.searchPoint());
			//~ individual1.penalizedFitness() += loss/2.0;
			//~ individual2.penalizedFitness() += (1-loss)/2.0;
		//~ }
		
		//~ //update population variance
		//~ double curPopVar = 0.0;
		//~ for(auto& individual : offspring){
			//~ curPopVar += sqr(individual.penalizedFitness()-0.5);
		//~ }
		
		//~ curPopVar /= offspring.size()-1;
		//~ double cPopVar = m_lambda*m_rate/(m_lambda *m_rate + m_numberOfVariables);
		//~ m_popVar = (1.0-cPopVar)*m_popVar + cPopVar * curPopVar;
		
		//~ updatePopulation(offspring);
	//~ }

	//~ double sigma() const {
		//~ return std::sqrt(m_var);
	//~ }
	
	//~ RealVector generatePolicy()const{
		//~ return m_mean + remora::normal(random::globalRng(), m_numberOfVariables, 0.0, m_var, remora::cpu_tag());
	//~ }

//~ protected:
	//~ /// \brief The type of individual used for the CMA
	//~ typedef Individual<RealVector, double, RealVector> IndividualType;
	
	//~ /// \brief Samples lambda individuals from the search distribution	
	//~ std::vector<IndividualType> & generateOffspring( ) const{
		//~ double sigma = std::sqrt(m_var);
		//~ auto sampler = [&](std::size_t i){
			//~ RealVector& z = m_offspring[i].chromosome();
			//~ RealVector& x = m_offspring[i].searchPoint();
			//~ noalias(z) = blas::normal(random::globalRng(), m_numberOfVariables, 0.0, 1.0, shark::blas::cpu_tag());
			//~ noalias(x) = m_mean + sigma * z;
		//~ };
		
		//~ threading::parallelND(m_offspring.size(), 0, sampler,threading::globalThreadPool());
		
		//~ return m_offspring;
	//~ }

	//~ /// \brief Updates the strategy parameters based on the supplied offspring population.
	//~ void updatePopulation( std::vector<IndividualType > const& offspring){	
		//~ //compute the weights
		//~ RealVector weights(m_lambda, 0.0);
		//~ for (std::size_t i = 0; i < m_lambda; i++){
			//~ weights(i) = -offspring[i].penalizedFitness();
		//~ }
		//~ weights -=min(weights);
		//~ weights /= norm_1(weights);

		//~ //update learning rates
		//~ double muEff = 1.0 / sum(sqr(weights));
		//~ double cPath = (muEff*m_rate + 2.)/(m_numberOfVariables + muEff*m_rate + 5.);
		//~ double dPath = m_rate*2.0*cPath/std::sqrt(cPath * (2-cPath));
		//~ double dPath = m_rate*2/(1+cPath);
		//~ double cPenalty = 0.1*cPath;
		
		//~ //first iteration: initialize all paths with true data from the function
		//~ if(m_firstIter){
			//~ m_firstIter = false;
			//~ cPath = 1.0;
			//~ dPath = 1.0;
		//~ }

		//~ //gradient of mean
		//~ RealVector dMean( m_numberOfVariables, 0. );
		//~ RealVector stepZ( m_numberOfVariables, 0. );
		//~ for (std::size_t i = 0; i < m_lambda; i++){
			//~ noalias(dMean) += (weights(i) - 1.0/m_lambda) * offspring[i].searchPoint();
			//~ noalias(stepZ) += weights(i) * offspring[i].chromosome();
		//~ }
		
		//~ noalias(m_path)= (1-cPath) * m_path + std::sqrt(cPath * (2-cPath) * muEff) * stepZ;
		//~ double deviationStepLen = norm_sqr(m_path)/m_numberOfVariables - 1.0;
		
		//~ //performing steps in variables
		//~ noalias(m_mean) +=  m_rate * dMean;
		//~ m_biasVar = std::max(0.0, m_biasVar - cPenalty * (m_popVar/m_popVarTarget - 1.0));
		//~ m_var *= std::exp((m_biasVar+deviationStepLen)*dPath);
		//~ //decide for learning-rate
		//~ boost::math::chi_squared_distribution<double> chi_squared(m_numberOfVariables);
		//~ double pQuot= pdf(chi_squared, m_numberOfVariables)/pdf(chi_squared, m_numberOfVariables*(1-0.5*m_biasVar));
		//~ m_rate =1.0/pQuot;
		
		//~ //store estimate for current loss
		//~ m_best.point = m_mean;
		//~ m_best.value = 0.0;
		//~ for (std::size_t i = 0; i < m_lambda; i++)
			//~ m_best.value += offspring[i].penalizedFitness()/m_lambda;
	//~ }

	//~ void doInit(
		//~ std::vector<SearchPointType> const& points,
		//~ std::vector<ResultType> const& functionValues,
		//~ std::size_t lambda,
		//~ double initialSigma
	//~ ){
		//~ SIZE_CHECK(points.size() > 0);
	
		//~ m_numberOfVariables =points[0].size();
		//~ lambda += lambda % 2;
		
		//~ m_lambda = lambda;
		
		//~ m_firstIter = true;
		
		//~ //variables for mean
		//~ m_mean = blas::repeat(0.0, m_numberOfVariables);
		
		//~ //variables for step size
		//~ m_path = blas::repeat(0.0, m_numberOfVariables);
		//~ m_var = sqr(initialSigma);
		

		
		
		//~ //variable for population-variance estimation
		//~ m_popVarTarget = 1.01/8.0;
		//~ m_popVar = 1.01/8.0;
		//~ m_biasVar = 0.0;
		//~ m_rate = 1.0;
		
		//~ m_mean = points[0];
		
		//~ //initialize offspring array
		//~ m_offspring.resize(m_lambda);
		//~ for( std::size_t i = 0; i < m_offspring.size(); i++ ) {
			//~ m_offspring[i].chromosome()  = blas::repeat(0.0, m_numberOfVariables);
			//~ m_offspring[i].searchPoint()  = blas::repeat(0.0, m_numberOfVariables);
		//~ }
	//~ }
//~ private:
	//~ mutable std::vector<IndividualType > m_offspring;
	//~ std::size_t m_numberOfVariables; ///< Stores the dimensionality of the search space.
	//~ std::size_t m_lambda; ///< The size of the offspring population, needs to be larger than mu.

	//~ //mean of search distribution
	//~ RealVector m_mean;
	
	//~ //Variables governing step size update
	//~ RealVector m_path;
	//~ double m_var;//global step-size


	//~ double m_popVar;
	//~ double m_popVarTarget;
	//~ double m_biasVar;
	//~ double m_rate;

	//~ bool m_firstIter;
//~ };
}
#endif
