#ifndef TDL_HPP
#define TDL_HPP

#include <shark/Models/LinearModel.h>//single dense layer

using namespace shark;

class TDL {
public:
    void init(int numVars ) {
        m_numVars = numVars;

        // init weights
        RealVector w(m_numVars);
        m_weights = subrange(w, 0, m_numVars);
        m_best = subrange(w, 0, m_numVars);
    }
    void setAlgorithmParams(double lambda, double rate, double gamma) {
        m_lambda = lambda;
        m_rate = rate;
        m_gamma = gamma;
    }
    void setValueFunction( void (*vf) ()  ) {

    }
    

private:
    double m_lambda;
    double m_rate;
    double m_gamma;
    int m_numVars;

    RealVector m_weights;
    RealVector m_best;
};
#endif