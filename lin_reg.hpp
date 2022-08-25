/********************************************************************************
* lin_reg.hpp: Innehåller funktionalitet för enkel implementering av
*              maskininlärningmodeller baserade på linjär regression via 
*              strukten lin_reg.
********************************************************************************/
#ifndef LIN_REG_HPP_
#define LIN_REG_HPP_

/* Inkluderingsdirektiv: */
#include <iostream>
#include <vector>

/********************************************************************************
* lin_reg: Strukt för implementering av maskininlärningsmodeller baserade på
*          linjär regression. Träningsdata passeras via referenser till vektorer 
*          innehållande träningsuppsättningarnas in- och utdata. Träning 
*          genomförs under angivet antal epoker med angiven lärhastighet.
********************************************************************************/
struct lin_reg
{
   /* Medlemmar: */
   std::vector<double> train_in;         /* Indata för träningsuppsättningar. */
   std::vector<double> train_out;        /* Utdata för träningsuppsättningar. */
   std::vector<std::size_t> train_order; /* Ordningsföljd för träningsuppsättningar. */ 
   double bias = 0.0;                    /* Vilovärde (m-värde). */ 
   double weight = 0.0;                  /* Vikt (k-värde). */

   /* Medlemsfunktioner: */
   std::size_t num_sets(void) { return this->train_order.size(); }
   void set_training_data(const std::vector<double>& train_in,
                          const std::vector<double>& train_out);
   void train(const std::size_t num_epochs,
              const double learning_rate);
   double predict(const double input) { return this->weight * input + this->bias; }
   void predict(std::ostream& ostream = std::cout);
   void predict_range(const double min,
                      const double max,
                      const double step = 1.0,
                      std::ostream& ostream = std::cout);
private:
   void shuffle(void);
   void optimize(const double input,
                 const double reference,
                 const double learning_rate);
};

#endif /* LIN_REG_HPP_ */