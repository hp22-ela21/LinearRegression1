/********************************************************************************
* lin_reg.hpp: Inneh�ller funktionalitet f�r enkel implementering av
*              maskininl�rningmodeller baserade p� linj�r regression via 
*              strukten lin_reg.
********************************************************************************/
#ifndef LIN_REG_HPP_
#define LIN_REG_HPP_

/* Inkluderingsdirektiv: */
#include <iostream>
#include <vector>

/********************************************************************************
* lin_reg: Strukt f�r implementering av maskininl�rningsmodeller baserade p�
*          linj�r regression. Tr�ningsdata passeras via referenser till vektorer 
*          inneh�llande tr�ningsupps�ttningarnas in- och utdata. Tr�ning 
*          genomf�rs under angivet antal epoker med angiven l�rhastighet.
********************************************************************************/
struct lin_reg
{
   /* Medlemmar: */
   std::vector<double> train_in;         /* Indata f�r tr�ningsupps�ttningar. */
   std::vector<double> train_out;        /* Utdata f�r tr�ningsupps�ttningar. */
   std::vector<std::size_t> train_order; /* Ordningsf�ljd f�r tr�ningsupps�ttningar. */ 
   double bias = 0.0;                    /* Vilov�rde (m-v�rde). */ 
   double weight = 0.0;                  /* Vikt (k-v�rde). */

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