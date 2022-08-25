/********************************************************************************
* lin_reg.hpp: Definition av funktionsmedlemmar tillh�rande strukten lin_reg, 
*              som anv�nds f�r implementering av enkla maskininl�rningsmodeller
*              som baseras p� linj�r regression.
********************************************************************************/
#include "lin_reg.hpp"

/********************************************************************************
* set_training_data: L�ser in tr�ningsdata f�r angiven regressionsmodell via
*                    passerad in- och utdata, tillsammans med att index
*                    f�r respektive tr�ningsupps�ttning lagras.
* 
*                    - train_in : Inneh�ller indata f�r tr�ningsupps�ttningar.
*                    - train_out: Inneh�ller utdata f�r tr�ningsupps�ttningar.
********************************************************************************/
void lin_reg::set_training_data(const std::vector<double>& train_in,
                                const std::vector<double>& train_out)
{
   const auto num_sets = train_in.size() <= train_out.size() ? train_in.size() : train_out.size();
   this->train_in.resize(num_sets);
   this->train_out.resize(num_sets);
   this->train_order.resize(num_sets);

   for (std::size_t i = 0; i < num_sets; ++i)
   {
      this->train_in[i] = train_in[i];
      this->train_out[i] = train_out[i];
      this->train_order[i] = i;
   }

   return;
}

/********************************************************************************
* train: Tr�nar angiven regressionsmodell med befintlig tr�ningsdata under
*        angivet antal epoker samt angiven l�rhastighet. I b�rjan av varje epok
*        randomiseras ordningen p� tr�ningsupps�ttningarna f�r att undvika att
*        eventuella icke avsedda m�nster i tr�ningsdatan p�verkar resultatet.
*
*        F�r varje tr�ningsupps�ttning sker en prediktion via aktuell indata.
*        Det predikterade v�rdet j�mf�rs mot aktuellt referensv�rde f�r att
*        ber�kna aktuell avvikelse. Modellens parametrar justeras d�refter.
*
*        - num_epochs   : Antalet epoker/omg�ngar som tr�ning skall genomf�ras.
*        - learning_rate: L�rhastigheten, som avg�r hur stor andel av uppm�tt
*                         avvikelse som modellens parametrar justeras med.
********************************************************************************/
void lin_reg::train(const std::size_t num_epochs,
                    const double learning_rate)
{
   if (!this->num_sets())
   {
      std::cerr << "Training data missing!\n\n";
      return;
   }

   for (std::size_t i = 0; i < num_epochs; ++i)
   {
      this->shuffle();

      for (auto& j : this->train_order)
      {
         this->optimize(this->train_in[j], this->train_out[j], learning_rate);
      }
   }
   return;
}

/********************************************************************************
* predict: Genomf�r prediktion med angiven regressionsmodell via indata fr�n
*          samtliga befintliga tr�ningsupps�ttningar och skriver ut varje
*          insignal samt motsvarande predikterat v�rde via angiven utstr�m
*          d�r standardutenheten std::cout anv�nds som default f�r utskrift
*          i terminalen.
*
*          - ostream: Angiven utstr�m (default = std::cout).
********************************************************************************/
void lin_reg::predict(std::ostream& ostream)
{
   if (!this->num_sets())
   {
      std::cerr << "Training data missing!\n\n";
      return;
   }

   const auto* end = &this->train_in[this->train_in.size() - 1];
   ostream << "--------------------------------------------------------------------------------\n";

   for (auto& i : this->train_in)
   {
      const auto prediction = this->weight * i + this->bias;

      ostream << "Input: " << i << "\n";
      ostream << "Predicted output: " << prediction << "\n";

      if (&i < end) ostream << "\n";
   }

   ostream << "--------------------------------------------------------------------------------\n\n";
   return;
}

/********************************************************************************
* predict_range: Genomf�r prediktion med angiven regressionsmodell f�r
*                datapunkter inom intervallet mellan angivet min- och maxv�rde
*                [min, max] med angiven stegringshastighet step, som s�tts till
*                1.0 som default.
*
*                Varje insignal skrivs ut tillsammans med motsvarande
*                predikterat v�rde via angiven utstr�m, d�r standardutenheten
*                std::cout anv�nds som default f�r utskrift i terminalen.
*
*                - min    : L�gsta v�rde f�r datatpunkter som skall testas.
*                - max    : H�gsta v�rde f�r datatpunkter som skall testas.
*                - step   : Stegringshastigheten, dvs. differensen mellan
*                           varje datapunkt som skall testas (default = 1.0).
                 - ostream: Angiven utstr�m (default = std::cout).
********************************************************************************/
void lin_reg::predict_range(const double min,
                            const double max,
                            const double step,
                            std::ostream& ostream)
{
   if (min >= max)
   {
      std::cerr << "Error: Minimum input value cannot be higher or equal to maximum input value!\n\n";
      return;
   }

   ostream << "--------------------------------------------------------------------------------\n";

   for (auto i = min; i <= max; i = i + step)
   {
      const auto prediction = this->weight * i + this->bias;

      ostream << "Input: " << i << "\n";
      ostream << "Predicted output: " << prediction << "\n";

      if (i < max) ostream << "\n";
   }

   ostream << "--------------------------------------------------------------------------------\n\n";
   return;
}

/********************************************************************************
* shuffle: Randomiserar den inb�rdes ordningen p� tr�ningsupps�ttningarna f�r
*          angiven regressionsmodell, vilket genomf�rs i syfte att minska risken
*          f�r att eventuella icke avsedda m�nster i tr�ningsdatan skall 
*          p�verka tr�ningen.
********************************************************************************/
void lin_reg::shuffle(void)
{
   for (std::size_t i = 0; i < this->num_sets(); ++i)
   {
      const auto r = std::rand() % this->num_sets();
      const auto temp = this->train_order[i];
      this->train_order[i] = this->train_order[r];
      this->train_order[r] = temp;
   }

   return;
}

/********************************************************************************
* optimize: Ber�knar aktuell avvikelse f�r angiven regressionsmodell och 
*           justerar modellens parametrar d�refter.
*
*           input        : Insignal som prediktion skall genomf�ras med.
*           reference    : Referensv�rde fr�n tr�ningsdatan, vilket utg�r det
*                          v�rde som modellen �nskas prediktera.
*           learning_rate: Modellens l�rhastighet, avg�r hur mycket modellens
*                          parametrar justeras vid avvikelse.
********************************************************************************/
void lin_reg::optimize(const double input,
                       const double reference,
                       const double learning_rate)
{
   const auto prediction = this->predict(input);
   const auto error = reference - prediction;
   const auto change_rate = error * learning_rate;

   this->bias += change_rate;
   this->weight += change_rate * input;
   return;
}