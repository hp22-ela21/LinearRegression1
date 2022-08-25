/********************************************************************************
* lin_reg.hpp: Definition av funktionsmedlemmar tillhörande strukten lin_reg, 
*              som används för implementering av enkla maskininlärningsmodeller
*              som baseras på linjär regression.
********************************************************************************/
#include "lin_reg.hpp"

/********************************************************************************
* set_training_data: Läser in träningsdata för angiven regressionsmodell via
*                    passerad in- och utdata, tillsammans med att index
*                    för respektive träningsuppsättning lagras.
* 
*                    - train_in : Innehåller indata för träningsuppsättningar.
*                    - train_out: Innehåller utdata för träningsuppsättningar.
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
* train: Tränar angiven regressionsmodell med befintlig träningsdata under
*        angivet antal epoker samt angiven lärhastighet. I början av varje epok
*        randomiseras ordningen på träningsuppsättningarna för att undvika att
*        eventuella icke avsedda mönster i träningsdatan påverkar resultatet.
*
*        För varje träningsuppsättning sker en prediktion via aktuell indata.
*        Det predikterade värdet jämförs mot aktuellt referensvärde för att
*        beräkna aktuell avvikelse. Modellens parametrar justeras därefter.
*
*        - num_epochs   : Antalet epoker/omgångar som träning skall genomföras.
*        - learning_rate: Lärhastigheten, som avgör hur stor andel av uppmätt
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
* predict: Genomför prediktion med angiven regressionsmodell via indata från
*          samtliga befintliga träningsuppsättningar och skriver ut varje
*          insignal samt motsvarande predikterat värde via angiven utström
*          där standardutenheten std::cout används som default för utskrift
*          i terminalen.
*
*          - ostream: Angiven utström (default = std::cout).
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
* predict_range: Genomför prediktion med angiven regressionsmodell för
*                datapunkter inom intervallet mellan angivet min- och maxvärde
*                [min, max] med angiven stegringshastighet step, som sätts till
*                1.0 som default.
*
*                Varje insignal skrivs ut tillsammans med motsvarande
*                predikterat värde via angiven utström, där standardutenheten
*                std::cout används som default för utskrift i terminalen.
*
*                - min    : Lägsta värde för datatpunkter som skall testas.
*                - max    : Högsta värde för datatpunkter som skall testas.
*                - step   : Stegringshastigheten, dvs. differensen mellan
*                           varje datapunkt som skall testas (default = 1.0).
                 - ostream: Angiven utström (default = std::cout).
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
* shuffle: Randomiserar den inbördes ordningen på träningsuppsättningarna för
*          angiven regressionsmodell, vilket genomförs i syfte att minska risken
*          för att eventuella icke avsedda mönster i träningsdatan skall 
*          påverka träningen.
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
* optimize: Beräknar aktuell avvikelse för angiven regressionsmodell och 
*           justerar modellens parametrar därefter.
*
*           input        : Insignal som prediktion skall genomföras med.
*           reference    : Referensvärde från träningsdatan, vilket utgör det
*                          värde som modellen önskas prediktera.
*           learning_rate: Modellens lärhastighet, avgör hur mycket modellens
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