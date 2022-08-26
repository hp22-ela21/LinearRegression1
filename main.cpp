/********************************************************************************
* main.cpp: Implementering av en enkel maskininl�rningsmodell baserad p� 
*           linj�r regression, med tr�ningsdata deklarerat direkt i funktionen 
*           main och lagrat via tv� vektorer. Tr�ningsdatan kan �ndras utefter 
*           behov, b�de via fler upps�ttningar eller via helt ny data.
*
*           I Windows, kompilera programkoden och skapa en k�rbar fil d�pt 
*           main.exe via f�ljande kommando:
*           $ g++ main.cpp lin_reg.cpp -o main.exe -Wall
*
*           Programmet kan sedan k�ras under 10 000 epoker med en l�rhastighet
*           p� 1 % via f�ljande kommando:
*           $ main.exe
*
*           F�r att mata in antalet epoker samt l�rhastighet som skall anv�ndas
*           vid tr�ning kan f�ljande kommando anv�ndas:
*           $ main.exe <num_epochs> <learning_rate>
*
*           Som exempel, f�r att genomf�ra tr�ning under 5000 epoker med en
*           l�rhastighet p� 2 % kan f�ljande kommando anv�ndas:
*           $ main.exe 5000 0.02
********************************************************************************/
#include "lin_reg.hpp"

/********************************************************************************
* main: Tr�nar en maskininl�rningsmodell baserad p� linj�r regression via 
*       tr�ningsdata best�ende av fem tr�ningsupps�ttningar, lagrade via var 
*       sin vektor. Modellen tr�nas som default under 10 000 epoker med en 
*       l�rhastighet p� 1 %. Dessa parametrar kan dock v�ljas av anv�ndaren 
*       via inmatning i samband med k�rning av programmet, vilket l�ses in 
*       via ing�ende argument argc samt argv.
*
*       Efter tr�ningen �r slutf�rd sker prediktion f�r samtliga insignaler
*       mellan -10 och 10 med en stegringshastighet p� 1.0. Varje insignal
*       i detta intervall skrivs ut i terminalen tillsammans med predikterad
*       utsignal.
*
*       - argc: Antalet argument som har matats in vid k�rning av programmet
*               (default = 1, vilket �r kommandot f�r att k�ra programmet).
*       - argc: Pekare till array inneh�llande samtliga inl�sta argument i
*               form av text (default = exekveringskommandot, exempelvis main).
********************************************************************************/
int main(const int argc,
         const char** argv)
{
   lin_reg l1;

   const std::vector<double> train_in = { 0, 1, 2, 3, 4 };
   const std::vector<double> train_out = { -2, 0, 2, 4, 6 };

   std::size_t num_epochs = 10000;
   double learning_rate = 0.2;

   if (argc == 3)
   {
      num_epochs = std::atoi(argv[1]);
      learning_rate = std::atof(argv[2]);
   }

   l1.set_training_data(train_in, train_out);
   l1.train(num_epochs, learning_rate);
   l1.predict();
   return 0;
}