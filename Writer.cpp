#include "lm.asswecan.pb.h"

int main(void)
{

lm:asswecan msg1;
   msg1.set_id(101);
   msg1.set_str("asswecan");

   fstream output("./log", ios::out | ios::trunc | ios::binary);

   if(!msg1.SerializeTo0stream(&output))
   {
       cerr << "Failed to write msg." << endl;
       return -1;
   }
   return 0;
}
