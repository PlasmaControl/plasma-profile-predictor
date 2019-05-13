// predictor.cpp
#include <fdeep/fdeep.hpp>



int main()
{
  const auto model =  fdeep::load_model("toy_model.json");

  fdeep::tensor5s inputs = {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 4), {1, 2, 3, 4})};

  fdeep::tensor5s outputs = {fdeep::tensor5(fdeep::shape5(1, 1, 1, 1, 3), {1, 2, 3})};

  outputs = model.predict(inputs);
  

  std::cout << fdeep::show_tensor5s(outputs) << std::endl; 

}
