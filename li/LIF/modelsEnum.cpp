//
// Created by timian on 12/20/18.
//
 enum ModelEnum {NN, MVR, LR};

inline const char* ToString(ModelEnum v)
{
 switch (v)
 {
  case NN:   return "NN ";
  case MVR:   return "MVR ";
  case LR: return "LR ";
  default:      return "UNKNOWN ";
 }
}
