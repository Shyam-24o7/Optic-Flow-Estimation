// Library anchor: keeps libfcw non-empty before the first module lands.
#include "fcw/types.hpp"
namespace fcw { const char* version() { return "0.1"; } }
