// Second translation unit that includes CAMBReader.h (see test_camb.cpp): the header must not define non-inline
// functions or variables, otherwise linking fails with multiple definitions
#include <FML/CAMBUtils/CAMBReader.h>

int camb_odr_check() { return FML::FILEUTILS::camb_transfer_format.transfer_col_k == 0 ? 1 : 0; }
