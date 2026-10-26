// Second translation unit that includes KDTree.h (see test_kdtree.cpp): the header must not define non-inline
// functions, otherwise linking fails with multiple definitions
#include <FML/KDTree/KDTree.h>

int kdtree_odr_check() {
    FML::KDTREE::KDTree tree(FML::KDTREE::pointVec{{0.1, 0.2}, {0.5, 0.5}}, 1.0);
    return int(tree.nearest_index(FML::KDTREE::point_t{0.45, 0.55}));
}
