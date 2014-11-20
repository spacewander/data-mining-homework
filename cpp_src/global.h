#ifndef GLOBAL_H
#define GLOBAL_H

#include <vector>

using std::vector;

#define USER_NUM 943
#define ITEM_NUN 1682
#define BETA 0.001

const int DIM = 1;

double alpha = 0.013;

// 注意用下标访问容器时记得使用id - 1
// 遍历时，注意item有些元素是不存在于训练集的，此时值为0
vector<float> bu(USER_NUM, 0);
vector<float> bi(ITEM_NUN, 0);
vector<int> un(USER_NUM, 0);
vector<int> in(ITEM_NUN, 0);
vector<double> p(USER_NUM, 0.0);
vector<double> q(ITEM_NUN, 0.0);
vector<double> yj(ITEM_NUN, -0.1);
vector<double> yi(USER_NUM, -0.1);
float mean = 0;

#endif /* GLOBAL_H */
