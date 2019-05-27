//
// Created by timian on 12/19/18.
//

#include "multivariateLinearReg.h"
#include <math.h>

using namespace std;

float multivariateLinearReg::predict(float key) {
    return pFunc(key);
}

multivariateLinearReg::multivariateLinearReg(std::string dataset) {
    type = MVR;
    if (dataset == "log") {
        pFunc = [this](double arg) { return this->logNorm(arg); };
    } else if (dataset == "osm") {
        pFunc = [this](double arg) { return this->osm(arg); };
    } else if (dataset == "norm") {
        pFunc = [this](double arg) { return this->norm(arg); };
    } else if (dataset == "sas") {
        pFunc = [this](double arg) { return this->sas(arg); };
    } else if (dataset == "beta1") {
        pFunc = [this](double arg) { return this->beta1(arg); };
    } else if (dataset == "beta2") {
        pFunc = [this](double arg) { return this->beta2(arg); };
    } else if (dataset == "beta3") {
        pFunc = [this](double arg) { return this->beta3(arg); };
    } else if (dataset == "beta4") {
        pFunc = [this](double arg) { return this->beta4(arg); };
    } else if (dataset == "beta5") {
        pFunc = [this](double arg) { return this->beta5(arg); };
    } else if (dataset == "beta6") {
        pFunc = [this](double arg) { return this->beta6(arg); };
    } else if (dataset == "beta7") {
        pFunc = [this](double arg) { return this->beta7(arg); };
    } else if (dataset == "beta8") {
        pFunc = [this](double arg) { return this->beta8(arg); };
    }

}

float multivariateLinearReg::logx(float base, float x) {
    return log(x) / log(base);
}

float multivariateLinearReg::logNorm(float x) {
    return static_cast<float>(-679949.8406229503 + (4019.49511586 * pow(x, 1)) + (-26.53840264 * pow(x, 2)) + (0.03474706 * pow(x, 3)) +
                              (-1.52837072e-05 * pow(x, 4)) + (2.09524487e-09 * pow(x, 5)) + (logx(2, x) * 187077.97010198) +
                              (logx(3, x) * 118033.05757509) + (logx(4, x) * 93538.98505099) + (logx(5, x) * 80570.09625488) +
                              (logx(6, x) * 72371.63790569) + (-10324.11537708 * cos(x)));
}

float multivariateLinearReg::osm(float x) {
    return -348635782327144.5 + (-1.37642258e+13 * pow(x, 1)) + (1.69118699e+11 * pow(x, 2)) +
           (-9.22047874e+08 * pow(x, 3)) + (logx(2, x) * 9.24174567e+13) + (logx(3, x) * 5.8308635e+13) +
           (-5.23953522e+08 * cos(x));
}

float multivariateLinearReg::sas(float x) {
    return 58396.41311842995 + (0.03214434 * pow(x, 1)) + (7.55483543e-09 * pow(x, 2)) + (-1.19851221e-16 * pow(x, 3)) +
           (logx(2, x) * 3.36695041e-08) + (logx(3, x) * 2.12430919e-08) + (logx(4, x) * 1.6834752e-08) +
           (logx(5, x) * 1.45006661e-08) + (logx(6, x) * 1.30251422e-08) + (2.13434679e-11 * cos(x));
}

float multivariateLinearReg::norm(float x) {
    return 130.84035600424613 + (-9.56665484e-13 * pow(x, 1)) + (-1.19392997e-08 * pow(x, 2)) +
           (-4.07037757e-09 * pow(x, 3)) +
           (-1.55889937e-07 * pow(x, 4)) + (-3.39943624e-06 * pow(x, 5)) + (1.5344262e-07 * pow(x, 6)) +
           (-2.10406744e-09 * pow(x, 7)) +
           (1.2787841e-11 * pow(x, 8)) + (-3.57960081e-14 * pow(x, 9)) + (3.70674919e-17 * pow(x, 10));
}

float multivariateLinearReg::beta1(float x) {
    return -5558893561.753906 + (3.71945667e+10 * pow(x, 1)) + (-2.10836832e+11 * pow(x, 2)) +
           (9.09456481e+11 * pow(x, 3)) + (-2.7886972e+12 * pow(x, 4)) + (6.02060212e+12 * pow(x, 5)) +
           (-9.05132382e+12 * pow(x, 6)) + (9.24492585e+12 * pow(x, 7)) + (-6.10029853e+12 * pow(x, 8)) +
           (2.34000035e+12 * pow(x, 9)) + (-3.95432551e+11 * pow(x, 10)) + (logx(2, x) * -9.68126281e+13) +
           (logx(3, x) * 1.91803494e+14) + (logx(4, x) * -4.84057771e+13);
}

float multivariateLinearReg::beta2(float x) {
    return 3361.539488876355 + (39784951.68015844 * pow(x, 1)) + (-3.70695123e+08 * pow(x, 2)) +
           (2.08655486e+09 * pow(x, 3)) + (-7.53265523e+09 * pow(x, 4)) + (1.68902359e+10 * pow(x, 5)) +
           (-2.01128444e+10 * pow(x, 6)) + (4.05750479e+09 * pow(x, 7)) + (1.65254159e+10 * pow(x, 8)) +
           (-1.2825345e+10 * pow(x, 9)) + (logx(2, x) * 288.06262058);
}

float multivariateLinearReg::beta3(float x) {
    return -204729544412359.97 + (-3.85660735e+12 * pow(x, 1)) + (1.14818788e+14 * pow(x, 2)) +
           (-2.9313814e+13 * pow(x, 3)) + (4.39443055e+13 * pow(x, 4)) + (-6.83717764e+13 * pow(x, 5)) +
           (6.42925135e+13 * pow(x, 6)) + (-4.20198641e+13 * pow(x, 7)) + (1.83755793e+13 * pow(x, 8)) +
           (-4.81621561e+12 * pow(x, 9)) + (5.72499318e+11 * pow(x, 10)) + (logx(2, x) * -2.317863e+13) +
           (logx(3, x) * 3.70414636e+13) + (2.05633286e+14 * cos(x));
}

float multivariateLinearReg::beta4(float x) {
    return 229736123888.09082 + (3.26586816e+10 * pow(x, 1)) + (-2.63166166e+11 * pow(x, 2)) +
           (5.01284204e+11 * pow(x, 3)) + (-1.2431022e+12 * pow(x, 4)) + (2.25007137e+12 * pow(x, 5)) +
           (-2.87286543e+12 * pow(x, 6)) + (2.54184725e+12 * pow(x, 7)) + (-1.48024648e+12 * pow(x, 8)) +
           (5.09496585e+11 * pow(x, 9)) + (-7.84020316e+10 * pow(x, 10)) + (logx(2, x) * 6.16007784e+11) +
           (logx(3, x) * -5.69707402e+11) + (logx(4, x) * 3.08003892e+11) + (logx(5, x) * -5.84580495e+11) +
           (logx(6, x) * -4.13388041e+11) + (-2.35627203e+11 * cos(x));
}

float multivariateLinearReg::beta5(float x) {
    return -35473.885498046875 + (708342.51966433 * pow(x, 1)) + (-12432488.5907828 * pow(x, 2)) +
           (5.70394048e+08 * pow(x, 3)) + (-3.782238e+09 * pow(x, 4)) + (1.26111573e+10 * pow(x, 5)) +
           (-2.58777068e+10 * pow(x, 6)) + (3.44621731e+10 * pow(x, 7)) + (-2.92403641e+10 * pow(x, 8)) +
           (1.43952358e+10 * pow(x, 9)) + (-3.12705484e+09 * pow(x, 10)) + (logx(2, x) * -1.4331013e+11) +
           (logx(3, x) * 2.27141175e+11);
}

float multivariateLinearReg::beta6(float x) {
    return 3.0860241578358492e+16 + (-2743862.82617828 * pow(x, 1)) + (-1.54301208e+16 * pow(x, 2)) +
           (-2.05739689e+08 * pow(x, 3)) + (1.28584437e+15 * pow(x, 4)) + (-2.83141167e+09 * pow(x, 5)) +
           (-4.2855719e+13 * pow(x, 6)) + (-8.29817496e+09 * pow(x, 7)) + (7.73489087e+11 * pow(x, 8)) +
           (-4.79436221e+09 * pow(x, 9)) + (-7.14323612e+09 * pow(x, 10)) + (logx(2, x) * 1.15615223e+16) +
           (logx(3, x) * -1.06567387e+16) + (logx(4, x) * 5.78099515e+15) + (logx(5, x) * -3.01341969e+16) +
           (logx(6, x) * 1.35703446e+16) + (-3.08602416e+16 * cos(x));
}

float multivariateLinearReg::beta7(float x) {
    return 28012163279538.9 + (1983578.47185127 * pow(x, 1)) + (-1.40060815e+13 * pow(x, 2)) +
           (-645481.4050293 * pow(x, 3)) + (1.16717637e+12 * pow(x, 4)) + (-13079233.07202148 * pow(x, 5)) +
           (-3.88662306e+10 * pow(x, 6)) + (-69424111.3684082 * pow(x, 7)) + (7.63379292e+08 * pow(x, 8)) +
           (-35714062.83911133 * pow(x, 9)) + (logx(2, x) * -6.00679138e+13) + (logx(3, x) * 1.19004478e+14) +
           (logx(4, x) * -3.00311045e+13) + (-2.80121633e+13 * cos(x));
}


float multivariateLinearReg::beta8(float x) {
    return 36357890679.5 + (-1.99760031e+11 * pow(x, 1)) + (8.79978706e+11 * pow(x, 2)) +
           (-3.00850212e+12 * pow(x, 3)) + (7.45749744e+12 * pow(x, 4)) + (-1.32751258e+13 * pow(x, 5)) +
           (1.67784523e+13 * pow(x, 6)) + (-1.46768593e+13 * pow(x, 7)) + (8.4388757e+12 * pow(x, 8)) +
           (-2.86638017e+12 * pow(x, 9)) + (4.35468185e+11 * pow(x, 10)) + (logx(2, x) * -2.46208705e+17) +
           (logx(3, x) * 4.88083023e+16) + (logx(4, x) * -1.2268457e+17) + (logx(5, x) * 5.36439202e+17) +
           (logx(6, x) * 1.18196371e+17);
}



