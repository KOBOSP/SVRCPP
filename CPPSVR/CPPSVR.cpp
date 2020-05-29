//https://github.com/dandongxue/SVR/commit/2807c9e11aaa7a9331379a1067491fb5e7c5c27c
//Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
#include "pch.h"
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>
#include <stdlib.h>

using namespace std;
const int INF = 0x7f7f7f7f;
const int C = 1000;//Paramter C
const double Ksi = 1e-3;//error
const double Epslion = 0.125;//Eplison Parameter,if ansy in sampy+-Epslion punish=0
const int TWO_SIGMA_SQUARED = 2;//RBF(Radial-Basis Function)Parameter
const int SampWN = 4; //The Sample SampWNsion
const int SampXN = 140;//Total Sample Num
const int SampTraN = 140;//Training Sample Num

double alph[2 * SampTraN];//alph to be double
int    IsLow[2 * SampTraN];//Get The Ans Y 1 is low,0 is high
double BorderY[2 * SampTraN];
double OffsetB = 0.0;//thresh b

double tau = 1e-12;
double SampY[SampXN];//Store All The Samples SampY
double SampX[SampXN][SampWN];//Store Samles，0-SampTraN-1 For Training;first_test_i-N For Testing
int Ansi = 0, Ansj = 0;
const int MaxIterNum = 40000;
int NowIter = 0;

double VectorDotMul(int i1, int i2)//Dot product
{
	double dot = 0;
	for (int i = 0; i < SampWN; i++) {
		dot += SampX[i1][i] * SampX[i2][i];
	}
	return dot;
}


//The kernel_func(int, int) is RBF(Radial-Basis Function).
//K(Xi, Xj)=exp(-||Xi-Xj||^2/(r))
//Kenel Of RBF for vector P140
double KernelMul(int i1, int i2) {
	double res = 0;
	int sign = 1;
	if ((i1 >= SampTraN) ^ (i2 >= SampTraN)) {
		sign = -1;
	}
	i1 = i1 >= SampTraN ? (i1 - SampTraN) : i1;
	i2 = i2 >= SampTraN ? (i2 - SampTraN) : i2;
	double s = -2*VectorDotMul(i1, i2);
	s += VectorDotMul(i1, i1) + VectorDotMul(i2, i2);
	res = sign*exp(-s / TWO_SIGMA_SQUARED);
	return res;
}


double GetPredictY(int k) {
	double s = 0.0;
	for (int i = 0; i < SampTraN; i++) {
		s += (alph[i] - alph[i + SampTraN])*-1* KernelMul(i, k);//A formula 11
	}
	s -= OffsetB;   //Because  b=-b
	return s;
}




void  OutPutFile(const char* fileName) {
	if (freopen(fileName, "w", stdout) == NULL) {
		printf("File could not be opened!");
		exit(1);//Exit To End
	}
	printf("SampWNsion=%d\n", SampWN);//sample dimension
	printf("b=%lf\n", OffsetB);//threshold
	printf("two_sigma_squared=%d\n", TWO_SIGMA_SQUARED);
	printf("C=%d\n", C);

	int n_support_vectors = 0;
	for (int i = 0; i < SampTraN; i++) {
		if (alph[i] > 0 && alph[i] < C) {
			n_support_vectors++;
		}
	}
	printf("n_support_vectors=%d\n", n_support_vectors);
	printf("support vector rate=%lf\n", (double)n_support_vectors / SampTraN);
	for (int i = 0; i < SampTraN; i++) {
		printf("alph[%d]=%lf\n", i, alph[i] - alph[i + SampTraN]);
	}
	printf("Iter=%d\n", NowIter);
	for (int i = 0; i < SampXN; i++)
		printf("SVR Precate(%d)=%0.4lf\t True Value=%0.4lf\t Fabs=%0.4lf\n", i, GetPredictY(i) + 6.7, SampY[i], fabs(GetPredictY(i) + 6.7 - SampY[i]));
	fclose(stdout);
}

void InPutFile(const char* fileName) //Reading Data
{
	if (freopen(fileName, "r", stdin) == NULL) {
		printf("File could not be opened!");
		exit(1);//Exit To End
	}
	int i = 0, j = 0;
	double temp;//temp为每次读到的数据, 默认为6位有效数字。
	while (scanf("%lf", &temp) != EOF) {
		if (temp < 0) {
			break;
		}
		if (j == SampWN) {
			SampY[i] = temp;
			IsLow[i] = 1;
			BorderY[i + SampTraN] = -(Epslion + SampY[i]);//up border
			BorderY[i] = SampY[i] - Epslion;// low border
			j = 0;
			i++;
		}
		else {
			SampX[i][j] = temp;//录入数据到SampX中  包括训练集和测试集
			j++;
		}
	}
	fclose(stdin);
}

void SMOSelectTwo()//To choose one datasets which needs to be optimization
{
	Ansi = Ansj = -1;
	double G_max = -INF, G_min = INF;//Gmax=highest -(Y+-epslion),Gmin=lowest Y+-epslion
	for (int i = 0; i < 2 * SampTraN; i++) {
		if (IsLow[i] == 1) {// low border
			if (alph[i] < C)
				if (-BorderY[i] >= G_max) {
					Ansi = i;
					G_max = -BorderY[i];//
				}
		}
		else {//up border
			if (alph[i] > 0)
				if (BorderY[i] >= G_max) {
					G_max = BorderY[i];
					Ansi = i;
				}
		}
	}
	double obj_min = INF;
	for (int j = 0; j < 2 * SampTraN; j++) {
		if ((IsLow[j] == 1 && alph[j] > 0) || (IsLow[j] == -1) && alph[j] < C) {
			OffsetB = IsLow[j] * BorderY[j]+G_max;//Find Ansi Then Bij=y[i]*G[i]+G_max
			if (-IsLow[j] * BorderY[j] <= G_min)
				G_min = -IsLow[j] * BorderY[j];
			if (OffsetB > 0) {
				double a = KernelMul(Ansi, Ansi) + KernelMul(j, j) - 2 * KernelMul(Ansi, j);
				if (a <= 0)
					a = tau;
				if ((-(OffsetB*OffsetB) / a) <= obj_min) {
					Ansj = j;
					obj_min = -(OffsetB*OffsetB) / a;
				}
			}
		}
	}
	if (G_max - G_min < Ksi) {
		Ansi = -1;
		Ansj = -1;
	}
}



int main() {
	memset(alph, 0, sizeof(alph));
	memset(IsLow, -1, sizeof(IsLow));
	InPutFile("battery_data.txt");//Reading Data

	while (1) {
		SMOSelectTwo();//Select Two Variable To Adjust
		if (Ansj == -1)
			break;
		double Eta, OldAi, OldAj,NewAj,NewAjCli, deltaAi, deltaAj, sum;
		Eta = KernelMul(Ansi, Ansi) + KernelMul(Ansj, Ansj) - 2 * KernelMul(Ansi, Ansj);//7.107
		if (Eta <= 0)
			Eta = tau;
		OldAi = alph[Ansi], OldAj = alph[Ansj];

		if (IsLow[Ansi] != IsLow[Ansj]) {//S formula 13
			double delta = (-BorderY[Ansi] - BorderY[Ansj]) / Eta;
			double diff = alph[Ansi] - alph[Ansj];
			alph[Ansi] += delta;
			alph[Ansj] += delta;
			//keep alph in 0 to C
			if (diff > 0) {
				if (alph[Ansj] < 0) {
					alph[Ansj] = 0;
					alph[Ansi] = diff;
				}
			}
			else {
				if (alph[Ansi] < 0) {
					alph[Ansi] = 0;
					alph[Ansj] = -diff;
				}
			}
			if (diff > 0) {
				if (alph[Ansi] > C) {
					alph[Ansi] = C;
					alph[Ansj] = C - diff;
				}
			}
			else {
				if (alph[Ansj] > C) {
					alph[Ansj] = C;
					alph[Ansi] = C + diff;
				}
			}
		}
		else {
			double delta = (BorderY[Ansi] - BorderY[Ansj]) / Eta;//S formula 16
			double sum = alph[Ansi] + alph[Ansj];
			alph[Ansi] -= delta;
			alph[Ansj] += delta;
			//keep alph in 0 to C
			if (sum > C) {
				if (alph[Ansi] > C) {
					alph[Ansi] = C;
					alph[Ansj] = sum - C;
				}
			}
			else {
				if (alph[Ansj] < 0) {
					alph[Ansj] = 0;
					alph[Ansi] = sum;
				}
			}
			if (sum > C) {
				if (alph[Ansj] > C) {
					alph[Ansj] = C;
					alph[Ansi] = sum - C;
				}
			}
			else {
				if (alph[Ansi] < 0) {
					alph[Ansi] = 0;
					alph[Ansj] = sum;
				}
			}
		}
		deltaAi = alph[Ansi] - OldAi, deltaAj = alph[Ansj] - OldAj;
		for (int i = 0; i < 2 * SampTraN; i++) {
			BorderY[i] += KernelMul(i, Ansi)*deltaAi + KernelMul(i, Ansj)*deltaAj;
		}
		if (NowIter++ > MaxIterNum)//The Bigest Iter Num!
			break;
		if (NowIter % 2000 == 0) {
			float pro = float(NowIter*100.0 / MaxIterNum);
			printf("Work Progres=>%0.2lf%%\r", pro);
		}
	}
	OutPutFile("data_result.txt");
	return 0;
}