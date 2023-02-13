#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#define INNODE 2 // 输入层神经元个数
#define HIDENODE 10 // 隐藏层神经元个数
#define OUTNODE 1 // 输出层神经元个数


/**
 * 步长（学习率）
 */
double StudyRate = 1.6;

/**
 * 允许最大误差
 */
double threshold = 1e-4;

/**
 * 最大迭代次数
 */
int mostTimes = 1e6;

/**
 * 训练集大小
 */
int trainSize = 0;

/**
 * 测试集大小
 */
int testSize = 0;

/**
 * 样本
 */
typedef struct Sample{
    double out[30][OUTNODE]; // 输出
    double in[30][INNODE]; // 输入
}Sample;


/**
 * 神经元结点
 */
typedef struct Node{
    double value; // 当前神经元结点输出的值
    double bias; // 当前神经元结点偏偏置值
    double bias_delta; // 当前神经元结点偏置值的修正值
    double *weight; // 当前神经元结点向下一层结点传播的权值
    double *weight_delta; // 当前神经元结点向下一层结点传播的权值的修正值
}Node;

/**
 *  输入层
 */
Node inputLayer[INNODE];
/**
 * 隐藏层
 */
Node hideLayer[HIDENODE];
/**
 * 输出层
 */
Node outLayer[OUTNODE];



double Max(double a, double b){
    return a > b ? a : b;
}

/**
 * 激活函数sigmoid
 * @param x 输入值
 * @return 输出值
 */
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}


/**
 * 读取训练集
 * @param filename 文件名
 * @return 训练集
 */
Sample * getTrainData(const char * filename){
    Sample * result = (Sample*)malloc(sizeof (Sample));
    FILE * file = fopen(filename, "r");
    if(file != NULL){
        int count = 0;
        while (fscanf(file, "%lf %lf %lf", &result->in[count][0], &result->in[count][1], &result->out[count][0]) != EOF){
            ++count;
        }
        trainSize = count;
        printf("%s 文件读取完毕\n", filename);
        fclose(file);
        return result;
    } else{
        fclose(file);
        printf("%s 文件打开错误!\n\a", filename);
        return NULL;
    }
}

/**
 * 读取测试集
 * @param filename 文件名
 * @return 测试集
 */
Sample * getTestData(const char * filename){
    Sample * result = (Sample*)malloc(sizeof (Sample));
    FILE * file = fopen(filename, "r");
    if(file != NULL){
        int count = 0;
        while (fscanf(file, "%lf %lf", &result->in[count][0], &result->in[count][1]) != EOF){
            ++count;
        }
        testSize = count;
        printf("%s 文件读取完毕\n", filename);
        fclose(file);
        return result;
    } else{
        fclose(file);
        printf("%s 文件打开错误!\n\a", filename);
        return NULL;
    }
}

/**
 * 打印样本
 * @param data 要打印的样本
 * @param size 样本大小
 */
void printData(Sample * data, int size){
    if(data == NULL){
        printf("样本为空！\n\a");
        return;
    }
    for (int i = 0; i < size; ++i) {
        printf("%d, x1 = %f, x2 = %f, y = %f\n", i + 1, data->in[i][0], data->in[i][1], data->out[i][0]);
    }
}

/**
 * 初始化函数
 */
void init(){
    // 设置时间戳为生成随机序列的种子
    srand(time(NULL));
    
    // 输入层的初始化
    for (int i = 0; i < INNODE; ++i) {
        inputLayer[i].weight = (double *)malloc(sizeof (double ) * HIDENODE);
        inputLayer[i].weight_delta = (double *) malloc(sizeof (double ) * HIDENODE);
        inputLayer[i].bias = 0.0;
        inputLayer[i].bias_delta = 0.0;
    }

    // 输出层权值初始化
    for (int i = 0; i < INNODE; ++i) {
        for (int j = 0; j < HIDENODE; ++j) {
            inputLayer[i].weight[j] = rand() % 10000 / (double )10000 * 2 - 1.0;
            inputLayer[i].weight_delta[j] = 0.0;
        }
    }


    // 初始化隐藏层结点
    for (int i = 0; i < HIDENODE; ++i) {
        hideLayer[i].weight = (double *) malloc(sizeof (double ) * OUTNODE);
        hideLayer[i].weight_delta = (double *) malloc(sizeof (double ) * OUTNODE);
        hideLayer[i].bias = rand() % 10000 / (double )10000 * 2 - 1.0;
        hideLayer[i].bias_delta = 0.0;
    }

    // 初始化隐藏层权值
    for (int i = 0; i < HIDENODE; ++i) {
        for (int j = 0; j < OUTNODE; ++j) {
            hideLayer[i].weight[j] = rand() % 10000 / (double )10000 * 2 - 1.0;
            hideLayer[i].weight_delta[j] = 0.0;
        }
    }

    for (int i = 0; i < OUTNODE; ++i) {
        outLayer[i].bias = rand() % 10000 / (double )10000 * 2 - 1.0;
        outLayer[i].bias_delta = 0.0;
    }
}

/**
 * 重置修正值
 */
void resetDelta(){
    for (int i = 0; i < INNODE; ++i) {
        for (int j = 0; j < HIDENODE; ++j) {
            inputLayer[i].weight_delta[j] = 0.0;
        }
    }

    for (int i = 0; i < HIDENODE; ++i) {
        hideLayer[i].bias_delta = 0.0;
        for (int j = 0; j < OUTNODE; ++j) {
            hideLayer[i].weight_delta[j] = 0.0;
        }
    }

    for (int i = 0; i < OUTNODE; ++i) {
        outLayer[i].bias_delta = 0.0;
    }
}


int main() {
    // 初始化
    init();
    // 获取训练集
    Sample * trainSample = getTrainData("TrainData.txt");
//    printData(trainSample, trainSize);


    for (int trainTime = 0; trainTime < mostTimes; ++trainTime) {
        // 重置梯度信息
        resetDelta();

        // 当前训练最大误差
        double error_max = 0.0;

        // 开始训练（累计bp）
        for (int currentTrainSample_Pos = 0; currentTrainSample_Pos < trainSize; ++currentTrainSample_Pos) {

            // 输入自变量
            for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
                inputLayer[inputLayer_Pos].value = trainSample->in[currentTrainSample_Pos][inputLayer_Pos];
            }

            /** ----- 开始正向传播 ----- */
            for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                double sum = 0.0;
                for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
                    sum += inputLayer[inputLayer_Pos].value * inputLayer[inputLayer_Pos].weight[hideLayer_Pos];
                }

                sum -= hideLayer[hideLayer_Pos].bias;
                hideLayer[hideLayer_Pos].value = sigmoid(sum);
            }

            for (int outLayer_Pos = 0;  outLayer_Pos < OUTNODE ; ++outLayer_Pos) {
                double sum = 0.0;
                for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                    sum += hideLayer[hideLayer_Pos].value * hideLayer[hideLayer_Pos].weight[outLayer_Pos];
                }
                sum -= outLayer[outLayer_Pos].bias;
                outLayer[outLayer_Pos].value = sigmoid(sum);
            }

            /** ----- 计算误差 ----- */
            double error = 0.0;
            for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                double temp = fabs(outLayer[outLayer_Pos].value - trainSample->out[currentTrainSample_Pos][outLayer_Pos]);
                // 损失函数
                error += temp * temp / 2.0;
            }
            
            error_max = Max(error_max, error);
            
            
            /** ----- 反向传播 ----- */
            for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                double bias_delta = -(trainSample->out[currentTrainSample_Pos][outLayer_Pos] - outLayer[outLayer_Pos].value)
                        * outLayer[outLayer_Pos].value * (1.0 - outLayer[outLayer_Pos].value);
                outLayer[outLayer_Pos].bias_delta += bias_delta;
            }

            for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                    double weight_delta = (trainSample->out[currentTrainSample_Pos][outLayer_Pos] - outLayer[outLayer_Pos].value)
                                          * outLayer[outLayer_Pos].value * (1.0 - outLayer[outLayer_Pos].value)
                                          * hideLayer[hideLayer_Pos].value;
                    hideLayer[hideLayer_Pos].weight_delta[outLayer_Pos] += weight_delta;
                }
            }

            //
            for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                double sum = 0.0;
                for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                    sum += -(trainSample->out[currentTrainSample_Pos][outLayer_Pos] - outLayer[outLayer_Pos].value)
                           * outLayer[outLayer_Pos].value * (1.0 - outLayer[outLayer_Pos].value)
                           * hideLayer[hideLayer_Pos].weight[outLayer_Pos];
                }
                hideLayer[hideLayer_Pos].bias_delta += sum * hideLayer[hideLayer_Pos].value * (1.0 - hideLayer[hideLayer_Pos].value);
            }


            for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
                for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                    double sum  = 0.0;
                    for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                        sum += (trainSample->out[currentTrainSample_Pos][outLayer_Pos] - outLayer[outLayer_Pos].value)
                               * outLayer[outLayer_Pos].value * (1.0 - outLayer[outLayer_Pos].value)
                               * hideLayer[hideLayer_Pos].weight[outLayer_Pos];
                    }
                    inputLayer[inputLayer_Pos].weight_delta[hideLayer_Pos] += sum * hideLayer[hideLayer_Pos].value * (1.0 - hideLayer[hideLayer_Pos].value)
                                                                              * inputLayer[inputLayer_Pos].value;
                }
            }

        }


        // 判断误差是否达到允许误差范围
        if(error_max < threshold){
            printf("\a训练完成！总训练次数：%d, 最大误差为：%f\n", trainTime + 1, error_max);
            break;
        }

        // 误差无法接受，开始修正

        for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
            for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                inputLayer[inputLayer_Pos].weight[hideLayer_Pos] += StudyRate
                                                                    * inputLayer[inputLayer_Pos].weight_delta[hideLayer_Pos] /
                                                                    (double) trainSize;
            }
        }

        for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
            hideLayer[hideLayer_Pos].bias += StudyRate
                    * hideLayer[hideLayer_Pos].bias_delta / (double )trainSize;
            for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                hideLayer[hideLayer_Pos].weight[outLayer_Pos] += StudyRate
                        * hideLayer[hideLayer_Pos].weight_delta[outLayer_Pos] / (double )trainSize;
            }
        }

        for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
            outLayer[outLayer_Pos].bias += StudyRate
                    * outLayer[outLayer_Pos].bias_delta / (double )trainSize;
        }
    }

    // 训练完成，读取测试集
    Sample * testSample = getTestData("TestData.txt");
    printf("预测结果如下：\n");
    for (int currentTestSample_Pos = 0; currentTestSample_Pos < testSize; ++currentTestSample_Pos) {
        for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
            inputLayer[inputLayer_Pos].value = testSample->in[currentTestSample_Pos][inputLayer_Pos];
        }

        for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
            double sum = 0.0;
            for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
                sum += inputLayer[inputLayer_Pos].value * inputLayer[inputLayer_Pos].weight[hideLayer_Pos];
            }
            sum -= hideLayer[hideLayer_Pos].bias;
            hideLayer[hideLayer_Pos].value = sigmoid(sum);
        }

        for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
            double sum = 0.0;
            for (int hideLayer_Pos = 0; hideLayer_Pos < HIDENODE; ++hideLayer_Pos) {
                sum += hideLayer[hideLayer_Pos].value * hideLayer[hideLayer_Pos].weight[outLayer_Pos];
            }
            sum -= outLayer[outLayer_Pos].bias;
            outLayer[outLayer_Pos].value = sigmoid(sum);
        }

        for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
            testSample->out[currentTestSample_Pos][outLayer_Pos] = outLayer[outLayer_Pos].value;
        }
    }

    printData(testSample, testSize);

    return 0;
}
