#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#define INNODE 2 // �������Ԫ����
#define HIDENODE 10 // ���ز���Ԫ����
#define OUTNODE 1 // �������Ԫ����


/**
 * ������ѧϰ�ʣ�
 */
double StudyRate = 1.6;

/**
 * ����������
 */
double threshold = 1e-4;

/**
 * ����������
 */
int mostTimes = 1e6;

/**
 * ѵ������С
 */
int trainSize = 0;

/**
 * ���Լ���С
 */
int testSize = 0;

/**
 * ����
 */
typedef struct Sample{
    double out[30][OUTNODE]; // ���
    double in[30][INNODE]; // ����
}Sample;


/**
 * ��Ԫ���
 */
typedef struct Node{
    double value; // ��ǰ��Ԫ��������ֵ
    double bias; // ��ǰ��Ԫ���ƫƫ��ֵ
    double bias_delta; // ��ǰ��Ԫ���ƫ��ֵ������ֵ
    double *weight; // ��ǰ��Ԫ�������һ���㴫����Ȩֵ
    double *weight_delta; // ��ǰ��Ԫ�������һ���㴫����Ȩֵ������ֵ
}Node;

/**
 *  �����
 */
Node inputLayer[INNODE];
/**
 * ���ز�
 */
Node hideLayer[HIDENODE];
/**
 * �����
 */
Node outLayer[OUTNODE];



double Max(double a, double b){
    return a > b ? a : b;
}

/**
 * �����sigmoid
 * @param x ����ֵ
 * @return ���ֵ
 */
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}


/**
 * ��ȡѵ����
 * @param filename �ļ���
 * @return ѵ����
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
        printf("%s �ļ���ȡ���\n", filename);
        fclose(file);
        return result;
    } else{
        fclose(file);
        printf("%s �ļ��򿪴���!\n\a", filename);
        return NULL;
    }
}

/**
 * ��ȡ���Լ�
 * @param filename �ļ���
 * @return ���Լ�
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
        printf("%s �ļ���ȡ���\n", filename);
        fclose(file);
        return result;
    } else{
        fclose(file);
        printf("%s �ļ��򿪴���!\n\a", filename);
        return NULL;
    }
}

/**
 * ��ӡ����
 * @param data Ҫ��ӡ������
 * @param size ������С
 */
void printData(Sample * data, int size){
    if(data == NULL){
        printf("����Ϊ�գ�\n\a");
        return;
    }
    for (int i = 0; i < size; ++i) {
        printf("%d, x1 = %f, x2 = %f, y = %f\n", i + 1, data->in[i][0], data->in[i][1], data->out[i][0]);
    }
}

/**
 * ��ʼ������
 */
void init(){
    // ����ʱ���Ϊ����������е�����
    srand(time(NULL));
    
    // �����ĳ�ʼ��
    for (int i = 0; i < INNODE; ++i) {
        inputLayer[i].weight = (double *)malloc(sizeof (double ) * HIDENODE);
        inputLayer[i].weight_delta = (double *) malloc(sizeof (double ) * HIDENODE);
        inputLayer[i].bias = 0.0;
        inputLayer[i].bias_delta = 0.0;
    }

    // �����Ȩֵ��ʼ��
    for (int i = 0; i < INNODE; ++i) {
        for (int j = 0; j < HIDENODE; ++j) {
            inputLayer[i].weight[j] = rand() % 10000 / (double )10000 * 2 - 1.0;
            inputLayer[i].weight_delta[j] = 0.0;
        }
    }


    // ��ʼ�����ز���
    for (int i = 0; i < HIDENODE; ++i) {
        hideLayer[i].weight = (double *) malloc(sizeof (double ) * OUTNODE);
        hideLayer[i].weight_delta = (double *) malloc(sizeof (double ) * OUTNODE);
        hideLayer[i].bias = rand() % 10000 / (double )10000 * 2 - 1.0;
        hideLayer[i].bias_delta = 0.0;
    }

    // ��ʼ�����ز�Ȩֵ
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
 * ��������ֵ
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
    // ��ʼ��
    init();
    // ��ȡѵ����
    Sample * trainSample = getTrainData("TrainData.txt");
//    printData(trainSample, trainSize);


    for (int trainTime = 0; trainTime < mostTimes; ++trainTime) {
        // �����ݶ���Ϣ
        resetDelta();

        // ��ǰѵ��������
        double error_max = 0.0;

        // ��ʼѵ�����ۼ�bp��
        for (int currentTrainSample_Pos = 0; currentTrainSample_Pos < trainSize; ++currentTrainSample_Pos) {

            // �����Ա���
            for (int inputLayer_Pos = 0; inputLayer_Pos < INNODE; ++inputLayer_Pos) {
                inputLayer[inputLayer_Pos].value = trainSample->in[currentTrainSample_Pos][inputLayer_Pos];
            }

            /** ----- ��ʼ���򴫲� ----- */
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

            /** ----- ������� ----- */
            double error = 0.0;
            for (int outLayer_Pos = 0; outLayer_Pos < OUTNODE; ++outLayer_Pos) {
                double temp = fabs(outLayer[outLayer_Pos].value - trainSample->out[currentTrainSample_Pos][outLayer_Pos]);
                // ��ʧ����
                error += temp * temp / 2.0;
            }
            
            error_max = Max(error_max, error);
            
            
            /** ----- ���򴫲� ----- */
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


        // �ж�����Ƿ�ﵽ������Χ
        if(error_max < threshold){
            printf("\aѵ����ɣ���ѵ��������%d, ������Ϊ��%f\n", trainTime + 1, error_max);
            break;
        }

        // ����޷����ܣ���ʼ����

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

    // ѵ����ɣ���ȡ���Լ�
    Sample * testSample = getTestData("TestData.txt");
    printf("Ԥ�������£�\n");
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
