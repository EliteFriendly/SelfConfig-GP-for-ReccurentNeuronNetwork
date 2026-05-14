#include <cmath>
#include <iostream>

using namespace std;

#pragma once
class SampleStorage
{
  private:
    double **trainData = nullptr;
    double **testData = nullptr;
    int size = 0;
    int trainSize = 0;
    int testSize = 0;
    int amDimensions = 0;
    double trainPart;

    void classificationSort(double **data)
    {
        int numbersOfClasses = 0;
        for (int i = 0; i < size; i++)
        {
            numbersOfClasses = max(numbersOfClasses, (int)data[i][amDimensions - 1]);
        }
        double *amOfEveryClass = new double[numbersOfClasses];
        for (int i = 0; i < numbersOfClasses; i++)
        {
            amOfEveryClass[i] = 0;
        }
        for (int i = 0; i < size; i++)
        {

            amOfEveryClass[(int)data[i][amDimensions - 1] - 1]++;
        }

        int *trainClass = new int[numbersOfClasses];
        int *usesClass = new int[numbersOfClasses];
        int amClasses = 0;
        for (int i = 0; i < numbersOfClasses; i++)
        {
            trainClass[i] = (int)(trainPart * amOfEveryClass[i]);
            amClasses += trainClass[i];
            usesClass[i] = 0;
        }
        if (amClasses != trainSize)
        {
            cout << "Er" << endl;
            trainClass[rand() % numbersOfClasses] += trainSize - amClasses;
        }
        int i1 = 0, i2 = 0;
        for (int j = 0; j < size; j++)
        {
            if (usesClass[(int)data[j][amDimensions - 1] - 1] < trainClass[(int)data[j][amDimensions - 1] - 1])
            {
                for (int w = 0; w < amDimensions; w++)
                {
                    trainData[i1][w] = data[j][w];
                }
                usesClass[(int)data[j][amDimensions - 1] - 1]++;
                i1++;
            }
            else
            {
                for (int w = 0; w < amDimensions; w++)
                {

                    testData[i2][w] = data[j][w];
                }
                i2++;
            }
        }

        delete[] amOfEveryClass;
        delete[] trainClass;
        delete[] usesClass;

    }

    double getDistance(double *a, double *b)
    {
        double sum = 0;
        for (int i = 0; i < amDimensions - 1; i++)
        {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(sum);
    }

    void regressionSort(double **data)
    {
        double **sortedData = new double *[size];

        for (int i = 0; i < size; i++)
        {
            sortedData[i] = new double[amDimensions];
            for (int j = 0; j < amDimensions; j++)
            {
                sortedData[i][j] = data[i][j];
            }
        }
        // Start sorting by closer point
        for (int i = 0; i < size; i++)
        {
            double Bestdistance = 99999, distance;
            int number;
            for (int j = 1; j < size - 1; j++)
            {
                distance = getDistance(sortedData[i], sortedData[j]);

                if (distance < Bestdistance)
                {
                    Bestdistance = distance;
                    number = j;
                }
            }
            double *temp = sortedData[i];
            sortedData[i] = sortedData[number];
            sortedData[number] = temp;
        }
        double remainder = 1 - trainPart;
        double tmp = 0;
        int i1 = 0, i2 = 0;
        for (int i = 0; i < size; i++)
        {
            tmp += remainder;
            if (tmp < 1)
            {
                for (int j = 0; j < amDimensions; j++)
                {
                    trainData[i1][j] = sortedData[i][j];
                }

                i1++;
            }
            else
            {
                for (int j = 0; j < amDimensions; j++)
                {
                    testData[i2][j] = sortedData[i][j];
                }

                i2++;
            }

            if (tmp >= 1.0)
            {
                tmp -= 1.0;
            }
        }

        for (int i = 0; i < size; i++)
        {
            delete[] sortedData[i];
        }
        delete[] sortedData;
    }

  public:
    SampleStorage(int size, int amDimensions, double **data, double trainPart, string target)
    {
        this->size = size;
        this->amDimensions = amDimensions + 1;
        trainSize = trainPart * size;
        testSize = (int)((1 - trainPart) * size);
        if (int(trainPart * size * 10) % 10 > 0)
        {
            testSize += 1;
        }
        this->trainData = new double *[trainSize];
        this->testData = new double *[testSize];
        for (int i = 0; i < trainSize; i++)
        {
            this->trainData[i] = new double[this->amDimensions];
        }
        for (int i = 0; i < testSize; i++)
        {
            this->testData[i] = new double[this->amDimensions];
        }

        this->trainPart = trainPart;
        if (target == "reg")
        {
            regressionSort(data);
        }
        if (target == "class")
        {
            classificationSort(data);
        }
    }

    double **getTrainData()
    {
        return trainData;
    }
    double **getTestData()
    {
        return testData;
    }
    int getTrainSize()
    {
        return trainSize;
    }
    int getTestSize()
    {
        return testSize;
    }
    ~SampleStorage()
    {
        if (trainData != nullptr)
        {
            delete[] trainData;
        }
        if (testData != nullptr)
        {
            delete[] testData;
        }
    }
};
