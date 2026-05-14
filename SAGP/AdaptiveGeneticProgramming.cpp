#include "AdaptiveGeneticProgramming.h"

void AdaptiveGeneticProgramming::findBest()
{
    for (int i = 0; i < numIndividuals; i++)
    {
        if (arrayIndividuals[i].getFitness() > bestIndividual.getFitness())
        {
            bestIndividual = arrayIndividuals[i];
        }
    }
    if (saveTrail) {
        
        fileTrail << bestIndividual.getFitness() << endl;
        
    }
}

void AdaptiveGeneticProgramming::setSelectionsArrays()
{
    selection[0]->setArrIndividuals(arrayIndividuals, numIndividuals);
    selection[1]->setArrIndividuals(arrayIndividuals, numIndividuals);
    selection[2]->setArrIndividuals(arrayIndividuals, numIndividuals);
    selection[3]->setArrIndividuals(arrayIndividuals, numIndividuals);
    selection[4]->setArrIndividuals(arrayIndividuals, numIndividuals);
}

Tree AdaptiveGeneticProgramming::createChild(int numInd)
{

    chosenSel[numInd] = probabilityChoice(selProbabilities, 5);

    // cout << "Номер генерации = " << i <<", Номер индивида = " << j << endl;
    int numParent1 = selection[chosenSel[numInd]]->getNumParents();
    int numParent2 = selection[chosenSel[numInd]]->getNumParents();
    while (numParent1 == numParent2)
    {
        numParent2 = selection[chosenSel[numInd]]->getNumParents();
    }

    chosenCross[numInd] = probabilityChoice(crossProbabilities, 4);
    Tree child = crossover[chosenCross[numInd]]->getChild(arrayIndividuals[numParent1], arrayIndividuals[numParent2]);

    chosenMut[numInd] = probabilityChoice(mutProbabilities, 4);

    mutation[chosenMut[numInd]]->doMutChild(child);
    return child;
}

void AdaptiveGeneticProgramming::threadsFitnessCalc(SampleStorage &data, int ammThread)
{
    if (ammThread == 1)
    {
        for (int i = 0; i < numIndividuals; i++)
        {
            arrayChildren[i].doNeuronNetwork();
            arrayChildren[i].trainWithDE(data, size, computingLimitation);
        }
        return;
    }

    std::vector<std::thread> workers;
    std::atomic<int> nextIndex{0}; // Атомарный счетчик для распределения задач

    auto worker_func = [&](int threadId) {
        while (true)
        {
            int idx = nextIndex.fetch_add(1); // Атомарно получаем следующий индекс
            if (idx >= numIndividuals)
                break;

            // std::cout << "Worker #" << threadId << ": Processing individual #" << idx << std::endl;
            arrayChildren[idx].doNeuronNetwork();
            arrayChildren[idx].trainWithDE(data, size, computingLimitation);
        }
    };

    // Запускаем потоки
    for (int i = 0; i < ammThread; ++i)
    {
        workers.emplace_back(worker_func, i);
    }

    // Ожидание завершения всех потоков
    for (auto &w : workers)
    {
        if (w.joinable())
            w.join();
    }
}
int AdaptiveGeneticProgramming::findWinner(int *arrPlayers, int ammPlayer, double *arrFitness)
{
    double *fitnessPlayers = new double[ammPlayer];
    int ammUses;

    for (int i = 0; i < ammPlayer; i++)
    {
        fitnessPlayers[i] = 0;
        ammUses = 0;
        for (int j = 0; j < numIndividuals; j++)
        {
            if (arrPlayers[j] == i)
            {
                fitnessPlayers[i] += arrFitness[j];
                ammUses++;
            }
        }
        fitnessPlayers[i] /= ammUses;
    }
    int best;
    double fitbest = -9999999;

    for (int i = 0; i < ammPlayer; i++)
    {
        if (fitnessPlayers[i] > fitbest)
        {
            fitbest = fitnessPlayers[i];
            best = i;
        }
    }

    delete[] fitnessPlayers;
    return best;
}

void AdaptiveGeneticProgramming::changeProbabilities(double *arrProba, int numWinner, int ammPlayers)
{
    double sumPenlty = 0;
    double penalty = 2.0 / (numGeneration * ammPlayers);

    for (int i = 0; i < ammPlayers; i++)
    {
        if (i != numWinner)
        {
            if (arrProba[i] > socialCard)
            {
                if ((arrProba[i] - penalty) < socialCard)
                {
                    sumPenlty += arrProba[i] - socialCard;
                    arrProba[i] = socialCard;
                }
                else
                {
                    arrProba[i] -= penalty;
                    sumPenlty += penalty;
                }
            }
        }
    }

    arrProba[numWinner] += sumPenlty;
}

void AdaptiveGeneticProgramming::recalcProbabilities()
{
    double sumFitness = 0;
    double *arrFitness = new double[numIndividuals];
    for (int i = 0; i < numIndividuals; i++)
    {
        arrFitness[i] = arrayChildren[i].getFitness();
        sumFitness += arrFitness[i];
    }
    int winner;
    // Селекция
    winner = findWinner(chosenSel, 5, arrFitness);
    changeProbabilities(selProbabilities, winner, 5);

    // Кроссовер
    winner = findWinner(chosenCross, 4, arrFitness);
    changeProbabilities(crossProbabilities, winner, 4);

    // Мутация
    winner = findWinner(chosenMut, 4, arrFitness);
    changeProbabilities(mutProbabilities, winner, 4);

    saveProbabilities();
}

void AdaptiveGeneticProgramming::startTrain(double **x, int ammInputs, int amOutPuts, int size, int numIndividuals,
                                            int numGeneration)
{
    ofstream fGen("algorithm_results/MaxGeneration/ReachedGeneration_" + markFile + ".txt");
    fSel.open("algorithm_results/Probabilities/ProbabilSel_" + markFile + ".txt");



    fCross.open("algorithm_results/Probabilities/ProbabilCross_" + markFile + ".txt");
    fMut.open("algorithm_results/Probabilities/ProbabilMut_" + markFile + ".txt");

    saveProbabilities();

    SampleStorage sampleStorage(size, ammInputs, x, dataTrainPart, typeTask);
    size = sampleStorage.getTrainSize();
    // try a sample storage

    AdaptiveGeneticProgramming::size = size;
    AdaptiveGeneticProgramming::ammInputs = ammInputs;
    AdaptiveGeneticProgramming::ammOutputs = amOutPuts;

    AdaptiveGeneticProgramming::numIndividuals = numIndividuals;
    AdaptiveGeneticProgramming::numGeneration = numGeneration;
    arrayIndividuals = new Tree[numIndividuals];
    arrayChildren = new Tree[numIndividuals];
    chosenCross = new int[numIndividuals];
    chosenMut = new int[numIndividuals];
    chosenSel = new int[numIndividuals];
    // Set limitations
    if (computingLimitation.getComputingLimitation() == 0)
    {
        int DEind = 30;
        int DEgen = 30;
        computingLimitation.setComputingLimitation(DEind * DEgen * numIndividuals * numGeneration);
    }
    cout << "Computing limitation = " << computingLimitation.getComputingLimitation() << endl;
    // Первая иницилизация поколения
    for (int i = 0; i < numIndividuals; i++)
    {
        Tree t(treeDepth - 1, ammInputs, amOutPuts, typeTask);
        // Подсчет узлов и уровней
        int nodes = 0, lvl = 0;
        t.recountLayers(lvl);
        t.countNodes(nodes);

        arrayIndividuals[i] = t;
        arrayIndividuals[i].doNeuronNetwork();

        arrayIndividuals[i].trainWithDE(sampleStorage, size, computingLimitation);
        cout << "Individual\t" << i << endl;

    }
    //cout << "Computing limitation = " << computingLimitation.getComputingLimitation() << endl;
    findBest(); // Первый поиск лучшего индивида
    // Основное начало алгоритма
    int numParent1, numParent2;

    int maxGeneration = 0;
    for (int i = 0; computingLimitation.getComputingLimitation() > 0; i++)

    {
        // cout << "Номер генерации = " << i << endl;
        setSelectionsArrays();
        for (int j = 0; j < numIndividuals; j++)
        {
            //cout << "Номер генерации = " << i <<", Номер индивида = " << j << endl;

            arrayChildren[j] = createChild(j);
            // arrayChildren[j].trainWithDE(x, y, size, K1);
        }
        cout << "Generation " << i << endl;

        threadsFitnessCalc(x, 6);

        recalcProbabilities();

        forming.replaceGeneration(arrayIndividuals, arrayChildren, numIndividuals);
        findBest();
        maxGeneration = i;
    }

    fGen << maxGeneration << endl;
    fGen.close();
    fSel.close();
    fCross.close();
    fMut.close();

}
