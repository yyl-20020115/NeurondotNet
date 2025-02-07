using System;
using MathNet.Numerics.LinearAlgebra;

namespace NN;

public class NeuronNetwork
{
    public int[] dimensions;
    public Random rand;
    private readonly Layer[] layers;

    public NeuronNetwork(int[] dimensions)
    {
        this.dimensions = dimensions;
        rand = new Random();
        layers = new Layer[dimensions.Length];
        layers[0] = new Layer(dimensions[0]);
        Console.WriteLine("Creating a neural network with " + layers.Length.ToString() + " layers.");
        Console.WriteLine("The input layer has " + InputLayer.Count.ToString() + " neurons.");
        for (int i = 1; i < layers.Length; i++)
        {
            layers[i] = new Layer(dimensions[i], layers[i - 1], rand);
        }
        for (int i = 1; i < layers.Length - 1; i++)
        {
            Console.WriteLine("Layer " + i.ToString() + " has " + layers[i].Count.ToString() + " neurons.");
        }
        Console.WriteLine("The output layer has " + OutputLayer.Count.ToString() + " neurons.");
        Console.WriteLine("");
    }

    private Layer InputLayer => layers[0];
    private Layer OutputLayer => layers[^1];

    public void Train(Matrix<double> inputMatrix, Matrix<double> outputMatrix, int maxEpochs = 10000, double targetError = 2.5)
    {
        int epoch = 0;
        double error = 100;
        while (epoch < maxEpochs + 1 && error > targetError)
        {
            error = RunEpoch(inputMatrix, outputMatrix);
            if (epoch % 25 == 0)
            {
                Console.WriteLine("Error for epoch " + epoch + ": " + error.ToString());
            }
            epoch++;
        }
        if (error <= targetError)
        {
            Console.WriteLine("Target error " + targetError.ToString("0.0000") + " reached - ending training.");
        }
        else if (epoch >= maxEpochs)
        {
            Console.WriteLine("Maximal epoch " + maxEpochs.ToString() + " reached - ending training.");
        }
        Console.WriteLine("");
    }

    public double CalculateSetError(Matrix<double> inputMatrix, Matrix<double> outputMatrix)
    {
        double error = 0;
        for (int row = 0; row < inputMatrix.RowCount; row++)
        {
            var input = new double[inputMatrix.ColumnCount];
            var expectedOutput = new double[outputMatrix.ColumnCount];
            //Get single row from test data
            for (int i = 0; i < inputMatrix.ColumnCount; i++)
            {
                input[i] = inputMatrix[row, i];
            }
            for (int i = 0; i < outputMatrix.ColumnCount; i++)
            {
                expectedOutput[i] = outputMatrix[row, i];
            }
            PredictOutput(input, expectedOutput);
            for (int i = 0; i < OutputLayer.Count; i++)
            {
                double delta = expectedOutput[i] - OutputLayer[i].output;
                error += Math.Pow(delta, 2);
            }
            AdjustWeights();
        }
        return error;
    }

    //Runs a single epoch. Returns the error for the epoch.
    public double RunEpoch(Matrix<double> inputMatrix, Matrix<double> outputMatrix)
    {
        double error = 0;
        for (int row = 0; row < inputMatrix.RowCount; row++)
        {
            double[] input = new double[inputMatrix.ColumnCount];
            var expectedOutput = new double[outputMatrix.ColumnCount];
            //Get single row from test data
            for (int i = 0; i < inputMatrix.ColumnCount; i++)
            {
                input[i] = inputMatrix[row, i];
            }
            for (int i = 0; i < outputMatrix.ColumnCount; i++)
            {
                expectedOutput[i] = outputMatrix[row, i];
            }
            //Forward propagation
            PredictOutput(input, expectedOutput);
            //Back propagation
            for (int i = 0; i < OutputLayer.Count; i++)
            {
                double delta = expectedOutput[i] - OutputLayer[i].output;
                OutputLayer[i].CollectError(delta);
                error += Math.Pow(delta, 2);
            }
            AdjustWeights();
        }
        return error;
    }

    public void PredictOutput(double[] input, double[] output)
    {
        for (int i = 0; i < InputLayer.Count; i++)
        {
            InputLayer[i].output = input[i];
        }
        for (int i = 1; i < layers.Length; i++)
        {
            for (int j = 0; j < layers[i].Count; j++)
            {
                layers[i][j].Activate();
            }
        }
    }

    public double[] GetOutput()
    {
        var output = new double[OutputLayer.Count];
        for (int i = 0; i < OutputLayer.Count; i++)
        {
            output[i] = OutputLayer[i].output;
        }
        return output;
    }

    private void AdjustWeights()
    {
        for (int i = layers.Length - 1; i > 0; i--)
        {
            for (int j = 0; j < layers[i].Count; j++)
            {
                layers[i][j].AdjustWeights();
            }
        }
    }
}