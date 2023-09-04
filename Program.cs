using System;
using MathNet.Numerics.LinearAlgebra;

namespace NN;

public class Program
{
    public static void Main(string[] args)
    {
        MatrixHelpers.ReadDataFile(@"iris_data_files/iris_training.dat", 4, 3, out Matrix<global::System.Double> trainingFeatures, out Matrix<global::System.Double> trainingOutput);
        
        var network = new NeuronNetwork(new int[] { 4, 3, 3, 3 });
        network.Train(trainingFeatures, trainingOutput, 10000, 2.5);
        network.PredictOutput(new double[] { 0.813331, 0.692682, 0.854032, 0.975 }, new double[] { 0, 0, 1 });
        var output = network.GetOutput();
        Console.WriteLine("The networks output for input 0.813331, 0.692682, 0.854032, 0.975 is: " + MatrixHelpers.OutputToString(output));
        /*double[] testPredict = nn.PredictOutput(new double[]{0.813331, 0.692682, 0.854032, 0.975});
        Console.WriteLine("Networks prediction for values (0.813331, 0.692682, 0.854032, 0.975):");
        for (int i = 0; i < testPredict.Length; i++){
            Console.WriteLine(testPredict[i]);
        }*/
        MatrixHelpers.ReadDataFile(@"iris_data_files/iris_test.dat", 4, 3, out Matrix<double> testFeatures, out Matrix<double> testOutput);
        Console.WriteLine("The networks error for the test set is: " + network.CalculateSetError(testFeatures, testOutput).ToString("0.0000"));
    }
}
