using System;
using System.Collections.Generic;

namespace NN;

public class Neuron
{
    private double bias;                       // Bias value.
    private double error;                      // Sum of error.
    private double gradient = 5;               // Steepness of sigmoid curve.
    private double learnRate = 0.01;           // Learning rate.
    public double output = double.MinValue;    // Preset value of neuron.
    private readonly List<Weight> weights = new();              // Collection of weights to inputs.
    public Neuron()
    {
    }

    public Neuron(Layer inputs, Random rnd)
    {
        for (int i = 0; i < inputs.Count; i++)
        {
            Weight weight = new Weight(inputs[i], rnd.NextDouble() * 2 - 1);
            weights.Add(weight);
        }
    }

    public double Derivative() => output * (1 - output);

    public void Activate()
    {
        error = 0;
        output = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            output += weights[i].Value * weights[i].Input.output;
        }
        output = 1 / (1 + Math.Exp(-gradient * (output + bias)));
    }

    public void CollectError(double delta)
    {
        if (weights != null)
        {
            error += delta;
            for (int i = 0; i < weights.Count; i++)
            {
                weights[i].Input.CollectError(error * weights[i].Value);
            }
        }
    }

    public void AdjustWeights()
    {
        for (int i = 0; i < weights.Count; i++)
        {
            weights[i].Value += error * Derivative() * learnRate * weights[i].Input.output;
        }
        bias += error * Derivative() * learnRate;
    }
}