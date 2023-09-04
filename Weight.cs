namespace NN;

public class Weight
{
    public readonly Neuron Input;
    public double Value;

    public Weight(Neuron input, double value)
    {
        this.Input = input;
        this.Value = value;
    }
}