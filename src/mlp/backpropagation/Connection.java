package mlp.backpropagation;

import mlp.backpropagation.Neuron;

public class Connection {
  private static int counter = 0;

  private double weight = 0;
  private double prevDeltaWeight = 0; // for momentum
  private double deltaWeight = 0;

  private final Neuron fromNeuron;
  private final Neuron toNeuron;

  private final int id; // auto increment, starts at 0

  public Connection(Neuron fromN, Neuron toN) {
    fromNeuron = fromN;
    toNeuron = toN;
    id = counter;
    counter++;
  }

  public static void resetCounter() {
    counter = 0;
  }

  public int getId() {
    return id;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(double w) {
    weight = w;
  }

  public void setDeltaWeight(double w) {
    prevDeltaWeight = deltaWeight;
    deltaWeight = w;
  }

  public double getPrevDeltaWeight() {
    return prevDeltaWeight;
  }

  public Neuron getFromNeuron() {
    return fromNeuron;
  }

  public Neuron getToNeuron() {
    return toNeuron;
  }
}