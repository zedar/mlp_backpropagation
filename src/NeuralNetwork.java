import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.*;
import java.util.*;

public class NeuralNetwork {
  static {
    Locale.setDefault(Locale.ENGLISH);
  }

  private static final DecimalFormat df = new DecimalFormat("#.0#");
  private static final int RANDOM_WEIGHT_MULTIPLIER = 1;
  private static final double EPSILON = 0.00000000001;
  private static final double LEARNING_RATE = 0.9f;
  private static final double MOMENTUM = 0.7f;

  private final Random rand = new Random();

  private final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
  private final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
  private final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
  private final int[] layers;

  private double[][] inputs;
  private double[][] expectedOutputs;
  private double[][] resultOutputs;
  private double[] output;

  private double[][] testInputs;
  private double[][] testExpectedOutputs;

  // for weight update all
  final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();

  public NeuralNetwork(int input, int hidden, int output,
                       double[][] inputs, double[][] expectedOutputs,
                       double[][] testInputs, double[][] testExpectedOutputs) {
    this.inputs =inputs;
    this.expectedOutputs = expectedOutputs;
    this.resultOutputs = new double[this.expectedOutputs.length][];
    this.testInputs = testInputs;
    this.testExpectedOutputs = testExpectedOutputs;

    this.layers = new int[] { input, hidden, output };

    //
    // Create all neurons and connections Connections are created in the
    // neuron class
    //
    for (int i = 0; i < layers.length; i++) {
      Neuron bias = new Neuron();
      if (i == 0) { // input layer
        for (int j = 0; j < layers[i]; j++) {
          Neuron neuron = new Neuron();
          inputLayer.add(neuron);
        }
      } else if (i == 1) { // hidden layer
        for (int j = 0; j < layers[i]; j++) {
          Neuron neuron = new Neuron();
          neuron.addInConnections(inputLayer);
          neuron.addBiasConnection(bias);
          hiddenLayer.add(neuron);
        }
      }

      else if (i == 2) { // output layer
        for (int j = 0; j < layers[i]; j++) {
          Neuron neuron = new Neuron();
          neuron.addInConnections(hiddenLayer);
          neuron.addBiasConnection(bias);
          outputLayer.add(neuron);
        }
      } else {
        System.out.println("!Error NeuralNetwork init");
      }
    }

    // initialize random weights
    for (Neuron neuron : hiddenLayer) {
      ArrayList<Connection> connections = neuron.getAllInConnections();
      for (Connection conn : connections) {
        double newWeight = getRandom();
        conn.setWeight(newWeight);
      }
    }
    for (Neuron neuron : outputLayer) {
      ArrayList<Connection> connections = neuron.getAllInConnections();
      for (Connection conn : connections) {
        double newWeight = getRandom();
        conn.setWeight(newWeight);
      }
    }

    // reset id counters
    Neuron.resetCounter();
    Connection.resetCounter();
  }

  // random
  private double getRandom() {
    return RANDOM_WEIGHT_MULTIPLIER * (rand.nextDouble() * 2 - 1); // [-1;1[
  }

  /**
   *
   * @param inputs
   *            There is equally many neurons in the input layer as there are
   *            in input variables
   */
  private void setInput(double inputs[]) {
    for (int i = 0; i < inputLayer.size(); i++) {
      inputLayer.get(i).setOutput(inputs[i]);
    }
  }

  private double[] getOutput() {
    double[] outputs = new double[outputLayer.size()];
    for (int i = 0; i < outputLayer.size(); i++)
      outputs[i] = outputLayer.get(i).getOutput();
    return outputs;
  }

  /**
   * Calculate the output of the neural network based on the input The forward
   * operation
   */
  private void activate() {
    for (Neuron n : hiddenLayer)
      n.calculateOutput();
    for (Neuron n : outputLayer)
      n.calculateOutput();
  }

  /**
   * all output propagate back
   *
   * @param expectedOutput
   *            first calculate the partial derivative of the error with
   *            respect to each of the weight leading into the output neurons
   *            bias is also updated here
   */
  private void applyBackpropagation(double expectedOutput[]) {

    // error check, normalize value ]0;1[
    for (int i = 0; i < expectedOutput.length; i++) {
      double d = expectedOutput[i];
      if (d < 0 || d > 1) {
        System.out.println("IMPORTANT: correction in expected values");
        if (d < 0)
          expectedOutput[i] = 0 + EPSILON;
        else
          expectedOutput[i] = 1 - EPSILON;
      }
    }

    int i = 0;
    for (Neuron n : outputLayer) {
      ArrayList<Connection> connections = n.getAllInConnections();
      for (Connection con : connections) {
        double ak = n.getOutput();
        double ai = con.getFromNeuron().getOutput();
        double desiredOutput = expectedOutput[i];

        double partialDerivative = -ak * (1 - ak) * ai
          * (desiredOutput - ak);
        double deltaWeight = -LEARNING_RATE * partialDerivative;
        double newWeight = con.getWeight() + deltaWeight;
        con.setDeltaWeight(deltaWeight);
        con.setWeight(newWeight + MOMENTUM * con.getPrevDeltaWeight());
      }
      i++;
    }

    // update weights for the hidden layer
    for (Neuron n : hiddenLayer) {
      ArrayList<Connection> connections = n.getAllInConnections();
      for (Connection con : connections) {
        double aj = n.getOutput();
        double ai = con.getFromNeuron().getOutput();
        double sumKoutputs = 0;
        int j = 0;
        for (Neuron out_neu : outputLayer) {
          double wjk = out_neu.getConnection(n.getId()).getWeight();
          double desiredOutput = (double) expectedOutput[j];
          double ak = out_neu.getOutput();
          j++;
          sumKoutputs = sumKoutputs
            + (-(desiredOutput - ak) * ak * (1 - ak) * wjk);
        }

        double partialDerivative = aj * (1 - aj) * ai * sumKoutputs;
        double deltaWeight = -LEARNING_RATE * partialDerivative;
        double newWeight = con.getWeight() + deltaWeight;
        con.setDeltaWeight(deltaWeight);
        con.setWeight(newWeight + MOMENTUM * con.getPrevDeltaWeight());
      }
    }
  }

  /**
   * Train neural network.
   * @param maxSteps if error does not coincide, max steps is a limit for iterations
   * @param maxError expected error rate
   * @return squered error
   */
  public double run(int maxSteps, double maxError, boolean printGnuplot) throws Exception {
    PrintWriter foutError = null;
    PrintWriter foutAccuracy = null;

    int i;
    double error = 1;

    try {
      if (printGnuplot) {
        foutError = new PrintWriter(new FileWriter("plot_error.dat"));
        foutError.println("#\tEPOCH\tERROR");
        if (testInputs != null) {
          foutAccuracy = new PrintWriter(new FileWriter("plot_accuracy.dat"));
          foutAccuracy.println("#\tEPOCH\tACCURACY");
        }
      }
      // Train neural network until maxError reached or maxSteps exceeded
      for (i = 0; i < maxSteps && error > maxError; i++) {
        error = 0;
        for (int p = 0; p < inputs.length; p++) {
          setInput(inputs[p]);

          activate();

          output = getOutput();
          resultOutputs[p] = output;

          for (int j = 0; j < expectedOutputs[p].length; j++) {
            double err = Math.pow((output[j] < 0.5 ? 0.0 : 1.0) - expectedOutputs[p][j], 2);
            error += err;
          }

          applyBackpropagation(expectedOutputs[p]);
        }
        if (printGnuplot) {
          foutError.println("\t" + i + "\t" + error);
          if (testInputs != null) {
            double accuracy = test(maxError, testInputs, testExpectedOutputs, false);
            foutAccuracy.println("\t" + i + "\t" + accuracy);
          }
        }
      }
    } catch (Exception ex) {
      throw ex;
    } finally {
      if (foutError != null) {
        foutError.close();
      }
      if (foutAccuracy != null) {
        foutAccuracy.close();
      }
    }

    printResult();

    System.out.println("Sum of squared errors = " + error);
    System.out.println("##### EPOCH " + i+"\n");
    if (i == maxSteps) {
      System.out.println("!Error training try again");
    } else {
      printAllWeights();
      printWeightUpdate();
    }
    return error;
  }

  /**
   * When neural network was trained, test it with testing dataset.
   * @param minError expected max error
   * @param testInputs input dataset for testing neural network
   * @param testExpectedOutputs expected output for test dataset
   * @return percent of correctly classified observations
   */
  public double test(double minError, double[][] testInputs, double[][] testExpectedOutputs, boolean printResults) {
    int correct = 0;
    for (int p=0; p<testInputs.length; p++) {
      setInput(testInputs[p]);
      activate();
      output = getOutput();

      double err = 0.0;
      for(int j=0; j<testExpectedOutputs[p].length; j++) {
        err += Math.pow((output[j] < 0.5 ? 0.0 : 1.0) - testExpectedOutputs[p][j], 2);
      }

      if (printResults) {
        System.out.print("INPUTS: ");
        for (int x = 0; x < testInputs[p].length; x++) {
          System.out.print(testInputs[p][x] + " ");
        }

        System.out.print("EXPECTED: ");
        for (int x = 0; x < testExpectedOutputs[p].length; x++) {
          System.out.print(testExpectedOutputs[p][x] + " ");
        }

        System.out.print("ACTUAL: ");
        for (double anOutput : output) {
          System.out.print((anOutput < 0.5 ? 0.0 : 1.0) + " ");
        }
        System.out.print("ERR: " + err);
        System.out.println();
      }
      if (err >= -minError && err <= minError) {
        correct++;
      }
    }
    double accuracy = ((double)correct/(double) testExpectedOutputs.length)*100.0;
    if (printResults) {
      System.out.println("CORRECTLY CLASSIFIED: " + df.format(accuracy) + "%");
    }
    return accuracy;
  }

  public void printResult() {
    System.out.println("NN EXECUTION RESULTS");
    for (int p = 0; p < inputs.length; p++) {
      System.out.print("INPUTS: ");
      for (int x = 0; x < layers[0]; x++) {
        System.out.print(inputs[p][x] + " ");
      }

      System.out.print("EXPECTED: ");
      for (int x = 0; x < layers[2]; x++) {
        System.out.print(expectedOutputs[p][x] + " ");
      }

      System.out.print("ACTUAL: ");
      for (int x = 0; x < layers[2]; x++) {
        System.out.print(resultOutputs[p][x] + " ");
      }
      System.out.println();
    }
    System.out.println();
  }

  String weightKey(int neuronId, int conId) {
    return "N" + neuronId + "_C" + conId;
  }

  private void printWeightUpdate() {
    System.out.println("printWeightUpdate, put this i trainedWeights() and set isTrained to true");
    // weights for the hidden layer
    for (Neuron n : hiddenLayer) {
      ArrayList<Connection> connections = n.getAllInConnections();
      for (Connection con : connections) {
        String w = df.format(con.getWeight());
        System.out.println("weightUpdate.put(weightKey(" + n.getId() + ", "
          + con.getId() + "), " + w + ");");
      }
    }
    // weights for the output layer
    for (Neuron n : outputLayer) {
      ArrayList<Connection> connections = n.getAllInConnections();
      for (Connection con : connections) {
        String w = df.format(con.getWeight());
        System.out.println("weightUpdate.put(weightKey(" + n.getId() + ", "
          + con.getId() + "), " + w + ");");
      }
    }
    System.out.println();
  }

  public void printAllWeights() {
    System.out.println("printAllWeights");
    // weights for the hidden layer
    for (Neuron n : hiddenLayer) {
      ArrayList<Connection> connections = n.getAllInConnections();
      for (Connection con : connections) {
        double w = con.getWeight();
        System.out.println("n=" + n.getId() + " c=" + con.getId() + " w=" + w);
      }
    }
    // weights for the output layer
    for (Neuron n : outputLayer) {
      ArrayList<Connection> connections = n.getAllInConnections();
      for (Connection con : connections) {
        double w = con.getWeight();
        System.out.println("n=" + n.getId() + " c=" + con.getId() + " w=" + w);
      }
    }
    System.out.println();
  }
}