package mlp;

import mlp.backpropagation.NeuralNetwork;

import javax.xml.crypto.Data;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {
  public static void main(String[] args) throws Exception {
    Config cfg = Config.build(args);
    if (!(cfg.getInLayerN() > 0)) {
      System.out.println("Input layer must have at least 1 neuron.");
      return;
    }
    if (!(cfg.getOutLayerN() > 0)) {
      System.out.println("Output layer must have at least 1 neuron.");
      return;
    }
    if (!(cfg.getHiddenLayerN() > 0)) {
      System.out.println("Hidden layer must have at least 1 neuron.");
      return;
    }
    if (cfg.getTrainDSPath() == null) {
      System.out.println("Undefined train data file.");
      return;
    }
    double[][][] trainDS = DataSetUtils.loadDS(
      cfg.getTrainDSPath(),
      cfg.getInLayerN(), cfg.getInLayerF(),
      cfg.getOutLayerN(), cfg.getOutLayerF(), cfg.getOutFeatureAs01());
    double[][] inDS = trainDS[0];
    double[][] outDS = trainDS[1];
    double[][] inTestDS = null;
    double[][] outTestDS = null;

    DataSetUtils.printDS("INPUT LAYER", inDS, 6);
    DataSetUtils.printDS("OUTPUT LAYER", outDS, 6);

    // Normalize values in the range of 0-1 for the good results
    if (cfg.getNormalizeFeatures()) {
      DataSetUtils.normalizeInRangeZeroOne(inDS);
      DataSetUtils.normalizeInRangeZeroOne(outDS);
      DataSetUtils.printDS("INPUT LAYER (NORMALIZED)", inDS, 6);
      DataSetUtils.printDS("OUTPUT LAYER (NORMALIZED)", outDS, 6);
    }

    if (cfg.getTestDSPath() != null) {
      double[][][] testDS = DataSetUtils.loadDS(
        cfg.getTestDSPath(),
        cfg.getInLayerN(), cfg.getInLayerF(),
        cfg.getOutLayerN(), cfg.getOutLayerF(), cfg.getOutFeatureAs01());
      inTestDS = testDS[0];
      outTestDS = testDS[1];

      // Normalize values in the range of 0-1 for the good results
      if (cfg.getNormalizeFeatures()) {
        DataSetUtils.normalizeInRangeZeroOne(inTestDS);
        DataSetUtils.normalizeInRangeZeroOne(outTestDS);
      }
    }

    NeuralNetwork nn = new NeuralNetwork(
      cfg.getInLayerN(),
      cfg.getHiddenLayerN(),
      cfg.getOutLayerN(),
      inDS,
      outDS,
      inTestDS,
      outTestDS);

    int maxRuns = cfg.getMaxEpoch();
    double minErrorCondition = 0.001;

    double error = nn.run(maxRuns, minErrorCondition, true, cfg.getPrintError(), cfg.getPrintApproximation());
    double accuracy = 0;
    if (cfg.getTestDSPath() != null) {
      accuracy = nn.test(minErrorCondition, inTestDS, outTestDS, true);
      System.out.printf("TEST FINAL ACCURACY: %f\n", accuracy);
    }
  }
}
