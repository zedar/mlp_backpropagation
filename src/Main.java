import backpropagation.JavaMLP;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {
  public static void main(String[] args) throws Exception {
    Config cfg = Config.build(args);
    if (!(cfg.getInLayerN() > 0 && cfg.getInLayerN() <= 4)) {
      System.out.println("Input layer must be 1-4 neurons");
      return;
    }
    if (!(cfg.getOutLayerN() > 0 && cfg.getOutLayerN() <= 3)) {
      System.out.println("Output layer must have 1-3 neurons.");
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
    double[][][] trainDS = loadDS(cfg.getTrainDSPath(), cfg.getInLayerN(), cfg.getInLayerF(), cfg.getOutLayerN());
    double[][] inDS = trainDS[0];
    double[][] outDS = trainDS[1];
    double[][] inTestDS = null;
    double[][] outTestDS = null;

    if (cfg.getTestDSPath() != null) {
      double[][][] testDS = loadDS(cfg.getTestDSPath(), cfg.getInLayerN(), cfg.getInLayerF(), cfg.getOutLayerN());
      inTestDS = testDS[0];
      outTestDS = testDS[1];
    }

    printDS("INPUT LAYER", inDS, 6);
    printDS("OUTPUT LAYER", outDS, 6);

    variant1(cfg, inDS, outDS, inTestDS, outTestDS);

  }

  private static void variant1(Config cfg, double[][] inDS, double[][] outDS, double[][] inTestDS, double[][] outTestDS) throws Exception {
    NeuralNetwork nn = new NeuralNetwork(
        cfg.getInLayerN(),
        cfg.getHiddenLayerN(),
        cfg.getOutLayerN(),
        inDS,
        outDS,
        inTestDS,
        outTestDS);

    int maxRuns = 150000;//50000;
    double minErrorCondition = 0.001;

    double error = nn.run(maxRuns, minErrorCondition, true);
    double accuracy = 0;
    if (cfg.getTestDSPath() != null) {
      System.out.println("NN TEST FINAL ACCURACY");
      accuracy = nn.test(minErrorCondition, inTestDS, outTestDS, true);
    }
  }

  private static void variant2(Config cfg, double[][] inDS, double[][] outDS, double[][] inTestDS, double[][] outTestDS) throws Exception {
    double[] expectedOutputs = new double[outDS.length];
    for (int i = 0; i < outDS.length; i++) {
      expectedOutputs[i] = outDS[i][0];
    }

    JavaMLP.run(cfg.getInLayerN(), cfg.getHiddenLayerN(), cfg.getOutLayerN(), inDS, expectedOutputs, inTestDS, outTestDS);

  }

  private static double[][][] loadDS(String trainDSPath, int inN, int[] inFs, int outN) throws Exception {
    ArrayList<double[]> in = new ArrayList<>();
    ArrayList<double[]> out = new ArrayList<>();

    // we assume that input file is in csv format with space as delimiter. First 4 values are features while 5th is label.
    BufferedReader br = new BufferedReader(new FileReader(trainDSPath));
    String line;
    while ((line = br.readLine()) != null) {
      String[] features = line.split(" ");
      if (features.length < inN+1) {
        throw new IllegalArgumentException("train DS has invalid number of features");
      }
      double[] inf = new double[inN];
      for (int i=0; i<inN; i++) {
        int pos = i;
        if (inFs != null && i < inFs.length) {
          pos = inFs[i]-1;
        }
        inf[i] = Double.valueOf(features[pos]);
      }
      in.add(inf);

//      int outv = Integer.valueOf(features[features.length-1]);
//      if (outv < 0 || outv > outN) {
//        throw new IllegalArgumentException("Invalid output feature value. Not inline with number of output neurons");
//      }
//      double[] outf = new double[outN];
//      Arrays.fill(outf, 0.0);
//      outf[outv-1] = 1.0;
      double[] outf = new double[outN];
      Arrays.fill(outf, 0.0);
      outf[0] = Double.valueOf(features[features.length-1]);
      out.add(outf);
    }
    return new double[][][] {in.toArray(new double[][]{{}}), out.toArray(new double[][]{{}})};
  }

  private static void printDS(String prompt, double[][] ds, Integer head) {
    System.out.println("-------------------------");
    System.out.println(prompt);
    if (head == null) head = ds.length;
    for (int i=0; i<ds.length && i<head; i++) {
      StringBuilder sb = new StringBuilder();
      sb.append(i).append(" : ");
      for (int j=0; j<ds[i].length; j++) {
        sb.append(ds[i][j]).append(" ");
      }
      System.out.println(sb.toString());
    }
  }
}
