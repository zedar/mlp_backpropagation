package mlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

public class DataSetUtils {
  /**
   * Load dataset from the path. The file should be in CSV format with space as a delimiter.
   * @param dsPath path to file with dataset
   * @param inN the number of neurons in input layer
   * @param inFs the indexes of features for input layer
   * @param outN the number of output layer neurons
   * @param outFs the indexes of features for output layer
   * @param outFeatureAs01 use output features as 0101 featurtes. Map its descriptive value to 0 and 1. 1=(1,0,0); 2=(0,1,0); etc.
   * @return index [0] => input dataset; index [1] => output dataset
   * @throws Exception any
   */
  public static double[][][] loadDS(String dsPath, int inN, int[] inFs, int outN, int[] outFs, boolean outFeatureAs01) throws Exception {
    ArrayList<double[]> in = new ArrayList<>();
    ArrayList<double[]> out = new ArrayList<>();

    // we assume that input file is in csv format with space as delimiter
    BufferedReader br = new BufferedReader(new FileReader(dsPath));
    String line;
    while ((line = br.readLine()) != null) {
      String[] features = line.split(" ");
      if (features.length < inN || features.length < outN) {
        throw new IllegalArgumentException("DataSet has invalid number of features");
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

      if (outFeatureAs01) {
        int outv = Integer.valueOf(features[outFs[0]-1]);
        if (outv < 0 || outv > outN) {
          throw new IllegalArgumentException("Invalid output feature value. Not inline with number of output neurons");
        }
        double[] outf = new double[outN];
        Arrays.fill(outf, 0.0);
        outf[outv-1] = 1.0;
        out.add(outf);
      } else {
        double[] outf = new double[outN];
        for (int i=0; i<outN; i++) {
          int pos = i;
          if (outFs != null && i < outFs.length) {
            pos = outFs[i]-1;
          }
          outf[i] = Double.valueOf(features[pos]);
        }
        out.add(outf);
      }
    }
    return new double[][][] {in.toArray(new double[][]{{}}), out.toArray(new double[][]{{}})};
  }

  /**
   * Print dataset to the standard output
   * @param prompt anything with meaning
   * @param ds columns and rows from dataset
   * @param head number of first rows to print. If null all rows print.
   */
  public static void printDS(String prompt, double[][] ds, Integer head) {
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

  /**
   * Normalize values from the dataset to [0,1] range.
   * @param ds
   */
  public static void normalizeInRangeZeroOne(double[][] ds) {
    double min = 0.0;
    double max = 0.0;
    for (int i = 0; i < ds.length; i++) {
      double d = ds[i][0];
      if (d < min) {
        min = d;
      }
      if (d > max) {
        max = d;
      }
    }
    System.out.printf("NORMALIZE: MIN: %f, MAX: %f\n", min, max);
    for (int i = 0; i < ds.length; i++) {
      //data[i][0] = (data[i][0] - min) / (max - min);
      // (b-a)(x-min)/(max-min) + a
      double a = 0.0;
      double b = 1.0;
      ds[i][0] = (b-a)*(ds[i][0] - min) / (max - min) + a;
    }
  }
}
