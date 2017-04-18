package mlp;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Config {
  private final int inLayerN;
  private final int[] inLayerF;
  private final int hiddenLayerN;
  private final int outLayerN;
  private final int[] outLayerF;
  private final String trainDSPath;
  private final String testDSPath;
  private final boolean outFeatureAs01;
  private final boolean printError;
  private final boolean printApproximation;
  private final int maxEpoch;
  private final boolean normalizeFeatures;

  private Config(
    Integer inN, Integer hiddenN, Integer outN,
    int[] inLayerF, int[] outLayerF,
    String trainDSPath, String testDSPath,
    boolean outAs01,
    boolean printError, boolean printApproximation,
    int maxEpoch,
    boolean normalizeFeatures) {
    if (inN != null) inLayerN = inN; else inLayerN = 2;
    if (hiddenN != null) hiddenLayerN = hiddenN; else hiddenLayerN = 4;
    if (outN != null) outLayerN = outN; else outLayerN = 3;
    this.inLayerF = inLayerF;
    this.outLayerF = outLayerF;
    this.trainDSPath = trainDSPath;
    this.testDSPath = testDSPath;
    this.outFeatureAs01 = outAs01;
    this.printError = printError;
    this.printApproximation = printApproximation;
    this.maxEpoch = maxEpoch;
    this.normalizeFeatures = normalizeFeatures;
  }

  public static Config build(String[] args) {
    Map<String,String> argsm = parseArgs(args);
    String in = argsm.get("in");
    Integer inN = in != null ? Integer.valueOf(in) : null;
    String inF = argsm.get("inf");
    int[] inFi = null;
    if (inF != null) {
      String[] inFs = inF.split(",");
      inFi = Arrays.stream(inFs).map(Integer::valueOf).mapToInt(Integer::intValue).toArray();
    }
    String outF = argsm.get("outf");
    int[] outFi = null;
    if (outF != null) {
      String[] outFs = outF.split(",");
      outFi = Arrays.stream(outFs).map(Integer::valueOf).mapToInt(Integer::intValue).toArray();
    }
    String hidden = argsm.get("hidden");
    Integer hiddenN = hidden != null ? Integer.valueOf(hidden) : null;
    String out = argsm.get("out");
    Integer outN = out != null ? Integer.valueOf(out) : null;
    String trainDSPath = argsm.get("trainDSPath");
    String testDSPath = argsm.get("testDSPath");

    String outFeatureAs01 = argsm.get("outFeatureAs01");
    boolean outfas01 = (outFeatureAs01 != null) ? Boolean.valueOf(outFeatureAs01) : false;

    String sPrintError = argsm.get("printError");
    boolean printError = (sPrintError != null) ? Boolean.valueOf(sPrintError) : false;
    String sPrintApproximation = argsm.get("printApproximation");
    boolean printApproximation = (sPrintApproximation != null) ? Boolean.valueOf(sPrintApproximation) : false;

    String sMaxEpoch = argsm.get("maxEpoch");
    int maxEpoch = (sMaxEpoch != null) ? Integer.valueOf(sMaxEpoch) : 50000;

    String sNormalizeFeatures = argsm.get("normalizeFeatures");
    boolean normalizeFeatures = (sNormalizeFeatures != null) ? Boolean.valueOf(sNormalizeFeatures) : true;

    return new Config(inN, hiddenN, outN, inFi, outFi, trainDSPath, testDSPath, outfas01, printError, printApproximation, maxEpoch, normalizeFeatures);
  }

  public int getInLayerN() {
    return inLayerN;
  }

  public int[] getInLayerF() {
    return inLayerF;
  }

  public int[] getOutLayerF() {
    return outLayerF;
  }

  public int getHiddenLayerN() {
    return hiddenLayerN;
  }

  public int getOutLayerN() {
    return outLayerN;
  }

  public String getTrainDSPath() {
    return trainDSPath;
  }

  public String getTestDSPath() {
    return testDSPath;
  }

  public boolean getOutFeatureAs01() {
    return outFeatureAs01;
  }

  public boolean getPrintError() {
    return printError;
  }

  public boolean getPrintApproximation() {
    return printApproximation;
  }

  public int getMaxEpoch() {
    return maxEpoch;
  }

  public boolean getNormalizeFeatures() {
    return normalizeFeatures;
  }

  private static Map<String, String> parseArgs(String[] args) {
    Map<String, String> argsm = new HashMap<>();
    for (String arg : args) {
      if (arg.contains("=")) {
        int idx = arg.indexOf("=");
        argsm.put(arg.substring(0, idx), arg.substring(idx+1));
      }
    }
    return argsm;
  }
}
