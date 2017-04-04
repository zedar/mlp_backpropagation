import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Config {
  private int inLayerN = 2;
  private int[] inLayerF;
  private int hiddenLayerN = 4;
  private int outLayerN = 3;
  private String trainDSPath;
  private String testDSPath;

  private Config(Integer inN, Integer hiddenN, Integer outN, String trainDSPath, String testDSPath) {
    if (inN != null) inLayerN = inN;
    if (hiddenN != null) hiddenLayerN = hiddenN;
    if (outN != null) outLayerN = outN;
    this.trainDSPath = trainDSPath;
    this.testDSPath = testDSPath;
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
    String hidden = argsm.get("hidden");
    Integer hiddenN = hidden != null ? Integer.valueOf(hidden) : null;
    String out = argsm.get("out");
    Integer outN = out != null ? Integer.valueOf(out) : null;
    String trainDSPath = argsm.get("trainDSPath");
    String testDSPath = argsm.get("testDSPath");
    return new Config(inN, hiddenN, outN, trainDSPath, testDSPath);
  }

  public int getInLayerN() {
    return inLayerN;
  }

  public int[] getInLayerF() {
    return inLayerF;
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
