import java.util.*;

public class Neuron {
  private static int counter = 0;

  private int activationFunc = 1; // 1-sigmoid, 2-identity
  private final int id;  // auto increment, starts at 0
  private Connection biasConnection;
  private double bias = -1;
  private double output = 0.0;

  private final ArrayList<Connection> inConnections = new ArrayList<Connection>();
  private final HashMap<Integer,Connection> connectionLookup = new HashMap<Integer,Connection>();

  public Neuron(){
    id = counter;
    counter++;
  }

  public Neuron(int activationFunc, double bias) {
    this();
    this.activationFunc = activationFunc;
    this.bias = bias;
  }

  public static void resetCounter() {
    counter = 0;
  }

  public int getId() {
    return id;
  }

  /**
   * Compute Sj = Wij*Aij + w0j*bias
   */
  public void calculateOutput(){
    double s = 0;
    for(Connection con : inConnections){
      Neuron leftNeuron = con.getFromNeuron();
      double weight = con.getWeight();
      double a = leftNeuron.getOutput(); //output from previous layer

      s = s + (weight*a);
    }
    if (biasConnection != null) {
      s = s + (biasConnection.getWeight() * bias);
    }

    output = g(s);
  }


  private double g(double x) {
    if (activationFunc == 1) {
      return sigmoid(x);
    } else {
      return x;
    }

  }

  private double sigmoid(double x) {
    return 1.0 / (1.0 +  (Math.exp(-x)));
  }

  public void addInConnections(ArrayList<Neuron> inNeurons){
    for(Neuron n: inNeurons){
      Connection con = new Connection(n,this);
      inConnections.add(con);
      connectionLookup.put(n.id, con);
    }
  }

  public Connection getConnection(int neuronIndex){
    return connectionLookup.get(neuronIndex);
  }

  public void addBiasConnection(Neuron n){
    Connection con = new Connection(n,this);
    biasConnection = con;
    inConnections.add(con);
  }

  public ArrayList<Connection> getAllInConnections(){
    return inConnections;
  }

  public double getBias() {
    return bias;
  }
  public double getOutput() {
    return output;
  }
  public void setOutput(double o){
    output = o;
  }
  public int getActivationFunc() {
    return activationFunc;
  }

}