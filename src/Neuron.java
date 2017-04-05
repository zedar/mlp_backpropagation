import java.util.*;

public class Neuron {
  private static int counter = 0;

  private final int id;  // auto increment, starts at 0
  private Connection biasConnection;
  private final double bias = -1;
  private double output;

  private final ArrayList<Connection> inConnections = new ArrayList<Connection>();
  private final HashMap<Integer,Connection> connectionLookup = new HashMap<Integer,Connection>();

  public Neuron(){
    id = counter;
    counter++;
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
    s = s + (biasConnection.getWeight()*bias);

    output = g(s);
  }


  private double g(double x) {
    return sigmoid(x);
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
}