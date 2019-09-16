import breeze.linalg.DenseMatrix

object BasicTest extends App {

  def test(mlp: MultiLayerPerceptron, features: List[Array[Double]], labels: List[Array[Double]]): Unit = {
    val _features = features.map(d => DenseMatrix.create(2, 1, d))
    val _labels = labels.map(d => DenseMatrix.create(2, 1, d))
    val data = _features.zip(_labels)
    mlp.sgdTraining(data, testData = Option(data))
    data.foreach { case (feature, label) =>
      val prediction = mlp.predict(feature)
      val p = prediction.data.toList
      val l = label.data.toList
      val predictionIndex = p.indexOf(p.max)
      val labelIndex = l.indexOf(l.max)
      println(s"r=${predictionIndex == labelIndex} feature=$feature, label=$label, prediction=$prediction")
    }
  }

  {
    val mlp = new MultiLayerPerceptron(2, 4, 2)
    val features = List(Array(0.0, 0.0), Array(1.0, 0.0), Array(1.0, 1.0), Array(0.0, 1.0))
    val labels = List(Array(1.0, 0.0), Array(0.0, 1.0), Array(0.0, 1.0), Array(0.0, 1.0))
    println("Learning the OR function")
    test(mlp, features, labels)
  }

  {
    val mlp = new MultiLayerPerceptron(2, 4, 2)
    val features = List(Array(0.0, 0.0), Array(1.0, 0.0), Array(1.0, 1.0), Array(0.0, 1.0))
    val labels = List(Array(1.0, 0.0), Array(1.0, 0.0), Array(0.0, 1.0), Array(1.0, 0.0))
    println("Learning the AND function")
    test(mlp, features, labels)
  }

  {
    val mlp = new MultiLayerPerceptron(2, 4, 2)
    val features = List(Array(0.0, 0.0), Array(1.0, 0.0), Array(1.0, 1.0), Array(0.0, 1.0))
    val labels = List(Array(1.0, 0.0), Array(0.0, 1.0), Array(1.0, 0.0), Array(0.0, 1.0))
    println("Learning the XOR function")
    test(mlp, features, labels)
  }
}
