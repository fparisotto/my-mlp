import breeze.linalg.{DenseMatrix, sum}
import breeze.stats.distributions.Gaussian

import scala.util.Random


case class HardcodedSimpleMLP(learningRate: Double = 0.1, random: Random = new Random()) {

  val g = Gaussian(0, 0.25)

  // random init weights between -0.25, 0.25
  val weight0: DenseMatrix[Double] = DenseMatrix.fill[Double](2, 4)(g.sample())
  val weight1: DenseMatrix[Double] = DenseMatrix.fill[Double](4, 1)(g.sample())

  def fit(feature: Array[Double], label: Array[Double]): Double = {
    // TODO input validation

    val _feature: DenseMatrix[Double] = DenseMatrix.create(1, feature.length, feature)
    val _label: DenseMatrix[Double] = DenseMatrix.create(1, label.length, label)

    // feed forward through layers
    // layer 0 is the input layer
    val layer0: DenseMatrix[Double] = _feature
    val layer1: DenseMatrix[Double] = (layer0 * weight0).map(math.tanh)
    val layer2: DenseMatrix[Double] = (layer1 * weight1).map(math.tanh)

    // back propagation
    // calculate the error of prediction (layer2)
    val layer2Error: DenseMatrix[Double] = _label - layer2
    // in what direction
    val layer2Delta: DenseMatrix[Double] = layer2Error *:* layer2.map(e => 1.0 - math.pow(e, 2))

    // how much did layer1 error contribute to prediction error
    val layer1Error: DenseMatrix[Double] = layer2Delta * weight1.t
    // in what direction
    val layer1Delta: DenseMatrix[Double] = layer1Error *:* layer1.map(e => 1.0 - math.pow(e, 2))

    // update the weights
    weight1 += layer1.t * layer2Delta *:* learningRate
    weight0 += layer0.t * layer1Delta *:* learningRate

    val rmse: Double = {
      val mean = sum(layer2Error) / layer2Delta.size
      math.sqrt(math.pow(mean, 2))
    }
    rmse
  }

  def predict(feature: Array[Double]): Array[Double] = {
    val layer0: DenseMatrix[Double] = DenseMatrix.create(1, 2, feature)
    val layer1: DenseMatrix[Double] = (layer0 * weight0).map(math.tanh)
    val layer2: DenseMatrix[Double] = (layer1 * weight1).map(math.tanh)
    layer2.toArray
  }

}

object TestHardCodedMlp extends App {

  def training(mlp: HardcodedSimpleMLP,
               samples: (Array[Array[Double]], Array[Array[Double]]),
               epoch: Int = 2500,
               debugStep: Int = 1000): Unit = {
    (0 to epoch).foreach { i =>
      var rmse = 0.0
      val featuresWithLabels = Random.shuffle(samples._1.zip(samples._2).toList)
      featuresWithLabels.foreach {
        case (feature, label) =>
          rmse = mlp.fit(feature, label)
      }
      if (i % debugStep == 0)
        println(f"rmse=$rmse%1.6f")
    }
  }

  {
    val features = Array(Array(0.0, 0.0), Array(1.0, 0.0), Array(1.0, 1.0), Array(0.0, 1.0))
    val labels = Array(Array(0.0), Array(1.0), Array(1.0), Array(1.0))
    val samples = (features, labels)
    println("Learning the OR function")
    val mlp = HardcodedSimpleMLP()
    training(mlp, samples)
    println("Predictions:")
    samples._1.zip(samples._2).foreach { case (_features, _labels) =>
      val prediction = mlp.predict(_features)
      println(s"features=${_features.mkString(",")}; prediction=${prediction.mkString(";")}, labels=${_labels.mkString(";")}")
    }
  }

  {
    val features = Array(Array(0.0, 0.0), Array(1.0, 0.0), Array(1.0, 1.0), Array(0.0, 1.0))
    val labels = Array(Array(0.0), Array(0.0), Array(1.0), Array(0.0))
    val samples = (features, labels)
    println("Learning the AND function")
    val mlp = HardcodedSimpleMLP()
    training(mlp, samples)
    println("Predictions:")
    samples._1.zip(samples._2).foreach { case (_features, _labels) =>
      val prediction = mlp.predict(_features)
      println(s"features=${_features.mkString(",")}; prediction=${prediction.mkString(";")}, labels=${_labels.mkString(";")}")
    }
  }

  {
    val features = Array(Array(0.0, 0.0), Array(1.0, 0.0), Array(1.0, 1.0), Array(0.0, 1.0))
    val labels = Array(Array(0.0), Array(1.0), Array(0.0), Array(1.0))
    val samples = (features, labels)
    println("Learning the XOR function")
    val mlp = HardcodedSimpleMLP()
    training(mlp, samples)
    println("Predictions:")
    samples._1.zip(samples._2).foreach { case (_features, _labels) =>
      val prediction = mlp.predict(_features)
      println(s"features=${_features.mkString(",")}; prediction=${prediction.mkString(";")}, labels=${_labels.mkString(";")}")
    }
  }

}
