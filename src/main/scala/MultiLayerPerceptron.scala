import java.util.logging.Logger

import breeze.linalg.{DenseMatrix, _}
import breeze.stats.distributions.Gaussian

import scala.collection.mutable
import scala.util.Random

class MultiLayerPerceptron(shape: Int*) {

  private val logger: Logger = Logger.getLogger(getClass.getName)

  type Feature = DenseMatrix[Double]
  type Label = DenseMatrix[Double]
  type FeatureAndLabel = (Feature, Label)

  val g = Gaussian(0, 1)

  var weights: List[DenseMatrix[Double]] = {
    val dimensions = shape.slice(1, shape.size).zip(shape.slice(0, shape.size - 1)).toList
    dimensions.map { case (rows, cols) => DenseMatrix.fill[Double](rows, cols)(g.sample()) }
  }

  var biases: List[DenseMatrix[Double]] = {
    val dimensions = shape.slice(1, shape.size).toList
    dimensions.map { case (rows) => DenseMatrix.fill[Double](rows, 1)(g.sample()) }
  }

  def sigmoid(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    1.0 / (1.0 +:+ m.map(e => Math.exp(-e)))
  }

  def derivativeSigmoid(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    sigmoid(m) *:* (1.0 - sigmoid(m))
  }

  def costDerivative(activation: DenseMatrix[Double], label: DenseMatrix[Double]): DenseMatrix[Double] = {
    activation - label
  }

  def predict(feature: Feature): Label = {
    biases.zip(weights).foldLeft(feature) {
      case (activation, (bias, weight)) =>
        sigmoid((weight * activation) + bias)
    }
  }

  def sgdTraining(trainingData: List[FeatureAndLabel],
                  epochs: Int = 2500,
                  batchSize: Int = 1,
                  learningRate: Double = 0.1,
                  testData: Option[List[FeatureAndLabel]] = None,
                  debugFrequency: Int = 500): Unit = {
    (0 until epochs).foreach { i =>
      val shuffle = Random.shuffle(trainingData)
      shuffle.grouped(batchSize).foreach(batch => fitBatch(batch, learningRate))
      testData match {
        case None if i % debugFrequency == 0 =>
          logger.info(s"Epoch $i complete")
        case Some(data) if i % debugFrequency == 0 =>
          logger.info(s"Epoch $i complete, evaluation = ${evaluate(data)} / ${data.size}")
        case _ => ()
      }
    }
  }

  def evaluate(testData: List[FeatureAndLabel]): Int = {
    val withPredictions = testData.map {
      case (feature, label) => (feature, label, predict(feature))
    }

    val score = withPredictions.map {
      case (_, label, prediction) => if (argmax(prediction) == argmax(label)) 1 else 0
    }.sum

    score
  }

  def fitBatch(batch: List[FeatureAndLabel], learningRate: Double): Unit = {
    val zeroBiases: List[DenseMatrix[Double]] = biases.map(bias => DenseMatrix.zeros[Double](bias.rows, bias.cols))
    val zeroWeights: List[DenseMatrix[Double]] = weights.map(weight => DenseMatrix.zeros[Double](weight.rows, weight.cols))

    val (gradientBiases, gradientWeights) = batch.foldLeft((zeroBiases, zeroWeights)) {
      case ((accBiases, accWeights), (feature, label)) =>
        val (deltaBias, deltaWeight) = gradientBiasAndWeight(feature, label)
        val gradientBiases = accBiases.zip(deltaBias).map { case (acc, delta) => acc + delta }
        val gradientWeights = accWeights.zip(deltaWeight).map { case (acc, delta) => acc + delta }
        (gradientBiases, gradientWeights)
    }

    weights = weights.zip(gradientWeights).map { case (weight, gradient) => weight - (learningRate / batch.size) *:* gradient }
    biases = biases.zip(gradientBiases).map { case (bias, gradient) => bias - (learningRate / batch.size) *:* gradient }
  }

  def gradientBiasAndWeight(feature: Feature, label: Label): (List[DenseMatrix[Double]], List[DenseMatrix[Double]]) = {
    var activation = feature
    val activations: mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer(activation)
    val layersOutput: mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()

    biases.zip(weights).foreach {
      case (bias, weight) =>
        val layerOutput = (weight * activation) + bias
        layersOutput += layerOutput
        activation = sigmoid(layerOutput)
        activations += activation
    }

    val gradientBiases: mutable.Buffer[DenseMatrix[Double]] =
      biases.map(bias => DenseMatrix.zeros[Double](bias.rows, bias.cols)).toBuffer

    val gradientWeights: mutable.Buffer[DenseMatrix[Double]] =
      weights.map(weight => DenseMatrix.zeros[Double](weight.rows, weight.cols)).toBuffer

    var delta = costDerivative(activations.last, label) *:* derivativeSigmoid(layersOutput.last)
    gradientBiases(gradientBiases.length - 1) = delta
    gradientWeights(gradientWeights.length - 1) = delta * activations(activations.size - 2).t

    (2 until shape.size).foreach { i =>
      val layerOutput = layersOutput(layersOutput.size - i)
      delta = (weights(weights.size - i + 1).t * delta) *:* derivativeSigmoid(layerOutput)
      gradientBiases(gradientBiases.size - i) = delta
      gradientWeights(gradientWeights.size - i) = delta * activations(activations.size - i - 1).t
    }

    (gradientBiases.toList, gradientWeights.toList)
  }

}
