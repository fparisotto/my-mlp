import java.util.logging.Logger

import breeze.linalg.DenseMatrix

import scala.io.Source

object MnistTest extends App {

  val logger = Logger.getLogger(getClass.getName)

  def loadMnist(filePath: String): List[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val labelTemplate = (0 until 10).map(_ => 0.0).toList
    val lines = Source.fromFile(filePath).getLines.toList
    lines.map { line =>
      val split = line.split(",")
      val labelVector = labelTemplate.updated(split.head.toInt, 1.0).toArray
      val label = DenseMatrix.create[Double](10, 1, labelVector)
      val featureVector = split.slice(1, split.size).map(_.toDouble)
      val feature = DenseMatrix.create[Double](784, 1, featureVector)
      feature -> label
    }
  }

  logger.info("Loading test dataset...")
  val test = loadMnist("scripts/mnist_test.csv")

  logger.info("Loading train dataset...")
  val training = loadMnist("scripts/mnist_train.csv")

  logger.info("Starting training...")
  val mlp = new MultiLayerPerceptron(784, 32, 32, 10)
  mlp.sgdTraining(training, epochs = 200, batchSize = 32, learningRate = 0.1, debugFrequency = 1, testData = Option(test))
  logger.info(s"evaluate ${mlp.evaluate(training.take(10000))} / ${10000}")
}
