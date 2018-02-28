import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD

val spam = sc.textFile("spam.txt") 
val ham = sc.textFile("ham.txt")

/*HashingTF is a Transformer which takes sets of terms and converts those sets into fixed-length feature vectors. 
In text processing, a “set of terms” might be a bag of words. HashingTF utilizes the hashing trick. A raw feature is 
mapped into an index (term) by applying a hash function. The hash function used here is MurmurHash 3. 
Then term frequencies are calculated based on the mapped indices. This approach avoids the need to compute a 
global term-to-index map, which can be expensive for a large corpus, but it suffers from potential hash collisions, 
where different raw features may become the same term after hashing.*/

//Creating a HashingTF instance to convert the text data into vectors 
val tf = new HashingTF(numFeatures = 10000)

//Each line in the text file is split into words and each word is mapped to features
val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
val hamFeatures = ham.map(email => tf.transform(email.split(" ")))

/*A labeled point is a local vector, either dense or sparse, associated with a label/response. 
In MLlib, labeled points are used in supervised learning algorithms */

//Create labeledpoints datasets for spam and ham examples
val spamdata = spamFeatures.map(features => LabeledPoint(1, features))
val hamdata = hamFeatures.map(features => LabeledPoint(1, features))

val trainingData = spamdata.union(hamdata)

//Since LogisticRegression is an iterative algorithm, we need to persist the data 
trainingData.cache() 

//Run LogisticRegression using the SGD algorithm
val model = new LogisticRegressionWithSGD().run(trainingData)


