
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
from sklearn.metrics import accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
	help="path to the tesitng images")
args = vars(ap.parse_args())

desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

for imagePath in paths.list_images(args["training"]):

	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)


	labels.append(imagePath.split("/")[-2])
	data.append(hist)

model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)
y_pred=[]
y_true=[]

for imagePath in paths.list_images(args["testing"]):

	image = cv2.imread(imagePath)
	y_true.append(str(imagePath[15]))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))
	y_pred.append(str(prediction[0]))

	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
print("y_true: {} \n y_pred:{}".format(y_true,y_pred))
print(accuracy_score(y_true,y_pred))
