from skimage.feature import hessian_matrix,hessian_matrix_eigvals
from sklearn.metrics import confusion_matrix
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
from sklearn.metrics import accuracy_score
import argparse
import cv2

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
    hxx,hxy,hyy = hessian_matrix(gray, sigma=5)
    i1,i2 = hessian_matrix_eigvals(hxx,hxy,hyy)
    hist=desc.describe(i1)
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
    hxx,hxy,hyy = hessian_matrix(gray,sigma=5)
    i1,i2 = hessian_matrix_eigvals(hxx,hxy,hyy)
    hist=desc.describe(i1)
    prediction = model.predict(hist.reshape(1, -1))
    y_pred.append(str(prediction[0]))
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
print("y_true: {} \n y_pred:{}".format(y_true,y_pred))
print("Accuracy using Ridge filter:{}".format(accuracy_score(y_true,y_pred)))
print("Confusion Matrix:")
print(confusion_matrix(y_true,y_pred))
