import turicreate as turi

turi.config.set_num_gpus(1)

print("data creation started")

url = "dataset/"
data = turi.image_analysis.load_images(url)
data["objectType"] = data["path"].apply(lambda path: "Car" if "car" in path else "NotACar")
data.save("car_classifier.sframe")
data.explore()

print("data creation finished")
print("model creation started")

dataBuffer = turi.SFrame("car_classifier.sframe")
trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)
model = turi.image_classifier.create(trainingBuffers, target="objectType", model="resnet-50", max_iterations=2000)
evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]
model.save("car_classifier.model")
model.export_coreml("CarClassifier.mlmodel")

print("model creation finished")