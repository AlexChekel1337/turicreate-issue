import turicreate as turi

turi.config.set_num_gpus(1)

url = "dataset/"
data = turi.image_analysis.load_images(url)
data["objectType"] = data["path"].apply(lambda path: "Car" if "car" in path else "NotACar")
data.save("car_or_phone.sframe")
data.explore()

dataBuffer = turi.SFrame("car_or_phone.sframe")
trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)
model = turi.image_classifier.create(trainingBuffers, target="objectType", model="squeezenet_v1.1", max_iterations=2000)
evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]
model.save("car_or_phone.model")
model.export_coreml("CarPhoneClassifier.mlmodel")
