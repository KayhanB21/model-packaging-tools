# MNIST Classifier in Tensorflow-Keras

- last updated on: Dec 26, 2021

# ONNX export for tensorflow saved model format

python -m tf2onnx.convert --saved-model ./model/epoch_checkpoint --opset 10 --output ./model/epoch_checkpoint.onnx

# Import and export the environment

- recommended: use pipenv for handling the environment: Pipfile

- to export the environment: pipenv lock -r > requirements.txt
