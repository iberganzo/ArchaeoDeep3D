import os
import subprocess
import shutil

class1 = 'Bere'
class2 = 'NoBere'

print("GENERAL ACCURACY")

subprocess.run(["bash", "./scripts/seeds/test.sh"])

print(f"CLASS 1 ACCURACY: {class1}")

shutil.copytree('datasets/testseeds/', 'datasets/testseeds2/')
shutil.rmtree(f'datasets/testseeds/{class2}/train/')
os.makedirs(f'datasets/testseeds/{class2}/train/')

subprocess.run(["bash", "./scripts/seeds/test.sh"])

print(f"CLASS 2 ACCURACY: {class2}")

shutil.rmtree('datasets/testseeds/')
shutil.move('datasets/testseeds2/', 'datasets/testseeds/')
shutil.rmtree(f'datasets/testseeds/{class1}/train/')
os.makedirs(f'datasets/testseeds/{class1}/train/')

subprocess.run(["bash", "./scripts/seeds/test.sh"])

shutil.rmtree('datasets/testseeds/')
os.makedirs('datasets/testseeds/')
shutil.rmtree('checkpoints/testweights/')
os.makedirs('checkpoints/testweights/')

print("Evaluation done")
