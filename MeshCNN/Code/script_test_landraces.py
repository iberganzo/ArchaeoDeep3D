import os
import subprocess
import shutil

class1 = 'Bere'
class2 = 'British'
class3 = 'Scandinavian'
class4 = 'Scottish'

print("GENERAL ACCURACY")

subprocess.run(["bash", "./scripts/seeds/test.sh"])

print(f"CLASS 1 ACCURACY: {class1}")

shutil.copytree('datasets/testseeds/', 'datasets/testseeds2/')
shutil.copytree('datasets/testseeds/', 'datasets/testseeds3/')
shutil.copytree('datasets/testseeds/', 'datasets/testseeds4/')
shutil.rmtree(f'datasets/testseeds/{class2}/train/')
os.makedirs(f'datasets/testseeds/{class2}/train/')
shutil.rmtree(f'datasets/testseeds/{class3}/train/')
os.makedirs(f'datasets/testseeds/{class3}/train/')
shutil.rmtree(f'datasets/testseeds/{class4}/train/')
os.makedirs(f'datasets/testseeds/{class4}/train/')

subprocess.run(["bash", "./scripts/seeds/test.sh"])

print(f"CLASS 2 ACCURACY: {class2}")

shutil.rmtree('datasets/testseeds/')
shutil.move('datasets/testseeds2/', 'datasets/testseeds/')
shutil.rmtree(f'datasets/testseeds/{class1}/train/')
os.makedirs(f'datasets/testseeds/{class1}/train/')
shutil.rmtree(f'datasets/testseeds/{class3}/train/')
os.makedirs(f'datasets/testseeds/{class3}/train/')
shutil.rmtree(f'datasets/testseeds/{class4}/train/')
os.makedirs(f'datasets/testseeds/{class4}/train/')

subprocess.run(["bash", "./scripts/seeds/test.sh"])

print(f"CLASS 3 ACCURACY: {class3}")

shutil.rmtree('datasets/testseeds/')
shutil.move('datasets/testseeds3/', 'datasets/testseeds/')
shutil.rmtree(f'datasets/testseeds/{class1}/train/')
os.makedirs(f'datasets/testseeds/{class1}/train/')
shutil.rmtree(f'datasets/testseeds/{class2}/train/')
os.makedirs(f'datasets/testseeds/{class2}/train/')
shutil.rmtree(f'datasets/testseeds/{class4}/train/')
os.makedirs(f'datasets/testseeds/{class4}/train/')

subprocess.run(["bash", "./scripts/seeds/test.sh"])

print(f"CLASS 4 ACCURACY: {class4}")

shutil.rmtree('datasets/testseeds/')
shutil.move('datasets/testseeds4/', 'datasets/testseeds/')
shutil.rmtree(f'datasets/testseeds/{class1}/train/')
os.makedirs(f'datasets/testseeds/{class1}/train/')
shutil.rmtree(f'datasets/testseeds/{class2}/train/')
os.makedirs(f'datasets/testseeds/{class2}/train/')
shutil.rmtree(f'datasets/testseeds/{class3}/train/')
os.makedirs(f'datasets/testseeds/{class3}/train/')

subprocess.run(["bash", "./scripts/seeds/test.sh"])

shutil.rmtree('datasets/testseeds/')
os.makedirs('datasets/testseeds/')
shutil.rmtree('checkpoints/testweights/')
os.makedirs('checkpoints/testweights/')

print("Evaluation done")
