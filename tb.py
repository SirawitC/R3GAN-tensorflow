import subprocess

# First, start TensorBoard normally (localhost only)
tb_process = subprocess.Popen([
    'tensorboard', 
    '--logdir', 'runs/gan_training_1754454892', 
    '--port', '6060'
])

print("TensorBoard + ngrok started!")
print("Check ngrok terminal for public URL")