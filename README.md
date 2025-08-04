# QuantCGW

# What is this?
This is a python program that crates simulated gravital wave data then runs it through a quantum computer simulator to see if it matches the template created. It is intended as a proof of concept to show how resarchers may be able to use quantum computers for continus gravital wave discovery. 

More work will be done on this project soon. The orignal project was done in: https://codeberg.org/AndrewWashburn/QuantCGW



# Instuctions for how to run on Linux Mint:

-To generate results first clone repository using 

'''

git clone ssh://git@codeberg.org/AndrewWashburn/QuantCGW.git

'''


-Enter the directory

'''

cd QuantCGW

'''


-Then install nessasry files

'''

pip install -r requirments.txt

''' 


-Next create a data set

'''

python3 CreateData.py

'''


-Run the quantum simulation using Qiskit and recive data output -  This step will take a while 

'''

python3 Quantum.py

'''
 

# Additional Notes
-I would reccomend using a virtual envirment due to the packages needed and possible problems that may occur.
Use "python3 -m venv .venv" to create an envirment then "source .venv/bin/activate" to activate it

-If you change the template generation prossess be careful of how many templates you create. Generating to many may take up large amounts of disk space

-If you encounter problems with installing qiskit as part of requirments.txt refer to https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit for more guidence





