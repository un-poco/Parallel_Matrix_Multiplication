# Parallel_Matrix_Multiplication
A team project for NYU graduate CS course Multicore Architecture and Programming.  

The experiment was conducted on the crunchy server. To connect with the server, the following commands need to be executed:
ssh <username>@access.cims.nyu.edu
where username is your Courant username.

Then you will be prompted to enter your Courant password followed by two-factor authentication.

Once successfully authenticated, you are in the Courant network.

As in this project, GPUs are not involved, we consider using crunchy, which are for CPU and memory-intensive processes. 

To ssh to crunchy1@cims.nyu.edu, you can use the command below:
ssh crunchy1

We should also move the local .zip file containing the matrix multiplication code to the CIMS server.
scp Parallel_Matrix_Multiplication.zip <username>@access.cims.nyu.edu:/home/<username>/<workspace>

Once connected to a Crunchy sever and have the Parellel_Matrix_Multiplication.zip unzipped through the unzip command, we can compile the code:
gcc -Wall -std=c99 -fopenmp -o matrixMultiplication main.c matrixOps.c matrixMultiplication.c

Finally, we are good to go by running the ./matrixMultiplication command, with parameters m, n, p, and block size.

Alternatively, we can run the script as follows to execute all commands in a batch.
chmod +x testPerformance.sh
./testPerformance.sh

For small size input matrix experiment, block size experiment and Extremely large input experiment, just need to run ./experiment1/2/3.sh accordingly.

For example:  ./experiment1.sh > experiment1.txt