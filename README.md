# NeuralNetwork

---

## Cat dataset

```text
Cat dataset loaded successfully
X_train_cat: (12288, 209)
y_train_cat: (1, 209)
X_test_cat: (12288, 50)
y_test_cat: (1, 50)
```

---

## Mnist dataset

```text
Mnist dataset loaded successfully
X_train_mnist: (784, 5000)
y_train_mnist: (10, 5000)
X_test_mnist: (784, 1000)
y_test_mnist: (10, 1000)
```

---

## Sequential cat dataset test

```text
Epoch: 0   || Train loss: 0.675588 || Test loss: 0.799319 || Train accuracy: 64.1148% || Test accuracy: 34% || Time: 1.34004 sec
Epoch: 100 || Train loss: 0.435139 || Test loss: 0.532536 || Train accuracy: 73.6842% || Test accuracy: 78% || Time: 117.136 sec
...
Epoch: 700 || Train loss: 0.053182 || Test loss: 0.669524 || Train accuracy: 100% || Test accuracy: 74% || Time: 805.184 sec

Sequential cat training time: 805.236 seconds

Truth: 1 || Pred: 0
```

---

## OpenMP cat dataset test

```text
Epoch: 0   || Train loss: 0.758618 || Test loss: 1.43023 || Train accuracy: 38.756% || Test accuracy: 34% || Time: 0.0594862 sec
Epoch: 100 || Train loss: 0.45634  || Test loss: 0.831209 || Train accuracy: 87.0813% || Test accuracy: 38% || Time: 6.83982 sec
...
Epoch: 700 || Train loss: 0.0591217 || Test loss: 0.686301 || Train accuracy: 99.5215% || Test accuracy: 72% || Time: 48.8558 sec

OpenMP cat training time: 48.9013 seconds

Truth: 1 || Pred: 0
```

---

## CUDA cat dataset test

```text
Epoch: 0   || Train loss: 0.73747 || Test loss: 0.734932 || Train accuracy: 65.5502% || Test accuracy: 42% || Time: 0.00626 sec
Epoch: 100 || Train loss: 0.492689 || Test loss: 0.533569 || Train accuracy: 70.8134% || Test accuracy: 76% || Time: 0.657742 sec
...
Epoch: 700 || Train loss: 0.0900744 || Test loss: 0.526929 || Train accuracy: 99.5215% || Test accuracy: 82% || Time: 3.70941 sec

CUDA cat training time: 3.91586 seconds

Truth: 1 || Pred: 0
```

---

## Sequential mnist dataset test

```text
Epoch: 0   || Train loss: 2.37501 || Test loss: 2.38058 || Train accuracy: 8.32% || Test accuracy: 8.5% || Time: 0.856481 sec
Epoch: 100 || Train loss: 1.50541 || Test loss: 1.58992 || Train accuracy: 65.52% || Test accuracy: 60.8% || Time: 78.1534 sec
...
Epoch: 700 || Train loss: 0.31902 || Test loss: 0.437212 || Train accuracy: 91.3% || Test accuracy: 86.2% || Time: 527.193 sec

Sequential mnist training time: 527.254 seconds

Truth: 9 || Pred: 9
```
```text
      .#
    .###
    ####.
   .######.
   ########
  .##  .####
  .#.   ####
  .#.   ###.
  .## ..####
   ###### ##.
    ###.  .##
     ..    .#.
           .##
            .#.
             ##
             .#.
              ##
               #.
               ##
                #.
                ##
                 #.
                 ##
                  #.
                  ##
                   #.
                   ##
                    #.
```

---

## OpenMP mnist dataset test

```text
Epoch: 0   || Train loss: 2.38729 || Test loss: 2.35986 || Train accuracy: 10.8% || Test accuracy: 11.4% || Time: 0.0668349 sec
Epoch: 100 || Train loss: 1.42917 || Test loss: 1.50612 || Train accuracy: 68.08% || Test accuracy: 62.8% || Time: 8.19211 sec
...
Epoch: 700 || Train loss: 0.31623 || Test loss: 0.43083 || Train accuracy: 91.38% || Test accuracy: 87.6% || Time: 58.7461 sec

OpenMP mnist training time: 58.759 seconds

Truth: 9 || Pred: 9
```
```text
      .#
    .###
    ####.
   .######.
   ########
  .##  .####
  .#.   ####
  .#.   ###.
  .## ..####
   ###### ##.
    ###.  .##
     ..    .#.
           .##
            .#.
             ##
             .#.
              ##
               #.
               ##
                #.
                ##
                 #.
                 ##
                  #.
                  ##
                   #.
                   ##
                    #.
```

---

## CUDA mnist dataset test

```text
Epoch: 0   || Train loss: 2.93916 || Test loss: 2.54614 || Train accuracy: 9.96% || Test accuracy: 12.1% || Time: 0.0771189 sec
Epoch: 100 || Train loss: 1.08881 || Test loss: 1.16974 || Train accuracy: 76.02% || Test accuracy: 71.3% || Time: 1.37293 sec
...
Epoch: 700 || Train loss: 0.313933 || Test loss: 0.433237 || Train accuracy: 91.38% || Test accuracy: 87.5% || Time: 7.8377 sec

CUDA mnist training time: 7.85793 seconds

Truth: 9 || Pred: 4
```
```text
      .#
    .###
    ####.
   .######.
   ########
  .##  .####
  .#.   ####
  .#.   ###.
  .## ..####
   ###### ##.
    ###.  .##
     ..    .#.
           .##
            .#.
             ##
             .#.
              ##
               #.
               ##
                #.
                ##
                 #.
                 ##
                  #.
                  ##
                   #.
                   ##
                    #.
```

---

## Program Exit

```text
C:\Users\ASUS\source\repos\NeuralNetwork\x64\Release\NeuralNetwork.exe (process 14772) exited with code 0 (0x0).
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
```
